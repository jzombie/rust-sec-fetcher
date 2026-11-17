//! Convert raw 10-K filings to itemized ZSTD-compressed Parquet.
//!
//! Rust companion to `tenk-raw-to-parquet`: walks the source tree
//! (`data/tenk_items/<TICKER>/*`), runs the 10-K section parser on each
//! filing, and writes one row per filing with
//! `(ticker_symbol, accession_number, item_1, item_7)`. Item columns hold
//! the full extracted section text, or null when the section is absent or
//! the document fails to parse. No adequacy filtering happens here;
//! consumers join on `accession_number` against
//! `data/tenk_audit_periods.csv` and filter
//! `is_canonical=true AND is_adequate=true` at scoring time.
//!
//! Parsing is CPU-bound (`html2text` rendering per filing), so filings are
//! parsed in bounded parallel chunks across all cores while writes stay
//! single-threaded in sorted order. Rows accumulate into 64 MiB row groups
//! before flushing; when the current output exceeds `--target-bytes`, the
//! writer is closed and a new `batch_NNNN.parquet` is started under
//! `hf_upload_folder/tenk_itemized/`. Shards are staged as `.tmp` files
//! and atomically renamed on close, and stale `batch_*.parquet` shards are
//! purged before writing.
//!
//! Peak RAM: one chunk of raw filings plus parsed sections, plus the
//! 64 MiB row-group buffer.
//!
//! ```sh
//! cargo run --release --bin tenk-itemized-to-parquet
//! ```

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use clap::{CommandFactory, FromArgMatches, Parser};
use rayon::prelude::*;
use sec_fetcher::parsers::extract_sections_from_document;
use sec_fetcher::utils::tenk_parquet::{
    DEFAULT_ROW_GROUP_TARGET_BYTES, DEFAULT_SHARD_TARGET_BYTES, ShardedParquetWriter,
    collect_tenk_filing_files, human, purge_stale_shards,
};

/// Filings parsed per parallel chunk: bounds peak RAM (raw text plus
/// extracted sections for 32 large filings) while keeping all cores busy.
const PARSE_CHUNK_SIZE: usize = 32;

/// Worker thread stack for the dedicated parse pool. `extract_sections_from_document`
/// runs html5ever DOM construction plus html2text layout recursion, which
/// assumes a main-thread-sized stack budget (see `LARGE_FILE_BYTES` in
/// `parse_10k_sections`). Rayon worker threads default to small OS stacks
/// that overflow on deeply nested filings, aborting the process, so parsing
/// runs on this explicit budget instead.
const WORKER_STACK_BYTES: usize = 8 * 1024 * 1024;

/// One parsed filing: ticker, accession, Item 1, Item 7, and whether the
/// document failed to parse (distinct from genuinely missing sections).
type ParsedFiling = (String, String, Option<String>, Option<String>, bool);

#[derive(Parser)]
#[command(
    name = "tenk-itemized-to-parquet",
    about = "Convert raw 10-K filings to itemized (Item 1 and Item 7) Parquet"
)]
struct Args {
    /// Source tree. Defaults to `data/tenk_items`.
    #[arg(long, short = 's')]
    source: Option<PathBuf>,

    /// Destination directory for `batch_NNNN.parquet`. Defaults to `hf_upload_folder/tenk_itemized`.
    #[arg(long, short = 'd')]
    dest: Option<PathBuf>,

    /// Start a new output file once the current one reaches this many bytes.
    #[arg(long, default_value_t = DEFAULT_SHARD_TARGET_BYTES)]
    target_bytes: u64,
}

/// Extract full Item 1 and Item 7 text from one filing document.
///
/// Returns `(item_1, item_7, failed)` with `None` for any section that is
/// absent. `failed` is true only when the document fails to parse, so
/// callers can count and report parser failures separately from genuinely
/// missing sections.
fn extract_item_texts(raw: &str) -> (Option<String>, Option<String>, bool) {
    match extract_sections_from_document(raw) {
        Ok(sections) => (
            sections.get("item_1").map(str::to_string),
            sections.get("item_7").map(str::to_string),
            false,
        ),
        Err(_) => (None, None, true),
    }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("ticker_symbol", DataType::Utf8, false),
        Field::new("accession_number", DataType::Utf8, false),
        Field::new("item_1", DataType::Utf8, true),
        Field::new("item_7", DataType::Utf8, true),
    ]))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::from_arg_matches(
        &Args::command()
            .mut_arg("source", |a| {
                a.help(format!(
                    "Source tree. Defaults to `{}`.",
                    sec_fetcher_shared::resolve_tenk_items_dir(None)
                ))
            })
            .mut_arg("dest", |a| {
                a.help(format!(
                    "Destination directory for `batch_NNNN.parquet`. Defaults to `{}`.",
                    sec_fetcher_shared::resolve_tenk_itemized_parquet_dir(None)
                ))
            })
            .get_matches(),
    )?;
    let source = args
        .source
        .unwrap_or_else(|| PathBuf::from(sec_fetcher_shared::resolve_tenk_items_dir(None)));
    let dest = args.dest.unwrap_or_else(|| {
        PathBuf::from(sec_fetcher_shared::resolve_tenk_itemized_parquet_dir(None))
    });
    fs::create_dir_all(&dest)?;

    if !source.is_dir() {
        return Err(format!("Source directory not found: {}", source.display()).into());
    }

    purge_stale_shards(&dest)?;

    print!("Counting files … ");
    use std::io::Write as _;
    std::io::stdout().flush()?;
    let files = collect_tenk_filing_files(&source)?;
    let total = files.len();
    println!("found {total} file(s)\n");

    let start = std::time::Instant::now();
    let schema = schema();
    let mut sharded = ShardedParquetWriter::new(
        dest.clone(),
        Arc::clone(&schema),
        args.target_bytes,
        DEFAULT_ROW_GROUP_TARGET_BYTES,
    );
    let mut done = 0usize;
    let mut parse_errors = 0usize;

    let pool = rayon::ThreadPoolBuilder::new()
        .stack_size(WORKER_STACK_BYTES)
        .build()?;

    for chunk in files.chunks(PARSE_CHUNK_SIZE) {
        let parsed: Vec<ParsedFiling> = pool.install(|| {
            chunk
                .par_iter()
                .map(|(ticker, accession, path)| -> Result<_, std::io::Error> {
                    let bytes = fs::read(path)?;
                    let text = match String::from_utf8(bytes) {
                        Ok(s) => s,
                        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
                    };
                    let (item_1, item_7, failed) = extract_item_texts(&text);
                    Ok((ticker.clone(), accession.clone(), item_1, item_7, failed))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        for (ticker, accession, item_1, item_7, failed) in parsed {
            if failed {
                parse_errors += 1;
                eprintln!("WARN: parse error for {ticker}/{accession}");
            }
            let item_bytes =
                item_1.as_ref().map_or(0, String::len) + item_7.as_ref().map_or(0, String::len);
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(StringArray::from(vec![ticker.as_str()])),
                    Arc::new(StringArray::from(vec![accession.as_str()])),
                    Arc::new(StringArray::from(vec![item_1.as_deref()])),
                    Arc::new(StringArray::from(vec![item_7.as_deref()])),
                ],
            )?;
            sharded.write_batch(&batch, item_bytes)?;
            done += 1;
        }
        let pct = done as f64 / total as f64 * 100.0;
        print!(
            "\r  [{done:>6}/{total}]  output {:>4}  ({pct:>5.1} %){}",
            sharded.shard_index(),
            " ".repeat(20)
        );
        std::io::stdout().flush()?;
    }
    let output_idx = sharded.finish()?;

    println!(
        "\n\nDone — {done} file(s) → {}  in {:.1} s",
        dest.display(),
        start.elapsed().as_secs_f64()
    );
    println!("       {output_idx} Parquet file(s) created");
    if parse_errors > 0 {
        println!("       {parse_errors} parse error(s): sections written as null");
    }
    let mut total_bytes = 0u64;
    let mut names: Vec<String> = Vec::new();
    for entry in fs::read_dir(&dest)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".parquet") {
            names.push(name);
        }
    }
    names.sort();
    for name in names {
        let size = fs::metadata(dest.join(&name))?.len();
        total_bytes += size;
        println!("         {name}:  {}", human(size));
    }
    println!("         total: {}", human(total_bytes));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_has_accession_and_nullable_items() {
        let schema = schema();
        assert_eq!(
            schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            vec!["ticker_symbol", "accession_number", "item_1", "item_7"]
        );
        assert!(!schema.field(0).is_nullable());
        assert!(!schema.field(1).is_nullable());
        assert!(schema.field(2).is_nullable());
        assert!(schema.field(3).is_nullable());
    }

    #[test]
    fn missing_sections_become_nulls_without_failure_flag() {
        let (item_1, item_7, failed) = extract_item_texts("no items here");
        assert!(item_1.is_none());
        assert!(item_7.is_none());
        assert!(!failed);
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(StringArray::from(vec!["TICK"])),
                Arc::new(StringArray::from(vec!["0001-01-000001"])),
                Arc::new(StringArray::from(vec![item_1.as_deref()])),
                Arc::new(StringArray::from(vec![item_7.as_deref()])),
            ],
        )
        .unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.column(2).null_count(), 1);
        assert_eq!(batch.column(3).null_count(), 1);
    }

    #[test]
    fn present_sections_return_text_without_failure_flag() {
        let raw = format!(
            "<p>ITEM 1. BUSINESS</p><p>{}</p><p>ITEM 7. MANAGEMENT'S DISCUSSION</p><p>{}</p><p>ITEM 8. FINANCIAL STATEMENTS</p><p>{}</p>",
            "Business. ".repeat(50),
            "MD&A. ".repeat(300),
            "Financials. ".repeat(50),
        );
        let (item_1, item_7, failed) = extract_item_texts(&raw);
        assert!(!failed);
        assert!(item_1.is_some());
        assert!(item_7.is_some());
        assert!(item_7.unwrap().len() >= item_1.unwrap().len());
    }
}
