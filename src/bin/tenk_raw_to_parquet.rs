//! Convert raw 10-K audit files to ZSTD-compressed Parquet.
//!
//! Rust port of `scripts/tenk_raw_to_parquet.py`: walks the source tree
//! (`data/tenk_items/<TICKER>/*`), one row per filing with
//! `(ticker_symbol, accession_number, raw_text)`. Rows accumulate into
//! 64 MiB row groups before flushing; when the current output exceeds
//! `--target-bytes`, the writer is closed and a new `batch_NNNN.parquet`
//! is started under `hf_upload_folder/tenk_raw/`, leaving the staging root
//! free for other datasets. Shards are staged as `.tmp` files and
//! atomically renamed on close, and stale `batch_*.parquet` shards are
//! purged before writing.
//!
//! Peak RAM: row-group buffer (64 MiB of text) plus the largest single file.
//!
//! ```sh
//! cargo run --release --bin tenk-raw-to-parquet
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use clap::{CommandFactory, FromArgMatches, Parser};
use sec_fetcher::utils::tenk_parquet::{
    DEFAULT_ROW_GROUP_TARGET_BYTES, DEFAULT_SHARD_TARGET_BYTES, ShardedParquetWriter,
    collect_tenk_filing_files, human, purge_stale_shards,
};

#[derive(Parser)]
#[command(
    name = "tenk-raw-to-parquet",
    about = "Convert raw 10-K files to ZSTD Parquet"
)]
struct Args {
    /// Source tree. Defaults to `data/tenk_items`.
    #[arg(long, short = 's')]
    source: Option<PathBuf>,

    /// Destination directory for `batch_NNNN.parquet`. Defaults to `hf_upload_folder/tenk_raw`.
    #[arg(long, short = 'd')]
    dest: Option<PathBuf>,

    /// Start a new output file once the current one reaches this many bytes.
    #[arg(long, default_value_t = DEFAULT_SHARD_TARGET_BYTES)]
    target_bytes: u64,
}

/// Build a field carrying a `description` entry in its Arrow metadata.
/// The `parquet` crate's `ArrowWriter` persists the Arrow schema (metadata
/// included) in the file footer, so `pyarrow` / HF `datasets` surface these
/// as `field.metadata["description"]` (see
/// [`sec_fetcher_shared::PARQUET_FIELD_METADATA_DESCRIPTION`]).
fn doc_field(
    name: impl Into<String>,
    data_type: DataType,
    nullable: bool,
    description: &str,
) -> Field {
    Field::new(name.into(), data_type, nullable).with_metadata(HashMap::from([(
        sec_fetcher_shared::PARQUET_FIELD_METADATA_DESCRIPTION.to_string(),
        description.to_string(),
    )]))
}

fn schema() -> Arc<Schema> {
    use sec_fetcher_shared::{
        TENK_AUDIT_COL_ACCESSION, TENK_PARQUET_COL_RAW_TEXT, TENK_PARQUET_COL_TICKER_SYMBOL,
    };
    Arc::new(Schema::new(vec![
        doc_field(
            TENK_PARQUET_COL_TICKER_SYMBOL,
            DataType::Utf8,
            false,
            "Directory ticker the filing was pulled under (sanitized on disk, e.g. BF_A for BF/A).",
        ),
        doc_field(
            TENK_AUDIT_COL_ACCESSION,
            DataType::Utf8,
            false,
            "EDGAR accession of the filing, canonical dashed form.",
        ),
        doc_field(
            TENK_PARQUET_COL_RAW_TEXT,
            DataType::Utf8,
            false,
            "Full filing document text, lossy UTF-8 decoded.",
        ),
    ]))
}

/// `errors="replace"` equivalent: lossy UTF-8 decode, reusing the read
/// buffer when the file is already valid UTF-8.
fn decode_lossy(bytes: Vec<u8>) -> String {
    match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    }
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
                    sec_fetcher_shared::resolve_tenk_raw_parquet_dir(None)
                ))
            })
            .get_matches(),
    )?;
    let source = args
        .source
        .unwrap_or_else(|| PathBuf::from(sec_fetcher_shared::resolve_tenk_items_dir(None)));
    let dest = args
        .dest
        .unwrap_or_else(|| PathBuf::from(sec_fetcher_shared::resolve_tenk_raw_parquet_dir(None)));
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

    for (ticker, accession, path) in &files {
        let bytes = fs::read(path)?;
        let text = decode_lossy(bytes);

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec![ticker.as_str()])),
                Arc::new(StringArray::from(vec![accession.as_str()])),
                Arc::new(StringArray::from(vec![text.as_str()])),
            ],
        )?;
        sharded.write_batch(&batch, text.len())?;
        done += 1;

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
    fn schema_has_accession_and_non_null_text() {
        let schema = schema();
        assert_eq!(
            schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            sec_fetcher_shared::TENK_RAW_PARQUET_COLUMNS
        );
        assert!(!schema.field(0).is_nullable());
        assert!(!schema.field(1).is_nullable());
        assert!(!schema.field(2).is_nullable());
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["TICK"])),
                Arc::new(StringArray::from(vec!["0001-01-000001"])),
                Arc::new(StringArray::from(vec!["filing text"])),
            ],
        )
        .unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.column(2).null_count(), 0);
    }

    #[test]
    fn every_field_carries_a_description() {
        for f in schema().fields() {
            let desc = f
                .metadata()
                .get(sec_fetcher_shared::PARQUET_FIELD_METADATA_DESCRIPTION);
            assert!(
                desc.is_some_and(|d| !d.trim().is_empty()),
                "field `{}` is missing description metadata",
                f.name()
            );
        }
    }

    #[test]
    fn decode_lossy_replaces_invalid_bytes() {
        assert_eq!(decode_lossy(b"plain".to_vec()), "plain");
        let text = decode_lossy(vec![0x68, 0x69, 0xff, 0x21]);
        assert!(text.starts_with("hi"));
        assert!(text.contains('�'));
    }
}
