//! Convert raw 10-K filings to itemized ZSTD-compressed Parquet.
//!
//! Rust companion to `tenk-raw-to-parquet`: walks the source tree
//! (`data/tenk_items/<TICKER>/*`), runs the 10-K section parser on each
//! filing, and writes one row per filing with `(ticker_symbol,
//! accession_number, filed_date, form_type, fy, fp, is_adequate,
//! is_canonical, canonical_accession, item_1, item_7)`. Item columns hold
//! the full extracted section text, or null when the section is absent or
//! the document fails to parse. Annotation columns are joined from the
//! frozen `data/tenk_audit_periods.csv` (keyed by `(ticker_symbol,
//! normalized accession)` so duplicate accessions shared across ticker
//! directories keep independent flags) so
//! the dataset is directly filterable with `WHERE is_canonical AND
//! is_adequate` — no sidecar CSV needed. Filings missing from the periods
//! CSV get blank annotations and `false` flags, and are counted in the
//! completion summary.
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

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{BooleanArray, RecordBatch, StringArray};
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

/// Audit/period annotation for one filing, joined from the frozen periods
/// CSV. Blank strings in the CSV become `None`; flags default to `false`
/// when the filing has no periods row.
#[derive(Clone, Default)]
struct FilingAnnotation {
    filed_date: Option<String>,
    form_type: Option<String>,
    fy: Option<String>,
    fp: Option<String>,
    is_adequate: bool,
    is_canonical: bool,
    canonical_accession: Option<String>,
}

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

    /// Periods CSV to join annotations from. Defaults to `data/tenk_audit_periods.csv`.
    #[arg(long, short = 'p')]
    periods: Option<PathBuf>,

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
    use sec_fetcher_shared::{TENK_PARQUET_COL_ITEM_1, TENK_PARQUET_COL_ITEM_7};
    match extract_sections_from_document(raw) {
        Ok(sections) => (
            sections.get(TENK_PARQUET_COL_ITEM_1).map(str::to_string),
            sections.get(TENK_PARQUET_COL_ITEM_7).map(str::to_string),
            false,
        ),
        Err(_) => (None, None, true),
    }
}

/// `errors="replace"` equivalent: lossy UTF-8 decode, reusing the read
/// buffer when the file is already valid UTF-8.
fn decode_lossy(bytes: Vec<u8>) -> String {
    match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    }
}

/// Load per-filing annotations from the periods CSV, keyed by
/// `(ticker, normalized accession)`. The ticker component is required:
/// duplicate accessions exist across ticker directories (e.g. dual-class
/// shares sharing a CIK and accession), and `tenk-period-join` assigns
/// independent `is_canonical` flags per `(ticker, accession)` row —
/// keying by accession alone would overwrite one row's flag with the
/// other's. Accession normalization is dash/padding/case-insensitive via
/// `sec_fetcher_shared::normalize_accession`. A missing file yields an
/// empty map (with a warning) so the export still runs with blank
/// annotations; a present file with missing columns is an error.
/// Malformed CSV records propagate as errors rather than being skipped.
///
/// In practice `main` refuses to reach this with a missing or stale file
/// (see [`check_periods_freshness`]); this fallback only serves direct
/// callers and tests.
fn load_annotations(
    path: &Path,
) -> Result<HashMap<(String, String), FilingAnnotation>, Box<dyn std::error::Error>> {
    let mut map = HashMap::new();
    let Ok(mut rdr) = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
    else {
        eprintln!(
            "WARN: periods CSV not found at {} — writing blank annotations",
            path.display()
        );
        return Ok(map);
    };
    let headers = rdr.headers()?.clone();
    let find = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("periods CSV is missing column `{name}`"))
    };
    let (
        ticker_idx,
        acc_idx,
        filed_idx,
        form_idx,
        fy_idx,
        fp_idx,
        adeq_idx,
        canon_idx,
        canon_acc_idx,
    ) = (
        find(sec_fetcher_shared::TENK_AUDIT_COL_TICKER)?,
        find(sec_fetcher_shared::TENK_AUDIT_COL_ACCESSION)?,
        find(sec_fetcher_shared::TENK_AUDIT_COL_FILED_DATE)?,
        find(sec_fetcher_shared::TENK_AUDIT_COL_FORM_TYPE)?,
        find(sec_fetcher_shared::TENK_PERIOD_COL_FY)?,
        find(sec_fetcher_shared::TENK_PERIOD_COL_FP)?,
        find(sec_fetcher_shared::TENK_AUDIT_COL_IS_ADEQUATE)?,
        find(sec_fetcher_shared::TENK_PERIOD_COL_IS_CANONICAL)?,
        find(sec_fetcher_shared::TENK_PERIOD_COL_CANONICAL_ACCESSION)?,
    );
    for result in rdr.records() {
        let rec = result?;
        let get = |idx: usize| rec.get(idx).unwrap_or_default();
        let nonempty = |idx: usize| {
            let s = get(idx).trim();
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        };
        map.insert(
            (
                get(ticker_idx).trim().to_string(),
                sec_fetcher_shared::normalize_accession(get(acc_idx)),
            ),
            FilingAnnotation {
                filed_date: nonempty(filed_idx),
                form_type: nonempty(form_idx),
                fy: nonempty(fy_idx),
                fp: nonempty(fp_idx),
                is_adequate: get(adeq_idx).eq_ignore_ascii_case("true"),
                is_canonical: get(canon_idx).eq_ignore_ascii_case("true"),
                canonical_accession: nonempty(canon_acc_idx),
            },
        );
    }
    Ok(map)
}

/// Refuse to export when the frozen join the flags embed is missing or
/// older than the audit it was built from. Embedding a stale join would
/// silently ship wrong `is_canonical`/`is_adequate` flags to Hugging Face;
/// re-running the seconds-long join is always cheaper than the hours-long
/// export, so this fails fast with the exact remediation command.
fn check_periods_freshness(periods: &Path, audit: &Path) -> Result<(), String> {
    if !periods.is_file() {
        return Err(format!(
            "periods CSV not found at {} — run `cargo run -p sec-fetcher-tenk-parser-audit --bin tenk-period-join --release` first",
            periods.display()
        ));
    }
    let (Ok(periods_modified), Ok(audit_modified)) = (
        fs::metadata(periods).and_then(|m| m.modified()),
        fs::metadata(audit).and_then(|m| m.modified()),
    ) else {
        return Ok(());
    };
    if audit_modified > periods_modified {
        return Err(format!(
            "periods CSV at {} is older than audit CSV at {} — re-run `cargo run -p sec-fetcher-tenk-parser-audit --bin tenk-period-join --release` first",
            periods.display(),
            audit.display()
        ));
    }
    Ok(())
}

fn schema() -> Arc<Schema> {
    use sec_fetcher_shared::{
        TENK_AUDIT_COL_ACCESSION, TENK_AUDIT_COL_FILED_DATE, TENK_AUDIT_COL_FORM_TYPE,
        TENK_AUDIT_COL_IS_ADEQUATE, TENK_PARQUET_COL_ITEM_1, TENK_PARQUET_COL_ITEM_7,
        TENK_PARQUET_COL_TICKER_SYMBOL, TENK_PERIOD_COL_CANONICAL_ACCESSION, TENK_PERIOD_COL_FP,
        TENK_PERIOD_COL_FY, TENK_PERIOD_COL_IS_CANONICAL,
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
            TENK_AUDIT_COL_FILED_DATE,
            DataType::Utf8,
            true,
            "Filing date from the audit manifest (YYYY-MM-DD); null when the periods CSV has no row.",
        ),
        doc_field(
            TENK_AUDIT_COL_FORM_TYPE,
            DataType::Utf8,
            true,
            "SEC form of the filing (10-K, 10-K/A, ...); null when unannotated.",
        ),
        doc_field(
            TENK_PERIOD_COL_FY,
            DataType::Utf8,
            true,
            "Fiscal year from the fundamentals join; null when unmatched (e.g. pre-XBRL).",
        ),
        doc_field(
            TENK_PERIOD_COL_FP,
            DataType::Utf8,
            true,
            "Fiscal period slot (FY, Q1-Q3); year-end Q4 normalized to FY; null when unmatched.",
        ),
        doc_field(
            TENK_AUDIT_COL_IS_ADEQUATE,
            DataType::Boolean,
            false,
            "True when item_1_chars >= 400 and item_7_chars >= 2000 on the true 10-K text.",
        ),
        doc_field(
            TENK_PERIOD_COL_IS_CANONICAL,
            DataType::Boolean,
            false,
            "True for the latest adequate row per (company, fy) group; the scored representative.",
        ),
        doc_field(
            TENK_PERIOD_COL_CANONICAL_ACCESSION,
            DataType::Utf8,
            true,
            "Accession of the canonical row for the period; equals own accession when canonical.",
        ),
        doc_field(
            TENK_PARQUET_COL_ITEM_1,
            DataType::Utf8,
            true,
            "Full Item 1 (Business) extracted text; null when absent or unparsable.",
        ),
        doc_field(
            TENK_PARQUET_COL_ITEM_7,
            DataType::Utf8,
            true,
            "Full Item 7 (MD&A) extracted text; null when absent or unparsable.",
        ),
    ]))
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
            .mut_arg("periods", |a| {
                a.help(format!(
                    "Periods CSV to join annotations from. Defaults to `{}`.",
                    sec_fetcher_shared::resolve_tenk_periods_csv(None)
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
    let periods = args
        .periods
        .unwrap_or_else(|| PathBuf::from(sec_fetcher_shared::resolve_tenk_periods_csv(None)));
    fs::create_dir_all(&dest)?;

    if !source.is_dir() {
        return Err(format!("Source directory not found: {}", source.display()).into());
    }

    // Fail fast before purging old shards: a missing or stale join would
    // otherwise embed wrong flags across a hours-long run.
    if let Err(msg) = check_periods_freshness(
        &periods,
        Path::new(&sec_fetcher::parsers::resolve_tenk_audit_csv(None)),
    ) {
        return Err(msg.into());
    }

    purge_stale_shards(&dest)?;

    print!("Loading annotations … ");
    use std::io::Write as _;
    std::io::stdout().flush()?;
    let annotations = load_annotations(&periods)?;
    println!("{} filing(s)", annotations.len());

    print!("Counting files … ");
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
    let mut unannotated = 0usize;

    let pool = rayon::ThreadPoolBuilder::new()
        .stack_size(WORKER_STACK_BYTES)
        .build()?;

    for chunk in files.chunks(PARSE_CHUNK_SIZE) {
        let parsed: Vec<ParsedFiling> = pool.install(|| {
            chunk
                .par_iter()
                .map(|(ticker, accession, path)| -> Result<_, std::io::Error> {
                    let bytes = fs::read(path)?;
                    let text = decode_lossy(bytes);
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
            let key = (
                ticker.trim().to_string(),
                sec_fetcher_shared::normalize_accession(&accession),
            );
            let ann = match annotations.get(&key) {
                Some(a) => a.clone(),
                None => {
                    unannotated += 1;
                    FilingAnnotation::default()
                }
            };
            let item_bytes =
                item_1.as_ref().map_or(0, String::len) + item_7.as_ref().map_or(0, String::len);
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(StringArray::from(vec![ticker.as_str()])),
                    Arc::new(StringArray::from(vec![accession.as_str()])),
                    Arc::new(StringArray::from(vec![ann.filed_date.as_deref()])),
                    Arc::new(StringArray::from(vec![ann.form_type.as_deref()])),
                    Arc::new(StringArray::from(vec![ann.fy.as_deref()])),
                    Arc::new(StringArray::from(vec![ann.fp.as_deref()])),
                    Arc::new(BooleanArray::from(vec![ann.is_adequate])),
                    Arc::new(BooleanArray::from(vec![ann.is_canonical])),
                    Arc::new(StringArray::from(vec![ann.canonical_accession.as_deref()])),
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
    if unannotated > 0 {
        println!("       {unannotated} filing(s) missing from periods CSV: blank annotations");
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
    fn schema_has_annotations_and_nullable_items() {
        // Order is pinned once in shared (`parquet_column_orders_match_wire_format`);
        // here we prove the built schema follows the shared arrays.
        let schema = schema();
        assert_eq!(
            schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            sec_fetcher_shared::TENK_ITEMIZED_PARQUET_COLUMNS
        );
        assert!(!schema.field(0).is_nullable());
        assert!(!schema.field(1).is_nullable());
        assert!(schema.field(2).is_nullable());
        assert!(schema.field(3).is_nullable());
        assert!(schema.field(4).is_nullable());
        assert!(schema.field(5).is_nullable());
        assert!(!schema.field(6).is_nullable());
        assert!(!schema.field(7).is_nullable());
        assert!(schema.field(8).is_nullable());
        assert!(schema.field(9).is_nullable());
        assert!(schema.field(10).is_nullable());
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
    fn missing_sections_become_nulls_without_failure_flag() {
        let (item_1, item_7, failed) = extract_item_texts("no items here");
        assert!(item_1.is_none());
        assert!(item_7.is_none());
        assert!(!failed);
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

    #[test]
    fn decode_lossy_replaces_invalid_bytes() {
        assert_eq!(decode_lossy(b"plain".to_vec()), "plain");
        let text = decode_lossy(vec![0x68, 0x69, 0xff, 0x21]);
        assert!(text.starts_with("hi"));
        assert!(text.contains('�'));
    }

    #[test]
    fn annotations_join_by_ticker_and_normalized_accession() {
        let dir = std::env::temp_dir().join("tenk_itemized_annotations_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("periods.csv");
        fs::write(
            &path,
            "ticker,accession_number,filed_date,form_type,fy,fp,is_adequate,is_canonical,canonical_accession\n\
             AAPL,0001-01-000001,2024-02-02,10-K,2024,FY,true,true,0001-01-000001\n\
             IBM,0002-02-000002,,,,,false,false,\n",
        )
        .unwrap();
        let map = load_annotations(&path).unwrap();
        assert_eq!(map.len(), 2);
        // Undashed accession plus ticker resolves to the dashed row.
        let key = (
            "AAPL".to_string(),
            sec_fetcher_shared::normalize_accession("000101000001"),
        );
        let ann = map[&key].clone();
        assert_eq!(ann.filed_date.as_deref(), Some("2024-02-02"));
        assert_eq!(ann.form_type.as_deref(), Some("10-K"));
        assert_eq!(ann.fy.as_deref(), Some("2024"));
        assert!(ann.is_adequate && ann.is_canonical);
        assert_eq!(ann.canonical_accession.as_deref(), Some("0001-01-000001"));
        let blank_key = (
            "IBM".to_string(),
            sec_fetcher_shared::normalize_accession("0002-02-000002"),
        );
        let blank = map[&blank_key].clone();
        assert!(blank.filed_date.is_none() && blank.fy.is_none());
        assert!(!blank.is_adequate && !blank.is_canonical);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn shared_accession_keeps_independent_canonical_flags_per_ticker() {
        let dir = std::env::temp_dir().join("tenk_itemized_annotations_collision_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("periods.csv");
        fs::write(
            &path,
            "ticker,accession_number,filed_date,form_type,fy,fp,is_adequate,is_canonical,canonical_accession\n\
             AAA,0001-01-000001,2024-02-02,10-K,2024,FY,true,true,0001-01-000001\n\
             BBB,0001-01-000001,2024-02-02,10-K,2024,FY,true,false,0001-01-000001\n",
        )
        .unwrap();
        let map = load_annotations(&path).unwrap();
        assert_eq!(map.len(), 2);
        let norm = sec_fetcher_shared::normalize_accession("0001-01-000001");
        let aaa = map
            .get(&("AAA".to_string(), norm.clone()))
            .expect("AAA row must survive");
        let bbb = map
            .get(&("BBB".to_string(), norm))
            .expect("BBB row must survive");
        assert!(aaa.is_canonical, "canonical winner must stay canonical");
        assert!(!bbb.is_canonical, "sibling must stay non-canonical");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_periods_record_is_an_error_not_a_skip() {
        let dir = std::env::temp_dir().join("tenk_itemized_annotations_malformed_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("periods.csv");
        // Invalid UTF-8 in the second record: `StringRecord` decoding must
        // fail, proving `result?` propagates instead of `.flatten()` drops.
        let bytes = b"ticker,accession_number,filed_date,form_type,fy,fp,is_adequate,is_canonical,canonical_accession\nAAPL,0001-01-000001,2024-02-02,10-K,2024,FY,true,true,0001-01-000001\nBBB,\xff\xfe,2024-02-02\n"
            .to_vec();
        fs::write(&path, &bytes).unwrap();
        assert!(
            load_annotations(&path).is_err(),
            "malformed CSV record must propagate, not be silently skipped"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_periods_file_yields_empty_map() {
        let map = load_annotations(Path::new("/nonexistent-xyz/periods.csv")).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn freshness_gate_rejects_missing_periods() {
        let err = check_periods_freshness(
            Path::new("/nonexistent-xyz/periods.csv"),
            Path::new("/nonexistent-xyz/audit.csv"),
        )
        .unwrap_err();
        assert!(err.contains("tenk-period-join"), "{err}");
    }

    #[test]
    fn freshness_gate_accepts_fresh_periods() {
        let dir = std::env::temp_dir().join("tenk_itemized_freshness_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Created in join order: audit first, periods second, so the
        // periods file is always at least as new as the audit.
        let audit = dir.join("tenk_audit.csv");
        let periods = dir.join("tenk_audit_periods.csv");
        fs::write(&audit, "ticker\nAAPL\n").unwrap();
        fs::write(&periods, "ticker,accession_number\nAAPL,0001\n").unwrap();
        assert!(check_periods_freshness(&periods, &audit).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn annotation_batch_marks_booleans_and_nulls() {
        let ann = FilingAnnotation {
            filed_date: Some("2024-02-02".to_string()),
            form_type: Some("10-K/A".to_string()),
            fy: None,
            fp: None,
            is_adequate: false,
            is_canonical: false,
            canonical_accession: Some("0001-01-000009".to_string()),
        };
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(StringArray::from(vec!["TICK"])),
                Arc::new(StringArray::from(vec!["0001-01-000001"])),
                Arc::new(StringArray::from(vec![ann.filed_date.as_deref()])),
                Arc::new(StringArray::from(vec![ann.form_type.as_deref()])),
                Arc::new(StringArray::from(vec![ann.fy.as_deref()])),
                Arc::new(StringArray::from(vec![ann.fp.as_deref()])),
                Arc::new(BooleanArray::from(vec![ann.is_adequate])),
                Arc::new(BooleanArray::from(vec![ann.is_canonical])),
                Arc::new(StringArray::from(vec![ann.canonical_accession.as_deref()])),
                Arc::new(StringArray::from(vec![None::<&str>])),
                Arc::new(StringArray::from(vec![None::<&str>])),
            ],
        )
        .unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.column(4).null_count(), 1);
        assert_eq!(batch.column(9).null_count(), 1);
    }
}
