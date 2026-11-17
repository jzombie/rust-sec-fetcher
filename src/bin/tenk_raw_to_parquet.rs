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

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use clap::{CommandFactory, FromArgMatches, Parser};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// 10 GiB per output file, matching the Python script's `TARGET_BYTES`.
const TARGET_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// Flush the writer once this many uncompressed text bytes have buffered,
/// so row groups stay large enough for effective ZSTD compression and
/// efficient downstream scans.
const ROW_GROUP_TARGET_BYTES: usize = 64 * 1024 * 1024;

#[derive(Parser)]
#[command(name = "tenk-raw-to-parquet", about = "Convert raw 10-K files to ZSTD Parquet")]
struct Args {
    /// Source tree. Defaults to `data/tenk_items`.
    #[arg(long, short = 's')]
    source: Option<PathBuf>,

    /// Destination directory for `batch_NNNN.parquet`. Defaults to `hf_upload_folder/tenk_raw`.
    #[arg(long, short = 'd')]
    dest: Option<PathBuf>,

    /// Start a new output file once the current one reaches this many bytes.
    #[arg(long, default_value_t = TARGET_BYTES)]
    target_bytes: u64,
}

/// Whether a file is a supported raw filing: `.htm`, `.html`, or `.txt`
/// in any letter case. Dotfiles (e.g. `.tenk_manifest.csv`) never match:
/// they have no extension under `Path::extension`.
fn is_supported_filing_file(path: &Path) -> bool {
    if path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with('.'))
    {
        return false;
    }
    path.extension().and_then(|e| e.to_str()).is_some_and(|ext| {
        ext.eq_ignore_ascii_case("htm")
            || ext.eq_ignore_ascii_case("html")
            || ext.eq_ignore_ascii_case("txt")
    })
}

/// Accession number from a filing path: the file stem, so `.htm`,
/// `.html`, `.txt`, and any letter case all resolve to the bare number.
fn accession_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string()
}

/// Collect `(ticker, accession, path)` triples in sorted order for
/// deterministic output. Dot-prefixed directories are skipped.
fn collect_files(source: &Path) -> Result<Vec<(String, String, PathBuf)>, std::io::Error> {
    let mut tickers: Vec<String> = Vec::new();
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type()?.is_dir() && !name.starts_with('.') {
            tickers.push(name);
        }
    }
    tickers.sort();

    let mut out = Vec::new();
    for ticker in tickers {
        let dir = source.join(&ticker);
        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_file() && is_supported_filing_file(&path) {
                paths.push(path);
            }
        }
        paths.sort();
        for path in paths {
            out.push((ticker.clone(), accession_from_path(&path), path));
        }
    }
    Ok(out)
}

/// Remove stale `batch_*.parquet` and `batch_*.parquet.tmp` shards from a
/// previous run so a smaller corpus never leaves orphaned outputs behind.
fn purge_stale_shards(dest: &Path) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(dest)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        if (name.starts_with("batch_") && name.ends_with(".parquet"))
            || (name.starts_with("batch_") && name.ends_with(".parquet.tmp"))
        {
            fs::remove_file(&path)?;
        }
    }
    Ok(())
}

fn human(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    for unit in UNITS {
        if value < 1024.0 || *unit == "TiB" {
            return format!("{value:.0} {unit}");
        }
        value /= 1024.0;
    }
    unreachable!()
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("ticker_symbol", DataType::Utf8, false),
        Field::new("accession_number", DataType::Utf8, false),
        Field::new("raw_text", DataType::Utf8, false),
    ]))
}

fn writer_for(path: &Path, schema: Arc<Schema>) -> Result<ArrowWriter<fs::File>, parquet::errors::ParquetError> {
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let file = fs::File::create(path)?;
    ArrowWriter::try_new(file, schema, Some(props))
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
    let files = collect_files(&source)?;
    let total = files.len();
    println!("found {total} file(s)\n");

    let start = std::time::Instant::now();
    let schema = schema();
    let mut writer: Option<ArrowWriter<fs::File>> = None;
    let mut cur_paths: Option<(PathBuf, PathBuf)> = None;
    let mut output_idx = 0usize;
    let mut done = 0usize;
    let mut row_group_bytes = 0usize;

    for (ticker, accession, path) in &files {
        if writer.is_none() {
            output_idx += 1;
            let final_path = dest.join(format!("batch_{output_idx:04}.parquet"));
            let tmp_path = dest.join(format!("batch_{output_idx:04}.parquet.tmp"));
            writer = Some(writer_for(&tmp_path, Arc::clone(&schema))?);
            cur_paths = Some((tmp_path, final_path));
            row_group_bytes = 0;
        }
        // `errors="replace"` equivalent: lossy UTF-8 decode, reusing the
        // read buffer when the file is already valid UTF-8.
        let bytes = fs::read(path)?;
        let text = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        };

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec![ticker.as_str()])),
                Arc::new(StringArray::from(vec![accession.as_str()])),
                Arc::new(StringArray::from(vec![text.as_str()])),
            ],
        )?;
        let w = writer.as_mut().expect("writer open");
        w.write(&batch)?;
        row_group_bytes += text.len();
        done += 1;

        // Flush only once a full row group has buffered, so ZSTD sees
        // enough contiguous text to compress well and readers see few,
        // large row groups instead of one per filing.
        if row_group_bytes >= ROW_GROUP_TARGET_BYTES {
            w.flush()?;
            row_group_bytes = 0;
            // Split check on bytes written so far, mirroring the Python
            // script's `os.path.getsize` check.
            if u64::try_from(w.bytes_written()).unwrap_or(0) >= args.target_bytes {
                let w = writer.take().expect("writer open");
                w.close()?;
                let (tmp_path, final_path) = cur_paths.take().expect("shard paths set");
                fs::rename(&tmp_path, &final_path)?;
            }
        }

        let pct = done as f64 / total as f64 * 100.0;
        print!("\r  [{done:>6}/{total}]  output {output_idx:>4}  ({pct:>5.1} %){}", " ".repeat(20));
        std::io::stdout().flush()?;
    }
    if let Some(w) = writer.take() {
        w.close()?;
        let (tmp_path, final_path) = cur_paths.take().expect("shard paths set");
        fs::rename(&tmp_path, &final_path)?;
    }

    println!("\n\nDone — {done} file(s) → {}  in {:.1} s", dest.display(), start.elapsed().as_secs_f64());
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
    fn accession_strips_known_suffixes_any_case() {
        let cases = [
            ("0001234567-23-000045.htm", "0001234567-23-000045"),
            ("0001234567-23-000045.HTM", "0001234567-23-000045"),
            ("0001234567-23-000045.html", "0001234567-23-000045"),
            ("0001234567-23-000045.HTML", "0001234567-23-000045"),
            ("0001234567-23-000045.txt", "0001234567-23-000045"),
            ("0001234567-23-000045.TXT", "0001234567-23-000045"),
        ];
        for (file, acc) in cases {
            assert_eq!(accession_from_path(Path::new(file)), acc);
        }
    }

    #[test]
    fn unsupported_files_are_skipped() {
        assert!(is_supported_filing_file(Path::new("a.htm")));
        assert!(is_supported_filing_file(Path::new("a.HTML")));
        assert!(is_supported_filing_file(Path::new("a.TXT")));
        assert!(!is_supported_filing_file(Path::new("a.csv")));
        assert!(!is_supported_filing_file(Path::new("a.pdf")));
        assert!(!is_supported_filing_file(Path::new(".tenk_manifest.csv")));
        assert!(!is_supported_filing_file(Path::new(".hidden.htm")));
    }

    #[test]
    fn collect_files_sorts_deterministically() {
        let dir = std::env::temp_dir().join("tenk_raw_parquet_sort_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("ZZZ")).unwrap();
        fs::create_dir_all(dir.join("AAA")).unwrap();
        fs::create_dir_all(dir.join(".hidden_dir")).unwrap();
        fs::write(dir.join("ZZZ").join("b.htm"), "b").unwrap();
        fs::write(dir.join("ZZZ").join("a.HTML"), "a").unwrap();
        fs::write(dir.join("ZZZ").join("u.TXT"), "u").unwrap();
        fs::write(dir.join("AAA").join("c.htm"), "c").unwrap();
        fs::write(dir.join("AAA").join("skip.csv"), "x").unwrap();
        fs::write(dir.join(".hidden_dir").join("h.htm"), "h").unwrap();
        let files = collect_files(&dir).unwrap();
        let keys: Vec<(&str, &str)> =
            files.iter().map(|(t, a, _)| (t.as_str(), a.as_str())).collect();
        assert_eq!(
            keys,
            vec![("AAA", "c"), ("ZZZ", "a"), ("ZZZ", "b"), ("ZZZ", "u")]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn purge_removes_stale_shards_only() {
        let dir = std::env::temp_dir().join("tenk_raw_parquet_purge_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for name in [
            "batch_0001.parquet",
            "batch_0002.parquet.tmp",
            "keep.csv",
            "batch_notes.txt",
        ] {
            fs::write(dir.join(name), "x").unwrap();
        }
        purge_stale_shards(&dir).unwrap();
        let mut remaining: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        remaining.sort();
        assert_eq!(remaining, vec!["batch_notes.txt", "keep.csv"]);
        let _ = fs::remove_dir_all(&dir);
    }
}
