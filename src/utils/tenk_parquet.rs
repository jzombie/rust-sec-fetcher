//! Shared corpus discovery and sharded Parquet writer for 10-K exports.
//!
//! Used by `tenk-raw-to-parquet` and `tenk-itemized-to-parquet` so shard
//! naming, rotation thresholds, file filtering, and atomic rename behavior
//! stay identical across datasets.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Target compressed size per output shard, matching the Python export.
pub const DEFAULT_SHARD_TARGET_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// Flush the writer once this many uncompressed bytes have buffered, so row
/// groups stay large enough for effective ZSTD compression.
pub const DEFAULT_ROW_GROUP_TARGET_BYTES: usize = 64 * 1024 * 1024;

/// Whether a file is a supported raw filing: `.htm`, `.html`, or `.txt`
/// in any letter case. Dotfiles (e.g. `.tenk_manifest.csv`) never match.
pub fn is_supported_filing_file(path: &Path) -> bool {
    if path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with('.'))
    {
        return false;
    }
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            ext.eq_ignore_ascii_case("htm")
                || ext.eq_ignore_ascii_case("html")
                || ext.eq_ignore_ascii_case("txt")
        })
}

/// Accession number from a filing path: the file stem, so `.htm`,
/// `.html`, `.txt`, and any letter case all resolve to the bare number.
pub fn accession_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string()
}

/// Collect `(ticker, accession, path)` triples in sorted order for
/// deterministic output. Dot-prefixed directories are skipped.
pub fn collect_tenk_filing_files(
    source: &Path,
) -> Result<Vec<(String, String, PathBuf)>, std::io::Error> {
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
pub fn purge_stale_shards(dest: &Path) -> Result<(), std::io::Error> {
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

/// Human-readable byte count for completion summaries.
pub fn human(bytes: u64) -> String {
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

/// Single-threaded sharded Parquet writer with `.tmp` staging and atomic
/// rename. File discovery stays sorted upstream; this writer preserves that
/// order. Parsing may run on worker threads, but all `write_batch` calls
/// must happen on one thread in sorted order.
pub struct ShardedParquetWriter {
    dest: PathBuf,
    schema: Arc<Schema>,
    target_bytes: u64,
    row_group_target_bytes: usize,
    writer: Option<ArrowWriter<fs::File>>,
    cur_paths: Option<(PathBuf, PathBuf)>,
    output_idx: usize,
    row_group_bytes: usize,
}

impl ShardedParquetWriter {
    /// Create a writer for `batch_NNNN.parquet` shards under `dest`.
    pub fn new(
        dest: PathBuf,
        schema: Arc<Schema>,
        target_bytes: u64,
        row_group_target_bytes: usize,
    ) -> Self {
        Self {
            dest,
            schema,
            target_bytes,
            row_group_target_bytes,
            writer: None,
            cur_paths: None,
            output_idx: 0,
            row_group_bytes: 0,
        }
    }

    /// One-based index of the current shard (0 when nothing written yet).
    pub fn shard_index(&self) -> usize {
        self.output_idx
    }

    /// Append one batch, flushing a full row group and rotating shards as
    /// thresholds require. `uncompressed_bytes` tracks the buffered payload
    /// size driving row-group flushes.
    pub fn write_batch(
        &mut self,
        batch: &RecordBatch,
        uncompressed_bytes: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.writer.is_none() {
            self.output_idx += 1;
            let final_path = self
                .dest
                .join(format!("batch_{:04}.parquet", self.output_idx));
            let tmp_path = self
                .dest
                .join(format!("batch_{:04}.parquet.tmp", self.output_idx));
            let props = WriterProperties::builder()
                .set_compression(Compression::ZSTD(Default::default()))
                .build();
            let file = fs::File::create(&tmp_path)?;
            self.writer = Some(ArrowWriter::try_new(
                file,
                Arc::clone(&self.schema),
                Some(props),
            )?);
            self.cur_paths = Some((tmp_path, final_path));
            self.row_group_bytes = 0;
        }

        let w = self.writer.as_mut().expect("writer open");
        w.write(batch)?;
        self.row_group_bytes += uncompressed_bytes;

        if self.row_group_bytes >= self.row_group_target_bytes {
            w.flush()?;
            self.row_group_bytes = 0;
            if u64::try_from(w.bytes_written()).unwrap_or(0) >= self.target_bytes {
                let w = self.writer.take().expect("writer open");
                w.close()?;
                let (tmp_path, final_path) = self.cur_paths.take().expect("shard paths set");
                fs::rename(&tmp_path, &final_path)?;
            }
        }
        Ok(())
    }

    /// Close the open shard (atomic rename) and return the shard count.
    pub fn finish(mut self) -> Result<usize, Box<dyn std::error::Error>> {
        if let Some(w) = self.writer.take() {
            w.close()?;
            let (tmp_path, final_path) = self.cur_paths.take().expect("shard paths set");
            fs::rename(&tmp_path, &final_path)?;
        }
        Ok(self.output_idx)
    }
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
        let dir = std::env::temp_dir().join("tenk_parquet_sort_test");
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
        let files = collect_tenk_filing_files(&dir).unwrap();
        let keys: Vec<(&str, &str)> = files
            .iter()
            .map(|(t, a, _)| (t.as_str(), a.as_str()))
            .collect();
        assert_eq!(
            keys,
            vec![("AAA", "c"), ("ZZZ", "a"), ("ZZZ", "b"), ("ZZZ", "u")]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn purge_removes_stale_shards_only() {
        let dir = std::env::temp_dir().join("tenk_parquet_purge_test");
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

    #[test]
    fn sharded_writer_rotates_and_renames_cleanly() {
        use arrow_array::StringArray;
        use arrow_schema::{DataType, Field};

        let dir = std::env::temp_dir().join("tenk_parquet_sharded_writer_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Utf8, false)]));
        let mut writer = ShardedParquetWriter::new(dir.clone(), Arc::clone(&schema), 1, 1);

        for v in ["row_one", "row_two"] {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(StringArray::from(vec![v]))],
            )
            .unwrap();
            writer.write_batch(&batch, 16).unwrap();
        }
        let shards = writer.finish().unwrap();
        assert_eq!(shards, 2);
        assert!(dir.join("batch_0001.parquet").is_file());
        assert!(dir.join("batch_0002.parquet").is_file());
        assert!(!dir.join("batch_0001.parquet.tmp").exists());
        assert!(!dir.join("batch_0002.parquet.tmp").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sharded_writer_empty_input_produces_no_shards() {
        use arrow_schema::{DataType, Field};

        let dir = std::env::temp_dir().join("tenk_parquet_sharded_writer_empty_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Utf8, false)]));
        let writer = ShardedParquetWriter::new(dir.clone(), Arc::clone(&schema), 1, 1);
        let shards = writer.finish().unwrap();
        assert_eq!(shards, 0);
        assert!(!dir.join("batch_0001.parquet").exists());
        assert!(!dir.join("batch_0001.parquet.tmp").exists());

        let _ = fs::remove_dir_all(&dir);
    }
}
