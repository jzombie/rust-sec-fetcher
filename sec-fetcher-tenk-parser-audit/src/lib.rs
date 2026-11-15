//! Shared utilities for auditing the 10-K section parser.
//!
//! Provides CSV row construction, file discovery, path heuristics, manifest
//! loading, and text helpers used by `tenk-parser-audit` (which runs the
//! parser on local files).
//!
//! This library keeps external dependencies to a minimum — plain
//! [`HashMap`]s and [`String`]s throughout, plus the `csv` crate for
//! manifest parsing — so that any crate (including `sec-fetcher`) can depend
//! on it without creating circular dependency issues.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

// ── CSV header construction ───────────────────────────────────────────────────

/// Returns the canonical column-header row for a 10-K audit CSV.
///
/// `item_keys` should be the list of item map keys (e.g. `"item_1"`,
/// `"item_1a"`, `"item_7"`, …) in display order.  One `*_chars` column is
/// emitted per key.
pub fn header_columns(item_keys: &[String]) -> Vec<String> {
    let mut header: Vec<String> = vec![
        "ticker".into(),
        "cik".into(),
        "accession_number".into(),
        "filed_date".into(),
        "form_type".into(),
        "is_adequate".into(),
        "item_1_snippet".into(),
        "item_7_snippet".into(),
        "items_found".into(),
    ];
    for key in item_keys {
        header.push(format!("{}_chars", key));
    }
    header.push("error".into());
    header
}

// ── CSV row builder ───────────────────────────────────────────────────────────

/// Build a single CSV record from the available data.
///
/// Every column defined by [`header_columns`] is populated.  Fields that are
/// not known at the call site should be passed as `""`.
///
/// * `item_keys` — the ordered list of item keys (e.g. `"item_1"`, `"item_7"`).
/// * `sections` — an optional map from item key to section text
///   (typically the output of the 10-K section parser).
/// * `is_adequate` — whether the filing is considered adequate.
///   When `sections` is `Some`, this is typically computed from it.
#[allow(clippy::too_many_arguments)]
pub fn build_row(
    ticker: &str,
    cik: &str,
    acc: &str,
    filed_date: &str,
    form_type: &str,
    item_keys: &[String],
    items_found: &str,
    sections: Option<&HashMap<String, String>>,
    is_adequate: bool,
    err: &str,
) -> Vec<String> {
    let mut record: Vec<String> = vec![
        ticker.into(),
        cik.into(),
        acc.into(),
        filed_date.into(),
        form_type.into(),
        if is_adequate { "true" } else { "false" }.into(),
        snippet(sections.and_then(|s| s.get("item_1")).map(|s| s.as_str())),
        snippet(sections.and_then(|s| s.get("item_7")).map(|s| s.as_str())),
        items_found.into(),
    ];
    for key in item_keys {
        let count = sections
            .and_then(|s| s.get(key))
            .map(|v| v.len())
            .unwrap_or(0);
        record.push(count.to_string());
    }
    record.push(err.into());
    record
}

// ── Text helpers ──────────────────────────────────────────────────────────────

/// First 150 chars of `text`, with runs of whitespace collapsed to a single
/// space.  Returns an empty string when `text` is `None`.
pub fn snippet(text: Option<&str>) -> String {
    match text {
        None => String::new(),
        Some(t) => {
            let collapsed: String = t.split_whitespace().collect::<Vec<_>>().join(" ");
            collapsed.chars().take(150).collect()
        }
    }
}

// ── File discovery ────────────────────────────────────────────────────────────

/// Recursively collect all files under `root` whose name matches the
/// optional extension filter.  Hidden files (names starting with `.`)
/// are skipped.
///
/// When `pattern` is `"*"`, every file is included; otherwise `pattern`
/// is matched as a case-insensitive suffix (e.g. `"*.htm"` matches both
/// `.htm` and `.html`).
pub fn collect_files(root: &Path, pattern: &str) -> io::Result<Vec<PathBuf>> {
    if !root.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            format!("{:?} is not a directory", root),
        ));
    }

    let mut files = Vec::new();
    let match_all = pattern == "*" || pattern == "*.*";
    collect_files_recursive(root, pattern, match_all, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_recursive(
    dir: &Path,
    pattern: &str,
    match_all: bool,
    files: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip hidden entries
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && name.starts_with('.')
        {
            continue;
        }

        if path.is_dir() {
            collect_files_recursive(&path, pattern, match_all, files)?;
        } else if path.is_file() {
            if match_all {
                files.push(path);
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let suffix = pattern.strip_prefix('*').unwrap_or(pattern);
                if name
                    .to_ascii_lowercase()
                    .ends_with(&suffix.to_ascii_lowercase())
                {
                    files.push(path);
                }
            }
        }
    }
    Ok(())
}

// ── Path heuristics ───────────────────────────────────────────────────────────

/// Try to extract the ticker symbol from a file path that follows the
/// `pull-tenk-items` layout: `<root>/<TICKER>/<ACCESSION>.<ext>`.
///
/// Walks up from the file's parent until `root` is reached and returns the
/// immediate child of `root` (so deeper layouts like
/// `<root>/<SECTOR>/<TICKER>/<FILE>` still resolve). Returns an empty string
/// when the file is not beneath `root`.
pub fn guess_ticker(file_path: &Path, root: &Path) -> String {
    let mut current = file_path.parent();
    let mut candidate: Option<&str> = None;
    while let Some(dir) = current {
        if dir == root {
            return candidate.unwrap_or_default().to_string();
        }
        if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
            candidate = Some(name);
        }
        current = dir.parent();
    }
    String::new()
}

/// Try to extract the accession number from the filename.
///
/// The `pull-tenk-items` naming convention is:
/// `<ACCESSION>.<ext>`.  Strips the extension.
pub fn guess_accession(file_path: &Path) -> String {
    file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_default()
}

// ── Accession→form manifest ───────────────────────────────────────────────────

/// One manifest record: `(cik, filed_date, form_type)`.
pub type ManifestEntry = (String, String, String);

/// Load the accession→form manifest written by `pull-tenk-items`
/// (`.tenk_manifest.csv`), keyed by `(ticker, accession_number)`.
///
/// Parsed with the `csv` crate (not line-splitting): fields may contain
/// quoted commas, quotes, or embedded newlines. Returns an empty map when
/// the file is missing or unreadable — callers fall back to blank `cik` /
/// `filed_date` / `form_type` columns, preserving the old offline behavior.
/// Malformed lines are skipped. Only the first five columns are read; extra
/// columns are ignored.
pub fn load_manifest(path: &Path) -> HashMap<(String, String), ManifestEntry> {
    let mut map = HashMap::new();
    let Ok(mut rdr) = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
    else {
        return map;
    };
    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        map.insert(
            (record[0].to_string(), record[2].to_string()),
            (
                record[1].to_string(),
                record[3].to_string(),
                record[4].to_string(),
            ),
        );
    }
    map
}

#[cfg(test)]
mod manifest_tests {
    use super::*;

    #[test]
    fn missing_manifest_loads_empty() {
        let map = load_manifest(Path::new("/nonexistent-xyz/.tenk_manifest.csv"));
        assert!(map.is_empty());
    }

    #[test]
    fn manifest_lookup_by_ticker_and_accession() {
        let dir = std::env::temp_dir().join("tenk_audit_manifest_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".tenk_manifest.csv");
        std::fs::write(
            &path,
            "ticker,cik,accession,filed_date,form,primary_document\n\
             ADM,7084,0000007084-05-000350,2005-08-12,10-K,adm10kfy05.htm\n\
             ABCB,70732,0001193125-07-251748,2007-11-29,10-K/A,d10ka.htm\n",
        )
        .unwrap();
        let map = load_manifest(&path);
        assert_eq!(map.len(), 2);
        assert_eq!(
            map[&("ADM".to_string(), "0000007084-05-000350".to_string())],
            (
                "7084".to_string(),
                "2005-08-12".to_string(),
                "10-K".to_string()
            )
        );
        assert!(
            map[&("ABCB".to_string(), "0001193125-07-251748".to_string())]
                .2
                .ends_with("/A"),
            "amendment form must be visible for scoring-time filtering"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_survives_quoted_multiline_field() {
        // A line-splitting parser fractures this record and drops it; the
        // `csv` reader must decode the embedded newline inside quotes.
        let dir = std::env::temp_dir().join("tenk_audit_manifest_multiline");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".tenk_manifest.csv");
        std::fs::write(
            &path,
            "ticker,cik,accession,filed_date,form,primary_document\n\
             ADM,7084,0000007084-05-000350,2005-08-12,10-K,\"doc\nname.htm\"\n",
        )
        .unwrap();
        let map = load_manifest(&path);
        assert_eq!(map.len(), 1);
        assert_eq!(
            map[&("ADM".to_string(), "0000007084-05-000350".to_string())].2,
            "10-K"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
