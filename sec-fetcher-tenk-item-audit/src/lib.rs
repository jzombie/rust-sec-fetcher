//! Shared utilities for auditing the 10-K section parser.
//!
//! Provides CSV row construction, file discovery, path heuristics, and text
//! helpers used by `audit-tenk-local` (which runs the parser on local files).
//!
//! This library intentionally has **zero** external dependencies — it works
//! with plain [`HashMap`]s and [`String`]s so that any crate (including
//! `sec-fetcher`) can depend on it without creating circular dependency
//! issues.

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
///   (typically the result of [`extract_sections_from_document`]).
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
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') {
                continue;
            }
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
/// Returns the ticker directory name when the file is exactly two levels deep
/// beneath `root`; returns an empty string otherwise.
pub fn guess_ticker(file_path: &Path, root: &Path) -> String {
    if let Some(parent) = file_path.parent() {
        if let Some(ticker_dir) = parent.file_name().and_then(|n| n.to_str()) {
            if let Some(grandparent) = parent.parent() {
                if grandparent == root {
                    return ticker_dir.to_string();
                }
            }
        }
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
