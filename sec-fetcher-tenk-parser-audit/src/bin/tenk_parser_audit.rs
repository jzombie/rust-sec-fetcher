//! Audit the 10-K section parser against a local directory of filing documents.
//!
//! Reads every file (or every `.htm`/`.html`/`.txt` file) under a given input
//! directory, runs [`extract_sections_from_document`] on each one, and writes
//! the results to a CSV.  There are **no network calls** — all documents are
//! assumed to already exist on disk.
//!
//! This is useful for:
//! - Testing parser changes against a known corpus without hitting EDGAR.
//! - Re-running the parser on filings previously saved by `pull-tenk-items`
//!   (one `<TICKER>/` subdir per ticker under the corpus directory).
//! - Bulk-evaluating parser quality or performance.
//!
//! # Usage
//!
//! ```sh
//! cargo run -p sec-fetcher-tenk-parser-audit --bin tenk-parser-audit --release
//! ```
//!
//! Defaults resolve from the shared data root; run with `--help` for paths.
//!
//! # Output columns
//!
//! Same columns originally produced by the old combined `pull-tenk-audit`
//! binary, except that `cik`, `filed_date`, and `form_type` come from the
//! `.tenk_manifest.csv` written by `pull-tenk-items` (blank when the audit
//! runs against a directory with no manifest).
//!
//! | Column | Description |
//! |--------|-------------|
//! | `ticker` | Ticker symbol (from path when available) |
//! | `cik` | CIK from the pull manifest (blank when no manifest) |
//! | `accession_number` | Accession number (from filename when available) |
//! | `filed_date` | Filing date from the pull manifest (blank when no manifest) |
//! | `form_type` | Form from the pull manifest, e.g. `10-K` or `10-K/A` (blank when no manifest) |
//! | `is_adequate` | `true` when `item_1_chars >= 400 AND item_7_chars >= 2000` |
//! | `item_1_snippet` | First 150 chars of Item 1, whitespace-collapsed |
//! | `item_7_snippet` | First 150 chars of Item 7, whitespace-collapsed |
//! | `items_found` | Semicolon-separated list of item keys extracted |
//! | `item_1_chars` ... `item_16_chars` | Character count for every `TenKItem` |
//! | `error` | Non-empty when the parse step failed |

use clap::{CommandFactory, FromArgMatches, Parser};
use csv::WriterBuilder;
use sec_fetcher::{
    enums::TenKItem,
    parsers::{
        extract_sections_from_document, resolve_tenk_audit_csv, resolve_tenk_items_dir,
        resolve_tenk_manifest_csv,
    },
};
use sec_fetcher_tenk_parser_audit::{
    build_row, collect_files, guess_accession, guess_ticker, header_columns, load_manifest,
    normalize_accession,
};
use std::collections::HashMap;
use std::path::PathBuf;
use strum::IntoEnumIterator;
use tracing::{error, info, warn};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "tenk-parser-audit",
    about = "Audit the 10-K section parser on a local directory of filing documents"
)]
struct Args {
    /// Directory containing filing documents to parse (searched recursively).
    #[arg(long, short = 'd')]
    dir: Option<PathBuf>,

    /// Path to the output CSV.
    #[arg(long, short = 'o')]
    output: Option<PathBuf>,

    /// Optional extension filter (e.g., "*.htm", "*.html").
    /// Default: process all files (skipping hidden files). Binary exhibits
    /// (PDFs, images, zips) are skipped by extension + size guard below, so a
    /// broad default won't load them into memory as strings.
    #[arg(long, short = 'p', default_value = "*")]
    pattern: String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy()
                .add_directive("html5ever=off".parse().unwrap())
                .add_directive("selectors=off".parse().unwrap()),
        )
        .init();

    let args = Args::from_arg_matches(
        &Args::command()
            .mut_arg("dir", |a| {
                a.help(format!(
                    "Directory containing filing documents to parse (searched recursively). Defaults to `{}`.",
                    resolve_tenk_items_dir(None)
                ))
            })
            .mut_arg("output", |a| {
                a.help(format!(
                    "Path to the output CSV. Defaults to `{}`.",
                    resolve_tenk_audit_csv(None)
                ))
            })
            .get_matches(),
    )?;
    let dir = PathBuf::from(resolve_tenk_items_dir(
        args.dir.as_ref().and_then(|p| p.to_str()),
    ));
    let output = PathBuf::from(resolve_tenk_audit_csv(
        args.output.as_ref().and_then(|p| p.to_str()),
    ));

    // Build the ordered item-key list once (used by both header and rows).
    let item_keys: Vec<String> = TenKItem::iter().map(|i| i.map_key()).collect();

    // ── Load the accession→form manifest (when present) ─────────────────────
    //
    // Written by `pull-tenk-items` into the corpus root. Absent for
    // hand-assembled directories — rows then keep blank cik/filed_date/
    // form_type, the old offline behavior.
    let dir_str = dir.to_string_lossy().into_owned();
    let manifest_path = resolve_tenk_manifest_csv(Some(&dir_str));
    let manifest = load_manifest(std::path::Path::new(&manifest_path));
    info!(
        "Manifest: {} tagged filing(s) loaded from {:?}",
        manifest.len(),
        manifest_path
    );

    // ── Build header ──────────────────────────────────────────────────────────

    let header = header_columns(&item_keys);

    // ── Collect files ─────────────────────────────────────────────────────────

    let all_files = collect_files(&dir, &args.pattern)?;
    let total = all_files.len();
    info!(
        "Found {} files under {:?} (pattern: {})",
        total, dir, args.pattern
    );

    if total == 0 {
        warn!("No files matched — nothing to do.");
        return Ok(());
    }

    // ── Open output CSV ───────────────────────────────────────────────────────

    let mut writer = WriterBuilder::new().has_headers(true).from_path(&output)?;
    writer.write_record(&header)?;

    // ── Process each file ─────────────────────────────────────────────────────

    let mut errors = 0usize;

    for (i, file_path) in all_files.iter().enumerate() {
        let relative = file_path
            .strip_prefix(&dir)
            .unwrap_or(file_path)
            .display()
            .to_string();

        info!("[{}/{}] {}", i + 1, total, relative);

        let ticker = guess_ticker(file_path, &dir);
        let accession = guess_accession(file_path);

        // Tag from the pull manifest when available; blanks otherwise.
        // Keys are normalized at the boundary (trimmed ticker, normalized
        // accession) to match `load_manifest`, so dash/padding/case variants
        // on either side still join.
        let (cik, filed_date, form_type) = manifest
            .get(&(ticker.trim().to_string(), normalize_accession(&accession)))
            .cloned()
            .unwrap_or_default();

        // Skip known-binary exhibits without reading them into memory, plus a
        // size guard against accidentally matched large binaries (max observed
        // 10-K is ~89MB, so 100MB covers the full corpus).
        let skip_reason = binary_skip_reason(file_path, 100 * 1024 * 1024);
        if let Some(reason) = skip_reason {
            let row = build_row(
                &ticker,
                &cik,
                &accession,
                &filed_date,
                &form_type,
                &item_keys,
                "",
                None,
                false,
                &reason,
            );
            writer.write_record(&row)?;
            errors += 1;
            continue;
        }

        let raw = match std::fs::read_to_string(file_path) {
            Ok(text) => text,
            Err(e) => {
                let row = build_row(
                    &ticker,
                    &cik,
                    &accession,
                    &filed_date,
                    &form_type,
                    &item_keys,
                    "",
                    None,
                    false,
                    &format!("read: {}", e),
                );
                writer.write_record(&row)?;
                errors += 1;
                continue;
            }
        };

        let (sections_map, err_msg) = match extract_sections_from_document(&raw) {
            Ok(sections) => {
                let n = sections.keys().count();
                info!("  {} — {} items", relative, n);

                // Convert TenKSections → HashMap<String, String> for the
                // generic audit library.
                let map: HashMap<String, String> = sections
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
                (Some(map), String::new())
            }
            Err(e) => {
                let msg = format!("parse: {}", e);
                error!("  {}: {}", relative, msg);
                (None, msg)
            }
        };

        let is_adequate = is_adequate_for(sections_map.as_ref());

        let items_found = items_found_str(sections_map.as_ref(), &item_keys);

        let row = build_row(
            &ticker,
            &cik,
            &accession,
            &filed_date,
            &form_type,
            &item_keys,
            &items_found,
            sections_map.as_ref(),
            is_adequate,
            &err_msg,
        );
        writer.write_record(&row)?;

        if sections_map.is_none() {
            errors += 1;
        }
    }

    writer.flush()?;

    if errors > 0 {
        warn!(
            "Completed with {} error(s) — see 'error' column in output.",
            errors
        );
    } else {
        info!("Completed successfully — {} files parsed.", total);
    }

    Ok(())
}

/// Returns a skip reason when `path` is a known-binary exhibit or exceeds
/// `max_bytes`; `None` means it is safe to `read_to_string`.
fn binary_skip_reason(path: &std::path::Path, max_bytes: u64) -> Option<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    const BINARY_EXTS: &[&str] = &[
        "pdf", "png", "jpg", "jpeg", "gif", "xlsx", "zip", "js", "css",
    ];
    if BINARY_EXTS.contains(&ext.as_str()) {
        return Some(format!("skip: binary .{} exhibit", ext));
    }
    if let Ok(meta) = std::fs::metadata(path)
        && meta.len() > max_bytes
    {
        return Some(format!("skip: file too large ({} bytes)", meta.len()));
    }
    None
}

/// Filing is adequate when Item 1 has at least 400 chars and Item 7 has at
/// least 2000 chars. Missing sections count as zero length, so `None`
/// (unparsed document) is never adequate.
fn is_adequate_for(sections_map: Option<&HashMap<String, String>>) -> bool {
    sections_map
        .map(|m| {
            let item1_len = m.get("item_1").map(|s| s.len()).unwrap_or(0);
            let item7_len = m.get("item_7").map(|s| s.len()).unwrap_or(0);
            item1_len >= 400 && item7_len >= 2000
        })
        .unwrap_or(false)
}

/// Semicolon-joined list of `item_keys` present in `sections_map`, in key
/// order. Empty when the document failed to parse.
fn items_found_str(sections_map: Option<&HashMap<String, String>>, item_keys: &[String]) -> String {
    sections_map
        .map(|m| {
            item_keys
                .iter()
                .filter(|key| m.contains_key(key.as_str()))
                .cloned()
                .collect::<Vec<_>>()
                .join(";")
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_extensions_are_skipped_case_insensitively() {
        assert_eq!(
            binary_skip_reason(std::path::Path::new("exhibit.PDF"), u64::MAX),
            Some("skip: binary .pdf exhibit".to_string())
        );
        assert_eq!(
            binary_skip_reason(std::path::Path::new("image.GIF"), u64::MAX),
            Some("skip: binary .gif exhibit".to_string())
        );
        assert_eq!(
            binary_skip_reason(std::path::Path::new("doc.htm"), u64::MAX),
            None
        );
        assert_eq!(
            binary_skip_reason(std::path::Path::new("noext"), u64::MAX),
            None
        );
    }

    #[test]
    fn oversized_files_are_skipped() {
        let dir = std::env::temp_dir().join("tenk_audit_skip_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("filing.htm");
        std::fs::write(&path, "0123456789").unwrap();
        assert_eq!(binary_skip_reason(&path, 100), None);
        let reason = binary_skip_reason(&path, 5).unwrap();
        assert!(reason.starts_with("skip: file too large (10 bytes)"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_file_is_not_skipped() {
        assert_eq!(
            binary_skip_reason(std::path::Path::new("/nonexistent-xyz/f.htm"), 10),
            None
        );
    }

    #[test]
    fn adequacy_requires_both_item_thresholds() {
        let mut full = HashMap::new();
        full.insert("item_1".to_string(), "x".repeat(400));
        full.insert("item_7".to_string(), "y".repeat(2000));
        assert!(is_adequate_for(Some(&full)));

        let mut short_item7 = HashMap::new();
        short_item7.insert("item_1".to_string(), "x".repeat(400));
        short_item7.insert("item_7".to_string(), "y".repeat(1999));
        assert!(!is_adequate_for(Some(&short_item7)));

        let mut short_item1 = HashMap::new();
        short_item1.insert("item_1".to_string(), "x".repeat(399));
        short_item1.insert("item_7".to_string(), "y".repeat(2000));
        assert!(!is_adequate_for(Some(&short_item1)));

        assert!(!is_adequate_for(None));
        assert!(!is_adequate_for(Some(&HashMap::new())));
    }

    #[test]
    fn items_found_lists_present_keys_in_order() {
        let keys = vec![
            "item_1".to_string(),
            "item_7".to_string(),
            "item_8".to_string(),
        ];
        let mut sections = HashMap::new();
        sections.insert("item_7".to_string(), "a".to_string());
        sections.insert("item_1".to_string(), "b".to_string());
        assert_eq!(items_found_str(Some(&sections), &keys), "item_1;item_7");
        assert_eq!(items_found_str(None, &keys), "");
    }
}
