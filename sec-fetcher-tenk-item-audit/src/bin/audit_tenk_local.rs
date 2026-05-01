//! Audit the 10-K section parser against a local directory of filing documents.
//!
//! Reads every file (or every `.htm`/`.html`/`.txt` file) under a given input
//! directory, runs [`extract_sections_from_document`] on each one, and writes
//! the results to a CSV.  There are **no network calls** — all documents are
//! assumed to already exist on disk.
//!
//! This is useful for:
//! - Testing parser changes against a known corpus without hitting EDGAR.
//! - Re-running the parser on filings previously saved by `pull-tenk-items` (which
//!   stores raw documents under `<output>/tenk_audit_raw/<TICKER>/`).
//! - Bulk-evaluating parser quality or performance.
//!
//! # Usage
//!
//! ```sh
//! cargo run -p sec-fetcher-tenk-item-audit --bin audit-tenk-local --release -- \
//!     --dir tenk_audit_raw \        # directory of filing documents
//!     --output audit_tenk.csv       # output CSV
//! ```
//!
//! # Output columns
//!
//! Same columns originally produced by the old combined `pull-tenk-audit`
//! binary, except that `cik`, `filed_date`, and `form_type` are always empty
//! (that information is not embedded in the raw file on disk).
//!
//! | Column | Description |
//! |--------|-------------|
//! | `ticker` | Ticker symbol (from path when available) |
//! | `cik` | Always empty |
//! | `accession_number` | Accession number (from filename when available) |
//! | `filed_date` | Always empty |
//! | `form_type` | Always empty |
//! | `items_found` | Semicolon-separated list of item keys extracted |
//! | `item_1_chars` ... `item_16_chars` | Character count for every `TenKItem` |
//! | `error` | Non-empty when the parse step failed |

use clap::Parser;
use csv::WriterBuilder;
use sec_fetcher::{enums::TenKItem, parsers::extract_sections_from_document};
use sec_fetcher_tenk_item_audit::{
    build_row, collect_files, guess_accession, guess_ticker, header_columns,
};
use std::collections::HashMap;
use std::path::PathBuf;
use strum::IntoEnumIterator;
use tracing::{error, info, warn};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "audit-tenk-local",
    about = "Audit the 10-K section parser on a local directory of filing documents"
)]
struct Args {
    /// Directory containing filing documents to parse (searched recursively).
    #[arg(long, short = 'd')]
    dir: PathBuf,

    /// Path to the output CSV.
    #[arg(long, short = 'o', default_value = "audit_tenk.csv")]
    output: PathBuf,

    /// Optional extension filter (e.g., "*.htm", "*.html").
    /// Default: process all files (skipping hidden files).
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

    let args = Args::parse();

    // Build the ordered item-key list once (used by both header and rows).
    let item_keys: Vec<String> = TenKItem::iter().map(|i| i.map_key()).collect();

    // ── Build header ──────────────────────────────────────────────────────────

    let header = header_columns(&item_keys);

    // ── Collect files ─────────────────────────────────────────────────────────

    let all_files = collect_files(&args.dir, &args.pattern)?;
    let total = all_files.len();
    info!(
        "Found {} files under {:?} (pattern: {})",
        total, args.dir, args.pattern
    );

    if total == 0 {
        warn!("No files matched — nothing to do.");
        return Ok(());
    }

    // ── Open output CSV ───────────────────────────────────────────────────────

    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_path(&args.output)?;
    writer.write_record(&header)?;

    // ── Process each file ─────────────────────────────────────────────────────

    let mut errors = 0usize;

    for (i, file_path) in all_files.iter().enumerate() {
        let relative = file_path
            .strip_prefix(&args.dir)
            .unwrap_or(file_path)
            .display()
            .to_string();

        info!("[{}/{}] {}", i + 1, total, relative);

        let ticker = guess_ticker(file_path, &args.dir);
        let accession = guess_accession(file_path);

        let raw = match std::fs::read_to_string(file_path) {
            Ok(text) => text,
            Err(e) => {
                let row = build_row(
                    &ticker,
                    "",
                    &accession,
                    "",
                    "",
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

        let is_adequate = sections_map
            .as_ref()
            .map(|m| {
                let item1_len = m.get("item_1").map(|s| s.len()).unwrap_or(0);
                let item7_len = m.get("item_7").map(|s| s.len()).unwrap_or(0);
                item1_len >= 400 && item7_len >= 2000
            })
            .unwrap_or(false);

        let items_found = sections_map
            .as_ref()
            .map(|m| {
                item_keys
                    .iter()
                    .filter(|key| m.contains_key(key.as_str()))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(";")
            })
            .unwrap_or_default();

        let row = build_row(
            &ticker,
            "",
            &accession,
            "",
            "",
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
