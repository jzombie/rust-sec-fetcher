//! Bulk-audits 10-K section extraction across every EDGAR primary listing.
//!
//! For every ticker in the full company-tickers dataset, fetches all 10-K and
//! 10-K405 filings and runs [`extract_sections_from_document`] on each one.
//! Results are written incrementally to a CSV so the run is safe to interrupt
//! and resume.
//!
//! # Usage
//!
//! ```sh
//! cargo run --bin pull-tenk-audit --release -- --output audit.csv
//! ```
//!
//! # Throughput
//!
//! Throughput is determined by `max_concurrent` and `min_delay_ms` in
//! `sec_fetcher_config.toml`.  Each concurrent slot contributes
//! `1000 / min_delay_ms` requests per second, so:
//!
//! | max_concurrent | min_delay_ms | req/s |
//! |---|---|---|
//! | 1 (default) | 500 (default) | 2 |
//! | 4 | 500 | 8 |
//! | 5 | 500 | 10 (SEC limit) |
//!
//! # Resuming
//!
//! If `--output` already exists, rows already present are read and those
//! `(cik, accession_number)` pairs are skipped.  Append new results to the
//! same file so incremental runs are cheap.
//!
//! # Output columns
//!
//! | Column | Description |
//! |--------|-------------|
//! | `ticker` | Ticker symbol |
//! | `cik` | CIK as zero-padded 10-digit string |
//! | `accession_number` | SEC accession number |
//! | `filed_date` | Filing date (YYYY-MM-DD) |
//! | `form_type` | `10-K` or `10-K405` |
//! | `items_found` | Semicolon-separated list of item keys extracted |
//! | `item_1_chars` … `item_16_chars` | Character count for every `TenKItem` |
//! | `error` | Non-empty when the fetch or parse step failed |

use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use futures::StreamExt;
use sec_fetcher::{
    config::ConfigManager,
    enums::TenKItem,
    network::{SecClient, fetch_10k_filings, fetch_best_10k_document, fetch_company_tickers},
    parsers::extract_sections_from_document,
};
use std::{collections::HashSet, fs::OpenOptions, path::PathBuf, sync::Arc};
use strum::IntoEnumIterator;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "pull-tenk-audit",
    about = "Audit 10-K section extraction for every EDGAR primary listing"
)]
struct Args {
    /// Path to the output CSV (created if absent; existing rows are skipped).
    #[arg(long, short = 'o', default_value = "tenk_audit.csv")]
    output: PathBuf,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy()
                // Suppress html5ever / html5 parser debug chatter which floods
                // stderr and dominates wall-clock time on large HTML documents.
                .add_directive("html5ever=off".parse().unwrap())
                .add_directive("selectors=off".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    let config_manager = ConfigManager::load()?;
    let max_concurrent = config_manager
        .get_config()
        .max_concurrent
        .unwrap_or(1)
        // Use at least 1; the SEC client enforces the real rate limit internally.
        .max(1);
    let client = Arc::new(SecClient::from_config_manager(&config_manager)?);
    info!("Processing with max_concurrent={}", max_concurrent);

    // ── Build header / column order once ─────────────────────────────────────

    let items: Vec<TenKItem> = TenKItem::iter().collect();

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
    for item in &items {
        header.push(format!("{}_chars", item.map_key()));
    }
    header.push("error".into());

    // ── Collect already-processed (cik, accession_number) pairs ──────────────

    let mut seen: HashSet<(String, String)> = HashSet::new();
    // A file counts as "existing" only if it already has content — an empty
    // file (left over from a previous aborted run before the header row was
    // flushed) should be treated the same as absent.
    let file_existed =
        args.output.exists() && args.output.metadata().map(|m| m.len() > 0).unwrap_or(false);

    if file_existed {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&args.output)?;
        // cik=index 1, accession_number=index 2
        for result in rdr.records() {
            let record = result?;
            let cik = record.get(1).unwrap_or("").to_string();
            let acc = record.get(2).unwrap_or("").to_string();
            if !cik.is_empty() && !acc.is_empty() {
                seen.insert((cik, acc));
            }
        }
        info!("Resuming — {} rows already in output", seen.len());
    }

    // ── Open output file ──────────────────────────────────────────────────────

    let out_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)?;
    // has_headers(false): we manage the header row ourselves so appending
    // to an existing file doesn't re-emit it.
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(out_file);

    if !file_existed {
        writer.write_record(&header)?;
    }

    // ── Fetch full ticker list ────────────────────────────────────────────────

    let company_tickers = fetch_company_tickers(&client, false).await?;
    let total_tickers = company_tickers.len();
    info!("Total primary listings: {}", total_tickers);

    // ── Pipelined producer / consumer ─────────────────────────────────────────
    //
    // The producer drives buffer_unordered(max_concurrent) over tickers.  Each
    // per-ticker future:
    //   1. Fetches the filing list for that ticker.
    //   2. Iterates every new (not-yet-seen) filing, fetching and parsing the
    //      document, then sends a completed row down an mpsc channel.
    //
    // The consumer reads from the channel and writes each row to CSV immediately.
    //
    // tokio::join! runs both futures in the *same task*, so neither needs to be
    // Send, and csv::Writer never crosses a thread boundary.
    //
    // The first CSV row appears as soon as the very first document fetch
    // completes — usually within a few seconds — regardless of how many tickers
    // remain.

    let items_arc = Arc::new(items);
    let seen_arc = Arc::new(seen);

    let (row_tx, row_rx) = mpsc::unbounded_channel::<Vec<String>>();

    let producer = {
        // Clone one sender for the producer scope; the outer `row_tx` is
        // dropped immediately after this block so the channel closes once this
        // clone (and every per-ticker clone derived from it) are dropped.
        let row_tx_p = row_tx.clone();
        async move {
            let mut stream = futures::stream::iter(company_tickers.into_iter().enumerate())
                .map(|(i, entry)| {
                    let client = Arc::clone(&client);
                    let items = Arc::clone(&items_arc);
                    let seen = Arc::clone(&seen_arc);
                    let tx = row_tx_p.clone();
                    async move {
                        let ticker = entry.symbol.to_string();
                        let cik = entry.cik.to_string();
                        info!("[{}/{}] {}", i + 1, total_tickers, ticker);

                        let filings = match fetch_10k_filings(&client, entry.cik).await {
                            Ok(f) => f,
                            Err(e) => {
                                let row = build_row(
                                    &ticker,
                                    &cik,
                                    "",
                                    "",
                                    "",
                                    &items,
                                    "",
                                    None,
                                    &format!("filings: {}", e),
                                );
                                let _ = tx.send(row);
                                return;
                            }
                        };

                        for filing in filings {
                            let acc = filing.accession_number.to_string();
                            if seen.contains(&(cik.clone(), acc.clone())) {
                                continue;
                            }

                            let filed_date = filing
                                .filing_date
                                .map(|d| d.to_string())
                                .unwrap_or_default();
                            let form_type = filing.form_type().to_string();

                            let (sections_opt, err_msg) =
                                match fetch_best_10k_document(&client, &filing).await {
                                    Ok(bytes) => {
                                        let text = String::from_utf8_lossy(&bytes);
                                        let sections = extract_sections_from_document(&text);
                                        let n = sections.keys().count();
                                        info!(
                                            "  {} {} {} — {} items",
                                            ticker, filed_date, form_type, n
                                        );
                                        (Some(sections), String::new())
                                    }
                                    Err(e) => {
                                        let msg = format!("doc: {}", e);
                                        error!(
                                            "  {} {} {}: {}",
                                            ticker, filed_date, form_type, msg
                                        );
                                        (None, msg)
                                    }
                                };

                            // Emit items in document order (Item 1 … Item 16)
                            // rather than HashMap iteration order.
                            let items_found = sections_opt
                                .as_ref()
                                .map(|s| {
                                    items
                                        .iter()
                                        .map(|item| item.map_key())
                                        .filter(|key| s.get(key.as_str()).is_some())
                                        .collect::<Vec<_>>()
                                        .join(";")
                                })
                                .unwrap_or_default();

                            let row = build_row(
                                &ticker,
                                &cik,
                                &acc,
                                &filed_date,
                                &form_type,
                                &items,
                                &items_found,
                                sections_opt.as_ref(),
                                &err_msg,
                            );
                            let _ = tx.send(row);
                        }
                    }
                })
                .buffer_unordered(max_concurrent);

            while stream.next().await.is_some() {}
        }
    };

    // Drop the original sender so the channel closes when the producer's own
    // clone (and all per-ticker clones) are dropped at stream exhaustion.
    drop(row_tx);

    let consumer = async move {
        let mut errors = 0usize;
        let mut row_rx = row_rx;
        while let Some(row) = row_rx.recv().await {
            if row.last().map(|e| !e.is_empty()).unwrap_or(false) {
                errors += 1;
            }
            if let Err(e) = writer.write_record(&row) {
                eprintln!("CSV write error: {e}");
            }
            let _ = writer.flush();
        }
        errors
    };

    let ((), error_count) = tokio::join!(producer, consumer);

    if error_count > 0 {
        warn!(
            "Completed with {} errors — see 'error' column in output.",
            error_count
        );
    } else {
        info!("Completed successfully.");
    }

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn build_row(
    ticker: &str,
    cik: &str,
    acc: &str,
    filed_date: &str,
    form_type: &str,
    items: &[TenKItem],
    items_found: &str,
    sections: Option<&sec_fetcher::parsers::TenKSections>,
    err: &str,
) -> Vec<String> {
    let adequate = sections.map(|s| s.is_adequate()).unwrap_or(false);
    let mut record: Vec<String> = vec![
        ticker.into(),
        cik.into(),
        acc.into(),
        filed_date.into(),
        form_type.into(),
        if adequate { "true" } else { "false" }.into(),
        snippet(sections.and_then(|s| s.get("item_1"))),
        snippet(sections.and_then(|s| s.get("item_7"))),
        items_found.into(),
    ];
    for item in items {
        let count = sections
            .and_then(|s| s.get(&item.map_key()))
            .map(|v| v.len())
            .unwrap_or(0);
        record.push(count.to_string());
    }
    record.push(err.into());
    record
}

/// First 150 chars of `text`, with runs of whitespace collapsed to a single
/// space.  Returns an empty string when `text` is `None`.
fn snippet(text: Option<&str>) -> String {
    match text {
        None => String::new(),
        Some(t) => {
            let collapsed: String = t.split_whitespace().collect::<Vec<_>>().join(" ");
            collapsed.chars().take(150).collect()
        }
    }
}
