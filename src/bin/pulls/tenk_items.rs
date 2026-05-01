//! Download 10-K filings from EDGAR to a local directory.
//!
//! For every ticker in the full company-tickers dataset, fetches all 10-K and
//! 10-K405 filings and downloads the best available document for each one.
//! Documents are saved to `<output_dir>/<TICKER>/<ACCESSION>.<ext>`.
//!
//! This binary does **no parsing or audit-CSV generation** — it only pulls
//! raw documents.  Run `audit-tenk-local` afterwards to evaluate the parser
//! against the downloaded corpus.
//!
//! # Usage
//!
//! ```sh
//! cargo run --bin tenk-items --release -- --output-dir tenk_audit_raw
//! ```
//!
//! # Resuming
//!
//! If a file already exists on disk for a given `(ticker, accession_number)`,
//! it is skipped.  This makes the run safe to interrupt and resume.
//!
//! To re-fetch all documents from scratch, delete the output directory or
//! use a fresh `--output-dir`.

use clap::Parser;
use futures::StreamExt;
use sec_fetcher::{
    config::ConfigManager,
    network::{SecClient, fetch_10k_filings, fetch_best_10k_document, fetch_company_tickers},
};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{error, info};

// ── Global counters ───────────────────────────────────────────────────────────

static GLOBAL_TICKERS_OK: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_TICKERS_ERR: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_DOWNLOADED: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_SKIPPED: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_FAILED: AtomicUsize = AtomicUsize::new(0);

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "pull-tenk-items",
    about = "Download 10-K filings from EDGAR to a local directory"
)]
struct Args {
    /// Directory to save raw documents into.
    #[arg(long, short = 'o', default_value = "data/tenk_items")]
    output_dir: PathBuf,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let config_manager = ConfigManager::load()?;
    let max_concurrent = config_manager
        .get_config()
        .max_concurrent
        .unwrap_or(1)
        .max(1);
    let client = Arc::new(SecClient::from_config_manager(&config_manager)?);
    info!(
        "Pulling 10-K documents into {:?} (max_concurrent={})",
        args.output_dir, max_concurrent,
    );

    // ── Ensure output directory exists ────────────────────────────────────────

    std::fs::create_dir_all(&args.output_dir)?;

    // ── Fetch full ticker list (deduplicated by CIK) ──────────────────────────

    let all_tickers = fetch_company_tickers(&client, false).await?;
    let mut seen_ciks = std::collections::HashSet::new();
    let company_tickers: Vec<_> = all_tickers
        .into_iter()
        .filter(|t| seen_ciks.insert(t.cik.to_string()))
        .collect();
    let total_tickers = company_tickers.len();
    info!("Total unique CIKs to process: {}", total_tickers);

    // ── Track current ticker index globally for progress ──────────────────────

    let ticker_index = Arc::new(AtomicUsize::new(0));

    // ── Pipelined producer / consumer ─────────────────────────────────────────
    //
    // Each per-ticker future:
    //   1. Fetches the filing list for that ticker.
    //   2. For each filing not yet on disk, downloads and saves the document.
    //   3. Logs a summary line with per-ticker status.

    let output_dir = args.output_dir.clone();
    let log_dir = output_dir.clone();

    let producer = async move {
        let mut stream = futures::stream::iter(company_tickers.into_iter())
            .map(|entry| {
                let output_dir = output_dir.clone();
                let client = Arc::clone(&client);
                let idx = ticker_index.fetch_add(1, Ordering::Relaxed);
                async move {
                    let ticker = entry.symbol.to_string();
                    let ticker_idx = idx + 1; // 1-based for display

                    // ── Per-ticker counters ───────────────────────────────────
                    let mut downloaded: usize = 0;
                    let mut skipped: usize = 0;
                    let mut failed: usize = 0;

                    let filings = match fetch_10k_filings(&client, entry.cik).await {
                        Ok(f) => f,
                        Err(e) => {
                            error!(
                                "[{ticker}] [{ticker_idx}/{total_tickers}] filing list failed: {e}"
                            );
                            GLOBAL_TICKERS_ERR.fetch_add(1, Ordering::Relaxed);
                            return;
                        }
                    };

                    let total_filings = filings.len();
                    info!(
                        "[{ticker}] [{ticker_idx}/{total_tickers}] found {total_filings} 10-K filing(s)"
                    );

                    for filing in &filings {
                        let acc = filing.accession_number.to_string();
                        let date_str = filing
                            .filing_date
                            .map(|d| d.to_string())
                            .unwrap_or_else(|| "unknown-date".to_string());

                        // Determine filename on disk.
                        let ticker_dir = output_dir.join(sanitize_dir_name(&ticker));
                        let ext = std::path::Path::new(&filing.primary_document)
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("txt");
                        let dest = ticker_dir.join(format!("{}.{}", sanitize_dir_name(&acc), ext));

                        // Skip if already downloaded.
                        if dest.exists() {
                            skipped += 1;
                            info!(
                                "[{ticker}]   {date_str}  {acc}  — SKIP (exists on disk)"
                            );
                            continue;
                        }

                        info!(
                            "[{ticker}]   {date_str}  {acc}  — DOWNLOADING ..."
                        );

                        let bytes = match fetch_best_10k_document(&client, filing).await {
                            Ok(b) => b,
                            Err(e) => {
                                failed += 1;
                                error!(
                                    "[{ticker}]   {date_str}  {acc}  — FAILED: {e}"
                                );
                                continue;
                            }
                        };

                        // Save to disk.
                        if let Err(e) = std::fs::create_dir_all(&ticker_dir) {
                            failed += 1;
                            error!("[{ticker}]   {date_str}  {acc}  — FAILED: cannot create dir {ticker_dir:?}: {e}");
                            continue;
                        }
                        if let Err(e) = std::fs::write(&dest, bytes.as_ref()) {
                            failed += 1;
                            error!("[{ticker}]   {date_str}  {acc}  — FAILED: cannot write {dest:?}: {e}");
                            continue;
                        }

                        downloaded += 1;
                        info!(
                            "[{ticker}]   {date_str}  {acc}  — saved ({size}B)",
                            size = bytes.len()
                        );
                    }

                    // ── Per-ticker summary ────────────────────────────────────
                    GLOBAL_DOWNLOADED.fetch_add(downloaded, Ordering::Relaxed);
                    GLOBAL_SKIPPED.fetch_add(skipped, Ordering::Relaxed);
                    GLOBAL_FAILED.fetch_add(failed, Ordering::Relaxed);
                    GLOBAL_TICKERS_OK.fetch_add(1, Ordering::Relaxed);

                    info!(
                        "[{ticker}] [{ticker_idx}/{total_tickers}] done — \
                         {total_filings} filing(s): \
                         {downloaded} downloaded, \
                         {skipped} skipped, \
                         {failed} failed"
                    );
                }
            })
            .buffer_unordered(max_concurrent);

        while stream.next().await.is_some() {}
    };

    producer.await;

    // ── Final report ──────────────────────────────────────────────────────────

    let final_downloaded = GLOBAL_DOWNLOADED.load(Ordering::Relaxed);
    let final_skipped = GLOBAL_SKIPPED.load(Ordering::Relaxed);
    let final_failed = GLOBAL_FAILED.load(Ordering::Relaxed);
    let final_ok = GLOBAL_TICKERS_OK.load(Ordering::Relaxed);
    let final_err = GLOBAL_TICKERS_ERR.load(Ordering::Relaxed);

    info!(
        "═══ Done — documents saved under {:?} ═══\n\
         Tickers processed: {final_ok} OK, {final_err} failed\n\
         Documents: {final_downloaded} downloaded, {final_skipped} skipped, {final_failed} failed",
        log_dir,
    );
    Ok(())
}

/// Replace characters unsafe for filesystem paths with `_`.
fn sanitize_dir_name(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect()
}
