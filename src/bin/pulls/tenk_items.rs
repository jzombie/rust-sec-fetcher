//! Download 10-K filings from EDGAR to a local directory.
//!
//! For every ticker in the full company-tickers dataset, fetches all 10-K and
//! 10-K405 filings and downloads the best available document for each one.
//! Amendment filings (`10-K/A`, `10-K405/A`) are included by default because
//! they carry corrections and restatements. Pass `--skip-amendments` to leave
//! them out, and add `--prune-amendments` to also delete previously-downloaded
//! amendment files. A plain pull never deletes anything on disk.
//! Documents are saved to `<output_dir>/<TICKER>/<ACCESSION>.<ext>`.
//!
//! This binary does **no parsing or audit-CSV generation** — it only pulls
//! raw documents.  Run `tenk-parser-audit` afterwards to evaluate the parser
//! against the downloaded corpus.
//!
//! # Usage
//!
//! ```sh
//! cargo run --bin pull-tenk-items --release
//! ```
//!
//! # Resuming
//!
//! If a file already exists on disk for a given `(ticker, accession_number)`,
//! it is skipped.  This makes the run safe to interrupt and resume.
//!
//! To re-fetch all documents from scratch, delete the output directory or
//! use a fresh `--output-dir`.

use clap::{CommandFactory, FromArgMatches, Parser};
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
    /// Output directory (default resolved from the shared data root).
    #[arg(long, short = 'o')]
    output_dir: Option<PathBuf>,

    /// Skip 10-K/A and 10-K405/A amendment filings (never download them).
    /// Amendments are included by default because they carry corrections and
    /// restatements — the unamended original alone can be stale. They often
    /// contain only the amended excerpt, so expect them to score inadequate
    /// as standalone audit rows.
    #[arg(long, default_value_t = false)]
    skip_amendments: bool,

    /// Delete previously-downloaded amendment files (`10-K/A`, `10-K405/A`)
    /// from the output directory. Requires `--skip-amendments`. Off by
    /// default — a plain pull never deletes anything.
    #[arg(long, default_value_t = false, requires = "skip_amendments")]
    prune_amendments: bool,
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

    let args = Args::from_arg_matches(
        &Args::command()
            .mut_arg("output_dir", |a| {
                a.help(format!(
                    "Output directory. Defaults to `{}`. Pass explicitly for an alternate location.",
                    sec_fetcher_shared::resolve_tenk_items_dir(None)
                ))
            })
            .get_matches(),
    )?;
    let output_dir = PathBuf::from(sec_fetcher_shared::resolve_tenk_items_dir(
        args.output_dir.as_ref().and_then(|p| p.to_str()),
    ));
    let include_amendments = !args.skip_amendments;
    // `requires = "skip_amendments"` on the flag guarantees prune implies
    // skip, so no runtime combination check is needed.
    let prune_amendments = args.prune_amendments;

    let config_manager = ConfigManager::load()?;
    let max_concurrent = config_manager
        .get_config()
        .max_concurrent
        .unwrap_or(1)
        .max(1);
    let client = Arc::new(SecClient::from_config_manager(&config_manager)?);
    info!(
        "Pulling 10-K documents into {:?} (max_concurrent={}, include_amendments={}, prune_amendments={})",
        output_dir, max_concurrent, include_amendments, prune_amendments,
    );

    // ── Ensure output directory exists ────────────────────────────────────────

    std::fs::create_dir_all(&output_dir)?;

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
                    // Amendment accession stems seen this run — pruned in one
                    // pass after the loop instead of scanning the directory
                    // per filing.
                    let mut prune_stems: std::collections::HashSet<String> =
                        std::collections::HashSet::new();

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

                        // Amendments are included by default (corrections and
                        // restatements); with `--skip-amendments` they are
                        // skipped here and, with `--prune-amendments`,
                        // previously-downloaded copies are removed in one
                        // pass after the loop. A plain pull never deletes.
                        if !include_amendments
                            && filing.form.trim().to_ascii_uppercase().ends_with("/A")
                        {
                            if prune_amendments {
                                prune_stems.insert(sanitize_dir_name(&acc));
                            }
                            skipped += 1;
                            info!(
                                "[{ticker}]   {date_str}  {acc}  ({}) — SKIP (amendment)",
                                filing.form,
                            );
                            continue;
                        }

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

                    // ── Prune skipped amendments in one pass ────────────────────
                    //
                    // Single `read_dir` scan per ticker (not per filing), run
                    // after all downloads so a mid-loop `remove_dir` can never
                    // force later filings to recreate the directory.
                    if !prune_stems.is_empty() {
                        let ticker_dir = output_dir.join(sanitize_dir_name(&ticker));
                        if ticker_dir.is_dir()
                            && let Ok(entries) = std::fs::read_dir(&ticker_dir)
                        {
                            for e in entries.flatten() {
                                let p = e.path();
                                if p.is_file()
                                    && let Some(stem) = p.file_stem().and_then(|s| s.to_str())
                                    && prune_stems.contains(stem)
                                {
                                    let _ = std::fs::remove_file(&p);
                                }
                            }
                            let _ = std::fs::remove_dir(&ticker_dir);
                        }
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

// TODO: Move into a shared utility?
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
