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

use clap::Parser;
use futures::StreamExt;
use sec_fetcher::{
    config::ConfigManager,
    network::{SecClient, fetch_10k_filings, fetch_best_10k_document, fetch_company_tickers},
};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info};

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

    // ── Pipelined producer / consumer ─────────────────────────────────────────
    //
    // Each per-ticker future:
    //   1. Fetches the filing list for that ticker.
    //   2. For each filing not yet on disk, downloads and saves the document.

    let output_dir = args.output_dir.clone();
    let log_dir = output_dir.clone();

    let producer = async move {
        let mut stream = futures::stream::iter(company_tickers.into_iter().enumerate())
            .map(|(i, entry)| {
                let output_dir = output_dir.clone();
                let client = Arc::clone(&client);
                async move {
                    let ticker = entry.symbol.to_string();
                    info!("[{}/{}] {}", i + 1, total_tickers, ticker);

                    let filings = match fetch_10k_filings(&client, entry.cik).await {
                        Ok(f) => f,
                        Err(e) => {
                            error!("  {}: filings fetch failed: {}", ticker, e);
                            return;
                        }
                    };

                    for filing in &filings {
                        let acc = filing.accession_number.to_string();

                        // Determine filename on disk.
                        let ticker_dir = output_dir.join(sanitize_dir_name(&ticker));
                        let ext = std::path::Path::new(&filing.primary_document)
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("txt");
                        let dest = ticker_dir.join(format!("{}.{}", sanitize_dir_name(&acc), ext));

                        // Skip if already downloaded.
                        if dest.exists() {
                            continue;
                        }

                        let bytes = match fetch_best_10k_document(&client, filing).await {
                            Ok(b) => b,
                            Err(e) => {
                                error!("  {} {}: document fetch failed: {}", ticker, acc, e);
                                continue;
                            }
                        };

                        // Save to disk.
                        if let Err(e) = std::fs::create_dir_all(&ticker_dir) {
                            error!("  Failed to create dir {:?}: {}", ticker_dir, e);
                            continue;
                        }
                        if let Err(e) = std::fs::write(&dest, bytes.as_ref()) {
                            error!("  Failed to write {:?}: {}", dest, e);
                            continue;
                        }
                    }
                }
            })
            .buffer_unordered(max_concurrent);

        while stream.next().await.is_some() {}
    };

    producer.await;

    info!("Done — documents saved under {:?}", log_dir);
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
