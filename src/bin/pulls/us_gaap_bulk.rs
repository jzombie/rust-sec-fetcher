//! Bulk-downloads US-GAAP XBRL fundamentals (and optionally custom→standard
//! tag anchoring) for every primary-listing ticker.
//!
//! Writes per-ticker CSVs into two subdirectories under `--output-dir`:
//!
//! ```text
//! data/12-jun-2026-us-gaap/
//!   us-gaap-fundamentals/   ← AAPL.csv, BAC.csv, … (companyfacts financial data)
//!   custom-tag-anchoring/   ← BAC.csv, … (extension-tag → GAAP-parent mappings)
//! ```
//! Anchoring is factual XBRL evidence only (`parent, weight, balances,
//! conflict`). `ValuePolarity` (`Higher/LowerIsBetter`) is NOT decided here —
//! single owner is textualizer (currently closed-source) `sec-xbrl-semantics-rules/src/heuristics.rs`,
//! which maps parent polarity + weight → custom polarity for the store that
//! hands trend-describer its per-field polarity.
//!
//! # Usage
//!
//! ```sh
//! cargo run --bin pull-us-gaap-bulk --release -- --output-dir data/12-jun-2026-us-gaap
//! cargo run --bin pull-us-gaap-bulk --release -- --output-dir data/12-jun-2026-us-gaap --skip-anchoring
//! ```

use clap::Parser;
use polars::prelude::{CsvWriter, SerWriter};
use sec_fetcher::{
    config::ConfigManager,
    models::TickerSymbol,
    network::{
        SecClient, fetch_company_tickers, fetch_custom_tag_anchoring, fetch_us_gaap_fundamentals,
    },
};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

const FUNDAMENTALS_DIR: &str = "us-gaap-fundamentals";
const ANCHORING_DIR: &str = "custom-tag-anchoring";

#[derive(Parser)]
#[command(
    name = "pull-us-gaap-bulk",
    about = "Bulk-download US-GAAP fundamentals + custom-tag anchoring"
)]
struct Args {
    /// Output directory. Defaults to `$SEC_FETCHER_DATA_DIR` when set,
    /// else `data/latest-us-gaap`. Pass explicitly for dated snapshots.
    #[arg(long, short = 'o')]
    output_dir: Option<PathBuf>,

    /// Skip fetching custom→standard tag anchoring.  Only writes the
    /// fundamentals (companyfacts) CSVs.
    #[arg(long, default_value_t = false)]
    skip_anchoring: bool,

    /// Only scan the N most-recent periodic filings per ticker for anchoring.
    /// Omit for full history (guarantees conflict detection back to inception).
    #[arg(long)]
    max_anchoring_filings: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::DEBUG.into())
                .from_env_lossy(),
        )
        .init();

    let args = Args::parse();
    let output_dir = PathBuf::from(sec_fetcher_shared::resolve_data_dir(
        args.output_dir.as_ref().and_then(|p| p.to_str()),
    ));

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client, false).await?;
    println!("Total primary listings: {}", company_tickers.len());

    // Create subdirectories.
    let fundamentals_dir = output_dir.join(FUNDAMENTALS_DIR);
    let anchoring_dir = output_dir.join(ANCHORING_DIR);
    tokio::fs::create_dir_all(&fundamentals_dir).await?;
    if !args.skip_anchoring {
        tokio::fs::create_dir_all(&anchoring_dir).await?;
    }

    let mut error_log: HashMap<String, String> = HashMap::new();

    for (i, company_ticker) in company_tickers.iter().enumerate() {
        let ticker: &TickerSymbol = &company_ticker.symbol;

        println!(
            "Processing ticker: {} ({} of {})",
            ticker,
            i + 1,
            company_tickers.len()
        );

        // ── Fundamentals (companyfacts) ─────────────────────────────────────
        match fetch_us_gaap_fundamentals(&client, &company_tickers, ticker).await {
            Ok(mut fundamentals_df) => {
                let file_path = fundamentals_dir.join(format!("{}.csv", ticker));
                match File::create(&file_path) {
                    Ok(mut file) => {
                        if let Err(e) = CsvWriter::new(&mut file)
                            .include_header(true)
                            .finish(&mut fundamentals_df)
                        {
                            error_log.insert(ticker.to_string(), format!("CSV write error: {}", e));
                        }
                    }
                    Err(e) => {
                        error_log.insert(ticker.to_string(), format!("File creation error: {}", e));
                    }
                }
            }
            Err(e) => {
                error_log.insert(
                    ticker.to_string(),
                    format!("Fundamentals fetch error: {}", e),
                );
                continue;
            }
        }

        // ── Custom→standard anchoring (optional) ────────────────────────────
        if args.skip_anchoring {
            continue;
        }

        print!("  Anchoring... ");
        match fetch_custom_tag_anchoring(
            &client,
            &company_tickers,
            ticker,
            args.max_anchoring_filings,
        )
        .await
        {
            Ok(mut anchoring_df) => {
                let count = anchoring_df.height();
                if count == 0 {
                    println!("no custom tags found");
                    continue;
                }
                println!("{} relationships", count);
                let file_path = anchoring_dir.join(format!("{}.csv", ticker));
                match File::create(&file_path) {
                    Ok(mut file) => {
                        if let Err(e) = CsvWriter::new(&mut file)
                            .include_header(true)
                            .finish(&mut anchoring_df)
                        {
                            error_log.insert(
                                ticker.to_string(),
                                format!("Anchoring CSV write error: {}", e),
                            );
                        }
                    }
                    Err(e) => {
                        error_log.insert(
                            ticker.to_string(),
                            format!("Anchoring file creation error: {}", e),
                        );
                    }
                }
            }
            Err(e) => {
                error_log.insert(ticker.to_string(), format!("Anchoring fetch error: {}", e));
            }
        }
    }

    if !error_log.is_empty() {
        println!("\nSummary of errors:");
        for (ticker, err) in &error_log {
            println!("- {}: {}", ticker, err);
        }
    } else {
        println!("\nAll tickers processed successfully.");
    }

    Ok(())
}
