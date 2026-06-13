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
    parsers::CURRENT_US_GAAP_DATA_DIR,
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
    /// Output directory (will be created if it does not exist).
    #[arg(long, short = 'o', default_value = CURRENT_US_GAAP_DATA_DIR)]
    output_dir: PathBuf,

    /// Skip fetching custom→standard tag anchoring.  Only writes the
    /// fundamentals (companyfacts) CSVs.
    #[arg(long, default_value_t = false)]
    skip_anchoring: bool,
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

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client, false).await?;
    println!("Total primary listings: {}", company_tickers.len());

    // Create subdirectories.
    let fundamentals_dir = args.output_dir.join(FUNDAMENTALS_DIR);
    let anchoring_dir = args.output_dir.join(ANCHORING_DIR);
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
                            error_log
                                .insert(ticker.to_string(), format!("CSV write error: {}", e));
                        }
                    }
                    Err(e) => {
                        error_log.insert(
                            ticker.to_string(),
                            format!("File creation error: {}", e),
                        );
                    }
                }
            }
            Err(e) => {
                error_log.insert(ticker.to_string(), format!("Fundamentals fetch error: {}", e));
                continue;
            }
        }

        // ── Custom→standard anchoring (optional) ────────────────────────────
        if args.skip_anchoring {
            continue;
        }

        print!("  Anchoring... ");
        match fetch_custom_tag_anchoring(&client, &company_tickers, ticker).await {
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
                error_log.insert(
                    ticker.to_string(),
                    format!("Anchoring fetch error: {}", e),
                );
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
