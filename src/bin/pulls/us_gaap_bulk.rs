//! Bulk-downloads US-GAAP XBRL fundamentals for every primary-listing ticker.
//!
//! Companyfacts-only scope: values come from `companyfacts.json`
//! (`us-gaap:*` only), balances from the FASB taxonomy. Filer customs
//! (`bac:*`) are out of scope — neither companyfacts nor FASB XSDs
//! contain them.
//!
//! Writes per-ticker CSVs into `--output-dir`:
//!
//! ```text
//! data/latest-us-gaap/
//!   us-gaap-fundamentals/   ← AAPL.csv, BAC.csv, … (companyfacts financial data)
//! ```
//!
//! # Usage
//!
//! ```sh
//! cargo run --bin pull-us-gaap-bulk --release -- --output-dir data/12-jun-2026-us-gaap
//! ```

use clap::Parser;
use polars::prelude::{CsvWriter, SerWriter};
use sec_fetcher::{
    config::ConfigManager,
    models::TickerSymbol,
    network::{SecClient, fetch_company_tickers, fetch_us_gaap_fundamentals},
};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

const FUNDAMENTALS_DIR: &str = "us-gaap-fundamentals";

#[derive(Parser)]
#[command(
    name = "pull-us-gaap-bulk",
    about = "Bulk-download US-GAAP fundamentals (companyfacts-only)"
)]
struct Args {
    /// Output directory. Defaults to `$SEC_FETCHER_DATA_DIR` when set,
    /// else `data/latest-us-gaap`. Pass explicitly for dated snapshots.
    #[arg(long, short = 'o')]
    output_dir: Option<PathBuf>,
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

    let fundamentals_dir = output_dir.join(FUNDAMENTALS_DIR);
    tokio::fs::create_dir_all(&fundamentals_dir).await?;

    let mut error_log: HashMap<String, String> = HashMap::new();

    for (i, company_ticker) in company_tickers.iter().enumerate() {
        let ticker: &TickerSymbol = &company_ticker.symbol;

        println!(
            "Processing ticker: {} ({} of {})",
            ticker,
            i + 1,
            company_tickers.len()
        );

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
