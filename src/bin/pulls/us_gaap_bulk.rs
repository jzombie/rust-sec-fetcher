//! Bulk-downloads US-GAAP XBRL fundamentals for every primary-listing ticker
//! and saves each company's data as a CSV in the given output directory.
//!
//! # Usage
//!
//! ```sh
//! # RUST_LOG=sec_fetcher=info,reqwest_drive=debug
//! cargo run --bin pull-us-gaap-bulk --release -- --output-dir data/16-mar-2026-us-gaap
//! ```

use clap::Parser;
use polars::prelude::{CsvWriter, SerWriter};
use sec_fetcher::{
    config::ConfigManager,
    models::TickerSymbol,
    network::{fetch_company_tickers, fetch_us_gaap_fundamentals, SecClient},
    utils::VecExtensions,
};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "pull-us-gaap-bulk",
    about = "Bulk-download US-GAAP fundamentals for all primary listings into CSV files"
)]
struct Args {
    /// Output directory (will be created if it does not exist).
    /// Example: data/16-mar-2026-us-gaap
    #[arg(long, short = 'o')]
    output_dir: PathBuf,
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

    // include_derived_instruments=false: primary listings only. Derived
    // instruments (warrants, units, prefs, delisted) share a CIK with their
    // parent and have no independent XBRL data, so they'd produce duplicates
    // or empty CSVs in this pipeline.
    let company_tickers = fetch_company_tickers(&client, false).await?;
    println!("Total primary listings: {}", company_tickers.len());
    println!("{:?}", company_tickers.head(60));

    tokio::fs::create_dir_all(&args.output_dir).await?;

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
                let mut file_path = args.output_dir.clone();
                file_path.push(format!("{}.csv", ticker));
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
                        eprintln!("File creation error: {}", e);
                        error_log.insert(ticker.to_string(), format!("File creation error: {}", e));
                    }
                }
            }
            Err(e) => {
                error_log.insert(ticker.to_string(), format!("Fetch error: {}", e));
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
