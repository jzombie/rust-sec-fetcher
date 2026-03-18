//! Iterates every investment company fund, fetches its latest NPORT-P holdings,
//! and saves each fund's positions as a CSV under `data/fund-holdings/<letter>/`.
//!
//! Output is organised by first letter of the ticker so the directory stays
//! manageable: `data/fund-holdings/S/SPY.csv`, `data/fund-holdings/Q/QQQ.csv`, …
//!
//! # Usage
//!
//! ```sh
//! # RUST_LOG=sec_fetcher=info
//! cargo run --bin pull-fund-holdings --release -- --output-dir data/fund-holdings
//! ```

use clap::Parser;
use log::{error, info};
use sec_fetcher::{
    config::ConfigManager,
    models::TickerSymbol,
    network::{
        fetch_cik_by_ticker_symbol, fetch_investment_company_series_and_class_dataset, fetch_nport,
        fetch_nport_filings, SecClient,
    },
    utils::VecExtensions,
};
use tokio::fs::create_dir_all;

#[derive(Parser)]
#[command(
    name = "pull-fund-holdings",
    about = "Download latest NPORT-P holdings for all investment company funds into CSV files"
)]
struct Args {
    /// Output directory (will be created if it does not exist).
    /// Files are organised as `<output-dir>/<letter>/<TICKER>.csv`.
    /// Example: data/fund-holdings
    #[arg(long, short = 'o')]
    output_dir: std::path::PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    env_logger::Builder::from_default_env()
        .filter(None, log::LevelFilter::Info)
        .init();

    let mut error_log: Vec<String> = Vec::new();

    let config_manager = match ConfigManager::load() {
        Ok(cfg) => cfg,
        Err(e) => {
            let msg = format!("Failed to load ConfigManager: {}", e);
            error!("{}", msg);
            error_log.push(msg);
            return Err(e);
        }
    };

    let client = match SecClient::from_config_manager(&config_manager) {
        Ok(client) => client,
        Err(e) => {
            let msg = format!("Failed to initialize SecClient: {}", e);
            error!("{}", msg);
            error_log.push(msg);
            return Err(e);
        }
    };

    let investment_companies =
        match fetch_investment_company_series_and_class_dataset(&client).await {
            Ok(data) => data,
            Err(e) => {
                let msg = format!("Failed to fetch investment companies: {}", e);
                error!("{}", msg);
                error_log.push(msg);
                return Err(e);
            }
        };

    let total_investment_companies = investment_companies.len();
    info!("Total investment companies: {}", total_investment_companies);

    for (i, fund) in investment_companies.iter().enumerate() {
        info!("Processing: {} of {}", i + 1, total_investment_companies);

        if let Some(raw_ticker) = &fund.class_ticker {
            let ticker = TickerSymbol::new(raw_ticker);
            info!("Ticker symbol: {}", ticker);
            info!("Fetching latest NPORT-P filing...");

            let cik = match fetch_cik_by_ticker_symbol(&client, &ticker).await {
                Ok(cik) => cik,
                Err(e) => {
                    let msg = format!("CIK lookup failed for {}: {}", ticker, e);
                    error!("{}", msg);
                    error_log.push(msg);
                    continue;
                }
            };

            let nport_filings = match fetch_nport_filings(&client, cik).await {
                Ok(filings) => filings,
                Err(e) => {
                    let msg = format!("Failed to fetch NPORT listings for {}: {}", ticker, e);
                    error!("{}", msg);
                    error_log.push(msg);
                    continue;
                }
            };

            let latest = match nport_filings.first() {
                Some(sub) => sub,
                None => {
                    let msg = format!("No NPORT-P filings found for {}", ticker);
                    error!("{}", msg);
                    error_log.push(msg);
                    continue;
                }
            };

            let investments = match fetch_nport(&client, latest).await {
                Ok(inv) => inv,
                Err(e) => {
                    let msg = format!("Failed to parse NPORT filing for {}: {}", ticker, e);
                    error!("{}", msg);
                    error_log.push(msg);
                    continue;
                }
            };
            info!("Fetched latest NPORT-P filing");

            let first_letter = ticker.chars().next().unwrap_or('_').to_ascii_uppercase();
            let dir_path = args.output_dir.join(first_letter.to_string());

            if let Err(e) = create_dir_all(&dir_path).await {
                let msg = format!("Failed to create directory {}: {}", dir_path.display(), e);
                error!("{}", msg);
                error_log.push(msg);
                continue;
            }

            info!("Total records for {}: {}", ticker, investments.len());

            info!("Writing CSV...");
            let file_path = dir_path.join(format!("{}.csv", ticker));
            if let Err(e) = investments.write_to_csv(file_path.to_str().unwrap()) {
                let msg = format!("Failed to write CSV for {}: {}", ticker, e);
                error!("{}", msg);
                error_log.push(msg);
            }
        }
    }

    if !error_log.is_empty() {
        error!("=== SUMMARY OF ERRORS ===");
        for err in &error_log {
            error!("{}", err);
        }
    } else {
        info!("All funds processed successfully!");
    }

    Ok(())
}
