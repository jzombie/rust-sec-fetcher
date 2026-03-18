//! Finds a company ticker entry by exact symbol match.
//!
//! Downloads the SEC company tickers dataset and performs an exact
//! case-sensitive lookup by ticker symbol.  Prints the full
//! [`sec_fetcher::models::Ticker`] record (CIK, company name) if found.
//!
//! # Usage
//!
//! ```text
//! cargo run --example ticker_show -- AAPL
//! cargo run --example ticker_show -- MSFT
//! cargo run --example ticker_show -- SPY
//! ```

use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{fetch_company_tickers, SecClient};
use std::error::Error;

#[derive(Parser)]
#[command(about = "Find a company ticker entry by exact symbol match")]
struct Args {
    /// Ticker symbol to look up (e.g. AAPL)
    ticker: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let ticker_symbol = TickerSymbol::new(&args.ticker);

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client, true).await.unwrap();
    // The SEC publishes thousands of registered company tickers.
    assert!(
        !company_tickers.is_empty(),
        "SEC company ticker list should not be empty"
    );

    let found = company_tickers.iter().find(|p| p.symbol == ticker_symbol);

    if let Some(t) = found {
        // Verify the returned entry actually matches the requested symbol.
        assert_eq!(
            t.symbol, ticker_symbol,
            "Returned ticker symbol must match the requested one"
        );
    }

    println!("Located: {:?}", found);

    Ok(())
}
