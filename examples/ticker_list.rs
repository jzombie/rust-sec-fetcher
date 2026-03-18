//! Lists SEC company tickers from the EDGAR company tickers dataset.
//!
//! Downloads the full SEC operating company list (and optionally the derived
//! instruments list) and prints each entry as a tab-separated row of
//! ticker symbol, CIK, and company name.  Supports filtering by a
//! case-insensitive substring match on either field.
//!
//! # Usage
//!
//! ```text
//! cargo run --example ticker_list
//! cargo run --example ticker_list -- --filter apple
//! cargo run --example ticker_list -- --filter MSFT
//! ```

use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::Ticker;
use sec_fetcher::network::{SecClient, fetch_company_tickers};
use std::error::Error;
use std::fmt;

#[derive(Parser)]
#[command(
    about = "List SEC company tickers (operating companies + optionally derived instruments)"
)]
struct Args {
    /// Filter by ticker symbol or company name substring (case-insensitive)
    #[arg(long)]
    filter: Option<String>,

    /// Include derived instruments: warrants, units, preferred shares, delisted
    #[arg(long, default_value_t = true)]
    include_derived: bool,
}

/// A single ticker row ready for tab-separated display.
struct TickerRow<'a>(&'a Ticker);

impl<'a> fmt::Display for TickerRow<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\t{}\t{}",
            self.0.symbol, self.0.cik, self.0.company_name,
        )
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let tickers = fetch_company_tickers(&client, args.include_derived).await?;
    // The SEC publishes thousands of registered company tickers.
    assert!(
        !tickers.is_empty(),
        "SEC company ticker list should not be empty"
    );

    let filtered: Vec<&Ticker> = match &args.filter {
        Some(pat) => {
            let pat = pat.to_lowercase();
            tickers
                .iter()
                .filter(|t| {
                    t.symbol.to_lowercase().contains(&pat)
                        || t.company_name.to_lowercase().contains(&pat)
                })
                .collect()
        }
        None => tickers.iter().collect(),
    };

    for t in &filtered {
        println!("{}", TickerRow(t));
    }

    println!("\nTotal tickers: {}", filtered.len());

    Ok(())
}
