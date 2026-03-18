//! Lists all 8-K filings for a given ticker symbol, printing the filing date
//! and a direct URL to the primary document of each filing.
//!
//! # Usage
//!
//! ```text
//! cargo run --example 8k_list -- <TICKER_SYMBOL>
//! cargo run --example 8k_list -- LLY
//! cargo run --example 8k_list -- AAPL
//! ```
use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{SecClient, fetch_8k_filings, fetch_cik_by_ticker_symbol};
use std::error::Error;
use std::fmt;

#[derive(Parser)]
#[command(
    about = "List all 8-K filings for a ticker with filing date and document URL",
    long_about = None
)]
struct Args {
    /// Ticker symbol (e.g. LLY)
    ticker: String,
}

/// A single 8-K filing row ready for display.
struct FilingRow {
    date: String,
    items: String,
    url: String,
}

impl fmt::Display for FilingRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<12}  {:<16}  {}", self.date, self.items, self.url)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let ticker_symbol = TickerSymbol::new(&args.ticker);

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let cik = fetch_cik_by_ticker_symbol(&client, &ticker_symbol).await?;
    let filings = fetch_8k_filings(&client, cik).await?;

    // Each filing's primary document URL follows the standard EDGAR archive path format.
    for filing in &filings {
        let url = filing.as_primary_document_url();
        assert!(
            url.starts_with("https://www.sec.gov/"),
            "Primary document URL must be an EDGAR URL, got: {url}"
        );
    }

    if filings.is_empty() {
        println!("No 8-K filings found for '{}'.", ticker_symbol);
        return Ok(());
    }

    println!("{} 8-K filings for {}:\n", filings.len(), ticker_symbol);
    println!("{:<12}  {:<16}  URL", "Date", "Items");
    println!("{}", "-".repeat(100));

    for filing in &filings {
        let row = FilingRow {
            date: filing
                .filing_date
                .map(|d| d.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            items: filing.items.join(","),
            url: filing.as_primary_document_url(),
        };
        println!("{}", row);
    }

    Ok(())
}
