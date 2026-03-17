/// Lists all 8-K filings for a given ticker symbol, printing the filing date
/// and a direct URL to the primary document of each filing.
///
/// Usage:
///   cargo run --example list_8k_filings -- <TICKER_SYMBOL>
///
/// Example:
///   cargo run --example list_8k_filings -- LLY
use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_8k_filings, fetch_cik_by_ticker_symbol, SecClient};
use std::error::Error;
use std::fmt;
use tokio;

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
    let ticker_symbol = args.ticker.to_uppercase();

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let cik = fetch_cik_by_ticker_symbol(&client, &ticker_symbol).await?;
    let filings = fetch_8k_filings(&client, cik).await?;

    if filings.is_empty() {
        println!("No 8-K filings found for '{}'.", ticker_symbol);
        return Ok(());
    }

    println!("{} 8-K filings for {}:\n", filings.len(), ticker_symbol);
    println!("{:<12}  {:<16}  {}", "Date", "Items", "URL");
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
