/// Lists all 8-K filings for a given ticker symbol, printing the filing date
/// and a direct URL to the primary document of each filing.
///
/// Usage:
///   cargo run --example list_8k_filings -- <TICKER_SYMBOL>
///
/// Example:
///   cargo run --example list_8k_filings -- LLY
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_8k_filings_by_ticker_symbol, fetch_company_tickers, SecClient};
use std::env;
use std::error::Error;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <TICKER_SYMBOL>", args[0]);
        std::process::exit(1);
    }

    let ticker_symbol = args[1].to_uppercase();

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client).await?;

    let filings =
        fetch_8k_filings_by_ticker_symbol(&client, &company_tickers, &ticker_symbol).await?;

    if filings.is_empty() {
        println!("No 8-K filings found for '{}'.", ticker_symbol);
        return Ok(());
    }

    println!("{} 8-K filings for {}:\n", filings.len(), ticker_symbol);
    println!("{:<12}  {:<16}  {}", "Date", "Items", "URL");
    println!("{}", "-".repeat(100));

    for filing in &filings {
        let date = filing
            .filing_date
            .map(|d| d.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let items = filing.items.join(",");
        println!(
            "{:<12}  {:<16}  {}",
            date,
            items,
            filing.as_primary_document_url()
        );
    }

    Ok(())
}
