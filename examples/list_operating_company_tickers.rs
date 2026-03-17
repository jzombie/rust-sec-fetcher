use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_operating_company_tickers, SecClient};
use std::error::Error;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    // true = include derived instruments (warrants, units, prefs, delisted)
    // alongside primary listings so the full symbol universe is shown.
    let tickers = fetch_operating_company_tickers(&client, true).await?;

    for t in &tickers {
        println!("{}\t{}\t{}", t.symbol, t.cik.to_string(), t.company_name);
    }

    println!("\nTotal tickers: {}", tickers.len());

    Ok(())
}
