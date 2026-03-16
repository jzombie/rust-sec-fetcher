use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_company_tickers, SecClient};
use std::error::Error;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let tickers = fetch_company_tickers(&client).await?;

    for t in &tickers {
        println!("{}\t{}\t{}", t.symbol, t.cik.to_string(), t.company_name);
    }

    println!("\nTotal tickers: {}", tickers.len());

    Ok(())
}
