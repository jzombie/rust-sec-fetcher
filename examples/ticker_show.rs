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

    let found = company_tickers.iter().find(|p| p.symbol == ticker_symbol);

    println!("Located: {:?}", found);

    Ok(())
}
