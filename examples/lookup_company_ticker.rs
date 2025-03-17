use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::CikSubmission;
use sec_fetcher::network::{
    fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_company_tickers, SecClient,
};
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

    let ticker_symbol = &args[1];

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client).await.unwrap();

    let found = company_tickers
        .iter()
        .find(|p| p.ticker_symbol == ticker_symbol.as_str());

    println!("Located: {:?}", found);

    Ok(())
}
