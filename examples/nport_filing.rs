use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_nport_filing_by_ticker_symbol, SecClient};
use std::env;
use std::error::Error;

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

    let investments = fetch_nport_filing_by_ticker_symbol(&client, ticker_symbol)
        .await
        .unwrap();

    for (i, investment) in investments.iter().enumerate() {
        println!("Investment {}: {:?}", i + 1, investment);
    }

    Ok(())
}
