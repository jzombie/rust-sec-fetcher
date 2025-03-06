use sec_fetcher::accessor::get_cik_by_ticker_symbol;
use sec_fetcher::network::{fetch_sec_tickers, CredentialManager, CredentialProvider, SecClient};
use std::env;
use std::error::Error;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let credential_manager = CredentialManager::from_prompt()?;

    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <TICKER_SYMBOL>", args[0]);
        std::process::exit(1);
    }

    let ticker_symbol = &args[1];

    let client = SecClient::from_credential_manager(&credential_manager, 1, 1000, Some(5))?;

    let tickers_df = fetch_sec_tickers(&client).await?;

    match get_cik_by_ticker_symbol(&tickers_df, ticker_symbol) {
        Ok(cik) => println!("Ticker: {}, CIK: {}", ticker_symbol, cik),
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}
