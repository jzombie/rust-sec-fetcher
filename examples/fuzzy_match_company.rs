use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::{CikSubmission, CompanyTicker};
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
        eprintln!("Usage: {} \"<SEARCH_STRING>\"", args[0]);
        std::process::exit(1);
    }

    let search_string = &args[1];

    println!("Searching for: {}", search_string);

    println!(
        "Tokenized: {:?}",
        CompanyTicker::tokenize_company_name(search_string)
    );

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client).await.unwrap();

    let fuzzy_matched = CompanyTicker::get_by_fuzzy_matched_name(&company_tickers, search_string);

    println!("Fuzzy matched: {:?}", fuzzy_matched);

    Ok(())
}
