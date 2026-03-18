//! Searches for and fuzzy-matches companies against the SEC operating company list.
//!
//! Downloads the full SEC company ticker list, checks for an exact symbol or name
//! match first, then falls back to [`Ticker::get_by_fuzzy_matched_name`] for
//! partial matches.  Prints the tokenization of the search string and the ranked
//! list of fuzzy-matched companies.
//!
//! See also `fuzzy_match_company` for a simpler version without the exact-match
//! pre-step.
//!
//! # Usage
//!
//! ```text
//! cargo run --example company_search -- "Apple"
//! cargo run --example company_search -- "AAPL"
//! cargo run --example company_search -- "Eli Lilly"
//! ```

use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::Ticker;
use sec_fetcher::network::{fetch_company_tickers, SecClient};
use std::error::Error;

#[derive(Parser)]
#[command(
    about = "Fuzzy-match a company name or ticker symbol against the SEC operating company list"
)]
struct Args {
    /// Search string: a company name, ticker symbol, or partial match (e.g. \"Apple\", \"AAPL\")
    query: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let search_string = &args.query;

    println!("Searching for: {}", search_string);

    println!(
        "Tokenized: {:?}",
        Ticker::tokenize_company_name(search_string)
    );

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client, true).await.unwrap();
    // The SEC publishes thousands of registered company tickers.
    assert!(
        !company_tickers.is_empty(),
        "SEC company ticker list should not be empty"
    );

    // Override search string with company name if using direct symbol
    let search_string = {
        let exact_company_ticker = company_tickers.iter().find(|p| {
            p.symbol.to_lowercase() == search_string.to_lowercase()
                || p.company_name.to_lowercase() == search_string.to_lowercase()
        });

        // Make it easier to test by doing symbol lookup to get the company name
        let search_string = match exact_company_ticker {
            Some(ticker) => {
                println!("Exact match: {:?}", ticker);

                ticker.company_name.to_string()
            }
            None => search_string.to_string(),
        };

        Box::new(search_string)
    };

    println!("Using search string: {}", search_string);

    let fuzzy_matched = Ticker::get_by_fuzzy_matched_name(&company_tickers, &search_string, None);

    println!("Fuzzy matched: {:?}", fuzzy_matched);

    Ok(())
}
