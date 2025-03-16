use csv::ReaderBuilder;
use sec_fetcher::accessors::get_company_cik_by_ticker_symbol;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::Cik;
use sec_fetcher::network::{
    fetch_cik_submissions, fetch_company_tickers,
    fetch_investment_company_series_and_class_dataset, CikSubmission, SecClient,
};
use std::env;
use std::error::Error;
use std::io::Cursor;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let funds = fetch_investment_company_series_and_class_dataset(&client, 2024).await?;

    for fund in funds {
        print!("{:?}\n\n\n", fund);
    }

    Ok(())
}
