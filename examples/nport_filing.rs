use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::CikSubmission;
use sec_fetcher::network::{
    fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_nport, SecClient,
};
use sec_fetcher::utils::VecExtensions;
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

    let cik = fetch_cik_by_ticker_symbol(&client, ticker_symbol).await?;
    let submissions = fetch_cik_submissions(&client, cik).await?;
    let latest = CikSubmission::by_form(&submissions, "NPORT-P")
        .into_iter()
        .next()
        .ok_or("No NPORT-P filings found")?;

    let investments = fetch_nport(&client, latest).await?;

    for (i, investment) in investments.head(510).iter().enumerate() {
        println!("Investment {}: {:?}", i + 1, investment);
    }

    Ok(())
}
