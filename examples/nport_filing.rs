use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::CikSubmission;
use sec_fetcher::network::{
    fetch_cik_by_ticker_symbol, fetch_cik_submissions,
    fetch_nport_filing_by_cik_and_accession_number, SecClient,
};
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

    let cik = fetch_cik_by_ticker_symbol(&client, ticker_symbol)
        .await
        .ok();

    // let cik_u64 = 884394; // Example CIK
    // let cik = Cik::from_u64(cik_u64)?;
    // let accession_number_str = "000175272425043826"; // Example accession number
    // let accession_number = AccessionNumber::from_str(accession_number_str)?;

    let cik_submissions = fetch_cik_submissions(&client, cik.unwrap()).await?;

    let latest_nport_p_submission =
        CikSubmission::most_recent_nport_p_submission(cik_submissions.as_slice()).unwrap();

    let investments = fetch_nport_filing_by_cik_and_accession_number(
        &client,
        latest_nport_p_submission.cik,
        latest_nport_p_submission.accession_number,
    )
    .await?;

    for (i, investment) in investments.iter().enumerate() {
        println!("Investment {}: {:?}", i + 1, investment);
    }

    Ok(())
}
