use sec_fetcher::{models::{Cik, AccessionNumber}, network::{fetch_nport_filing, SecClient}};
use sec_fetcher::config::ConfigManager;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager, 1, 1000, Some(5))?;

    let cik_u64 = 884394; // Example CIK
    let cik = Cik::from_u64(cik_u64)?;
    let accession_number_str = "000175272425043826"; // Example accession number
    let accession_number = AccessionNumber::from_str(accession_number_str)?;

    let investments = fetch_nport_filing(&client, cik, accession_number).await?;

    for (i, investment) in investments.iter().enumerate() {
        println!("Investment {}: {:?}", i + 1, investment);
    }

    Ok(())
}
