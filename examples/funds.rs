use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_investment_company_series_and_class_dataset, SecClient};
use std::error::Error;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let funds = fetch_investment_company_series_and_class_dataset(&client).await?;

    for fund in funds {
        print!("{:?}\n\n\n", fund);
    }

    Ok(())
}
