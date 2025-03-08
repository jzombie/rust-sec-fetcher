use sec_fetcher::network::{
    fetch_investment_company_series_and_class_dataset,
    SecClient,
};
use sec_fetcher::config::{
    CredentialManager, CredentialProvider
};
use std::error::Error;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let credential_manager = CredentialManager::from_prompt()?;

    let client = SecClient::from_credential_manager(&credential_manager, 1, 1000, Some(5))?;

    let byte_array = fetch_investment_company_series_and_class_dataset(&client, 2024).await?;

    print!("{:?}", byte_array);

    Ok(())
}
