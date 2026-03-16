use crate::enums::Url;
use crate::types::{Cik, CompanyProfile};
use crate::network::SecClient;
use std::error::Error;

/// Fetches the company profile for the given CIK from the SEC EDGAR submissions
/// endpoint.
///
/// This reuses the same cached HTTP response as [`fetch_cik_submissions`] — both
/// call `https://data.sec.gov/submissions/CIK{cik}.json`. When the cache is
/// warm the second call is free.
///
/// [`fetch_cik_submissions`]: crate::network::fetch_cik_submissions
///
/// # Example
///
/// ```no_run
/// # use sec_fetcher::network::{SecClient, fetch_company_profile};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::types::Cik;
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&cfg)?;
/// let cik = Cik::from_str("0000320193")?;
/// let profile = fetch_company_profile(&client, cik).await?;
/// println!("{} — {} ({})", profile.name, profile.sic_description.unwrap_or_default(), profile.exchanges.join(", "));
/// # Ok(()) }
/// ```
pub async fn fetch_company_profile(
    sec_client: &SecClient,
    cik: Cik,
) -> Result<CompanyProfile, Box<dyn Error>> {
    let url = Url::CikSubmission(cik.clone()).value();
    let data = sec_client.fetch_json(&url, None).await?;

    let name = data["name"].as_str().unwrap_or_default().to_string();

    let entity_type = data["entityType"].as_str().map(|s| s.to_string());
    let sic = data["sic"].as_str().map(|s| s.to_string());
    let sic_description = data["sicDescription"].as_str().map(|s| s.to_string());
    let owner_org = data["ownerOrg"].as_str().map(|s| s.to_string());
    let category = data["category"].as_str().map(|s| s.to_string());
    let state_of_incorporation = data["stateOfIncorporationDescription"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .or_else(|| {
            data["stateOfIncorporation"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        });
    let fiscal_year_end = data["fiscalYearEnd"].as_str().map(|s| s.to_string());
    let website = data["website"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let investor_website = data["investorWebsite"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let phone = data["phone"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let description = data["description"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let tickers = data["tickers"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let exchanges = data["exchanges"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Ok(CompanyProfile {
        cik,
        name,
        entity_type,
        sic,
        sic_description,
        owner_org,
        tickers,
        exchanges,
        category,
        state_of_incorporation,
        fiscal_year_end,
        website,
        investor_website,
        phone,
        description,
    })
}
