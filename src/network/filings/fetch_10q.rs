use crate::models::{Cik, CikSubmission};
use crate::network::{fetch_cik_submissions, SecClient};
use std::error::Error;

/// Fetches all 10-Q quarterly report filings for a given CIK, ordered
/// newest-first.
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_10q_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "AAPL").await?;
/// let filings = fetch_10q_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{:?}  {}", f.filing_date, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_10q_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    Ok(CikSubmission::by_form(&submissions, "10-Q")
        .into_iter()
        .cloned()
        .collect())
}
