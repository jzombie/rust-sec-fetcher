use crate::enums::Url;
use crate::models::{Cik, CikSubmission, NportInvestment};
use crate::network::{fetch_cik_submissions, fetch_company_tickers, SecClient};
use crate::parsers::parse_nport_xml;
use std::error::Error;

/// Fetches all NPORT-P filings for a given CIK, ordered newest-first.
///
/// Returns every [`CikSubmission`] whose form type is `NPORT-P`.  Pass each
/// submission to [`fetch_nport`] to retrieve the parsed investment holdings.
pub async fn fetch_nport_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    Ok(CikSubmission::by_form(&submissions, "NPORT-P")
        .into_iter()
        .cloned()
        .collect())
}

/// Fetches and parses an NPORT-P filing given its [`CikSubmission`].
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_nport, SecClient};
/// # use sec_fetcher::models::CikSubmission;
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "VFINX").await?;
/// let submissions = fetch_cik_submissions(&client, cik).await?;
/// let latest = CikSubmission::by_form(&submissions, "NPORT-P").into_iter().next().unwrap();
/// let investments = fetch_nport(&client, latest).await?;
/// for inv in &investments {
///     println!("{:?}", inv);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_nport(
    client: &SecClient,
    submission: &CikSubmission,
) -> Result<Vec<NportInvestment>, Box<dyn Error>> {
    let company_tickers = fetch_company_tickers(client).await?;

    let url =
        Url::CikAccessionPrimaryDocument(submission.cik.clone(), submission.accession_number.clone())
            .value();

    let response = client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;
    let xml_data = response.text().await?;

    parse_nport_xml(&xml_data, &company_tickers)
}
