use crate::enums::Url;
use crate::models::{Cik, CikSubmission, Form4Transaction};
use crate::network::{fetch_cik_submissions, SecClient};
use crate::parsers::parse_form4_xml;
use std::error::Error;

/// Fetches all Form 4 and Form 4/A filings for a given CIK, ordered
/// newest-first.
///
/// Both the original filing (`4`) and amendments (`4/A`) are included.  Pass
/// each submission to [`fetch_form4`] to retrieve the parsed transactions.
pub async fn fetch_form4_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    let mut results: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "4")
        .into_iter()
        .cloned()
        .collect();
    let mut amendments: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "4/A")
        .into_iter()
        .cloned()
        .collect();
    results.append(&mut amendments);
    results.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
    Ok(results)
}

/// Fetches and parses a Form 4 filing given its [`CikSubmission`].
///
/// The primary document is the Form 4 XML file listed in the submission.
/// Returns one [`Form4Transaction`] per transaction row in the filing,
/// sorted by transaction date descending.
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_form4, SecClient};
/// # use sec_fetcher::models::CikSubmission;
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "AAPL").await?;
/// let submissions = fetch_cik_submissions(&client, cik).await?;
/// let latest = CikSubmission::by_form(&submissions, "4").into_iter().next().unwrap();
/// let txns = fetch_form4(&client, latest).await?;
/// for txn in &txns {
///     println!("{:?}", txn);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_form4(
    client: &SecClient,
    submission: &CikSubmission,
) -> Result<Vec<Form4Transaction>, Box<dyn Error>> {
    // SEC's submissions JSON sometimes contains an XSLT-prefixed path for
    // Form 4 primary documents (e.g. "xslF345X05/form4.xml"). That URL
    // returns rendered HTML, not the raw XML. Strip any directory prefix so
    // we always fetch the raw XML from the archive root.
    let doc_name = submission
        .primary_document
        .rfind('/')
        .map(|pos| submission.primary_document[pos + 1..].to_string())
        .unwrap_or_else(|| submission.primary_document.clone());

    let url = Url::CikAccessionDocument(
        submission.cik.clone(),
        submission.accession_number.clone(),
        doc_name,
    )
    .value();

    let response = client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;
    let xml = response.text().await?;
    parse_form4_xml(&xml, submission.filing_date)
}
