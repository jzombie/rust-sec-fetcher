use crate::enums::Url;
use crate::types::MasterIndexEntry;
use crate::network::SecClient;
use crate::parsers::parse_master_idx;
use std::error::Error;

/// Downloads and parses the EDGAR full-index `master.idx` for the requested
/// `year` and `quarter` (1–4).
///
/// # What is the EDGAR master index?
///
/// EDGAR publishes a `master.idx` pipe-delimited flat file for each calendar
/// quarter, listing **every filing accepted by EDGAR during that period**
/// across all registrants and all form types.  Each row contains: CIK,
/// company name, form type, date filed, and the relative archive path to the
/// primary document.
///
/// Typical uses:
/// - Bulk discovery: find all 10-K filings in a given quarter without knowing
///   CIKs in advance.
/// - Cross-sectional research: build a universe of companies for a time period.
/// - Back-fill pipelines: iterate through historical quarters to process filings
///   that predate the EDGAR Atom feed.
///
/// History is available from **Q4 1993** onwards.  The current quarter's file
/// is updated nightly through the previous business day.
///
/// For real-time / recent filings, prefer [`fetch_edgar_feed`] which delivers
/// new submissions within seconds of EDGAR processing them.
///
/// [`fetch_edgar_feed`]: crate::network::fetch_edgar_feed
///
/// # Example
///
/// ```no_run
/// # use sec_fetcher::network::{SecClient, fetch_edgar_master_index};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&cfg)?;
/// let entries = fetch_edgar_master_index(&client, 2024, 1).await?;
/// println!("Q1 2024 had {} filings", entries.len());
/// # Ok(()) }
/// ```
pub async fn fetch_edgar_master_index(
    sec_client: &SecClient,
    year: u16,
    quarter: u8,
) -> Result<Vec<MasterIndexEntry>, Box<dyn Error>> {
    let url = Url::EdgarFullIndex { year, quarter }.value();
    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;

    if !response.status().is_success() {
        return Err(format!(
            "EDGAR full-index returned HTTP {} for {year} Q{quarter}",
            response.status()
        )
        .into());
    }

    let text = response.text().await?;
    parse_master_idx(&text)
}
