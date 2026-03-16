use crate::enums::Url;
use crate::models::MasterIndexEntry;
use crate::network::SecClient;
use crate::parsers::parse_master_idx;
use std::error::Error;

/// Downloads and parses the EDGAR full-index `master.idx` for the requested
/// `year` and `quarter` (1–4).
///
/// The index covers every filing submitted to EDGAR during that quarter.
/// History is available from Q4 1993 onwards. The current quarter's file is
/// updated nightly, so it is always complete through the previous business day.
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
