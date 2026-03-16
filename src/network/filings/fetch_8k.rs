use crate::types::{Cik, CikSubmission};
use crate::network::{fetch_cik_submissions, SecClient};
use std::error::Error;

/// Fetches all 8-K filings for a given CIK, ordered newest-first.
///
/// # What is an 8-K?
///
/// An **8-K** ("Current Report") is the notification a public company must
/// file with the SEC within **4 business days** of any *material event* that
/// shareholders need to know about immediately.  SEC rules define about 20
/// standardised disclosure items; the most common are:
///
/// | Item | Meaning |
/// |------|---------|
/// | 1.01 | Material definitive agreement |
/// | 1.02 | Termination of a material agreement |
/// | 2.02 | Results of operations / earnings release |
/// | 5.02 | Departure / appointment of officers or directors |
/// | 7.01 | Regulation FD disclosure |
/// | 8.01 | Other events |
/// | 9.01 | Exhibits |
///
/// Items 2.02 and 7.01 almost always accompany an earnings press release
/// (see [`CikSubmission::is_earnings_release`]).  All other items are
/// non-earnings events (see [`CikSubmission::is_mid_quarter_event`]).
///
/// # Return value
///
/// Returns every [`CikSubmission`] whose form type is `8-K`.  Each submission
/// exposes:
/// - [`CikSubmission::as_primary_document_url`] — direct URL to the 8-K HTML body
/// - [`CikSubmission::as_edgar_archive_url`] — archive directory (use with [`crate::network::fetch_filing_index`] to list exhibits)
/// - [`CikSubmission::is_earnings_release`] — `true` if Item 2.02 is present (earnings)
/// - [`CikSubmission::is_mid_quarter_event`] — `true` for any other substantive disclosure
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_8k_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "AAPL").await?;
/// let filings = fetch_8k_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{:?}  {}", f.filing_date, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_8k_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    Ok(CikSubmission::by_form(&submissions, "8-K")
        .into_iter()
        .cloned()
        .collect())
}
