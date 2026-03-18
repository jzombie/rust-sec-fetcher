use crate::models::{Cik, CikSubmission};
use crate::network::{SecClient, fetch_cik_submissions};
use std::error::Error;

/// Fetches all 10-Q quarterly report filings for a given CIK, ordered
/// newest-first.
///
/// # What is a 10-Q?
///
/// A **10-Q** is the **quarterly report** filed three times per year (Q1, Q2,
/// Q3).  The fourth-quarter financial statements appear in the annual 10-K
/// rather than a separate 10-Q.  Key differences from a 10-K:
///
/// - Financial statements are **unaudited** (reviewed, not full audit).
/// - Required within **40 days** of quarter end for large accelerated filers,
///   45 days for all others.
/// - MD&A section covers only the current quarter and year-to-date changes,
///   rather than a full-year narrative.
///
/// 10-Qs are the primary source of quarter-over-quarter comparisons for
/// income statements, balance sheets, and cash flows.  The XBRL-tagged values
/// for all quarters are available through [`fetch_us_gaap_fundamentals`].
///
/// [`fetch_us_gaap_fundamentals`]: crate::network::fetch_us_gaap_fundamentals()
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_10q_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
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
