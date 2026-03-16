use crate::models::{Cik, CikSubmission};
use crate::network::{fetch_cik_submissions, SecClient};
use std::error::Error;

/// Fetches all 10-K and 10-K405 annual report filings for a given CIK,
/// ordered newest-first.
///
/// 10-K405 is an older variant of the 10-K used before the SEC retired it in
/// 2002; both form types represent the same annual report and are returned
/// together.  Use [`CikSubmission::most_recent_10k`] if you only need the
/// latest filing.
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_10k_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "AAPL").await?;
/// let filings = fetch_10k_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{:?}  {}", f.filing_date, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_10k_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    let mut results: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "10-K")
        .into_iter()
        .cloned()
        .collect();
    let mut k405: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "10-K405")
        .into_iter()
        .cloned()
        .collect();
    results.append(&mut k405);
    // Re-sort newest-first by filing_date (submissions list is already ordered
    // newest-first per form, but mixing the two types may interleave them).
    results.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
    Ok(results)
}
