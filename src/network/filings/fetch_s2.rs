use crate::models::{Cik, CikSubmission};
use crate::network::{SecClient, fetch_cik_submissions};
use std::error::Error;

/// Fetches all S-2 and S-2/A registration statement filings for a given CIK,
/// ordered newest-first.
///
/// # What is an S-2?
///
/// An **S-2** was a **secondary-tier registration statement** available to
/// companies that had been filing Exchange Act reports for at least 12
/// consecutive months and had timely filed all required reports.  It occupied
/// the middle tier in the old three-form registration hierarchy:
///
/// | Form | Eligibility |
/// |------|-------------|
/// | S-1  | All companies (most detail required) |
/// | S-2  | 12+ months of reporting history (moderate detail) |
/// | S-3  | Well-known seasoned issuers (least detail, mostly by reference) |
///
/// The S-2 allowed companies to incorporate their most recent annual report
/// (Form 10-K) into the prospectus by reference, reducing duplicative
/// disclosure compared to a full S-1.
///
/// # Historical note — form retired in 2005
///
/// The SEC **eliminated the S-2 form** as part of its 2005 Securities Offering
/// Reform (Release No. 33-8591, effective December 1, 2005).  The three-tier
/// system was replaced with a revised S-1/S-3 framework based on public float.
/// No new S-2 filings have been accepted by EDGAR since December 2005.
///
/// This function therefore returns **historical filings only** — useful for
/// research on pre-2005 offering activity.  For current offerings, use
/// [`fetch_s1_filings`] or [`fetch_s3_filings`].
///
/// [`fetch_s1_filings`]: crate::network::fetch_s1_filings
/// [`fetch_s3_filings`]: crate::network::fetch_s3_filings
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_s2_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("GE")).await?;
/// let filings = fetch_s2_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{:?}  {}  {}", f.filing_date, f.form, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_s2_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    let mut results: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "S-2")
        .into_iter()
        .cloned()
        .collect();
    let mut amendments: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "S-2/A")
        .into_iter()
        .cloned()
        .collect();
    results.append(&mut amendments);
    results.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
    Ok(results)
}
