use crate::network::{fetch_cik_submissions, SecClient};
use crate::types::{Cik, CikSubmission};
use std::error::Error;

/// Fetches all S-3 and S-3/A shelf registration statement filings for a given
/// CIK, ordered newest-first.
///
/// # What is an S-3?
///
/// An **S-3** is a **short-form shelf registration** available only to
/// companies that meet the SEC's "well-known seasoned issuer" (WKSI) criteria
/// or certain other eligibility tests.  It is significantly shorter than an
/// S-1 because it incorporates by reference all of the company's outstanding
/// Exchange Act reports (10-K, 10-Q, 8-K), avoiding the need to repeat that
/// information in the prospectus itself.
///
/// # Eligibility
///
/// To use Form S-3 a company generally must:
/// - Have been a reporting company for at least **12 calendar months**
/// - Have timely filed all Exchange Act reports over the previous 12 months
/// - Meet one of several alternative tests, most commonly:
///   - **Public float** of at least **$75 million** held by non-affiliates; or
///   - Qualify as a **WKSI** (≥ $700 million public float or ≥ $1 billion in
///     non-convertible securities issued in the prior 3 years)
///
/// # What is a shelf registration?
///
/// A "shelf" registration lets a company register securities in advance and
/// then draw down ("take off the shelf") specific tranches at any time within
/// the next three years, without filing a new full registration statement for
/// each offering.  Companies use this to raise capital quickly when market
/// conditions are favourable, issuing a short prospectus supplement (`424B`)
/// that references the underlying S-3.
///
/// # Amendments (S-3/A)
///
/// S-3/A amendments are typically filed to address SEC comments, update
/// financial information, or add new securities to the shelf.  This function
/// returns both the initial S-3 and all S-3/A amendments together, sorted
/// newest-first.
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_s3_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "MSFT").await?;
/// let filings = fetch_s3_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{:?}  {}  {}", f.filing_date, f.form, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_s3_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    let mut results: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "S-3")
        .into_iter()
        .cloned()
        .collect();
    let mut amendments: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "S-3/A")
        .into_iter()
        .cloned()
        .collect();
    results.append(&mut amendments);
    results.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
    Ok(results)
}
