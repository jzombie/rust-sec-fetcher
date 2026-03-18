use crate::models::{Cik, CikSubmission};
use crate::network::{fetch_cik_submissions, SecClient};
use std::error::Error;

/// Fetches all S-1 and S-1/A registration statement filings for a given CIK,
/// ordered newest-first.
///
/// # What is an S-1?
///
/// An **S-1** is the **initial registration statement** a company must file
/// with the SEC before it can sell securities to the public for the first
/// time.  It is the primary disclosure document for an IPO (Initial Public
/// Offering).
///
/// Key sections of an S-1:
///
/// | Section | Contents |
/// |---------|----------|
/// | Prospectus summary | Business overview, offering terms |
/// | Risk factors | Comprehensive risks to the business and investment |
/// | Use of proceeds | How IPO funds will be deployed |
/// | MD&A | Management discussion and analysis |
/// | Business | Products, markets, competition, growth strategy |
/// | Financial statements | Audited historical financials (typically 2–3 years) |
/// | Executive compensation | Pay packages for named executive officers |
/// | Selling shareholders | Any secondary shares being offered by insiders |
///
/// The SEC review process typically takes 4–8 weeks.  During this period the
/// company files one or more **S-1/A** amendments (incorporated by this
/// function) to address SEC comments and update the financials.  The final
/// amendment becomes the definitive prospectus once the SEC declares it
/// effective.
///
/// # Amendments (S-1/A)
///
/// S-1/A filings are amendments filed during the SEC comment period or to
/// update the prospectus before the offering price is set.  This function
/// returns both the initial S-1 and all S-1/A amendments together, sorted
/// newest-first, so the most recent (definitive) version appears first.
///
/// # EDGAR search note
///
/// S-1 filings are searchable by the **company's CIK** — the company that is
/// registering the offering.  After the IPO, subsequent equity offerings use
/// [`fetch_s3_filings`] if the company qualifies as a well-known seasoned
/// issuer.
///
/// [`fetch_s3_filings`]: crate::network::fetch_s3_filings
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_s1_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("META")).await?;
/// let filings = fetch_s1_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{:?}  {}  {}", f.filing_date, f.form, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_s1_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    let mut results: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "S-1")
        .into_iter()
        .cloned()
        .collect();
    let mut amendments: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "S-1/A")
        .into_iter()
        .cloned()
        .collect();
    results.append(&mut amendments);
    results.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
    Ok(results)
}
