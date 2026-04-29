use crate::models::{Cik, CikSubmission};
use crate::network::{SecClient, fetch_all_entity_submissions};
use std::error::Error;

/// Fetches all DEF 14A (definitive proxy statement) filings for a given CIK,
/// ordered newest-first.
///
/// # What is a DEF 14A?
///
/// A **DEF 14A** ("Definitive Proxy Statement") is the document a public
/// company sends to shareholders before the **annual meeting** (or any special
/// meeting) asking them to vote on various matters.  It is one of the richest
/// sources of corporate-governance and executive-compensation data available
/// in public disclosures.
///
/// Typical contents:
///
/// | Section | Contents |
/// |---------|----------|
/// | Election of directors | Board nominees, bios, committee assignments, independence assessments |
/// | Executive compensation | Summary compensation table, pay-versus-performance, CEO pay ratio |
/// | Say-on-pay | Advisory vote on executive pay packages |
/// | Equity plan approval | New or expanded stock option / RSU plans |
/// | Auditor ratification | Ratification of the independent auditor (with audit fees) |
/// | Shareholder proposals | Any shareholder-submitted resolutions |
/// | Related-party transactions | Dealings with insiders above the disclosure threshold |
///
/// The DEF 14A must be filed with the SEC **at least 40 calendar days** before
/// the shareholder meeting.
///
/// # Preliminary proxy (PRE 14A)
///
/// Before the definitive proxy is filed, companies subject to SEC review must
/// first file a **PRE 14A** (preliminary proxy).  PRE 14A filings are not
/// returned by this function; they can be retrieved directly via
/// [`fetch_cik_submissions`] and filtering for `"PRE 14A"`.
///
/// [`fetch_cik_submissions`]: crate::network::fetch_cik_submissions()
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_def14a_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
/// let filings = fetch_def14a_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{:?}  {}", f.filing_date, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_def14a_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_all_entity_submissions(client, cik).await?;
    Ok(CikSubmission::by_form(&submissions, "DEF 14A")
        .into_iter()
        .cloned()
        .collect())
}
