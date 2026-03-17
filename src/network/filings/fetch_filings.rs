use crate::models::{Cik, CikSubmission};
use crate::network::{fetch_cik_submissions, SecClient};
use std::error::Error;

/// Fetches all filings of a given form type for a CIK, ordered newest-first.
///
/// This is the generic counterpart to the form-specific fetchers such as
/// [`fetch_8k_filings`], [`fetch_10k_filings`], and [`fetch_10q_filings`].
/// Use it when the form type is determined at runtime, or when you need a
/// form type that doesn't have a dedicated function.
///
/// # Form type matching
///
/// `form_type` is matched **exactly** against the EDGAR form field (e.g.
/// `"8-K"` does not include `"8-K/A"` amendments).  Pass the amendment
/// suffix explicitly if you want amendments:
///
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let config = ConfigManager::load()?;
/// # let client = SecClient::from_config_manager(&config)?;
/// # let cik = fetch_cik_by_ticker_symbol(&client, "AAPL").await?;
/// // Fetch original filings and amendments separately, then merge.
/// let mut filings = fetch_filings(&client, cik, "8-K").await?;
/// let mut amendments = fetch_filings(&client, cik, "8-K/A").await?;
/// filings.append(&mut amendments);
/// filings.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
/// # Ok(())
/// # }
/// ```
///
/// # Common form type strings
///
/// | Form type | Description |
/// |-----------|-------------|
/// | `"8-K"` | Current report — material events |
/// | `"10-K"` | Annual report |
/// | `"10-Q"` | Quarterly report |
/// | `"DEF 14A"` | Definitive proxy statement |
/// | `"S-1"` | Registration statement (IPO) |
/// | `"SC 13D"` | Beneficial ownership > 5 % (activist) |
/// | `"SC 13G"` | Beneficial ownership > 5 % (passive) |
/// | `"424B4"` | Prospectus supplement (priced offering) |
/// | `"NPORT-P"` | Monthly portfolio holdings (funds) |
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "AAPL").await?;
/// let filings = fetch_filings(&client, cik, "10-K").await?;
/// for f in &filings {
///     println!("{:?}  {}", f.filing_date, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_filings(
    client: &SecClient,
    cik: Cik,
    form_type: &str,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    Ok(CikSubmission::by_form(&submissions, form_type)
        .into_iter()
        .cloned()
        .collect())
}
