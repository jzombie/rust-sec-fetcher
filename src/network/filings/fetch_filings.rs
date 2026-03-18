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
/// The `form_type` parameter accepts anything that implements `AsRef<str>`,
/// which includes:
/// - A `&str` literal: `fetch_filings(&client, cik, "10-K")`
/// - A `&FormType` reference: `fetch_filings(&client, cik, &FormType::TenK)`
/// - A `String` reference: `fetch_filings(&client, cik, &args.form)`
///
/// # Form type matching
///
/// The match is **case-insensitive** but otherwise exact: `"8-K"` does not
/// include `"8-K/A"` amendments.  Pass the amendment suffix explicitly if
/// you want amendments:
///
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::enums::FormType;
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let config = ConfigManager::load()?;
/// # let client = SecClient::from_config_manager(&config)?;
/// # let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
/// // Using FormType enum variants — no magic strings.
/// let mut filings = fetch_filings(&client, cik.clone(), &FormType::EightK).await?;
/// let mut amendments = fetch_filings(&client, cik, &FormType::EightKA).await?;
/// filings.append(&mut amendments);
/// filings.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
/// # Ok(())
/// # }
/// ```
///
/// # Common form types
///
/// Use [`FormType::named_variants`] for the full list, or see the table in
/// [`FormType`]'s documentation.
///
/// # Example — with a runtime string from user input
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
/// // Plain string also works.
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
    form_type: impl AsRef<str>,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    Ok(CikSubmission::by_form(&submissions, form_type.as_ref())
        .into_iter()
        .cloned()
        .collect())
}
