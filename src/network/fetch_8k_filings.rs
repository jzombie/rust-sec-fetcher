use crate::models::{Cik, CikSubmission, Ticker};
use crate::network::{fetch_cik_submissions, SecClient};
use std::error::Error;

/// Fetches all 8-K filings for a given ticker symbol.
///
/// Returns a list of [`CikSubmission`] records filtered to the `8-K` form type,
/// ordered from most recent to oldest (as returned by the SEC submissions API).
///
/// Each [`CikSubmission`] exposes:
/// - [`CikSubmission::as_edgar_archive_url`] — the filing's EDGAR archive directory
/// - [`CikSubmission::as_primary_document_url`] — the direct URL to the primary filing document
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_8k_filings_by_ticker_symbol, fetch_company_tickers, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let tickers = fetch_company_tickers(&client).await?;
///
/// let filings = fetch_8k_filings_by_ticker_symbol(&client, &tickers, "AAPL").await?;
/// for filing in &filings {
///     println!("{} | {} | {}", filing.filing_date.map_or("".into(), |d| d.to_string()), filing.form, filing.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_8k_filings_by_ticker_symbol(
    client: &SecClient,
    company_tickers: &[Ticker],
    ticker_symbol: &str,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let cik = Cik::get_company_cik_by_ticker_symbol(company_tickers, ticker_symbol)?;
    let all_submissions = fetch_cik_submissions(client, cik).await?;
    let filings_8k = CikSubmission::filter_8k_submissions(&all_submissions);
    Ok(filings_8k)
}
