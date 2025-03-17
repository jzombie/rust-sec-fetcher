use crate::models::{Cik, CompanyTicker};
use std::error::Error;

// TODO: Refactor to optionally work off of funding data source as well
// (reference `lookup_cik.rs` example)

/// Retrieves the **CIK (Central Index Key)** for a given **ticker symbol**
/// from the **SEC tickers DataFrame**.
///
///
/// # Arguments
/// - `company_tickers` - A slice of `CompanyTicker` instances.
/// - `ticker_symbol` - A **stock ticker symbol** (case-sensitive) as a `&str`.
///
/// # Returns
/// - `Ok(Cik)` - A `Cik` model instance.
/// - `Err(Box<dyn Error>)` - If:
///   - The ticker column is not a string type.
///   - The CIK column is not a string type.
///   - The ticker symbol is not found in the DataFrame.
///   - The corresponding CIK value is missing.
///
/// # Response Format
/// The returned **CIK** will be a **zero-padded 10-digit string** (e.g., `"0000320193"`),
/// ensuring consistency with SEC filings.
///
/// # Errors
/// - Returns an **error** if the dataset does not contain the given ticker,
///   or if the data is incorrectly formatted.
pub fn get_company_cik_by_ticker_symbol(
    company_tickers: &[CompanyTicker],
    ticker_symbol: &str,
) -> Result<Cik, Box<dyn Error>> {
    company_tickers
        .iter()
        .find(|pred| pred.ticker_symbol == ticker_symbol)
        .map(|company_ticker| company_ticker.cik.clone())
        .ok_or_else(|| format!("Ticker symbol '{}' not found", ticker_symbol).into())
}
