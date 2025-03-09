use crate::models::Cik;
use crate::network::SecTickersDataFrame;
use polars::prelude::*;
use std::error::Error;

// TODO: Refactor to optionally work off of funding data source as well
// (reference `lookup_cik.rs` example)

/// Retrieves the **CIK (Central Index Key)** for a given **ticker symbol**
/// from the **SEC tickers DataFrame**.
///
/// This function searches the `df_tickers` DataFrame for a row where the
/// "ticker" column matches the provided `ticker_symbol` and returns the
/// corresponding CIK value from the "cik_str" column.
///
/// # Arguments
/// - `df_tickers` - A reference to a **Polars DataFrame** containing SEC ticker data.
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
    df_tickers: &SecTickersDataFrame,
    ticker_symbol: &str,
) -> Result<Cik, Box<dyn Error>> {
    let ticker_series = df_tickers.column("ticker")?;

    // Ensure ticker column is a UTF-8 (string) type
    if ticker_series.dtype() != &DataType::String {
        return Err("Ticker column is not a string type".into());
    }

    let ticker_utf8 = ticker_series.str()?; // Convert to Utf8Chunked

    // Create a boolean mask where ticker matches
    let filter_mask = ticker_utf8.equal(ticker_symbol);
    let filtered = df_tickers.filter(&filter_mask)?;

    if filtered.height() == 0 {
        return Err(format!("Ticker symbol '{}' not found", ticker_symbol).into());
    }

    let cik_series = filtered.column("cik_str")?;

    // Ensure CIK column is also a UTF-8 string type
    if cik_series.dtype() != &DataType::String {
        return Err("CIK column is not a string type".into());
    }

    let cik_utf8 = cik_series.str()?; // Convert to Utf8Chunked
    let cik_str = cik_utf8.get(0).ok_or("CIK value not found")?;

    let cik = Cik::from_str(cik_str)?;

    Ok(cik)
}
