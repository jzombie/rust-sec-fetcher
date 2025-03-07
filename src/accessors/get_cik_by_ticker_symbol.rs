use polars::prelude::*;
use std::error::Error;

use crate::network::SecTickersDataFrame;

// TODO: Refactor to optionally work off of funding data source as well
// (reference `lookup_cik.rs` example)
/// Fetches the formatted CIK for a given ticker symbol from the DataFrame
pub fn get_cik_by_ticker_symbol(
    df_tickers: &SecTickersDataFrame,
    ticker_symbol: &str,
) -> Result<String, Box<dyn Error>> {
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
    let cik_value = cik_utf8.get(0).ok_or("CIK value not found")?;

    Ok(cik_value.to_string())
}
