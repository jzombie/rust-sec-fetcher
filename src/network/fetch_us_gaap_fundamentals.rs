use crate::accessors::get_company_cik_by_ticker_symbol;
use crate::network::{SecClient, CompanyTickersDataFrame};
use crate::parsers::parse_us_gaap_fundamentals;
use polars::prelude::*;
use serde_json::Value;
use std::error::Error;

pub type TickerFundamentalsDataFrame = DataFrame;

/// Fetches US-GAAP SEC fundamentals for a given ticker symbol
pub async fn fetch_us_gaap_fundamentals(
    client: &SecClient,
    df_tickers: &CompanyTickersDataFrame,
    ticker_symbol: &str,
) -> Result<TickerFundamentalsDataFrame, Box<dyn Error>> {
    // Get the formatted CIK for the ticker
    let cik = get_company_cik_by_ticker_symbol(df_tickers, ticker_symbol)?;

    let url = format!(
        "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json",
        cik.to_string()
    );

    // TODO: Debug log
    println!("Using URL: {}", url);

    let data: Value = client.fetch_json(&url).await?;

    parse_us_gaap_fundamentals(data)
}
