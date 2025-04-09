use crate::models::{Cik, Ticker};
use crate::network::SecClient;
use crate::parsers::parse_us_gaap_fundamentals;
use polars::prelude::*;
use serde_json::Value;
use std::error::Error;

pub type TickerFundamentalsDataFrame = DataFrame;

/// Fetches US-GAAP SEC fundamentals for a given ticker symbol
pub async fn fetch_us_gaap_fundamentals(
    client: &SecClient,
    company_tickers: &[Ticker],
    ticker_symbol: &str,
) -> Result<TickerFundamentalsDataFrame, Box<dyn Error>> {
    // Get the formatted CIK for the ticker
    let cik = Cik::get_company_cik_by_ticker_symbol(company_tickers, ticker_symbol)?;

    // TODO: Move to enum
    let url = format!(
        "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json",
        cik.to_string()
    );

    // TODO: Debug log
    println!("Using URL: {}", url);

    let data: Value = client.fetch_json(&url, None).await?;

    parse_us_gaap_fundamentals(data)
}
