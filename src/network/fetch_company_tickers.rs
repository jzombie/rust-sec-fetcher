use crate::models::Cik;
use crate::network::SecClient;
use polars::prelude::*;
use std::error::Error;

pub type CompanyTickersDataFrame = DataFrame;

// TODO: Use struct instead of a dataframe

// TODO: Make distinction how these are not fund tickers
pub async fn fetch_company_tickers(
    client: &SecClient,
) -> Result<CompanyTickersDataFrame, Box<dyn Error>> {
    // TODO: Also incorporate: https://www.sec.gov/include/ticker.txt

    let company_tickers_url = "https://www.sec.gov/files/company_tickers.json";
    let company_tickers_data = client.fetch_json(company_tickers_url).await?;

    // TODO: Move the following into `parsers`

    let mut cik_raw_values = Vec::new();
    let mut cik_transformed_values = Vec::new();
    let mut ticker_values = Vec::new();
    let mut title_values = Vec::new();

    if let Some(ticker_map) = company_tickers_data.as_object() {
        for (_, ticker_info) in ticker_map.iter() {
            let cik_u64 = ticker_info["cik_str"].as_u64().unwrap_or_default();

            let cik = Cik::from_u64(cik_u64)?;

            cik_raw_values.push(cik.to_u64());
            cik_transformed_values.push(cik.to_string());
            ticker_values.push(ticker_info["ticker"].as_str().unwrap_or("").to_string());
            title_values.push(ticker_info["title"].as_str().unwrap_or("").to_string());
        }
    }

    let mut df = df!(
        // TODO: Just use cik_u64(
        "cik_raw" => &cik_raw_values,  // Original numeric CIK
        "cik_str" => &cik_transformed_values, // Transformed zero-padded CIK
        "ticker" => &ticker_values,
        "title" => &title_values
    )?;

    // **Explicitly cast columns to UTF-8**
    df.try_apply("cik_str", |s| s.cast(&DataType::String))?;
    df.try_apply("ticker", |s| s.cast(&DataType::String))?;

    Ok(df)
}
