use crate::network::SecClient;
use crate::transform::format_cik;
use polars::prelude::*;
use std::error::Error;

pub type SecTickersDataFrame = DataFrame;

pub async fn fetch_sec_tickers(client: &SecClient) -> Result<SecTickersDataFrame, Box<dyn Error>> {
    // TODO: Also incorporate: https://www.sec.gov/include/ticker.txt

    let url = "https://www.sec.gov/files/company_tickers.json";
    let data = client.fetch_json(url).await?;

    let mut cik_raw_values = Vec::new();
    let mut cik_transformed_values = Vec::new();
    let mut ticker_values = Vec::new();
    let mut title_values = Vec::new();

    if let Some(ticker_map) = data.as_object() {
        for (_, ticker_info) in ticker_map.iter() {
            let cik = ticker_info["cik_str"].as_i64().unwrap_or_default();
            let formatted_cik = format_cik(cik); // Transform CIK to zero-padded format

            cik_raw_values.push(cik);
            cik_transformed_values.push(formatted_cik);
            ticker_values.push(ticker_info["ticker"].as_str().unwrap_or("").to_string());
            title_values.push(ticker_info["title"].as_str().unwrap_or("").to_string());
        }
    }

    let mut df = df!(
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
