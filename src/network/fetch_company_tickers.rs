use crate::models::{Cik, CompanyTicker};
use crate::network::SecClient;
use std::error::Error;

// TODO: Make distinction how these are not fund tickers
pub async fn fetch_company_tickers(
    client: &SecClient,
) -> Result<Vec<CompanyTicker>, Box<dyn Error>> {
    // TODO: Also incorporate: https://www.sec.gov/include/ticker.txt

    let company_tickers_url = "https://www.sec.gov/files/company_tickers.json";
    let company_tickers_data = client.fetch_json(company_tickers_url, None).await?;

    // TODO: Move the following into `parsers`

    let mut company_tickers: Vec<CompanyTicker> = Vec::new();

    if let Some(ticker_map) = company_tickers_data.as_object() {
        for (_, ticker_info) in ticker_map.iter() {
            let cik_u64 = ticker_info["cik_str"].as_u64().unwrap_or_default();

            let cik = Cik::from_u64(cik_u64)?;

            company_tickers.push(CompanyTicker {
                cik,
                ticker_symbol: ticker_info["ticker"].as_str().unwrap_or("").to_string(),
                company_name: ticker_info["title"].as_str().unwrap_or("").to_string(),
            });
        }
    }

    Ok(company_tickers)
}
