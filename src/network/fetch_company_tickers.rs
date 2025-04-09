use crate::enums::{TickerOrigin, Url};
use crate::models::{Cik, Ticker};
use crate::network::SecClient;
use std::error::Error;

// TODO: Make distinction how these are not fund tickers
// TODO: Rename to `fetch_operating_company_tickers`? https://www.sec.gov/data-research/standard-taxonomies/operating-companies
pub async fn fetch_company_tickers(sec_client: &SecClient) -> Result<Vec<Ticker>, Box<dyn Error>> {
    // TODO: Also incorporate: https://www.sec.gov/include/ticker.txt

    let company_tickers_url = Url::CompanyTickers.value();
    let company_tickers_data = sec_client.fetch_json(&company_tickers_url, None).await?;

    // TODO: Move the following into `parsers`

    let mut company_tickers: Vec<Ticker> = Vec::new();

    if let Some(ticker_map) = company_tickers_data.as_object() {
        for (_, ticker_info) in ticker_map.iter() {
            let cik_u64 = ticker_info["cik_str"].as_u64().unwrap_or_default();

            let cik = Cik::from_u64(cik_u64)?;

            company_tickers.push(Ticker {
                cik,
                symbol: ticker_info["ticker"].as_str().unwrap_or("").to_string(),
                company_name: ticker_info["title"].as_str().unwrap_or("").to_string(),
                origin: TickerOrigin::CompanyTickers,
            });
        }
    }

    Ok(company_tickers)
}
