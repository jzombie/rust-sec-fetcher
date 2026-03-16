use crate::enums::{TickerOrigin, Url};
use crate::models::{Cik, Ticker};
use crate::network::SecClient;
use std::error::Error;

/// Fetches the SEC's operating-company ticker-to-CIK mapping.
///
/// # What this returns
///
/// The file at `https://www.sec.gov/files/company_tickers.json` is the SEC's
/// canonical list of **exchange-listed operating companies**:
/// stocks, ETPs, and REITs that trade under a ticker symbol and file as
/// operating companies rather than under the Investment Company Act.
///
/// Each [`Ticker`] entry carries:
/// - `cik` — the registrant's SEC Central Index Key
/// - `symbol` — the exchange ticker (e.g. `"AAPL"`)
/// - `company_name` — the name as it appears in EDGAR
///
/// # Coverage gaps
///
/// Mutual-fund share classes and ETFs registered under the Investment Company
/// Act are **not** included here.  Use
/// [`fetch_investment_company_series_and_class_dataset`] to look up those
/// tickers, or call [`fetch_cik_by_ticker_symbol`] which searches both sources
/// automatically.
///
/// [`fetch_investment_company_series_and_class_dataset`]: crate::network::fetch_investment_company_series_and_class_dataset
/// [`fetch_cik_by_ticker_symbol`]: crate::network::fetch_cik_by_ticker_symbol
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
