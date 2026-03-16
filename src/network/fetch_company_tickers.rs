use crate::enums::Url;
use crate::models::Ticker;
use crate::network::SecClient;
use crate::parsers::{parse_company_tickers_json, parse_ticker_txt};
use std::collections::HashMap;
use std::error::Error;

/// Fetches the SEC's operating-company ticker-to-CIK mapping from both
/// available sources, returning a merged, deduplicated, alphabetically-sorted
/// list of [`Ticker`]s.
///
/// # What this returns
///
/// Two complementary SEC files are fetched in parallel and merged:
///
/// 1. **`company_tickers.json`** — the primary JSON source; includes CIK,
///    ticker symbol, and company name for exchange-listed operating companies.
/// 2. **`ticker.txt`** — a supplementary plain-text tab-separated file
///    (`symbol\cik`); may include tickers not present in the JSON.  Company
///    names are not available from this source.
///
/// When the same symbol appears in both sources the JSON entry takes
/// precedence (it carries the company name).  The final list is sorted
/// alphabetically by symbol.
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
    let json_url = Url::CompanyTickers.value();
    let txt_url = Url::TickerTxt.value();

    let (json_data, txt_response) = tokio::try_join!(
        sec_client.fetch_json(&json_url, None),
        sec_client.raw_request(reqwest::Method::GET, &txt_url, None, None),
    )?;
    let txt_text = txt_response.text().await?;

    let json_tickers = parse_company_tickers_json(&json_data)?;
    let txt_tickers = parse_ticker_txt(&txt_text);

    // Merge: txt entries go in first (no company name); JSON entries
    // overwrite since they carry the authoritative company name.
    // Both sources are already normalized by the parsers.
    let mut map: HashMap<String, Ticker> = HashMap::new();
    for t in txt_tickers {
        map.insert(t.symbol.clone(), t);
    }
    for t in json_tickers {
        map.insert(t.symbol.clone(), t);
    }

    let mut result: Vec<Ticker> = map.into_values().collect();
    result.sort_by(|a, b| a.symbol.cmp(&b.symbol));
    Ok(result)
}
