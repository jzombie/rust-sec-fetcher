use crate::enums::TickerOrigin;
use crate::types::{Cik, Ticker};
use serde_json::Value;
use std::error::Error;

/// Parses the SEC `company_tickers.json` response body into a list of [`Ticker`]s.
///
/// The JSON is a map from an arbitrary integer key to an object with fields
/// `cik_str`, `ticker`, and `title` (company name).  This is the primary source
/// for operating-company tickers and always includes the company name.
pub fn parse_company_tickers_json(data: &Value) -> Result<Vec<Ticker>, Box<dyn Error>> {
    let mut tickers = Vec::new();
    if let Some(ticker_map) = data.as_object() {
        for (_, ticker_info) in ticker_map.iter() {
            let cik_u64 = ticker_info["cik_str"].as_u64().unwrap_or_default();
            let cik = Cik::from_u64(cik_u64)?;
            tickers.push(Ticker {
                cik,
                symbol: Ticker::normalize_symbol(ticker_info["ticker"].as_str().unwrap_or("")),
                company_name: ticker_info["title"]
                    .as_str()
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                origin: TickerOrigin::PrimaryListing,
            });
        }
    }
    Ok(tickers)
}

/// Parses the SEC `ticker.txt` plain-text file into a list of [`Ticker`]s.
///
/// The file is tab-separated with no header: each line is `symbol\cik`.
/// Symbols are lowercase in the source; this function normalizes them to
/// uppercase.  Company names are not available in this source; the
/// `company_name` field is left empty for entries that come only from this
/// file and are not present in `company_tickers.json`.
///
/// Malformed lines are silently skipped.
pub fn parse_ticker_txt(text: &str) -> Vec<Ticker> {
    let mut tickers = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let symbol_raw = parts.next().unwrap_or("").trim();
        let cik_raw = parts.next().unwrap_or("").trim();
        if symbol_raw.is_empty() || cik_raw.is_empty() {
            continue;
        }
        let cik_u64: u64 = match cik_raw.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let cik = match Cik::from_u64(cik_u64) {
            Ok(c) => c,
            Err(_) => continue,
        };
        tickers.push(Ticker {
            cik,
            symbol: Ticker::normalize_symbol(symbol_raw),
            company_name: String::new(),
            origin: TickerOrigin::DerivedInstrument,
        });
    }
    tickers
}
