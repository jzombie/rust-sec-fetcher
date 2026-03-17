use crate::enums::Url;
use crate::models::{Cik, Ticker};
use crate::network::SecClient;
use crate::parsers::{parse_company_tickers_json, parse_ticker_txt};
use std::collections::HashMap;
use std::error::Error;

/// Fetches operating-company equity tickers from SEC EDGAR, returning a
/// deduplicated, alphabetically-sorted list of [`Ticker`]s.
///
/// # What this returns — and what it does NOT
///
/// This function covers **operating companies** only: exchange-listed stocks
/// and most ETPs registered as operating companies.  It does **not** include
/// mutual fund or ETF share classes registered under the Investment Company
/// Act.  For those, use [`fetch_investment_company_series_and_class_dataset`];
/// to resolve any ticker (operating company or fund), use
/// [`fetch_cik_by_ticker_symbol`].
///
/// # Sources
///
/// The primary source is always fetched:
///
/// 1. **`company_tickers.json`** — one entry per primary common-stock ticker;
///    includes CIK and company name; tagged [`TickerOrigin::PrimaryListing`].
///
/// When `include_derived_instruments` is `true`, a second source is merged:
///
/// 2. **`ticker.txt`** — tab-separated `symbol\tcik`; tagged
///    [`TickerOrigin::DerivedInstrument`].  Company names are not available
///    from this source, but names for instruments that share a CIK with a
///    primary listing (warrants, units, preferreds) are backfilled.
///
/// When the same symbol appears in both sources the JSON entry takes
/// precedence.  The final list is sorted alphabetically by symbol.
///
/// # `include_derived_instruments` flag
///
/// Pass `false` (the common default) to fetch only primary listings.  This is
/// faster and sufficient whenever you need operating-company XBRL/US-GAAP data.
///
/// Pass `true` when you need the broadest possible symbol coverage — for
/// example, when resolving arbitrary symbols to CIKs or enriching NPORT
/// portfolio holdings.  [`Cik::get_company_cik_by_ticker_symbol`] will always
/// resolve a derived instrument to its parent registrant's CIK; see that
/// method for details.
///
/// [`Cik::get_company_cik_by_ticker_symbol`]: crate::models::Cik::get_company_cik_by_ticker_symbol
///
/// Derived instruments include warrants (`-WT`), units (`-UN`), preferred
/// share classes (`-PA`/`-PB`/…), and tickers for defunct, delisted, or
/// OTC-only issuers.
///
/// [`TickerOrigin::PrimaryListing`]: crate::enums::TickerOrigin::PrimaryListing
/// [`TickerOrigin::DerivedInstrument`]: crate::enums::TickerOrigin::DerivedInstrument
/// [`fetch_investment_company_series_and_class_dataset`]: crate::network::fetch_investment_company_series_and_class_dataset
/// [`fetch_cik_by_ticker_symbol`]: crate::network::fetch_cik_by_ticker_symbol
pub async fn fetch_operating_company_tickers(
    sec_client: &SecClient,
    include_derived_instruments: bool,
) -> Result<Vec<Ticker>, Box<dyn Error>> {
    let json_url = Url::CompanyTickersJson.value();

    let json_data = sec_client.fetch_json(&json_url, None).await?;
    let json_tickers = parse_company_tickers_json(&json_data)?;

    if !include_derived_instruments {
        let mut result = json_tickers;
        result.sort_by(|a, b| a.symbol.cmp(&b.symbol));
        return Ok(result);
    }

    let txt_url = Url::CompanyTickersTxt.value();
    let txt_response = sec_client
        .raw_request(reqwest::Method::GET, &txt_url, None, None)
        .await?;
    let txt_text = txt_response.text().await?;
    let txt_tickers = parse_ticker_txt(&txt_text);

    // Build a CIK → company name map from the JSON entries (which always have
    // names). This is used below to backfill names for txt-only entries such
    // as warrants (-WT), units (-UN), and preferred share classes (-PA/-PB/…)
    // that share a CIK with a JSON entry but have no name source of their own.
    let cik_to_name: HashMap<Cik, String> = json_tickers
        .iter()
        .filter(|t| !t.company_name.is_empty())
        .map(|t| (t.cik.clone(), t.company_name.clone()))
        .collect();

    // Merge: txt entries go in first; JSON entries overwrite since they carry
    // the authoritative company name.  Both sources are already normalized by
    // the parsers.
    let mut map: HashMap<String, Ticker> = HashMap::new();
    for mut t in txt_tickers {
        if t.company_name.is_empty() {
            if let Some(name) = cik_to_name.get(&t.cik) {
                t.company_name = name.clone();
            }
        }
        map.insert(t.symbol.clone(), t);
    }
    for t in json_tickers {
        map.insert(t.symbol.clone(), t);
    }

    let mut result: Vec<Ticker> = map.into_values().collect();
    result.sort_by(|a, b| a.symbol.cmp(&b.symbol));
    Ok(result)
}
