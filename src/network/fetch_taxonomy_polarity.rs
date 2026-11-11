use std::error::Error;

use polars::prelude::*;

use crate::network::SecClient;
use crate::parsers::parse_all_taxonomy_balances;

/// Full standard-tag polarity table for downstream consumers.
///
/// Companyfacts-only scope: values come from `companyfacts.json`
/// (`us-gaap:*` only), balances come from here. Filer customs (`bac:*`)
/// are out of scope — neither companyfacts nor FASB XSDs contain them.
///
/// Call once per taxonomy year (e.g. `"2025"`) and join on concept name
/// to get `credit`/`debit` for every `us-gaap:*` column in
/// `us-gaap-fundamentals/*.csv`.
///
/// Columns: `concept` (`us-gaap:Revenues`), `balance` (`credit`/`debit`).
pub async fn fetch_us_gaap_polarity_table(
    client: &SecClient,
    year: &str,
) -> Result<DataFrame, Box<dyn Error>> {
    let url = format!(
        "https://xbrl.fasb.org/us-gaap/{}/elts/us-gaap-{}.xsd",
        year, year
    );
    let response = client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?
        .error_for_status()?;
    let xml = response.text().await?;
    let map = parse_all_taxonomy_balances(&xml)?;
    if map.is_empty() {
        return Err(format!("taxonomy XSD at {} yielded zero concept balances", url).into());
    }
    let mut entries: Vec<(String, String)> = map
        .into_iter()
        .map(|(concept, balance)| (format!("us-gaap:{}", concept), balance))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let (concepts, balances): (Vec<String>, Vec<String>) = entries.into_iter().unzip();
    let df = df!(
        "concept" => concepts,
        "balance" => balances,
    )?;
    Ok(df)
}
