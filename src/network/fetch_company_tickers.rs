use crate::models::Cik;
use crate::network::SecClient;
use polars::prelude::*;
use std::error::Error;

pub type CompanyTickersDataFrame = DataFrame;

// TODO: Use struct instead of a dataframe and add a `best_match_cik` method,
// based on string similarity for the purpose of mapping LEI (on `NPORT` filings) to CIK
/*
fn best_match_cik(json_data: &str, query: &str) -> Option<u64> {
    let data: Value = serde_json::from_str(json_data).ok()?;

    let mut cik_counts: HashMap<u64, usize> = HashMap::new();
    let mut candidates: Vec<(u64, String, String, i32)> = Vec::new(); // (CIK, ticker, title, score)

    // Iterate over dataset
    if let Some(objects) = data.as_object() {
        for obj in objects.values() {
            let cik = obj.get("cik_str")?.as_u64()?;
            let ticker = obj.get("ticker")?.as_str()?.to_string();
            let title = obj.get("title")?.as_str()?.to_lowercase();

            // Only consider companies that match the query
            if title.contains(&query.to_lowercase()) {
                // Scoring system
                let mut score = 0;

                if title == query.to_lowercase() {
                    score += 5; // Exact match
                }
                if ticker.len() <= 4 { // Common stock (e.g., "MS")
                    score += 3;
                } else if ticker.contains('-') { // Preferred stock (e.g., "MS-PA")
                    score -= 2;
                }

                // Count occurrences of CIK
                *cik_counts.entry(cik).or_insert(0) += 1;
                candidates.push((cik, ticker, title, score));
            }
        }
    }

    // Boost score based on CIK frequency
    for candidate in &mut candidates {
        candidate.3 += *cik_counts.get(&candidate.0).unwrap_or(&0) as i32;
    }

    // Return the best-matching CIK
    candidates.into_iter().max_by_key(|c| c.3).map(|(cik, _, _, _)| cik)
}
*/

// TODO: Make distinction how these are not fund tickers
pub async fn fetch_company_tickers(
    client: &SecClient,
) -> Result<CompanyTickersDataFrame, Box<dyn Error>> {
    // TODO: Also incorporate: https://www.sec.gov/include/ticker.txt

    let url = "https://www.sec.gov/files/company_tickers.json";
    let data = client.fetch_json(url).await?;

    // TODO: Move the following into `parsers`

    let mut cik_raw_values = Vec::new();
    let mut cik_transformed_values = Vec::new();
    let mut ticker_values = Vec::new();
    let mut title_values = Vec::new();

    if let Some(ticker_map) = data.as_object() {
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
