use crate::models::Cik;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct CompanyTicker {
    pub cik: Cik,
    pub ticker_symbol: String,
    pub company_name: String,
}

// TODO: Add unit tests for (at least)
//  - Container: SPYV
//  ---- ZION:  Investment 4347: NportInvestment { company_ticker: None, name: "ZIONS BANCORP NA"
//  ---- SHW: Investment 4551: NportInvestment { company_ticker: Some(CompanyTicker { cik: Cik { value: 89800 }, ticker_symbol: "SHW", company_name: "SHERWIN WILLIAMS CO" }),
//  ---- HSY: Investment 4522: NportInvestment { company_ticker: None, name: "HERSHEY COMPANY", lei: "21X2CX66SU2BR6QTAD08", title: "Hershey Co/The", cusip: "427866AX6",
//  ---- PGR: Investment 4391: NportInvestment { company_ticker: None, name: "PROGRESSIVE CORP", lei: "529900TACNVLY9DCR586", title: "Progressive Corp/The",
//  ---- [subsidiary; probably ok] DE: Investment 4814: NportInvestment { company_ticker: None, name: "JOHN DEERE CAPITAL CORP", lei: "E0KSF7PFQ210NWI8Z391", title: "John Deere Capital Corp",

const TOKEN_MATCH_THRESHOLD: f64 = 0.6; // At least 60% of tokens must match
const EXACT_MATCH_BOOST: f64 = 10.0;
const COMMON_STOCK_BOOST: f64 = 2.0;
const PREFERRED_STOCK_PENALTY: f64 = -3.0;
const CIK_FREQUENCY_BOOST: f64 = 2.0;

impl CompanyTicker {
    pub fn get_by_fuzzy_matched_name(
        company_tickers: &[CompanyTicker],
        query: &str,
    ) -> Option<CompanyTicker> {
        let query_tokens = tokenize_text(query);

        let mut cik_counts = HashMap::new();
        let mut candidates = Vec::new();

        for company in company_tickers {
            let cik = company.cik.value;
            let title_tokens = tokenize_text(&company.company_name);

            // **Step 1: Compute Token Overlap Percentage**
            let intersection_size = query_tokens.intersection(&title_tokens).count();
            let total_size = query_tokens.len().max(title_tokens.len());

            let match_score = (intersection_size as f64) / (total_size as f64);
            if match_score < TOKEN_MATCH_THRESHOLD {
                continue; // Skip weak matches
            }

            let mut score = match_score * 100.0; // Scale score

            if query_tokens == title_tokens {
                score += EXACT_MATCH_BOOST;
            }
            // if company.ticker_symbol.len() <= 4 {
            //     score += COMMON_STOCK_BOOST;
            // } else if company.ticker_symbol.contains('-') {
            //     score += PREFERRED_STOCK_PENALTY;
            // }
            if company.ticker_symbol.contains('-') {
                score += PREFERRED_STOCK_PENALTY;
            } else {
                score += COMMON_STOCK_BOOST;
            }

            *cik_counts.entry(cik).or_insert(0) += 1;
            candidates.push((company, score));
        }

        // **Step 2: Apply CIK Frequency Boost**
        for (company, score) in &mut candidates {
            if let Some(count) = cik_counts.get(&company.cik.value) {
                *score += *count as f64 * CIK_FREQUENCY_BOOST;
            }
        }

        // **Step 3: Return best match**
        candidates
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(company, _)| company.clone())
    }
}

/// **Tokenize text into uppercase words (alphanumeric only)**
fn tokenize_text(text: &str) -> HashSet<String> {
    text.chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' }) // Replace non-alphanums with space
        .collect::<String>()
        .split_whitespace() // Now safely split by actual words
        .map(|word| word.to_uppercase()) // Convert to uppercase for case-insensitive matching
        .collect()
}
