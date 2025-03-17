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

// - Container: SPY
//  ---- LOW: Investment 67: NportInvestment { company_ticker: None, name: "Lowe's Cos Inc", lei: "WAFCR4OKGSC504WU3E95", title: "Lowe's Cos Inc", cu
//  ---- TJX: Investment 69: NportInvestment { company_ticker: None, name: "TJX Cos Inc/The", lei: "V167QI9I69W364E2DY52", title: "TJX Cos Inc/The",
//  ---- WMB: Investment 145: NportInvestment { company_ticker: None, name: "Williams Cos Inc/The", lei: "D71FAKCBLFS2O0RBPG08", title: "Williams Cos Inc/The",
//  ---- DHI: Investment 218: NportInvestment { company_ticker: None, name: "DR Horton Inc", lei: "529900ZIUEYVSB8QDD25", title: "DR Horton Inc", cusip: "23331A109", isin: "US23331A1097",
//  ---- LYB: Investment 367: NportInvestment { company_ticker: None, name: "LyondellBasell Industries NV", lei: "BN6WCCZ8OVP3ITUUVN49", title: "LyondellBasell Industries NV",
//  ---- KEY: Investment 390: NportInvestment { company_ticker: None, name: "KeyCorp", lei: "RKPI3RZGV1V1FJTH5T61", title: "KeyCorp",
//  ---- DPZ: Investment 424: NportInvestment { company_ticker: None, name: "Domino's Pizza Inc", lei: "25490005ZWM1IF9UXU57", title: "Domino's Pizza Inc",
//  ---- CPB: Investment 483: NportInvestment { company_ticker: None, name: "The Campbell's Company", lei: "5493007JDSMX8Z5Z1902", title: "The Campbell's Company", cusip: "134429109",

const TOKEN_MATCH_THRESHOLD: f64 = 0.6; // At least 60% of tokens must match
const EXACT_MATCH_BOOST: f64 = 10.0;
const COMMON_STOCK_BOOST: f64 = 2.0;
const PREFERRED_STOCK_PENALTY: f64 = -3.0;
const CIK_FREQUENCY_BOOST: f64 = 2.0;

impl CompanyTicker {
    // TODO: Rename to `from_fuzzy_matched_name`?
    pub fn get_by_fuzzy_matched_name(
        company_tickers: &[CompanyTicker],
        query: &str,
    ) -> Option<CompanyTicker> {
        let query_tokens = Self::tokenize_company_name(query);

        let mut cik_counts = HashMap::new();
        let mut candidates = Vec::new();

        for company in company_tickers {
            let cik = company.cik.value;
            let title_tokens = Self::tokenize_company_name(&company.company_name);

            let query_set: HashSet<_> = query_tokens.iter().collect();
            let title_set: HashSet<_> = title_tokens.iter().collect();

            // **Step 1: Compute Token Overlap Percentage**
            let intersection_size = query_set.intersection(&title_set).count();
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

    // TODO: Vectorize
    // TODO: Use some common replacements
    //  - Company / CO
    //  - Companies / Cos
    /// **Tokenize text into uppercase words (alphanumeric only)**
    pub fn tokenize_company_name(text: &str) -> Vec<String> {
        text.chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' }) // Replace non-alphanums with space
            .collect::<String>()
            .split_whitespace() // Split into words
            .map(|word| word.to_uppercase()) // Convert to uppercase
            .collect::<Vec<String>>() // Preserve order
    }

    /*
    use std::collections::HashSet;
    use std::simd::{u8x16, SimdPartialEq};

    /// **Tokenize text into uppercase words (alphanumeric only) using SIMD**
    fn tokenize_text_simd(text: &str) -> HashSet<String> {
        let mut cleaned = String::with_capacity(text.len());

        // Process in chunks of 16 bytes using SIMD
        let bytes = text.as_bytes();
        let mut i = 0;
        while i + 16 <= bytes.len() {
            let chunk = u8x16::from_slice(&bytes[i..i + 16]);

            // Create a mask for alphanumeric characters (0-9, A-Z, a-z)
            let is_alnum = (chunk.ge(u8x16::splat(b'0')) & chunk.le(u8x16::splat(b'9')))
                | (chunk.ge(u8x16::splat(b'A')) & chunk.le(u8x16::splat(b'Z')))
                | (chunk.ge(u8x16::splat(b'a')) & chunk.le(u8x16::splat(b'z')));

            // Replace non-alphanumeric bytes with spaces
            let filtered_chunk = is_alnum.select(chunk, u8x16::splat(b' '));
            cleaned.push_str(&String::from_utf8_lossy(filtered_chunk.as_array()));

            i += 16;
        }

        // Process remaining bytes
        for &b in &bytes[i..] {
            cleaned.push(if b.is_ascii_alphanumeric() { b as char } else { ' ' });
        }

        // Tokenize by whitespace and convert to uppercase
        cleaned
            .split_whitespace()
            .map(|word| word.to_ascii_uppercase())
            .collect()
    }

    */
}
