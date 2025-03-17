use crate::models::Cik;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::LazyLock;

static TOKEN_CACHE: LazyLock<DashMap<String, Vec<String>>> = LazyLock::new(DashMap::new);

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
// ---- [Incorrect stock symbol for PNC] PNC: Investment 109: NportInvestment { company_ticker: Some(CompanyTicker { cik: Cik { value: 1393612 }, ticker_symbol: "DFS", company_name: "Discover Financial Services" }), name: "PNC FINANCIAL SERVICES",
// ---- PCG: Investment 487: NportInvestment { company_ticker: None, name: "PACIFIC GAS & ELECTRIC", lei: "1HNPXZSMMB7HMBMVBS46", title: "Pacific Gas and Electric Co"

// - Container: SPY
//  ---- LOW: Investment 67: NportInvestment { company_ticker: None, name: "Lowe's Cos Inc", lei: "WAFCR4OKGSC504WU3E95", title: "Lowe's Cos Inc", cu
//  ---- TJX: Investment 69: NportInvestment { company_ticker: None, name: "TJX Cos Inc/The", lei: "V167QI9I69W364E2DY52", title: "TJX Cos Inc/The",
//  ---- WMB: Investment 145: NportInvestment { company_ticker: None, name: "Williams Cos Inc/The", lei: "D71FAKCBLFS2O0RBPG08", title: "Williams Cos Inc/The",
//  ---- DHI: Investment 218: NportInvestment { company_ticker: None, name: "DR Horton Inc", lei: "529900ZIUEYVSB8QDD25", title: "DR Horton Inc", cusip: "23331A109", isin: "US23331A1097",
//  ---- LYB: Investment 367: NportInvestment { company_ticker: None, name: "LyondellBasell Industries NV", lei: "BN6WCCZ8OVP3ITUUVN49", title: "LyondellBasell Industries NV",
//  ---- KEY: Investment 390: NportInvestment { company_ticker: None, name: "KeyCorp", lei: "RKPI3RZGV1V1FJTH5T61", title: "KeyCorp",
//  ---- DPZ: Investment 424: NportInvestment { company_ticker: None, name: "Domino's Pizza Inc", lei: "25490005ZWM1IF9UXU57", title: "Domino's Pizza Inc",
//  ---- CPB: Investment 483: NportInvestment { company_ticker: None, name: "The Campbell's Company", lei: "5493007JDSMX8Z5Z1902", title: "The Campbell's Company", cusip: "134429109",
//  ---- JNJ: Investment 22: NportInvestment { company_ticker: None, name: "Johnson & Johnson", lei: "549300G0CFPGEF6X2043", title: "Johnson & Johnson",

const TOKEN_MATCH_THRESHOLD: f64 = 0.6; // At least 60% of tokens must match
const TICKER_SYMBOL_MATCH_BOOST: f64 = 2.0;
const EXACT_MATCH_BOOST: f64 = 10.0;
const COMMON_STOCK_BOOST: f64 = 2.0;
const PREFERRED_STOCK_PENALTY: f64 = -3.0;
const CIK_FREQUENCY_BOOST: f64 = 2.0;
const TICKER_SYMBOL_LENGTH_PENALTY: f64 = -1.0;

impl CompanyTicker {
    // TODO: Rename to `from_fuzzy_matched_name`?
    pub fn get_by_fuzzy_matched_name(
        company_tickers: &[CompanyTicker],
        query: &str,
    ) -> Option<CompanyTicker> {
        // TODO: Lookup in persistent cache before further processing

        let query_tokens = Self::tokenize_company_name(query);

        let mut cik_counts = HashMap::new();
        let mut candidates = Vec::new();

        for company in company_tickers {
            let cik = company.cik.value;
            let ticker_symbol = &company.ticker_symbol;
            let title_tokens = Self::tokenize_company_name(&company.company_name);

            let query_freq = Self::token_frequencies(&query_tokens);
            let title_freq = Self::token_frequencies(&title_tokens);

            // **Compute Weighted Token Overlap**
            let intersection_size: usize = query_freq
                .iter()
                .filter_map(|(token, &query_count)| {
                    title_freq
                        .get(*token)
                        .map(|&title_count| query_count.min(title_count))
                })
                .sum();

            let total_size = query_tokens.len().max(title_tokens.len());

            let match_score = (intersection_size as f64) / (total_size as f64);

            if match_score < TOKEN_MATCH_THRESHOLD {
                continue; // Skip weak matches
            }

            let mut score = match_score * 100.0; // Scale score

            if query_tokens == title_tokens {
                score += EXACT_MATCH_BOOST;
            }

            if query_tokens.contains(&ticker_symbol) {
                score += TICKER_SYMBOL_MATCH_BOOST;
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

            score += TICKER_SYMBOL_LENGTH_PENALTY * company.ticker_symbol.len() as f64;

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

    /// Converts a vector of tokens into a frequency map
    fn token_frequencies(tokens: &[String]) -> HashMap<&String, usize> {
        let mut map = HashMap::new();
        for token in tokens {
            *map.entry(token).or_insert(0) += 1;
        }
        map
    }

    /// **Tokenize text into uppercase words (alphanumeric only) using SIMD**
    pub fn tokenize_company_name(text: &str) -> Vec<String> {
        if let Some(tokens) = TOKEN_CACHE.get(text) {
            return tokens.clone();
        }

        // **Manual Removal Before Normalization**
        let preprocessed = text
            .replace("/NEW/", "")
            .replace("COMPANY", "CO")
            .replace("COMPANIES", "COS")
            .replace("BANCORPORATION", "BANK BANCORP")
            .replace("NATIONAL ASSOCIATION", "NA")
            //
            // Specific patch for `PNC FINANCIAL SERVICES GROUP, INC.`
            .replace("GROUP, INC.", "")
            .replace("PG&E", "PACIFIC GAS AND ELECTRIC CO");

        let mut cleaned = Vec::with_capacity(preprocessed.len());

        for &b in preprocessed.as_bytes() {
            if b == b'\'' {
                continue; // Remove apostrophes
            }
            cleaned.push(if b.is_ascii_alphanumeric() {
                b.to_ascii_uppercase()
            } else {
                b' ' // Replace non-alphanumeric with space
            });
        }

        let normalized = String::from_utf8(cleaned).unwrap();
        let mut tokens: Vec<String> = Vec::new();
        let mut single_letter_buffer = String::new();

        for word in normalized.split_whitespace() {
            let upper = word.to_string();

            // Join single-letter words together
            if upper.len() == 1 {
                single_letter_buffer.push_str(&upper);
            } else {
                if !single_letter_buffer.is_empty() {
                    tokens.push(single_letter_buffer.clone());
                    single_letter_buffer.clear();
                }
                tokens.push(upper);
            }
        }

        // If any single letters remain in the buffer, push them
        if !single_letter_buffer.is_empty() {
            tokens.push(single_letter_buffer);
        }

        TOKEN_CACHE.insert(text.to_string(), tokens.clone());
        tokens
    }

    // TODO: Vectorize
    // **Tokenize text into uppercase words (alphanumeric only)**
    // pub fn tokenize_company_name(text: &str) -> Vec<String> {
    //     // Define common replacements
    //     let replacements: HashMap<&str, &str> = [("COMPANY", "CO"), ("COMPANIES", "COS")]
    //         .iter()
    //         .cloned()
    //         .collect();

    //     text.chars()
    //         .scan(None, |last, c| {
    //             // Remove possessives (apostrophe followed by 's' or standalone apostrophe)
    //             if c == '\'' {
    //                 *last = Some(c);
    //                 return Some(None);
    //             }
    //             if let Some('\'') = *last {
    //                 if c == 'S' || c == 's' {
    //                     *last = None;
    //                     return Some(None); // Skip both apostrophe and 's'
    //                 }
    //             }
    //             *last = None;
    //             Some(Some(if c.is_alphanumeric() { c } else { ' ' }))
    //         })
    //         .flatten() // Remove None values (filtered possessives)
    //         .collect::<String>()
    //         .split_whitespace()
    //         .map(|word| {
    //             let upper = word.to_uppercase();
    //             replacements
    //                 .get(upper.as_str())
    //                 .cloned()
    //                 .unwrap_or(&upper)
    //                 .to_string()
    //         })
    //         .collect()
    // }

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
