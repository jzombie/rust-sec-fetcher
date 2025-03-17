use crate::models::Cik;
use std::collections::HashMap;

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

impl CompanyTicker {
    pub fn get_by_fuzzy_matched_name(
        company_tickers: &[CompanyTicker],
        query: &str,
    ) -> Option<CompanyTicker> {
        let mut cik_counts: HashMap<u64, usize> = HashMap::with_capacity(company_tickers.len());
        let mut candidates: Vec<(&CompanyTicker, f64)> = Vec::new(); // (CompanyTicker reference, score)

        let query_lower = query.to_lowercase();

        let normalized_query = normalize_text(&query_lower);

        // if normalize_text(&query_lower) == normalize_text(&title) {
        //     // Count occurrences of CIK
        //     *cik_counts.entry(cik).or_insert(0) += 1;

        //     candidates.push((company, 1.0));
        // }

        for company in company_tickers {
            let cik = company.cik.value; // Extract raw u64
                                         // let ticker = company.ticker_symbol.as_str();
            let title_lower = company.company_name.to_lowercase();

            // let normalized_title = normalize_text(text)

            let normalized_title = normalize_text(&title_lower);

            if normalized_query == normalized_title {
                // Count occurrences of CIK
                *cik_counts.entry(cik).or_insert(0) += 1;

                candidates.push((company, 1.0));
            }

            // Compute Jaro-Winkler similarity (range: 0.0 - 1.0)
            // let similarity = jaro_winkler(&query_lower, &title);

            // TODO: Extract weights as constants

            // Only consider if similarity > 0.7 (adjust threshold as needed)
            // if similarity > 0.7 {
            //     let mut score = similarity * 10.0; // Convert to a weighted score

            //     if title == query_lower {
            //         score += 5.0; // Exact match boost
            //     }
            //     if ticker.len() <= 4 {
            //         score += 3.0; // Common stock
            //     } else if ticker.contains('-') {
            //         score -= 2.0; // Preferred stock penalty
            //     }

            //     // Count occurrences of CIK
            //     *cik_counts.entry(cik).or_insert(0) += 1;

            //     candidates.push((company, score));
            // }
        }

        // Boost score based on CIK frequency
        for candidate in &mut candidates {
            candidate.1 += *cik_counts.get(&candidate.0.cik.value).unwrap_or(&0) as f64;
        }

        // Return the best-matching `CompanyTicker`
        // Return the best-matching `CompanyTicker` (cloning the selected reference)
        candidates
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(company, _)| (*company).clone())
    }
}

fn normalize_text(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len()); // Preallocate for speed
    let mut last_was_space = false;

    for c in text.chars() {
        if c.is_alphanumeric() {
            cleaned.push(c.to_ascii_lowercase());
            last_was_space = false;
        } else if !last_was_space {
            cleaned.push(' '); // Add a single space for non-alphanumeric
            last_was_space = true;
        }
    }

    cleaned.trim().to_string() // Trim trailing space
}
