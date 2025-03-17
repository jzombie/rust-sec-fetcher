use crate::models::Cik;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CompanyTicker {
    pub cik: Cik,
    pub ticker_symbol: String,
    pub company_name: String,
}

impl CompanyTicker {
    pub fn get_by_fuzzy_matched_name(
        company_tickers: &[CompanyTicker],
        query: &str,
    ) -> Option<CompanyTicker> {
        let mut cik_counts: HashMap<u64, usize> = HashMap::with_capacity(company_tickers.len());
        let mut candidates: Vec<(&CompanyTicker, f64)> = Vec::new(); // (CompanyTicker reference, score)

        let query_lower = query.to_lowercase();

        for company in company_tickers {
            let cik = company.cik.value; // Extract raw u64
            let ticker = company.ticker_symbol.as_str();
            let title = company.company_name.to_lowercase();

            if query_lower == title {
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
