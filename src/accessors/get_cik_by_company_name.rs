// TODO: Based on string similarity for the purpose of mapping LEI (on `NPORT` filings) to CIK
// A good query to test this with is "Morgan Stanley".

// pub fn get_cik_by_company_name(json_data: &str, query: &str) -> Option<u64> {
//     let data: Value = serde_json::from_str(json_data).ok()?;

//     let mut cik_counts: HashMap<u64, usize> = HashMap::new();
//     let mut candidates: Vec<(u64, String, String, i32)> = Vec::new(); // (CIK, ticker, title, score)

//     // Iterate over dataset
//     if let Some(objects) = data.as_object() {
//         for obj in objects.values() {
//             let cik = obj.get("cik_str")?.as_u64()?;
//             let ticker = obj.get("ticker")?.as_str()?.to_string();
//             let title = obj.get("title")?.as_str()?.to_lowercase();

//             // Only consider companies that match the query
//             if title.contains(&query.to_lowercase()) {
//                 // Scoring system
//                 let mut score = 0;

//                 if title == query.to_lowercase() {
//                     score += 5; // Exact match
//                 }
//                 if ticker.len() <= 4 {
//                     // Common stock (e.g., "MS")
//                     score += 3;
//                 } else if ticker.contains('-') {
//                     // Preferred stock (e.g., "MS-PA")
//                     score -= 2;
//                 }

//                 // Count occurrences of CIK
//                 *cik_counts.entry(cik).or_insert(0) += 1;
//                 candidates.push((cik, ticker, title, score));
//             }
//         }
//     }

//     // Boost score based on CIK frequency
//     for candidate in &mut candidates {
//         candidate.3 += *cik_counts.get(&candidate.0).unwrap_or(&0) as i32;
//     }

//     // Return the best-matching CIK
//     candidates
//         .into_iter()
//         .max_by_key(|c| c.3)
//         .map(|(cik, _, _, _)| cik)
// }
