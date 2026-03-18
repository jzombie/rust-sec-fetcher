use crate::enums::FormType;
use crate::models::{Cik, CikSubmission, FeedEntry};
use crate::network::{fetch_edgar_feeds_since, fetch_filings, SecClient};
use chrono::{DateTime, FixedOffset};
use std::collections::HashSet;
use std::error::Error;

/// Fetches all IPO registration filings (S-1, S-1/A, F-1, F-1/A) for a CIK,
/// returning them sorted newest-first.
///
/// Queries all form types in [`FormType::IPO_REGISTRATION_FORM_TYPES`] to
/// handle both domestic (S-1 family) and foreign private issuers (F-1 family)
/// without requiring the caller to know which applies.
///
/// # Example
///
/// ```rust,no_run
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::Cik;
/// # use sec_fetcher::network::SecClient;
/// # use sec_fetcher::ops::get_ipo_registration_filings;
/// # use std::str::FromStr;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = Cik::from_str("0002039972")?;
/// let filings = get_ipo_registration_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{} — {:?}", f.form, f.filing_date);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn get_ipo_registration_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let mut all = Vec::new();
    for ft in FormType::IPO_REGISTRATION_FORM_TYPES {
        let mut filings = fetch_filings(client, cik.clone(), ft).await?;
        all.append(&mut filings);
    }

    // Sort newest-first so callers can use index 0 as "latest".
    all.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));

    Ok(all)
}

/// Fetches EDGAR feed entries for the given form types, returning a
/// deduplicated, exact-match-filtered list sorted newest-first.
///
/// EDGAR's `type=` URL parameter is a *prefix* match, so requesting `"S-1"`
/// also returns `"S-11"` (REIT forms, not IPOs). This function exact-matches
/// and deduplicates by accession number so each filing appears exactly once
/// regardless of how many form-type streams may include it.
///
/// # Returns
///
/// A `(entries, high_water_mark)` tuple.  The `high_water_mark` is the
/// acceptance timestamp of the newest entry, suitable for delta-poll `--since`
/// arguments on the next run.
///
/// # Example
///
/// ```rust,no_run
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::enums::FormType;
/// # use sec_fetcher::network::SecClient;
/// # use sec_fetcher::ops::get_ipo_feed_entries;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let (entries, hw) = get_ipo_feed_entries(&client, FormType::IPO_REGISTRATION_FORM_TYPES, None, 5).await?;
/// for entry in &entries {
///     println!("{} — {} — {}", entry.form_type, entry.company_name, entry.filing_href);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn get_ipo_feed_entries(
    client: &SecClient,
    form_types: &[FormType],
    since: Option<DateTime<FixedOffset>>,
    max_pages: usize,
) -> Result<(Vec<FeedEntry>, Option<DateTime<FixedOffset>>), Box<dyn Error>> {
    let ft_strs: Vec<&str> = form_types.iter().map(|ft| ft.as_edgar_str()).collect();
    let delta = fetch_edgar_feeds_since(client, &ft_strs, since, max_pages).await?;
    let high_water = delta.high_water;

    let mut seen = HashSet::new();
    let entries: Vec<FeedEntry> = delta
        .entries
        .into_iter()
        // EDGAR prefix-match produces false positives (e.g. S-11 from S-1
        // stream); exact-match filter removes them.
        .filter(|e| {
            form_types
                .iter()
                .any(|ft| e.form_type.eq_ignore_ascii_case(ft.as_edgar_str()))
        })
        // Each form type is fetched as a separate stream; dedup by accession
        // number so filings appearing in multiple streams appear exactly once.
        .filter(|e| seen.insert(e.accession_number.clone()))
        .collect();

    Ok((entries, high_water))
}
