use crate::enums::Url;
use crate::models::{Cik, CikSubmission};
use crate::network::SecClient;
use serde_json::Value;
use std::error::Error;

/// Fetches the complete EDGAR filing history for a registrant, returned as a
/// flat list of [`CikSubmission`]s ordered **newest-first**.
///
/// # What is a CIK submissions response?
///
/// The SEC's `https://data.sec.gov/submissions/CIK{cik}.json` endpoint is the
/// authoritative source of metadata for every filing a registrant has ever made
/// with EDGAR.  A single JSON response contains:
///
/// - Registrant identity fields: `name`, `entityType`, `sic`, `sicDescription`,
///   `category`, `stateOfIncorporation`, `fiscalYearEnd`, `exchanges`, etc.
/// - A `filings.recent` block: the most recent ~1,000 filings, each row
///   carrying `accessionNumber`, `form`, `primaryDocument`, `filingDate`, and
///   `items` (for 8-Ks).
/// - An optional `filings.files` array listing additional paginated JSON files
///   (`CIK{cik}-submissions-001.json`, etc.) for registrants with very long
///   histories.  This function **follows all pages** automatically.
///
/// This function is the common data source for every `fetch_*_filings` helper
/// (8-K, 10-K, 10-Q, 13F-HR, NPORT-P, Form 4).  The HTTP response is cached
/// locally, so calling multiple per-form helpers for the same CIK only hits
/// the network once.
///
/// # Example
///
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_cik_submissions, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
/// let submissions = fetch_cik_submissions(&client, cik).await?;
/// println!("Total filings: {}", submissions.len());
/// # Ok(())
/// # }
/// ```
pub async fn fetch_cik_submissions(
    sec_client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let url = Url::CikSubmission(cik.clone()).value();
    let data: Value = sec_client.fetch_json(&url, None).await?;

    let entity_type: Option<String> = data["entityType"].as_str().map(|s| s.to_string());

    let mut submissions: Vec<CikSubmission> = Vec::new();

    // Parse the primary `recent` block
    if let Some(recent) = data["filings"]["recent"].as_object() {
        submissions = crate::parsers::parse_cik_submissions_block(
            &Value::Object(recent.clone()),
            &cik,
            &entity_type,
        );
    }

    // Follow any paginated files listed in filings.files
    // Each entry: { "name": "CIK0000320193-submissions-001.json", "filingCount": N, ... }
    if let Some(files) = data["filings"]["files"].as_array() {
        for file_entry in files {
            if let Some(filename) = file_entry["name"].as_str() {
                let page_url = Url::CikSubmissionPage(filename.to_string()).value();
                match sec_client.fetch_json(&page_url, None).await {
                    Ok(page_data) => {
                        let page_subs = crate::parsers::parse_cik_submissions_block(
                            &page_data,
                            &cik,
                            &entity_type,
                        );
                        submissions.extend(page_subs);
                    }
                    Err(e) => {
                        // Non-fatal: skip this page rather than failing the whole fetch
                        eprintln!(
                            "Warning: failed to fetch submissions page {}: {}",
                            filename, e
                        );
                    }
                }
            }
        }
    }

    Ok(submissions)
}
