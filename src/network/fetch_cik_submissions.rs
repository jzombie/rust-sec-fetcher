use crate::enums::Url;
use crate::models::{AccessionNumber, Cik, CikSubmission};
use crate::network::SecClient;
use chrono::NaiveDate;
use serde_json::Value;
use std::error::Error;

/// Parses a `filings` block (either `recent` or a paginated page) and appends
/// results into the supplied output vectors.
fn extract_filings_from_block(
    block: &Value,
    cik: &Cik,
    entity_type: &Option<String>,
    out: &mut Vec<CikSubmission>,
) {
    let accession_numbers = block["accessionNumber"].as_array();
    let forms = block["form"].as_array();
    let primary_documents = block["primaryDocument"].as_array();
    let filing_dates = block["filingDate"].as_array();
    let items_field = block["items"].as_array();

    let (Some(accns), Some(forms), Some(docs), Some(dates)) =
        (accession_numbers, forms, primary_documents, filing_dates)
    else {
        return;
    };

    for (idx, (accn_val, form_val, doc_val, date_val)) in
        itertools::izip!(accns, forms, docs, dates).enumerate()
    {
        let accession_number_str = accn_val.as_str().unwrap_or_default();
        let accession_number = match AccessionNumber::from_str(accession_number_str) {
            Ok(a) => a,
            Err(_) => continue,
        };

        let filing_date_parsed = date_val
            .as_str()
            .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

        // items is a comma-separated string like "2.02,9.01"
        let items = items_field
            .and_then(|arr| arr.get(idx))
            .and_then(|v| v.as_str())
            .map(|s| {
                s.split(',')
                    .map(|i| i.trim().to_string())
                    .filter(|i| !i.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        out.push(CikSubmission {
            cik: cik.clone(),
            entity_type: entity_type.clone(),
            accession_number,
            form: form_val.as_str().unwrap_or("").to_string(),
            primary_document: doc_val.as_str().unwrap_or("").to_string(),
            filing_date: filing_date_parsed,
            items,
        });
    }
}

/// Parses a raw SEC submissions JSON value into a list of [`CikSubmission`]s.
///
/// This is the pure parsing core of [`fetch_cik_submissions`], exposed for
/// testing and offline processing.  Pass the full JSON body returned by
/// `https://data.sec.gov/submissions/CIK{cik}.json` (or a loaded fixture).
/// Pagination (`filings.files`) is **not** followed; only the `filings.recent`
/// block present in the value is parsed.
///
/// The `cik` argument is stamped onto every returned submission because the
/// SEC JSON does not repeat the CIK in each filing row.
pub fn parse_cik_submissions_json(data: &Value, cik: Cik) -> Vec<CikSubmission> {
    let entity_type: Option<String> = data["entityType"].as_str().map(|s| s.to_string());
    let mut submissions: Vec<CikSubmission> = Vec::new();
    if let Some(recent) = data["filings"]["recent"].as_object() {
        extract_filings_from_block(
            &Value::Object(recent.clone()),
            &cik,
            &entity_type,
            &mut submissions,
        );
    }
    submissions
}

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
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "AAPL").await?;
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
        extract_filings_from_block(
            &Value::Object(recent.clone()),
            &cik,
            &entity_type,
            &mut submissions,
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
                        extract_filings_from_block(
                            &page_data,
                            &cik,
                            &entity_type,
                            &mut submissions,
                        );
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
