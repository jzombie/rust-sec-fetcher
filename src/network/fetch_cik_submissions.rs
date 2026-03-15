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
        extract_filings_from_block(&Value::Object(recent.clone()), &cik, &entity_type, &mut submissions);
    }

    // Follow any paginated files listed in filings.files
    // Each entry: { "name": "CIK0000320193-submissions-001.json", "filingCount": N, ... }
    if let Some(files) = data["filings"]["files"].as_array() {
        for file_entry in files {
            if let Some(filename) = file_entry["name"].as_str() {
                let page_url = Url::CikSubmissionPage(filename.to_string()).value();
                match sec_client.fetch_json(&page_url, None).await {
                    Ok(page_data) => {
                        extract_filings_from_block(&page_data, &cik, &entity_type, &mut submissions);
                    }
                    Err(e) => {
                        // Non-fatal: skip this page rather than failing the whole fetch
                        eprintln!("Warning: failed to fetch submissions page {}: {}", filename, e);
                    }
                }
            }
        }
    }

    Ok(submissions)
}
