use crate::models::{AccessionNumber, Cik, CikSubmission};
use chrono::NaiveDate;
use serde_json::Value;

/// Parses a single `filings` block (either `recent` or a paginated page) and
/// returns the parsed [`CikSubmission`]s.
///
/// The block is expected to contain parallel arrays under the keys
/// `accessionNumber`, `form`, `primaryDocument`, `filingDate`, and `items`
/// — the standard EDGAR CIK submissions JSON format.
///
/// This is the low-level parser; prefer [`parse_cik_submissions_json`] when
/// working with the top-level CIK submissions response.
pub fn parse_cik_submissions_block(
    block: &Value,
    cik: &Cik,
    entity_type: &Option<String>,
) -> Vec<CikSubmission> {
    let accession_numbers = block["accessionNumber"].as_array();
    let forms = block["form"].as_array();
    let primary_documents = block["primaryDocument"].as_array();
    let filing_dates = block["filingDate"].as_array();
    let items_field = block["items"].as_array();

    let (Some(accns), Some(forms), Some(docs), Some(dates)) =
        (accession_numbers, forms, primary_documents, filing_dates)
    else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(accns.len());
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
    out
}

/// Parses a raw SEC submissions JSON value into a list of [`CikSubmission`]s.
///
/// This is the pure parsing core, exposed for testing and offline processing.
/// Pass the full JSON body returned by
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
        submissions =
            parse_cik_submissions_block(&Value::Object(recent.clone()), &cik, &entity_type);
    }
    submissions
}
