use crate::enums::Url;
use crate::models::{AccessionNumber, Cik, CikSubmission};
use crate::network::SecClient;
use chrono::NaiveDate;
use serde_json::Value;
use std::error::Error;

pub async fn fetch_cik_submissions(
    sec_client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let url = Url::CikSubmission(cik.clone()).value();

    let data: Value = sec_client.fetch_json(&url, None).await?;

    // TODO: Move the following into `parsers`

    let entity_type_value: Option<String> = data["entityType"].as_str().map(|s| s.to_string());

    let mut accession_number_values = Vec::new();
    let mut form_values = Vec::new();
    let mut primary_document_values = Vec::new();
    let mut filing_dates = Vec::new();

    if let Some(filings) = data["filings"].as_object() {
        if let Some(recent) = filings["recent"].as_object() {
            if let Some(accession_numbers) = recent["accessionNumber"].as_array() {
                for accession_number in accession_numbers {
                    accession_number_values.push(accession_number);
                }
            }

            if let Some(forms) = recent["form"].as_array() {
                for form in forms {
                    form_values.push(form);
                }
            }

            if let Some(primary_documents) = recent["primaryDocument"].as_array() {
                for primary_document in primary_documents {
                    primary_document_values.push(primary_document);
                }
            }

            if let Some(filing_dates_array) = recent["filingDate"].as_array() {
                for filing_date in filing_dates_array {
                    filing_dates.push(filing_date);
                }
            }
        }
    }

    let mut cik_submissions: Vec<CikSubmission> = Vec::with_capacity(accession_number_values.len());

    for (accession_number_value, form, primary_document, filing_date) in itertools::izip!(
        &accession_number_values,
        &form_values,
        &primary_document_values,
        &filing_dates
    ) {
        let filing_date_parsed = filing_date
            .as_str()
            .and_then(|date_str| NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok());

        let accession_number_str = accession_number_value.as_str().unwrap_or_default();
        let accession_number = AccessionNumber::from_str(accession_number_str)?;

        cik_submissions.push(CikSubmission {
            cik: cik.clone(),
            entity_type: entity_type_value.clone(),
            accession_number,
            form: form.as_str().unwrap_or("").to_string(),
            primary_document: primary_document.as_str().unwrap_or("").to_string(),
            filing_date: filing_date_parsed,
        });
    }

    Ok(cik_submissions)
}
