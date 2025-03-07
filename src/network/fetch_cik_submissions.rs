use crate::network::SecClient;
use crate::transform::format_cik;
use chrono::NaiveDate;
use serde_json::Value;
use std::error::Error;
use string_replace_all::StringReplaceAll;

#[derive(Clone, Debug)]
pub struct CikSubmission {
    pub cik: u64,
    // TODO: Add `sic` here and provide the ability elsewhere to map to sector and industry
    pub entity_type: Option<String>,
    pub accession_number: String,
    pub accession_number_stripped: String,
    pub form: String,
    pub primary_document: String,
    pub filing_date: Option<NaiveDate>, // New field for the date
}

impl CikSubmission {
    pub fn filter_nport_p_submissions(cik_submissions: &[Self]) -> Vec<Self> {
        cik_submissions
            .iter()
            .filter(|submission| submission.form.to_uppercase() == "NPORT-P")
            .cloned()
            .collect()
    }

    pub fn most_recent_nport_p_submission(cik_submissions: &[Self]) -> Option<Self> {
        let nport_p_submissions = Self::filter_nport_p_submissions(cik_submissions);

        nport_p_submissions.first().cloned()
    }

    pub fn as_edgar_archive_url(&self) -> String {
        format!(
            "https://www.sec.gov/Archives/edgar/data/{}/{}/",
            self.cik, self.accession_number_stripped
        )
    }
}

// TODO: Move into impl?
pub async fn fetch_cik_submissions(
    sec_client: &SecClient,
    cik: u64,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let formatted_cik = format_cik(cik);
    let url = format!("https://data.sec.gov/submissions/CIK{}.json", formatted_cik);
    let data: Value = sec_client.fetch_json(&url).await?;

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

    for (accession_number, form, primary_document, filing_date) in itertools::izip!(
        &accession_number_values,
        &form_values,
        &primary_document_values,
        &filing_dates
    ) {
        let filing_date_parsed = filing_date
            .as_str()
            .and_then(|date_str| NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok());

        cik_submissions.push(CikSubmission {
            cik,
            entity_type: entity_type_value.clone(),
            accession_number: accession_number.as_str().unwrap_or("").to_string(),
            accession_number_stripped: accession_number
                .as_str()
                .unwrap_or("")
                .to_string()
                .replace_all("-", ""),
            form: form.as_str().unwrap_or("").to_string(),
            primary_document: primary_document.as_str().unwrap_or("").to_string(),
            filing_date: filing_date_parsed,
        });
    }

    Ok(cik_submissions)
}
