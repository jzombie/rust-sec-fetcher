use crate::models::{AccessionNumber, Cik};
use chrono::NaiveDate;

#[derive(Clone, Debug)]
pub struct CikSubmission {
    pub cik: Cik,
    // TODO: Add these fields (and more), but not per submission; capture as a single separate entity
    // pub name: Option<String>,            // i.e. "Apple"
    pub entity_type: Option<String>, // i.e. "operating"
    // pub sic: Option<u64>,                // i.e. 3571
    // pub sic_description: Option<String>, // i.e. "Electronic Computers"
    // pub owner_org: Option<String>,       // i.e. "06 Technology"
    // insiderTransactionForOwnerExists
    // insiderTransactionForIssuerExists
    pub accession_number: AccessionNumber,
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

    // TODO: Dedupe
    pub fn as_edgar_archive_url(&self) -> String {
        format!(
            "https://www.sec.gov/Archives/edgar/data/{}/{}/",
            self.cik.to_string(),
            self.accession_number.to_string()
        )
    }
}
