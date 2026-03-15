use crate::enums::Url;
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

    pub fn filter_8k_submissions(cik_submissions: &[Self]) -> Vec<Self> {
        cik_submissions
            .iter()
            .filter(|submission| submission.form.to_uppercase() == "8-K")
            .cloned()
            .collect()
    }

    pub fn as_edgar_archive_url(&self) -> String {
        Url::CikAccession(self.cik.clone(), self.accession_number.clone()).value()
    }

    /// Returns the URL of the primary document for this filing.
    ///
    /// This points directly to the main document (e.g., the HTML 8-K or 10-K body),
    /// rather than the filing index directory.
    pub fn as_primary_document_url(&self) -> String {
        format!("{}/{}", self.as_edgar_archive_url(), self.primary_document)
    }
}
