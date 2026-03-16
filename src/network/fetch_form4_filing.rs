use crate::enums::Url;
use crate::models::{CikSubmission, Form4Transaction};
use crate::network::SecClient;
use crate::parsers::parse_form4_xml;
use std::error::Error;

/// Fetches and parses a Form 4 filing given its [`CikSubmission`].
///
/// The primary document is the Form 4 XML file listed in the submission.
/// Returns one [`Form4Transaction`] per transaction row in the filing,
/// sorted by transaction date descending.
pub async fn fetch_form4_filing(
    sec_client: &SecClient,
    submission: &CikSubmission,
) -> Result<Vec<Form4Transaction>, Box<dyn Error>> {
    // SEC's submissions JSON sometimes contains an XSLT-prefixed path for
    // Form 4 primary documents (e.g. "xslF345X05/form4.xml"). That URL
    // returns rendered HTML, not the raw XML. Strip any directory prefix so
    // we always fetch the raw XML from the archive root.
    let doc_name = submission.primary_document
        .rfind('/')
        .map(|pos| submission.primary_document[pos + 1..].to_string())
        .unwrap_or_else(|| submission.primary_document.clone());

    let url = Url::CikAccessionDocument(
        submission.cik.clone(),
        submission.accession_number.clone(),
        doc_name,
    ).value();

    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;
    let xml = response.text().await?;
    parse_form4_xml(&xml, submission.filing_date)
}
