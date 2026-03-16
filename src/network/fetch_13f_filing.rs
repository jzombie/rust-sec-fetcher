use crate::enums::Url;
use crate::models::{CikSubmission, ThirteenfHolding};
use crate::network::{fetch_filing_index, SecClient};
use crate::parsers::parse_13f_xml;
use std::error::Error;

/// Fetches and parses a 13F-HR filing given its [`CikSubmission`].
///
/// 13F holdings live in a separate XML document (named `INFORMATION TABLE` in
/// the filing index) rather than in `primary_doc.xml`. Because filers use
/// inconsistent filenames (e.g. `form13fInfoTable.xml`, `infotable.xml`,
/// `xslForm13F_X02.xml`), this function fetches the filing index first to
/// discover the actual filename, then fetches and parses that file.
pub async fn fetch_13f_filing(
    sec_client: &SecClient,
    submission: &CikSubmission,
) -> Result<Vec<ThirteenfHolding>, Box<dyn Error>> {
    // Discover the informationTable filename from the filing index.
    let index = fetch_filing_index(sec_client, submission).await?;

    let info_doc = index
        .documents
        .iter()
        .find(|d| d.document_type.to_uppercase().contains("INFORMATION TABLE"))
        .ok_or_else(|| {
            format!(
                "No INFORMATION TABLE document found in 13F index for {}",
                submission.accession_number.to_string()
            )
        })?;

    let url = Url::CikAccessionDocument(
        submission.cik.clone(),
        submission.accession_number.clone(),
        info_doc.name.clone(),
    )
    .value();

    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;
    let xml = response.text().await?;
    parse_13f_xml(&xml)
}
