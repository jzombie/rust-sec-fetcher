use crate::models::{AccessionNumber, Cik, NportInvestment};
use crate::network::SecClient;
use crate::parsers::parse_nport_xml;
use std::error::Error;

// TODO: Include expense ratios!
pub async fn fetch_nport_filing(
    sec_client: &SecClient,
    cik: Cik,
    accession_number: AccessionNumber,
) -> Result<Vec<NportInvestment>, Box<dyn Error>> {
    // TODO: Dedupe
    let url = format!(
        "https://www.sec.gov/Archives/edgar/data/{}/{}/primary_doc.xml",
        cik.to_string(),
        accession_number.to_unformatted_string()
    );

    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None)
        .await?;
    let xml_data = response.text().await?;

    parse_nport_xml(&xml_data)
}
