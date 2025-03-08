use crate::models::{Cik, NportInvestment};
use crate::network::SecClient;
use crate::parsers::parse_nport_xml;
use std::error::Error;

pub async fn fetch_nport_filing(
    sec_client: &SecClient,
    cik: Cik,
    accession_number: &str, // TODO: Use `AccessionNumber` model
) -> Result<Vec<NportInvestment>, Box<dyn Error>> {
    // TODO: Move to transformer and dedupe
    let url = format!(
        "https://www.sec.gov/Archives/edgar/data/{}/{}.xml",
        cik.to_string(), accession_number
    );

    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None)
        .await?;
    let xml_data = response.text().await?;

    parse_nport_xml(&xml_data)
}
