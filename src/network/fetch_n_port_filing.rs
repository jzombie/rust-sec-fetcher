use crate::network::SecClient;
use crate::parsers::{parse_nport_xml, Investment};
use std::error::Error;

// TODO: Rename to `nport_p`
pub async fn fetch_n_port_filing(
    sec_client: &SecClient,
    cik: u64,
    accession_number: &str, // TODO: Strip
) -> Result<Vec<Investment>, Box<dyn Error>> {
    let url = format!(
        "https://www.sec.gov/Archives/edgar/data/{}/{}.xml",
        cik, accession_number
    );

    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None)
        .await?;
    let xml_data = response.text().await?;

    parse_nport_xml(&xml_data)
}
