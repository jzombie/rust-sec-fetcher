use crate::enums::Url;
use crate::models::{AccessionNumber, Cik, CikSubmission, NportInvestment};
use crate::network::{
    fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_company_tickers, SecClient,
};
use crate::parsers::parse_nport_xml;
use std::error::Error;

// TODO: Document
pub async fn fetch_nport_filing_by_ticker_symbol(
    sec_client: &SecClient,
    ticker_symbol: &str,
) -> Result<Vec<NportInvestment>, Box<dyn Error>> {
    let cik = fetch_cik_by_ticker_symbol(&sec_client, ticker_symbol).await?;

    let cik_submissions = fetch_cik_submissions(&sec_client, cik).await?;

    let latest_nport_p_submission =
        CikSubmission::most_recent_nport_p_submission(cik_submissions.as_slice())
            .ok_or_else(|| "Could not obtain most recent NPORT-P submission.")?;

    let investments = fetch_nport_filing_by_cik_and_accession_number(
        &sec_client,
        latest_nport_p_submission.cik,
        latest_nport_p_submission.accession_number,
    )
    .await?;

    Ok(investments)
}

// TODO: Document
// TODO: Include expense ratios!
pub async fn fetch_nport_filing_by_cik_and_accession_number(
    sec_client: &SecClient,
    cik: Cik,
    accession_number: AccessionNumber,
) -> Result<Vec<NportInvestment>, Box<dyn Error>> {
    let company_tickers = fetch_company_tickers(&sec_client).await?;

    let url = Url::CikAccessionPrimaryDocument(cik.clone(), accession_number.clone()).value();

    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;
    let xml_data = response.text().await?;

    parse_nport_xml(&xml_data, &company_tickers)
}
