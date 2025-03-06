use crate::network::SecClient;
use std::error::Error;

pub async fn fetch_investment_company_series_and_class_dataset(
    sec_client: &SecClient,
    year: usize,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let url = format!(
        "https://www.sec.gov/files/investment/data/other/investment-company-series-and-class-information/investment-company-series-class-{}.csv",
        year
    );

    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None)
        .await?;
    let bytes = response.bytes().await?;
    Ok(bytes.to_vec())
}
