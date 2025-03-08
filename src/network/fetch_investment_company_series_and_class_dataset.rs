use crate::network::SecClient;
use std::error::Error;

// TODO: Fetch as dataframe or vector of structs

/// Fetches the **Investment Company Series and Class Report** from the SEC's dataset.
/// 
/// This dataset provides **identification and classification details** for registered
/// investment company series and share classes. It includes details such as:
/// - **CIK** (Central Index Key)  
/// - **Series & Class IDs** (Unique SEC-assigned identifiers)  
/// - **Fund Names** (Legal names of investment entities)  
/// - **Organization Types** (Open-end mutual funds, variable annuities, etc.)  
/// - **Ticker Symbols** (If available)  
/// - **Registrant Addresses**  
/// 
/// Data is available in **CSV** format, with historical reports going back to 2010.
/// 
/// # Arguments
/// - `sec_client` - A reference to an instance of `SecClient` used for making HTTP requests.
/// - `year` - The target **year** of the dataset to fetch (e.g., 2024).
///
/// # Returns
/// Returns a `Result<Vec<u8>, Box<dyn Error>>`, where:
/// - `Ok(Vec<u8>)` contains the **raw CSV data** as bytes.
/// - `Err(Box<dyn Error>)` if the request fails.
/// 
/// # Reference
/// - SEC Dataset: [Investment Company Series and Class Information](https://www.sec.gov/about/opendatasets)
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
