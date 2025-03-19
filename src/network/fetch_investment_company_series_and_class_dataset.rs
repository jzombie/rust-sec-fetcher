use crate::models::InvestmentCompany;
use crate::network::SecClient;
use crate::parsers::parse_investment_companies_csv;
use chrono::{Datelike, Utc};
use std::error::Error;

/// Fetches the Investment Company Series and Class dataset for a specific year.
///
/// # Arguments
/// - `sec_client` - A reference to an instance of `SecClient` used for HTTP requests.
/// - `year` - The target **year** to fetch (e.g., 2024).
///
/// # Returns
/// Returns `Result<Vec<InvestmentCompany>, Box<dyn Error>>`, where:
/// - `Ok(Vec<InvestmentCompany>)` contains the parsed investment companies.
/// - `Err(Box<dyn Error>)` if the request fails.
pub async fn fetch_investment_company_series_and_class_dataset_for_year(
    year: usize,
    sec_client: &SecClient,
) -> Result<Vec<InvestmentCompany>, Box<dyn Error>> {
    let url = format!(
        "https://www.sec.gov/files/investment/data/other/investment-company-series-and-class-information/investment-company-series-class-{}.csv",
        year
    );

    let throttle_policy_override = {
        let mut policy = sec_client.get_throttle_policy();

        policy.max_retries = 2;

        policy
    };

    let response = sec_client
        .raw_request(
            reqwest::Method::GET,
            &url,
            None,
            Some(throttle_policy_override),
        )
        .await?;
    let byte_array = response.bytes().await?;

    parse_investment_companies_csv(byte_array)
}

/// Attempts to fetch the latest Investment Company Series and Class dataset,
/// falling back to previous years if the request fails.
///
/// This function starts from the **current year** and attempts to fetch data.
/// If the request fails (e.g., 404 error), it retries with the previous year,
/// continuing until successful or reaching a reasonable fallback limit.
///
/// # Arguments
/// - `sec_client` - A reference to an instance of `SecClient` used for HTTP requests.
///
/// # Returns
/// Returns `Result<Vec<InvestmentCompany>, Box<dyn Error>>`, where:
/// - `Ok(Vec<InvestmentCompany>)` contains the parsed investment companies.
/// - `Err(Box<dyn Error>)` if all attempts fail.
pub async fn fetch_investment_company_series_and_class_dataset(
    sec_client: &SecClient,
) -> Result<Vec<InvestmentCompany>, Box<dyn Error>> {
    let current_year = Utc::now().year() as usize;
    let mut year = current_year;

    while year >= 2024 {
        match fetch_investment_company_series_and_class_dataset_for_year(year, sec_client).await {
            Ok(data) => return Ok(data),
            Err(_) => {
                // Try previous year if the current year fails
                year -= 1;
            }
        }
    }

    Err("Failed to fetch dataset from any available year.".into())
}

// pub async fn fetch_investment_company_series_and_class_dataset(
//     sec_client: &SecClient,
//     year: usize,
// ) -> Result<Vec<u8>, Box<dyn Error>> {
//     let url = format!(
//         "https://www.sec.gov/files/investment/data/other/investment-company-series-and-class-information/investment-company-series-class-{}.csv",
//         year
//     );

//     let response = sec_client
//         .raw_request(reqwest::Method::GET, &url, None)
//         .await?;
//     let byte_array = response.bytes().await?;

//     // TODO: Move to `parsers`

//     let cursor = Cursor::new(&byte_array);
//     let mut reader = ReaderBuilder::new().from_reader(cursor);

//     // Extract headers first
//     let headers = reader.headers()?.clone();

//     // let ticker_index = headers
//     //     .iter()
//     //     .position(|h| h == "Class Ticker")
//     //     .ok_or("Column 'Class Ticker' not found")?;
//     // let cik_index = headers
//     //     .iter()
//     //     .position(|h| h == "CIK Number")
//     //     .ok_or("Column 'CIK Number' not found")?;

//     for result in reader.records() {
//         let record = result?;

//         for (col, header) in headers.iter().enumerate() {
//             record.get(col);
//         }

//         // TODO: Remove
//         println!("{:?}", record);

//         // if record.get(ticker_index) == Some(ticker_symbol.as_str()) {
//         //     if let Some(cik_str) = record.get(cik_index) {
//         //         println!("Ticker: {}, CIK: {} (fund)", ticker_symbol, cik_str);

//         //         let cik = Cik::from_str(cik_str)?;
//         //         result_cik = Some(cik);
//         //     }
//         // }
//     }

//     // TODO: Remove
//     Ok(vec![])

//     // Reporting File Number
//     // CIK Number
//     // Entity Name
//     // Entity Org Type
//     // Series ID
//     // Series Name
//     // Class ID
//     // Class Name
//     // Class Ticker
//     // Address_1
//     // Address_2
//     // City
//     // State
//     // Zip Code

// }
