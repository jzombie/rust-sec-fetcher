use crate::models::InvestmentCompany;
use crate::network::SecClient;
use crate::parsers::parse_investment_companies_csv;
use std::error::Error;

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
/// - `Ok(Vec<InvestmentCompany>)` contains a list of parsed investment companies.
/// - `Err(Box<dyn Error>)` if the request fails.
///
/// # Reference
/// - SEC Dataset: [Investment Company Series and Class Information](https://www.sec.gov/about/opendatasets)
pub async fn fetch_investment_company_series_and_class_dataset(
    sec_client: &SecClient,
    year: usize,
) -> Result<Vec<InvestmentCompany>, Box<dyn Error>> {
    let url = format!(
        "https://www.sec.gov/files/investment/data/other/investment-company-series-and-class-information/investment-company-series-class-{}.csv",
        year
    );

    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None)
        .await?;
    let byte_array = response.bytes().await?;

    parse_investment_companies_csv(byte_array)
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
