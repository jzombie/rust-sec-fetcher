use crate::models::Cik;
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize)]
pub struct InvestmentCompany {
    #[serde(rename = "Reporting File Number")]
    pub reporting_file_number: Option<String>,

    // TODO: Rework so that this is a `Cik` object
    #[serde(rename = "CIK Number")]
    pub cik_number: Option<String>,

    #[serde(rename = "Entity Name")]
    pub entity_name: Option<String>,

    #[serde(rename = "Entity Org Type")]
    pub entity_org_type: Option<String>,

    #[serde(rename = "Series ID")]
    pub series_id: Option<String>,

    #[serde(rename = "Series Name")]
    pub series_name: Option<String>,

    #[serde(rename = "Class ID")]
    pub class_id: Option<String>,

    #[serde(rename = "Class Name")]
    pub class_name: Option<String>,

    #[serde(rename = "Class Ticker")]
    pub class_ticker: Option<String>,

    #[serde(rename = "Address_1")]
    pub address_1: Option<String>,

    #[serde(rename = "Address_2")]
    pub address_2: Option<String>,

    #[serde(rename = "City")]
    pub city: Option<String>,

    #[serde(rename = "State")]
    pub state: Option<String>,

    #[serde(rename = "Zip Code")]
    pub zip_code: Option<String>,
}

impl InvestmentCompany {
    pub fn get_fund_cik_by_ticker_symbol(
        investment_companies: &[InvestmentCompany],
        ticker_symbol: &str,
    ) -> Result<Cik, Box<dyn Error>> {
        for result in investment_companies.iter() {
            if result.class_ticker == Some(ticker_symbol.to_string()) {
                if let Some(cik_str) = &result.cik_number {
                    let cik = Cik::from_str(&cik_str)?;

                    return Ok(cik);
                }
            }
        }

        Err("CIK not found".into())
    }
}
