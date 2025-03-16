use crate::models::{Cik, InvestmentCompany};
use std::error::Error;

// TODO: Move directly to `InvestmentCompany`
// TODO: Document
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
