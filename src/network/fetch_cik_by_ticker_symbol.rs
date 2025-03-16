use std::error::Error;

use crate::network::SecClient;
use crate::network::fetch_company_tickers;
use crate::network::fetch_investment_company_series_and_class_dataset;

use crate::accessors::get_company_cik_by_ticker_symbol;
use crate::accessors::get_fund_cik_by_ticker_symbol;

use crate::models::Cik;

pub async fn fetch_cik_by_ticker_symbol(
    sec_client: &SecClient,
    ticker_symbol: &str
) -> Result<Cik, Box<dyn Error>> {
    // First, look at companies
    let tickers_df = fetch_company_tickers(&sec_client).await?;
    if let Ok(company_cik) = get_company_cik_by_ticker_symbol(&tickers_df, ticker_symbol) {
        return Ok(company_cik);
    }

    // TODO: Determine dynammc year
    let year = 2024;

    // Then, look at funds
    let investment_companies = fetch_investment_company_series_and_class_dataset(&sec_client, year).await?;
    let fund_cik = get_fund_cik_by_ticker_symbol(&investment_companies, ticker_symbol)?;

    Ok(fund_cik)
}
