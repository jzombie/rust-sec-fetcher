use std::error::Error;

use crate::network::fetch_investment_company_series_and_class_dataset;
use crate::network::fetch_operating_company_tickers;
use crate::network::SecClient;

use crate::models::InvestmentCompany;

use crate::models::Cik;

/// Resolves a ticker symbol to its SEC CIK (Central Index Key).
///
/// # What is a CIK?
///
/// A **CIK** (Central Index Key) is the permanent numeric identifier the SEC
/// assigns to every registrant in EDGAR — companies, funds, individuals, and
/// foreign private issuers.  All EDGAR endpoints are keyed by CIK; you
/// typically start any data pipeline by looking up the CIK for the entity you
/// care about.
///
/// CIKs are 10-digit zero-padded numbers (e.g. `0000320193` for Apple Inc.).
/// They never change, even through corporate name changes, mergers, or
/// exchange transfers.
///
/// # Resolution strategy
///
/// This function searches two separate SEC datasets in priority order:
///
/// 1. **Operating-company tickers** (`/files/company_tickers.json`) — covers
///    exchange-listed stocks and most ETPs registered as operating companies.
/// 2. **Investment-company series/class dataset** — covers mutual fund share
///    classes, ETF tickers registered under the Investment Company Act, and
///    closed-end fund tickers that are not in the first file.
///
/// Returns `Err` if the ticker is not found in either source.  Ticker lookup
/// is case-insensitive.
///
/// # Example
///
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "AAPL").await?;
/// println!("{}", cik.to_string());  // "0000320193"
/// # Ok(())
/// # }
/// ```
pub async fn fetch_cik_by_ticker_symbol(
    sec_client: &SecClient,
    ticker_symbol: &str,
) -> Result<Cik, Box<dyn Error>> {
    // First, look at companies
    // include_derived_instruments=true so that warrant, unit, and preferred
    // symbols (-WT, -UN, -PA…) are searchable. get_company_cik_by_ticker_symbol
    // resolves any derived instrument to its parent registrant's CIK.
    let company_tickers = fetch_operating_company_tickers(&sec_client, true).await?;
    if let Ok(company_cik) = Cik::get_company_cik_by_ticker_symbol(&company_tickers, ticker_symbol)
    {
        return Ok(company_cik);
    }

    // Then, look at funds
    let investment_companies =
        fetch_investment_company_series_and_class_dataset(&sec_client).await?;
    let fund_cik =
        InvestmentCompany::get_fund_cik_by_ticker_symbol(&investment_companies, ticker_symbol)?;

    Ok(fund_cik)
}
