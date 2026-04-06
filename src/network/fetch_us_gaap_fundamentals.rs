use crate::enums::Url;
use crate::models::{AccessionNumber, Cik, Ticker, TickerSymbol};
use crate::network::{SecClient, fetch_cik_submissions};
use polars::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;

pub type TickerFundamentalsDataFrame = DataFrame;

/// Fetches all US-GAAP XBRL-tagged financial data for a company as a
/// structured [`DataFrame`].
///
/// # What this fetches
///
/// The SEC "company facts" endpoint
/// (`https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`) returns the
/// complete set of **XBRL-tagged values** the company has disclosed in its
/// periodic filings (10-K, 10-Q, and related amendments).  XBRL tagging is
/// mandatory for all public companies, so this endpoint covers the full
/// time-series financial history available on EDGAR.
///
/// Event-driven filings (8-K, 8-K/A) are excluded by the parser: they do not
/// carry a canonical fiscal period and are not part of the periodic financial
/// time-series.
///
/// This is **structured financial data** — the numbers extracted from
/// financial statements.  It is entirely separate from the filings/submissions
/// API used by `fetch_*_filings`.  You do not need to fetch filing documents
/// to get financial figures; this endpoint provides them directly.
///
/// # Returned DataFrame columns
///
/// | Column       | Type     | Description |
/// |--------------|----------|-------------|
/// | `concept`    | `Utf8`   | US-GAAP concept name (e.g. `"Revenues"`, `"NetIncomeLoss"`) |
/// | `label`      | `Utf8`   | Human-readable label from the XBRL taxonomy |
/// | `unit`       | `Utf8`   | Unit of measure (`"USD"`, `"shares"`, `"USD/shares"`) |
/// | `start`      | `Date`   | Period start date (absent for instant-type concepts) |
/// | `end`        | `Date`   | Period end date (balance-sheet date for instants) |
/// | `val`        | `Float64` | Reported value |
/// | `form`       | `Utf8`   | SEC form type that carried this tag (`"10-K"`, `"10-Q"`, etc.) |
/// | `accn`       | `Utf8`   | Accession number of the source filing |
/// | `filing_url` | `Utf8`   | Direct URL to the filing primary document |
///
/// # Common concepts
///
/// - `Revenues` / `RevenueFromContractWithCustomerExcludingAssessedTax` — top-line revenue
/// - `NetIncomeLoss` — bottom-line net income
/// - `EarningsPerShareBasic` / `EarningsPerShareDiluted`
/// - `Assets` / `Liabilities` / `StockholdersEquity`
/// - `CashAndCashEquivalentsAtCarryingValue`
/// - `OperatingCashFlow` (often tagged as `NetCashProvidedByUsedInOperatingActivities`)
///
/// To explore which concepts a company has reported, filter the returned
/// DataFrame on the `concept` column.  To compare across companies, join on
/// `concept` + normalized `end` date.
pub async fn fetch_us_gaap_fundamentals(
    client: &SecClient,
    company_tickers: &[Ticker],
    ticker: &TickerSymbol,
) -> Result<TickerFundamentalsDataFrame, Box<dyn Error>> {
    // Get the formatted CIK for the ticker
    let cik = Cik::get_company_cik_by_ticker_symbol(company_tickers, ticker)?;

    let url = Url::CompanyFacts(cik.clone()).value();

    // TODO: Debug log
    println!("Using URL: {}", url);

    let data: Value = client.fetch_json(&url, None).await?;

    let mut fundamentals_df = crate::parsers::parse_us_gaap_fundamentals(data)?;

    // Fetch submissions to resolve accession numbers -> primary document URLs.
    // The companyfacts API only provides accession numbers; the primary document
    // filename (e.g. "aapl-20241228.htm") comes from the submissions API.
    match fetch_cik_submissions(client, cik.clone()).await {
        Ok(submissions) => {
            // Build map: accn (dashed, e.g. "0000320193-25-000008") -> primary document URL
            let primary_doc_map: HashMap<String, String> = submissions
                .iter()
                .filter(|s| !s.primary_document.is_empty())
                .map(|s| (s.accession_number.to_string(), s.as_primary_document_url()))
                .collect();

            let updated_urls: Vec<Option<String>> = fundamentals_df
                .column("accn")?
                .str()?
                .into_iter()
                .map(|opt| {
                    opt.map(|accn| {
                        primary_doc_map.get(accn).cloned().unwrap_or_else(|| {
                            // Fall back to the filing index page for any accn not in submissions
                            AccessionNumber::from_str(accn)
                                .map(|a| Url::CikAccessionIndex(cik.clone(), a).value())
                                .unwrap_or_default()
                        })
                    })
                })
                .collect();

            fundamentals_df.with_column(Series::new("filing_url".into(), updated_urls))?;
        }
        Err(e) => {
            // Non-fatal: keep the index-page URLs already set by the parser
            println!("Warning: could not fetch submissions for {}: {}", ticker, e);
        }
    }

    Ok(fundamentals_df)
}
