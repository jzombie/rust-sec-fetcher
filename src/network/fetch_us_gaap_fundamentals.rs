use crate::enums::Url;
use crate::models::{AccessionNumber, Cik, Ticker};
use crate::network::{fetch_cik_submissions, SecClient};
use polars::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;

pub type TickerFundamentalsDataFrame = DataFrame;

/// Fetches US-GAAP SEC fundamentals for a given ticker symbol
pub async fn fetch_us_gaap_fundamentals(
    client: &SecClient,
    company_tickers: &[Ticker],
    ticker_symbol: &str,
) -> Result<TickerFundamentalsDataFrame, Box<dyn Error>> {
    // Get the formatted CIK for the ticker
    let cik = Cik::get_company_cik_by_ticker_symbol(company_tickers, ticker_symbol)?;

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
            println!("Warning: could not fetch submissions for {}: {}", ticker_symbol, e);
        }
    }

    Ok(fundamentals_df)
}
