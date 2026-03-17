use crate::enums::Url;
use crate::models::{Cik, CikSubmission, NportInvestment};
use crate::network::{fetch_cik_submissions, fetch_company_tickers, SecClient};
use crate::parsers::parse_nport_xml;
use std::error::Error;

/// Fetches all NPORT-P filings for a given CIK, ordered newest-first.
///
/// # What is an NPORT-P?
///
/// An **NPORT-P** is the **monthly portfolio holdings report** filed by
/// registered open-end investment companies (mutual funds and ETFs) under
/// Rule 30b1-9.  Unlike the 13F, which covers only institutional equity
/// managers, NPORT-P covers the *entire* portfolio of registered funds:
///
/// - Domestic and foreign equities
/// - Fixed income (corporate, government, municipal, ABS)
/// - Derivatives (futures, swaps, options — including notional amounts and
///   counterparty names)
/// - Cash equivalents and repo agreements
/// - Private / illiquid instruments
///
/// Key facts:
/// - Funds with net assets ≥ **$1 billion** file monthly; smaller funds file
///   quarterly on Form NPORT-P.
/// - Reports are due within **30 days** after each month end.
/// - Only the **third month** of each fiscal quarter is publicly available;
///   the other two months are filed but kept non-public until the following
///   quarter.
/// - The filing XML includes both aggregate portfolio statistics (duration,
///   credit quality, liquidity breakdown) and line-item holdings.
///
/// Returns every [`CikSubmission`] whose form type is `NPORT-P`.  Pass each
/// submission to [`fetch_nport`] to retrieve the parsed investment holdings.
pub async fn fetch_nport_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    Ok(CikSubmission::by_form(&submissions, "NPORT-P")
        .into_iter()
        .cloned()
        .collect())
}

/// Fetches and parses an NPORT-P filing given its [`CikSubmission`].
///
/// The primary document in an NPORT-P submission is a single XML file
/// (pointed to by `submission.primary_document`).  This function fetches that
/// XML and parses each `<invstOrSec>` element into an [`NportInvestment`].
/// The ticker-symbol lookup table ([`fetch_company_tickers`]) is used to enrich
/// holdings with exchange-listed ticker symbols where available.
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_nport, SecClient};
/// # use sec_fetcher::models::CikSubmission;
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "VFINX").await?;
/// let submissions = fetch_cik_submissions(&client, cik).await?;
/// let latest = CikSubmission::by_form(&submissions, "NPORT-P").into_iter().next().unwrap();
/// let investments = fetch_nport(&client, latest).await?;
/// for inv in &investments {
///     println!("{:?}", inv);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_nport(
    client: &SecClient,
    submission: &CikSubmission,
) -> Result<Vec<NportInvestment>, Box<dyn Error>> {
    // include_derived_instruments=true so all instrument symbols (warrants,
    // units, preferred classes) are available when resolving portfolio holdings.
    let company_tickers = fetch_company_tickers(client, true).await?;

    let url = Url::CikAccessionPrimaryDocument(
        submission.cik.clone(),
        submission.accession_number.clone(),
    )
    .value();

    let response = client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;
    let xml_data = response.text().await?;

    parse_nport_xml(&xml_data, &company_tickers)
}
