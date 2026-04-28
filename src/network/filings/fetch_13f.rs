use crate::enums::Url;
use crate::models::{Cik, CikSubmission, ThirteenfHolding};
use crate::network::filings::fetch_filing_index;
use crate::network::{SecClient, fetch_all_entity_submissions};
use crate::parsers::parse_13f_xml;
use std::error::Error;

/// Fetches all 13F-HR filings for a given CIK, ordered newest-first.
///
/// # What is a 13F-HR?
///
/// A **13F-HR** ("Holding Report") is the **quarterly institutional holdings
/// disclosure** required from every investment manager whose Section 13(f)
/// securities portfolio exceeds **$100 million** in assets under management.
/// Covered securities include exchange-listed equities, certain ETFs, and
/// equity options; bonds, private investments, and short positions are
/// excluded.
///
/// Key facts:
/// - Filed within **45 calendar days** after each calendar quarter end
///   (i.e., by May 15, Aug 14, Nov 14, Feb 14).
/// - Minimum reporting threshold: positions ≥ **$200,000** *or* ≥ **10,000
///   shares** (whichever is smaller).
/// - The form has two parts: a cover page (`primary_doc.xml`) and a separate
///   **information table** XML that lists every position.  Use [`fetch_13f`]
///   to retrieve the parsed holdings table.
/// - Filers include hedge funds, mutual funds, pension funds, banks, and
///   insurance companies — essentially any institutional buyer of US equities.
///
/// Returns every [`CikSubmission`] whose form type is `13F-HR`.  Pass each
/// submission to [`fetch_13f`] to retrieve the parsed holdings.
pub async fn fetch_13f_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_all_entity_submissions(client, cik).await?;
    Ok(CikSubmission::by_form(&submissions, "13F-HR")
        .into_iter()
        .cloned()
        .collect())
}

/// Fetches and parses a 13F-HR filing given its [`CikSubmission`].
///
/// # What this fetches
///
/// A 13F-HR archive always contains at least two documents:
/// 1. The **cover sheet** (`primary_doc.xml`) — filer identity, report period,
///    AUM summary.  This is what the [`CikSubmission`] `primary_document` field
///    points to; it does **not** contain the holdings.
/// 2. The **information table** (variable filename, e.g.
///    `form13fInfoTable.xml`, `infotable.xml`) — the actual per-position rows.
///
/// Because filers use inconsistent filenames for the information table,
/// this function fetches the filing index first to discover the actual
/// filename, then fetches and parses that file.
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_13f, SecClient};
/// # use sec_fetcher::models::{CikSubmission, TickerSymbol};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("BRK-B")).await?;
/// let submissions = fetch_cik_submissions(&client, cik).await?;
/// let latest = CikSubmission::by_form(&submissions, "13F-HR").into_iter().next().unwrap();
/// let holdings = fetch_13f(&client, latest).await?;
/// for h in &holdings {
///     println!("{}: {} shares", h.name, h.shares);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_13f(
    client: &SecClient,
    submission: &CikSubmission,
) -> Result<Vec<ThirteenfHolding>, Box<dyn Error>> {
    // Discover the informationTable filename from the filing index.
    let index = fetch_filing_index(client, submission).await?;

    let info_doc = index
        .documents
        .iter()
        .find(|d| d.document_type.to_uppercase().contains("INFORMATION TABLE"))
        .ok_or_else(|| {
            format!(
                "No INFORMATION TABLE document found in 13F index for {}",
                submission.accession_number
            )
        })?;

    let url = Url::CikAccessionDocument(
        submission.cik.clone(),
        submission.accession_number.clone(),
        info_doc.name.clone(),
    )
    .value();

    let response = client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;
    let xml = response.text().await?;
    parse_13f_xml(&xml, submission.filing_date)
}
