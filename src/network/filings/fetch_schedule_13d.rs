use crate::models::{Cik, CikSubmission};
use crate::network::{SecClient, fetch_all_entity_submissions};
use std::error::Error;

/// Fetches all SC 13D and SC 13D/A beneficial ownership report filings for a
/// given CIK, ordered newest-first.
///
/// # What is a Schedule 13D?
///
/// A **Schedule 13D** (filed as form type `SC 13D` on EDGAR) is a beneficial
/// ownership disclosure required from any person or group that acquires **more
/// than 5%** of a class of a publicly registered equity security *with the
/// intent to influence or control the issuer*.  It must be filed within
/// **10 calendar days** of crossing the 5% threshold.
///
/// Key fields disclosed:
///
/// | Item | Contents |
/// |------|----------|
/// | 2 | Identity and background of the filer and all group members |
/// | 3 | Source and amount of funds used to acquire the position |
/// | 4 | **Purpose of the transaction** — this is the activist heart of a 13D |
/// | 5 | Interest in securities: exact share count and percentage |
/// | 6 | Contracts, arrangements, or understandings related to the securities |
/// | 7 | Exhibit: any agreements between group members |
///
/// Item 4 is where activist investors announce intentions such as seeking
/// board seats, pushing for a sale of the company, or demanding strategic
/// changes.
///
/// # 13D vs 13G
///
/// Schedule 13G is the passive-investor counterpart: filers who cross 5%
/// *without* an intent to influence the issuer may use the shorter 13G form
/// with a 45-day filing window.  Institutional investors (mutual funds,
/// index funds) almost always file 13G; activist hedge funds typically
/// file 13D.  A 13G filer that later becomes active must **convert** to a
/// 13D within 10 days.
///
/// # Important: filed by the investor, not the issuer
///
/// Unlike most EDGAR forms which are filed by the company being reported on,
/// **SC 13D is filed by the beneficial owner** (the investor).  To look up a
/// company's 13D filings, pass the **CIK of the beneficial owner** — not the
/// target company's CIK.
///
/// To find all 13D filings against a specific target company, use the EDGAR
/// full-text search endpoint with a filter on the company name or CIK as the
/// subject of Item 5.
///
/// # Amendments (SC 13D/A)
///
/// Amendments must be filed whenever any material fact in Item 2–6 changes,
/// and within **2 business days** if the aggregate ownership crosses another
/// 1% threshold.  This function returns both the initial SC 13D and all
/// SC 13D/A amendments together, sorted newest-first.
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_schedule_13d_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// // Pass the CIK of the beneficial owner (activist investor), not the target company.
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("BRK-B")).await?;
/// let filings = fetch_schedule_13d_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{:?}  {}  {}", f.filing_date, f.form, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_schedule_13d_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_all_entity_submissions(client, cik).await?;
    let mut results: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "SC 13D")
        .into_iter()
        .cloned()
        .collect();
    let mut amendments: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "SC 13D/A")
        .into_iter()
        .cloned()
        .collect();
    results.append(&mut amendments);
    results.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
    Ok(results)
}
