use crate::models::{Cik, CikSubmission};
use crate::network::{fetch_cik_submissions, SecClient};
use std::error::Error;

/// Fetches all SC 13G and SC 13G/A passive beneficial ownership filings for a
/// given CIK, ordered newest-first.
///
/// # What is a Schedule 13G?
///
/// A **Schedule 13G** (`SC 13G` on EDGAR) is the passive-investor counterpart
/// to Schedule 13D.  Any person or group that crosses the **5% ownership
/// threshold** in a registered equity security *without* intending to influence
/// or control the issuer may use the shorter 13G form instead of the activist
/// 13D.
///
/// Institutional investors — mutual funds, index funds, pension funds, and
/// other passive asset managers — almost always file 13G.  A 13G filer who
/// later becomes active must **convert to a 13D within 10 days**.
///
/// # Filing deadlines
///
/// | Filer type | Initial filing deadline |
/// |---|---|
/// | Institutional investors (§ 13(d)(6)(A)) | 45 days after calendar year-end |
/// | Passive investors (§ 13(d)(6)(B)) | 10 days after the month-end in which the 5% threshold was crossed |
/// | All others | 10 days after crossing the 5% threshold |
///
/// Annual amendments are due within 45 days of calendar year-end; immediate
/// amendments are required when ownership changes by ±1% or if the filer can
/// no longer certify passivity (triggering conversion to 13D).
///
/// # What's disclosed
///
/// The 13G is shorter than the 13D and omits Items 3 (source of funds) and 4
/// (purpose of transaction).  The primary purpose is ownership transparency:
///
/// | Item | Contents |
/// |------|-----------| 
/// | 2 | Identity and background of the filer |
/// | 5 | Ownership interest: share count and percentage |
/// | 6 | Certifications (passive intent) |
///
/// # 13G vs 13D
///
/// | | SC 13G | SC 13D |
/// |---|---|---|
/// | Intent | Passive — no influence over management | Active — may seek control |
/// | Ownership threshold | > 5 % | > 5 % |
/// | Deadline | 45 days after year-end (institutional) | 10 calendar days |
/// | Form length | Short — ~2 pages | Long — full narrative |
/// | Typical filers | Index funds, mutual funds | Activist hedge funds |
///
/// # Important: filed by the investor, not the issuer
///
/// Like SC 13D, this form is filed by the beneficial owner, not the target
/// company.  Pass the **CIK of the institutional investor** to look up their
/// holdings disclosures.
///
/// # Amendments (SC 13G/A)
///
/// This function returns both initial `SC 13G` filings and all `SC 13G/A`
/// amendments, merged and sorted newest-first.
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_schedule_13g_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// // Pass the CIK of the institutional investor (beneficial owner).
/// let cik = fetch_cik_by_ticker_symbol(&client, "BLK").await?;
/// let filings = fetch_schedule_13g_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{:?}  {}  {}", f.filing_date, f.form, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_schedule_13g_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(client, cik).await?;
    let mut results: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "SC 13G")
        .into_iter()
        .cloned()
        .collect();
    let mut amendments: Vec<CikSubmission> = CikSubmission::by_form(&submissions, "SC 13G/A")
        .into_iter()
        .cloned()
        .collect();
    results.append(&mut amendments);
    results.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
    Ok(results)
}
