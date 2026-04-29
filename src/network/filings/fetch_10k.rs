use crate::enums::FormType;
use crate::models::{Cik, CikSubmission};
use crate::network::{SecClient, fetch_all_entity_submissions};
use std::error::Error;

/// Fetches all 10-K and 10-K405 annual report filings for a given CIK,
/// ordered newest-first.
///
/// # What is a 10-K?
///
/// A **10-K** is the comprehensive **annual report** every public company must
/// file with the SEC after the close of its fiscal year.  It is the single
/// most information-dense document a company publishes:
///
/// | Section | Contents |
/// |---------|----------|
/// | Item 1  | Business description — products, markets, competition |
/// | Item 1A | Risk factors |
/// | Item 2  | Properties |
/// | Item 7  | MD&A — management discussion & analysis |
/// | Item 8  | Audited financial statements (income, balance sheet, cash flow) |
/// | Item 9A | Internal controls assessment |
///
/// The financial statements in a 10-K are **audited** by an independent
/// accounting firm, making them the most reliable periodic disclosures.
/// Large accelerated filers must file within 60 days of fiscal year end;
/// accelerated filers have 75 days; non-accelerated filers have 90 days.
///
/// Use [`fetch_company_description`] to extract the Item 1 business section as
/// plain text.  Use [`fetch_us_gaap_fundamentals`] to query the XBRL-tagged
/// financial figures across all annual and quarterly periods.
///
/// # Form 10-K405
///
/// 10-K405 is an older variant of the 10-K used before the SEC retired it in
/// 2002; both form types represent the same annual report and are returned
/// together.  Use [`CikSubmission::most_recent_10k`] if you only need the
/// latest filing.
///
/// [`fetch_company_description`]: crate::network::fetch_company_description()
/// [`fetch_us_gaap_fundamentals`]: crate::network::fetch_us_gaap_fundamentals()
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_10k_filings, fetch_cik_by_ticker_symbol, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
/// let filings = fetch_10k_filings(&client, cik).await?;
/// for f in &filings {
///     println!("{:?}  {}", f.filing_date, f.as_primary_document_url());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_10k_filings(
    client: &SecClient,
    cik: Cik,
) -> Result<Vec<CikSubmission>, Box<dyn Error>> {
    let submissions = fetch_all_entity_submissions(client, cik).await?;
    Ok(collect_10k_filings(&submissions))
}

/// Merges 10-K and 10-K405 submissions from a combined list, sorts
/// newest-first, and deduplicates by accession number.
///
/// Extracted as a pure function so it can be tested offline against fixture
/// data without a live network connection.
pub fn collect_10k_filings(submissions: &[CikSubmission]) -> Vec<CikSubmission> {
    // Collect all four form types: original and amended variants of both the
    // current (10-K) and pre-2002 legacy (10-K405) form.  Each amendment has
    // its own accession number and may supersede only a subset of the original
    // filing's disclosures; both documents are preserved as separate rows so
    // callers can choose how to handle them.
    const ANNUAL_REPORT_FORMS: &[FormType] = &[
        FormType::TenK,
        FormType::TenKA,
        FormType::TenK405,
        FormType::TenK405A,
    ];
    let mut results: Vec<CikSubmission> = ANNUAL_REPORT_FORMS
        .iter()
        .flat_map(|ft| {
            CikSubmission::by_form(submissions, ft.as_edgar_str())
                .into_iter()
                .cloned()
        })
        .collect();
    // Re-sort newest-first by filing_date (submissions list is already ordered
    // newest-first per form, but mixing the two types may interleave them).
    results.sort_by_key(|b| std::cmp::Reverse(b.filing_date));
    // Deduplicate by accession number: EDGAR can tag the same submission under
    // both "10-K" and "10-K405" form types, and co-registrant subsidiaries
    // list the same accession numbers as their parent.  Without this, each
    // accession can appear once per co-registrant CIK in the merged list.
    let mut seen_acc = std::collections::HashSet::new();
    results.retain(|s| seen_acc.insert(s.accession_number.to_string()));
    results
}
