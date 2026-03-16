use crate::models::Cik;
use chrono::NaiveDate;

/// Top-level company profile from the SEC EDGAR submissions endpoint.
///
/// This covers the identifying and classification fields returned alongside the
/// filings list at `https://data.sec.gov/submissions/CIK{cik}.json`. The
/// filings themselves are handled separately by [`fetch_cik_submissions`].
///
/// [`fetch_cik_submissions`]: crate::network::fetch_cik_submissions
#[derive(Debug, Clone)]
pub struct CompanyProfile {
    pub cik: Cik,
    /// Display name (e.g. `"Apple Inc."`).
    pub name: String,
    /// SEC entity type (e.g. `"operating"`, `"investment-company"`).
    pub entity_type: Option<String>,
    /// 4-digit SIC code (e.g. `"3571"`).
    pub sic: Option<String>,
    /// Human-readable SIC description (e.g. `"Electronic Computers"`).
    pub sic_description: Option<String>,
    /// SEC owner-org sector grouping (e.g. `"06 Technology"`).
    pub owner_org: Option<String>,
    /// Ticker symbols listed for this entity (e.g. `["AAPL"]`).
    pub tickers: Vec<String>,
    /// Exchanges the entity is listed on (e.g. `["Nasdaq"]`).
    pub exchanges: Vec<String>,
    /// EDGAR filer category (e.g. `"Large accelerated filer"`).
    pub category: Option<String>,
    /// State of incorporation as a human-readable label (e.g. `"CA"`, `"Delaware"`,
    /// `"Cayman Islands"`). For foreign filers SEC uses an opaque code in the raw
    /// `stateOfIncorporation` field while `stateOfIncorporationDescription` is the
    /// readable name — this field stores the latter.
    pub state_of_incorporation: Option<String>,
    /// Fiscal year end as MMDD (e.g. `"0926"` = September 26).
    pub fiscal_year_end: Option<String>,
    /// Company website URL.
    pub website: Option<String>,
    /// Investor-relations website URL.
    pub investor_website: Option<String>,
    /// Phone number as reported.
    pub phone: Option<String>,
    /// Free-text business description, if provided by the filer. Often empty
    /// for large established companies; more commonly populated for smaller or
    /// newly registered entities.
    pub description: Option<String>,
}

impl CompanyProfile {
    /// Returns the sector label derived from [`owner_org`](CompanyProfile::owner_org)
    /// by stripping the leading numeric prefix the SEC prepends.
    ///
    /// `"06 Technology"` → `Some("Technology")`,
    /// `None` → `None`.
    pub fn sector(&self) -> Option<&str> {
        self.owner_org.as_deref().map(|s| {
            match s.find(' ') {
                Some(pos) => s[pos + 1..].trim(),
                None => s,
            }
        })
    }

    /// Parses [`fiscal_year_end`](CompanyProfile::fiscal_year_end) (MMDD format,
    /// e.g. `"0926"`) into a [`chrono::NaiveDate`].
    ///
    /// The year component is set to 2000 as a stable placeholder — callers
    /// should use the date only for month/day formatting:
    ///
    /// ```
    /// # use sec_fetcher::models::CompanyProfile;
    /// # use sec_fetcher::models::Cik;
    /// let profile = CompanyProfile {
    ///     cik: Cik::new(320193),
    ///     name: "Apple Inc.".into(),
    ///     entity_type: None, sic: None, sic_description: None, owner_org: None,
    ///     tickers: vec![], exchanges: vec![], category: None,
    ///     state_of_incorporation: None,
    ///     fiscal_year_end: Some("0926".into()),
    ///     website: None, investor_website: None, phone: None, description: None,
    /// };
    /// let d = profile.fiscal_year_end_date().unwrap();
    /// assert_eq!(format!("{} {}", d.format("%b"), d.day()), "Sep 26");
    /// ```
    pub fn fiscal_year_end_date(&self) -> Option<NaiveDate> {
        let s = self.fiscal_year_end.as_deref()?;
        if s.len() != 4 {
            return None;
        }
        let month: u32 = s[0..2].parse().ok()?;
        let day: u32 = s[2..4].parse().ok()?;
        NaiveDate::from_ymd_opt(2000, month, day)
    }
}
