/// A single entry from the SEC EDGAR Standard Industrial Classification (SIC)
/// code list at `https://www.sec.gov/info/edgar/siccodes.htm`.
///
/// The SEC reviews filings by office. Each SIC code is assigned to one EDGAR
/// reviewing office (e.g. `"Office of Technology"`, `"Office of Manufacturing"`)
/// and given an industry title (e.g. `"ELECTRONIC COMPUTERS"`).  The industry
/// title is identical to the `sicDescription` field returned by the submissions
/// endpoint, so this struct's main added value is the `office` field.
///
/// Obtain via [`crate::network::fetch_sic_codes`].
#[derive(Debug, Clone)]
pub struct SicCode {
    /// The 4-digit SEC SIC code (e.g. `3571`).
    pub code: u16,
    /// Human-readable industry title as used by the SEC (e.g.
    /// `"ELECTRONIC COMPUTERS"`).  Matches the `sicDescription` in the
    /// company submissions JSON.
    pub description: String,
    /// The EDGAR reviewing office responsible for filings in this SIC code
    /// (e.g. `"Office of Technology"`, `"Industrial Applications and Services"`).
    pub office: String,
}

impl SicCode {
    /// Returns [`office`](SicCode::office) with any leading `"Office of "`
    /// prefix stripped.
    ///
    /// ```
    /// use sec_fetcher::models::SicCode;
    /// let s = SicCode { code: 3571, description: "ELECTRONIC COMPUTERS".into(), office: "Office of Technology".into() };
    /// assert_eq!(s.office_short(), "Technology");
    /// ```
    pub fn office_short(&self) -> &str {
        self.office
            .strip_prefix("Office of ")
            .unwrap_or(&self.office)
    }
}
