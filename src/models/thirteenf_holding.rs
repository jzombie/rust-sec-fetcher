use rust_decimal::Decimal;

/// One row in a Form 13F-HR informationTable — a single equity position held
/// by an institutional manager at the end of a reporting period.
///
/// The `value_usd` field contains the position's fair market value in **actual
/// US dollars**, stored verbatim from the `<value>` XML element (the modern
/// EDGAR 13F-HR XML schema uses actual dollars, not thousands).
#[derive(Debug, Clone)]
pub struct ThirteenfHolding {
    /// Issuer name as reported (e.g. "APPLE INC").
    pub name: String,
    /// 9-character CUSIP identifying the security.
    pub cusip: String,
    /// Title of the share class (e.g. "COM", "PFD").
    pub title_of_class: String,
    /// Market value in USD, stored verbatim from the `<value>` XML element.
    pub value_usd: Decimal,
    /// Number of shares or face value of bonds held.
    pub shares: Decimal,
    /// "SH" (shares) or "PRN" (principal amount for bonds).
    pub shares_type: String,
    /// Only present for options: "Put" or "Call".
    pub put_call: Option<String>,
    /// Investment discretion: "SOLE", "SHARED", or "OTHER".
    pub investment_discretion: String,
    /// Portfolio weight computed by [`crate::parsers::parse_13f_xml`] as
    /// `value_usd / total_value_usd × 100`.  **0–100 percentage scale**
    /// (e.g. `7.7546` means 7.7546%).  Set to `0` if the total is zero.
    pub weight_pct: Decimal,
}
