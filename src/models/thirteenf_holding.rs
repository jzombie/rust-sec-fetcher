use rust_decimal::Decimal;

/// One row in a Form 13F-HR informationTable — a single equity position held
/// by an institutional manager at the end of a reporting period.
///
/// The `value_usd` field has been converted from the thousands-of-USD unit
/// used in the raw XML to actual dollars for consistency with `NportInvestment`.
#[derive(Debug, Clone)]
pub struct ThirteenfHolding {
    /// Issuer name as reported (e.g. "APPLE INC").
    pub name: String,
    /// 9-character CUSIP identifying the security.
    pub cusip: String,
    /// Title of the share class (e.g. "COM", "PFD").
    pub title_of_class: String,
    /// Market value in USD (raw XML value × 1 000 — 13F reports in thousands).
    pub value_usd: Decimal,
    /// Number of shares or face value of bonds held.
    pub shares: Decimal,
    /// "SH" (shares) or "PRN" (principal amount for bonds).
    pub shares_type: String,
    /// Only present for options: "Put" or "Call".
    pub put_call: Option<String>,
    /// Investment discretion: "SOLE", "SHARED", or "OTHER".
    pub investment_discretion: String,
}
