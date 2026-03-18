use rust_decimal::Decimal;

/// One row in a Form 13F-HR informationTable — a single equity position held
/// by an institutional manager at the end of a reporting period.
///
/// All numeric fields are stored in their **normalized form** — the raw XML
/// values have been processed by [`crate::normalize`] before storage.
#[derive(Debug, Clone)]
pub struct ThirteenfHolding {
    /// Issuer name as reported (e.g. "APPLE INC").
    pub name: String,
    /// 9-character CUSIP identifying the security.
    pub cusip: String,
    /// Title of the share class (e.g. "COM", "PFD").
    pub title_of_class: String,
    /// Market value in **actual US dollars**, normalized by
    /// [`crate::normalize::normalize_13f_value_usd`].  Legacy filings
    /// (pre-2023) reported `<value>` in thousands; this field always contains
    /// the fully-converted dollar amount regardless of the filing era.
    pub value_usd: Decimal,
    /// Number of shares or face value of bonds held.
    pub shares: Decimal,
    /// "SH" (shares) or "PRN" (principal amount for bonds).
    pub shares_type: String,
    /// Only present for options: "Put" or "Call".
    pub put_call: Option<String>,
    /// Investment discretion: "SOLE", "SHARED", or "OTHER".
    pub investment_discretion: String,
    /// Portfolio weight on the **canonical 0–100 percentage scale**
    /// (e.g. `7.7546` means 7.7546%).  Computed by
    /// [`crate::normalize::compute_13f_weight_pct`] as
    /// `value_usd / total_value_usd × 100`, rounded to 4 decimal places.
    /// Set to `0` when total portfolio value is zero.
    pub weight_pct: Decimal,
}
