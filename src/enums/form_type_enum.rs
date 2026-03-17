use std::fmt;
use std::str::FromStr;
use strum_macros::{EnumIter, EnumProperty};

/// An EDGAR form type such as `"8-K"`, `"10-K"`, or `"SC 13G"`.
///
/// Named variants cover the form types this library explicitly handles.
/// Any other form type string is represented by [`FormType::Other`], which
/// preserves the original EDGAR string verbatim.
///
/// # EDGAR as the authoritative source
///
/// The SEC publishes a quarterly full-index at:
///
/// ```text
/// https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{n}/master.idx
/// ```
///
/// Every filing accepted by EDGAR during that quarter appears in that file —
/// the third pipe-delimited column is the form type.  This is the most
/// comprehensive machine-readable list of form types actually in use.
///
/// Fetch it with [`crate::network::fetch_edgar_master_index`].  To verify that
/// every named variant appears in real EDGAR data and all high-frequency form
/// types have named variants, run:
///
/// ```sh
/// cargo run --bin check_form_type_coverage
/// ```
///
/// Iterate all named variants with [`strum::IntoEnumIterator`]:
///
/// ```rust
/// use sec_fetcher::enums::FormType;
/// use strum::IntoEnumIterator;
/// assert!(FormType::iter().count() > 30);
/// ```
///
/// # Parsing
///
/// `FromStr` is case-insensitive and **always succeeds**: unknown form type
/// strings become [`FormType::Other`].
///
/// ```rust
/// use sec_fetcher::enums::FormType;
///
/// let ft: FormType = "10-k".parse().unwrap();
/// assert_eq!(ft, FormType::TenK);
///
/// let ft: FormType = "WEIRD-NEW-FORM".parse().unwrap();
/// assert_eq!(ft, FormType::Other("WEIRD-NEW-FORM".to_string()));
/// ```
///
/// # Display / EDGAR string
///
/// `Display` (and [`FormType::as_edgar_str`]) returns the canonical EDGAR
/// capitalisation, e.g. `"10-K"` not `"10-k"`.
///
/// ```rust
/// use sec_fetcher::enums::FormType;
/// assert_eq!(FormType::EightK.to_string(), "8-K");
/// assert_eq!(FormType::Sc13G.to_string(), "SC 13G");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, EnumIter, EnumProperty)]
pub enum FormType {
    // ── Periodic reports ─────────────────────────────────────────────────────

    /// Annual report — `"10-K"`.
    #[strum(props(edgar = "10-K"))]
    TenK,
    /// Amendment to an annual report — `"10-K/A"`.
    #[strum(props(edgar = "10-K/A"))]
    TenKA,
    /// Historical annual report form used before May 2003 — `"10-K405"`.
    ///
    /// Functionally identical to 10-K; the `405` suffix indicated whether the
    /// filer met the requirements of Rule 405.  Treat the same as `TenK` in
    /// most analysis.
    #[strum(props(edgar = "10-K405", retired = "true"))]
    TenK405,

    /// Quarterly report — `"10-Q"`.
    #[strum(props(edgar = "10-Q"))]
    TenQ,
    /// Amendment to a quarterly report — `"10-Q/A"`.
    #[strum(props(edgar = "10-Q/A"))]
    TenQA,

    // ── Current reports ───────────────────────────────────────────────────────

    /// Current report (material event) — `"8-K"`.
    #[strum(props(edgar = "8-K"))]
    EightK,
    /// Amendment to a current report — `"8-K/A"`.
    #[strum(props(edgar = "8-K/A"))]
    EightKA,

    // ── Proxy statements ──────────────────────────────────────────────────────

    /// Definitive proxy statement — `"DEF 14A"`.
    #[strum(props(edgar = "DEF 14A"))]
    Def14A,
    /// Amendment to a definitive proxy statement — `"DEF 14A/A"`.
    ///
    /// Retired: EDGAR no longer accepts new filings of this form type.
    #[strum(props(edgar = "DEF 14A/A", retired = "true"))]
    Def14AA,
    /// Definitive additional proxy soliciting materials — `"DEFA14A"`.
    #[strum(props(edgar = "DEFA14A"))]
    Defa14A,

    // ── Registration statements ───────────────────────────────────────────────

    /// Registration statement, Form S-1 (full, often an IPO) — `"S-1"`.
    #[strum(props(edgar = "S-1"))]
    S1,
    /// Amendment to Form S-1 — `"S-1/A"`.
    #[strum(props(edgar = "S-1/A"))]
    S1A,

    /// Registration statement, Form S-3 (shelf, established issuers) — `"S-3"`.
    #[strum(props(edgar = "S-3"))]
    S3,
    /// Amendment to Form S-3 — `"S-3/A"`.
    #[strum(props(edgar = "S-3/A"))]
    S3A,

    /// Registration statement, Form S-2 — `"S-2"`.
    ///
    /// Retired: superseded by Form S-3 for established issuers.
    #[strum(props(edgar = "S-2", retired = "true"))]
    S2,
    /// Amendment to Form S-2 — `"S-2/A"`.
    #[strum(props(edgar = "S-2/A", retired = "true"))]
    S2A,

    /// Registration statement, Form S-4 (mergers and acquisitions) — `"S-4"`.
    #[strum(props(edgar = "S-4"))]
    S4,
    /// Amendment to Form S-4 — `"S-4/A"`.
    #[strum(props(edgar = "S-4/A"))]
    S4A,

    /// Registration statement for securities offered pursuant to employee
    /// benefit plans — `"S-8"`.
    ///
    /// The most common registration statement form by filing count.  Major
    /// employers file S-8s to register shares for stock option plans, RSUs,
    /// and employee stock purchase plans (ESPPs).  The primary document is
    /// short; the substantive content (plan details) is in the exhibits.
    #[strum(props(edgar = "S-8"))]
    S8,
    /// Amendment to Form S-8 — `"S-8/A"`.
    ///
    /// Retired: amendments are now filed as post-effective amendments (POS AM).
    #[strum(props(edgar = "S-8/A", retired = "true"))]
    S8A,

    // ── Insider ownership ─────────────────────────────────────────────────────

    /// Initial statement of beneficial ownership (> 10% insiders) — `"3"`.
    #[strum(props(edgar = "3"))]
    Form3,
    /// Statement of changes in beneficial ownership (insider transaction) — `"4"`.
    #[strum(props(edgar = "4"))]
    Form4,
    /// Amendment to Form 4 — `"4/A"`.
    #[strum(props(edgar = "4/A"))]
    Form4A,
    /// Amendment to Form 3 — `"3/A"`.
    #[strum(props(edgar = "3/A"))]
    Form3A,
    /// Annual statement of beneficial ownership (> 10% insiders) — `"5"`.
    #[strum(props(edgar = "5"))]
    Form5,

    // ── Beneficial ownership — Schedule 13 ───────────────────────────────────

    /// Activist beneficial ownership > 5% — `"SCHEDULE 13D"`.
    ///
    /// Filed by the **investor** (not the target company) when crossing the
    /// 5% threshold with intent to influence management.  See also
    /// [`Sc13G`][FormType::Sc13G].
    ///
    /// Note: EDGAR previously used `"SC 13D"`; the current canonical string
    /// is `"SCHEDULE 13D"`.
    #[strum(props(edgar = "SCHEDULE 13D"))]
    Sc13D,
    /// Amendment to Schedule 13D — `"SCHEDULE 13D/A"`.
    #[strum(props(edgar = "SCHEDULE 13D/A"))]
    Sc13DA,

    /// Passive beneficial ownership > 5% — `"SCHEDULE 13G"`.
    ///
    /// Filed by institutional investors (index funds, pension funds) who cross
    /// 5% ownership without intent to influence the issuer.  A 13G filer who
    /// becomes active must convert to a 13D within 10 days.
    ///
    /// Note: EDGAR previously used `"SC 13G"`; the current canonical string
    /// is `"SCHEDULE 13G"`.
    #[strum(props(edgar = "SCHEDULE 13G"))]
    Sc13G,
    /// Amendment to Schedule 13G — `"SCHEDULE 13G/A"`.
    #[strum(props(edgar = "SCHEDULE 13G/A"))]
    Sc13GA,

    // ── Institutional / fund forms ────────────────────────────────────────────

    /// Institutional investment manager quarterly holdings report — `"13F-HR"`.
    ///
    /// Required from managers with > $100 million AUM.  Lists every long
    /// position in U.S.-listed equities as of quarter-end.
    #[strum(props(edgar = "13F-HR"))]
    F13fHr,
    /// Amendment to 13F-HR — `"13F-HR/A"`.
    #[strum(props(edgar = "13F-HR/A"))]
    F13fHrA,

    /// Monthly portfolio holdings report (registered investment companies) — `"NPORT-P"`.
    ///
    /// Filed by mutual funds and ETFs; includes full position-level detail
    /// and fair value levels.
    #[strum(props(edgar = "NPORT-P"))]
    NportP,
    /// Amendment to NPORT-P — `"NPORT-P/A"`.
    #[strum(props(edgar = "NPORT-P/A"))]
    NportPA,

    /// Investment company prospectus filing — `"497"`.
    ///
    /// Rule 497 materials — definitive copies of fund prospectuses and
    /// statements of additional information.  One of the highest-volume form
    /// types on EDGAR by filing count; mutual funds and ETFs file these
    /// continuously throughout the year.
    #[strum(props(edgar = "497"))]
    FundMaterial497,
    /// ETF / fund summary prospectus — `"497K"`.
    ///
    /// Summary prospectus for exchange-traded funds and other investment
    /// companies, filed under Rule 498.  Required for ever ETF share class;
    /// filing volume tracks ETF launches and annual updates.
    #[strum(props(edgar = "497K"))]
    FundMaterial497K,
    /// Post-effective amendment to an investment company registration
    /// statement — `"485BPOS"`.
    ///
    /// Filed by mutual funds and ETFs to update their registration statements
    /// (prospectuses) that have already gone effective.  The most common
    /// amendment form in the fund universe.
    #[strum(props(edgar = "485BPOS"))]
    FundAmendment485BPos,

    // ── Prospectuses ──────────────────────────────────────────────────────────

    /// Rule 424(b)(4) prospectus — `"424B4"`.
    ///
    /// Filed after pricing a firm-commitment underwritten public offering;
    /// contains final deal terms (price, size, underwriter economics).
    #[strum(props(edgar = "424B4"))]
    Prospectus424B4,
    /// Rule 424(b)(2) prospectus supplement — `"424B2"`.
    ///
    /// Pricing supplement for medium-term note programs and structured
    /// products filed off a shelf registration statement.
    #[strum(props(edgar = "424B2"))]
    Prospectus424B2,
    /// Rule 424(b)(3) prospectus — `"424B3"`.
    ///
    /// Used for secondary resale offerings, rights offerings, and prospectus
    /// supplements that do not qualify under other 424 sub-rules.
    #[strum(props(edgar = "424B3"))]
    Prospectus424B3,
    /// Rule 424(b)(5) prospectus supplement — `"424B5"`.
    ///
    /// Filed for take-downs from a shelf registration statement, including
    /// at-the-market (ATM) equity programs.  One of the highest-volume
    /// prospectus types in recent years.
    #[strum(props(edgar = "424B5"))]
    Prospectus424B5,

    // ── Other high-frequency form types ──────────────────────────────────────

    /// Notice of proposed sale of restricted securities — `"144"`.
    ///
    /// Filed by insiders and affiliates before selling restricted or control
    /// securities under Rule 144.  One of the highest-volume form types on
    /// EDGAR; most are filed electronically and do not generate substantive
    /// documents of interest to fundamental analysts.
    #[strum(props(edgar = "144"))]
    Form144,

    /// Notice of exempt offering of securities (Regulation D) — `"D"`.
    ///
    /// Filed by issuers raising capital in private placements under
    /// Regulation D (e.g. Rule 506(b), 506(c)).  Includes hedge funds, VC
    /// rounds, real-estate syndicates, and other private offerings.
    #[strum(props(edgar = "D"))]
    FormD,

    /// Amendment to a Form D — `"D/A"`.
    #[strum(props(edgar = "D/A"))]
    FormDA,

    /// Foreign private issuer current report — `"6-K"`.
    ///
    /// The non-US equivalent of an 8-K; filed by foreign private issuers to
    /// furnish material information disclosed in their home jurisdiction.
    #[strum(props(edgar = "6-K"))]
    SixK,

    /// Free writing prospectus — `"FWP"`.
    ///
    /// A prospectus that does not meet the formal requirements of a Rule 424
    /// filing but is used to condition the market before or during a
    /// registered offering.
    #[strum(props(edgar = "FWP"))]
    Fwp,

    /// ABS issuer periodic distribution report — `"10-D"`.
    ///
    /// Monthly or quarterly report filed by asset-backed securities (ABS)
    /// issuers on distribution activity, collateral performance, and
    /// servicer compliance.
    #[strum(props(edgar = "10-D"))]
    TenD,

    /// Investment company application or report amendment — `"40-APP/A"`.
    ///
    /// Amendment to an application filed under the Investment Company Act
    /// of 1940 (e.g. exemptive relief, no-action requests).
    #[strum(props(edgar = "40-APP/A"))]
    FortyAppA,

    /// Asset-backed securities exhibit file — `"ABS-EE"`.
    ///
    /// Contains the underlying asset data (loan-level) for ABS issuances
    /// in machine-readable XML format, required since 2016 under Regulation AB II.
    #[strum(props(edgar = "ABS-EE"))]
    AbsEe,

    // ── Catch-all ─────────────────────────────────────────────────────────────

    /// Any EDGAR form type not listed above.
    ///
    /// The inner `String` is the original value received from EDGAR,
    /// preserved verbatim (original case, no trimming beyond leading/trailing
    /// whitespace).
    ///
    /// This variant is excluded from [`FormType::iter()`] — only named
    /// variants are yielded by the iterator.
    #[strum(disabled)]
    Other(String),
}

impl FormType {
    /// Returns the canonical EDGAR form type string for this variant.
    ///
    /// This is identical to `&self.to_string()` for named variants, but
    /// avoids an allocation by borrowing from a static table.
    ///
    /// For `FormType::Other(s)`, borrows from the inner `String`.
    ///
    /// ```rust
    /// use sec_fetcher::enums::FormType;
    /// assert_eq!(FormType::EightK.as_edgar_str(), "8-K");
    /// assert_eq!(FormType::Other("FOO".to_string()).as_edgar_str(), "FOO");
    /// ```
    pub fn as_edgar_str(&self) -> &str {
        use strum::EnumProperty;
        match self {
            FormType::Other(s) => s.as_str(),
            _ => self
                .get_str("edgar")
                .expect("all named FormType variants have an edgar prop"),
        }
    }

    /// Returns `true` if this variant represents an EDGAR form type that has
    /// been retired (no longer accepted for new filings).
    ///
    /// Retired variants are preserved for historical data parsing but are
    /// excluded from the forward coverage check run by `check_form_type_coverage`.
    ///
    /// ```rust
    /// use sec_fetcher::enums::FormType;
    /// assert!(FormType::TenK405.is_retired());
    /// assert!(!FormType::TenK.is_retired());
    /// ```
    pub fn is_retired(&self) -> bool {
        use strum::EnumProperty;
        self.get_str("retired").is_some()
    }

}

// ── Display ───────────────────────────────────────────────────────────────────

impl fmt::Display for FormType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_edgar_str())
    }
}

// ── FromStr (case-insensitive, always succeeds) ───────────────────────────────

impl FromStr for FormType {
    /// Parsing is **infallible** — unknown strings become [`FormType::Other`].
    ///
    /// Matching is case-insensitive and delegates to [`FormType::as_edgar_str`],
    /// which is the single source of truth for EDGAR string representations.
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use strum::IntoEnumIterator;
        let trimmed = s.trim();
        Ok(
            FormType::iter()
                .find(|ft| ft.as_edgar_str().eq_ignore_ascii_case(trimmed))
                .unwrap_or_else(|| FormType::Other(trimmed.to_string())),
        )
    }
}

// ── AsRef<str> ────────────────────────────────────────────────────────────────

impl AsRef<str> for FormType {
    fn as_ref(&self) -> &str {
        self.as_edgar_str()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    #[test]
    fn all_named_variants_round_trip() {
        for ft in FormType::iter() {
            let edgar_str = ft.to_string();
            let parsed: FormType = edgar_str.parse().unwrap();
            assert_eq!(
                parsed, ft,
                "round-trip failed: \"{}\" did not parse back to {:?}",
                edgar_str, ft
            );
        }
    }

    #[test]
    fn parsing_is_case_insensitive() {
        assert_eq!("10-k".parse::<FormType>().unwrap(), FormType::TenK);
        assert_eq!("8-k".parse::<FormType>().unwrap(), FormType::EightK);
        assert_eq!("schedule 13g".parse::<FormType>().unwrap(), FormType::Sc13G);
    }

    #[test]
    fn whitespace_trimmed_on_parse() {
        assert_eq!("  10-K  ".parse::<FormType>().unwrap(), FormType::TenK);
    }

    #[test]
    fn unknown_form_type_becomes_other() {
        let ft: FormType = "WEIRD-FORM-XYZ".parse().unwrap();
        assert_eq!(ft, FormType::Other("WEIRD-FORM-XYZ".to_string()));
    }

    #[test]
    fn display_returns_edgar_capitalisation() {
        assert_eq!(FormType::EightK.to_string(), "8-K");
        assert_eq!(FormType::Sc13G.to_string(), "SCHEDULE 13G");
        assert_eq!(FormType::Def14A.to_string(), "DEF 14A");
        assert_eq!(FormType::Form4.to_string(), "4");
        assert_eq!(
            FormType::Other("FOO-BAR".to_string()).to_string(),
            "FOO-BAR"
        );
    }

    #[test]
    fn as_ref_matches_display() {
        for ft in FormType::iter() {
            assert_eq!(ft.as_ref(), ft.to_string().as_str());
        }
    }
}
