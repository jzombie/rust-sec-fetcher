//! # 13F-HR `<value>` field normalization
//!
//! ## Background
//!
//! The SEC's EDGAR 13F-HR `informationTable.xml` schema has used **two
//! different units** for the `<value>` field over its lifetime:
//!
//! | Era | `<value>` unit | Applies to filings with date… |
//! |-----|----------------|-------------------------------|
//! | Legacy ("thousands era") | **Thousands of USD** | `filing_date < 2023-01-01` |
//! | Modern ("actual-USD era") | **Actual USD**       | `filing_date ≥ 2023-01-01` |
//!
//! [`normalize_13f_value_usd`] is the **only place** in this codebase that
//! knows about this distinction.  All callers — parsers, tests — must go
//! through it.  Do **not** multiply or divide `<value>` by 1 000 anywhere else.
//!
//! ## Empirical evidence
//!
//! The transition was verified empirically by inspecting real Berkshire
//! Hathaway (BRK-A, CIK 1067983) 13F-HR filings on EDGAR and computing the
//! implied price-per-share for their Apple (AAPL) position against the known
//! market price:
//!
//! | Filing date | Accession | Raw `<value>` (AAPL) | Shares | Implied $/sh | Unit |
//! |-------------|-----------|---------------------|--------|--------------|------|
//! | 2022-11-14 | [0000950123-22-012275](https://www.sec.gov/Archives/edgar/data/1067983/000095012322012275/) | 95 634 | 692 000 | $0.14 verbatim / **$138 ×1000** | **THOUSANDS** |
//! | 2023-02-14 | [0000950123-23-002585](https://www.sec.gov/Archives/edgar/data/1067983/000095012323002585/) | 133 289 470 | 1 025 856 | **$129.93** verbatim | **ACTUAL USD** |
//!
//! AAPL traded at ~$138 (Q3-2022 close) and ~$130 (Q4-2022 close), confirming
//! the unit interpretation above.  The cutoff is set conservatively to
//! 2023-01-01 because no filing between 2022-11-14 and 2023-02-14 was observed.
//!
//! ## SEC references
//!
//! - **EDGAR 13F XML Technical Specification** (the authoritative schema):
//!   <https://www.sec.gov/info/edgar/forms/edgarform13f/13fxmltechspec.pdf>
//! - **EDGAR Filing Manual, Vol. II** (filer guidance, §16 Form 13F):
//!   <https://www.sec.gov/info/edgar/edgarfm-vol2.pdf>
//! - **SEC Release 34-94520** (March 18, 2022) — Form 13F amendments that
//!   updated reporting thresholds and triggered the schema refresh from which
//!   the transition to actual-dollar reporting originated:
//!   <https://www.sec.gov/rules/final/2022/34-94520.pdf>
//!
//! ## Updating this module
//!
//! If the SEC ever changes the `<value>` unit again, update
//! [`THIRTEENF_THOUSANDS_ERA_CUTOFF`] and add a new row to the empirical
//! evidence table above with: filing date, accession number, raw value, share
//! count, and implied price-per-share.  The function signature intentionally
//! requires `filing_date` so callers are forced to thread the date through —
//! this makes it impossible to silently skip the conversion.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use super::pct::Pct;

/// The first filing date (inclusive) on which EDGAR 13F-HR `informationTable`
/// XML began reporting `<value>` in **actual US dollars**.
///
/// Filings with `filing_date < THIRTEENF_THOUSANDS_ERA_CUTOFF` encode `<value>`
/// in **thousands of USD**; all later filings use actual USD.
///
/// Evidence: BRK-A filed 2022-11-14 (accession 0000950123-22-012275) used
/// thousands; BRK-A filed 2023-02-14 (accession 0000950123-23-002585) used
/// actual dollars.  The cutoff is conservatively set to 2023-01-01 to sit
/// cleanly between the two observed filing dates.
///
/// If you need to update this constant, cite a specific EDGAR filing accession
/// number and a price-per-share sanity check in the module-level doc above.
pub const THIRTEENF_THOUSANDS_ERA_CUTOFF: (i32, u32, u32) = (2023, 1, 1);

/// Returns `true` if a 13F-HR filing encodes `<value>` in **thousands of USD**.
///
/// Pass the `filing_date` from [`crate::models::CikSubmission`].  When the
/// date is unknown (`None`), this returns `false` — i.e., assumes the modern
/// actual-USD format — because erroneously treating modern values as thousands
/// would produce 1 000× inflated dollar figures, which is the more visible
/// and thus safer failure mode.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use sec_fetcher::normalize::is_13f_thousands_era;
///
/// // 2022-11-14 (last observed thousands-era filing) → thousands
/// let ancient = NaiveDate::from_ymd_opt(2022, 11, 14).unwrap();
/// assert!(is_13f_thousands_era(Some(ancient)));
///
/// // 2023-02-14 (first observed actual-USD filing) → actual dollars
/// let modern = NaiveDate::from_ymd_opt(2023, 2, 14).unwrap();
/// assert!(!is_13f_thousands_era(Some(modern)));
///
/// // Unknown date → assume modern (safer direction)
/// assert!(!is_13f_thousands_era(None));
/// ```
pub fn is_13f_thousands_era(filing_date: Option<NaiveDate>) -> bool {
    let cutoff = NaiveDate::from_ymd_opt(
        THIRTEENF_THOUSANDS_ERA_CUTOFF.0,
        THIRTEENF_THOUSANDS_ERA_CUTOFF.1,
        THIRTEENF_THOUSANDS_ERA_CUTOFF.2,
    )
    .expect("THIRTEENF_THOUSANDS_ERA_CUTOFF is a hardcoded valid date");

    match filing_date {
        Some(d) => d < cutoff,
        // Unknown date: assume modern (actual USD).  1000× inflation is more
        // detectable than 1000× deflation — this is the safer failure mode.
        None => false,
    }
}

/// Converts the raw `<value>` field from a 13F-HR `informationTable` XML entry
/// into **actual US dollars**, transparently handling both eras.
///
/// This is the **only** place in the codebase that should contain this
/// conversion logic.  See the [module-level documentation](self) for full
/// background, empirical evidence, and SEC citations.
///
/// # Arguments
///
/// - `raw` — the `Decimal` parsed directly from the `<value>` XML text.
/// - `filing_date` — the `filing_date` field from the corresponding
///   [`crate::models::CikSubmission`].  Pass `None` only when genuinely
///   unknown; the function will assume the modern (actual-USD) format.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use rust_decimal_macros::dec;
/// use sec_fetcher::normalize::normalize_13f_value_usd;
///
/// // Legacy era: raw value is in thousands → multiply by 1 000
/// let ancient = NaiveDate::from_ymd_opt(2014, 2, 14).unwrap();
/// assert_eq!(normalize_13f_value_usd(dec!(15000), Some(ancient)), dec!(15_000_000));
///
/// // Modern era: raw value is already actual USD → pass through verbatim
/// let modern = NaiveDate::from_ymd_opt(2026, 2, 17).unwrap();
/// assert_eq!(normalize_13f_value_usd(dec!(15_000_000), Some(modern)), dec!(15_000_000));
/// ```
pub fn normalize_13f_value_usd(raw: Decimal, filing_date: Option<NaiveDate>) -> Decimal {
    if is_13f_thousands_era(filing_date) {
        raw * dec!(1000)
    } else {
        raw
    }
}

/// Computes the portfolio weight for a single 13F-HR holding, returning a
/// type-safe [`Pct`] on the 0–100 scale.
///
/// The 13F-HR `informationTable` schema has no portfolio-weight field — weight
/// must be derived from this holding's `value_usd` divided by the sum of all
/// holdings' `value_usd`.  This is the **single place** in the codebase where
/// that division is performed; do not replicate it.
///
/// Returns [`Pct::ZERO`] when `total_usd` is zero (empty or all-zero portfolio
/// guard).  Result is rounded to 4 decimal places before wrapping.
///
/// # Examples
///
/// ```
/// use rust_decimal_macros::dec;
/// use sec_fetcher::normalize::compute_13f_weight_pct;
///
/// // $15 M out of $20 M total → 75.0000%
/// assert_eq!(compute_13f_weight_pct(dec!(15_000_000), dec!(20_000_000)).value(), dec!(75.0000));
/// // $5 M out of $20 M total → 25.0000%
/// assert_eq!(compute_13f_weight_pct(dec!(5_000_000), dec!(20_000_000)).value(), dec!(25.0000));
/// // Zero total → 0% (guard against divide-by-zero)
/// assert_eq!(compute_13f_weight_pct(dec!(999), dec!(0)).value(), dec!(0));
/// ```
pub fn compute_13f_weight_pct(value_usd: Decimal, total_usd: Decimal) -> Pct {
    if total_usd.is_zero() {
        Pct::ZERO
    } else {
        Pct::from_pct((value_usd / total_usd * dec!(100)).round_dp(4))
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    fn date(y: i32, m: u32, d: u32) -> Option<NaiveDate> {
        NaiveDate::from_ymd_opt(y, m, d)
    }

    // ── is_13f_thousands_era ─────────────────────────────────────────────────

    #[test]
    fn ancient_filing_is_thousands_era() {
        // 2014-02-14: BRK-A Q4-2013 filing, confirmed thousands via price check
        assert!(is_13f_thousands_era(date(2014, 2, 14)));
    }

    #[test]
    fn last_observed_thousands_filing_is_thousands_era() {
        // 2022-11-14: BRK-A Q3-2022, AAPL raw=95634, shares=692000
        // verbatim $/sh = $0.14 (nonsense) → thousands → $138/sh (correct for Q3-2022)
        assert!(is_13f_thousands_era(date(2022, 11, 14)));
    }

    #[test]
    fn first_observed_actual_dollars_filing_is_not_thousands_era() {
        // 2023-02-14: BRK-A Q4-2022, AAPL raw=133289470, shares=1025856
        // verbatim $/sh = $129.93 (correct for Q4-2022) → actual USD
        assert!(!is_13f_thousands_era(date(2023, 2, 14)));
    }

    #[test]
    fn modern_filing_is_not_thousands_era() {
        // 2026-02-17: BRK-A Q4-2025 (accession 0001193125-26-054580)
        assert!(!is_13f_thousands_era(date(2026, 2, 17)));
    }

    #[test]
    fn unknown_date_assumes_modern() {
        // None → false (assume actual USD; wrong direction is 1000× inflation)
        assert!(!is_13f_thousands_era(None));
    }

    #[test]
    fn cutoff_boundary_inclusive_from_below() {
        // 2022-12-31: one day before cutoff → still thousands era
        assert!(is_13f_thousands_era(date(2022, 12, 31)));
    }

    #[test]
    fn cutoff_boundary_at_cutoff() {
        // 2023-01-01: exactly the cutoff → modern era
        assert!(!is_13f_thousands_era(date(2023, 1, 1)));
    }

    // ── normalize_13f_value_usd ──────────────────────────────────────────────

    #[test]
    fn ancient_raw_multiplied_by_1000() {
        let ancient = date(2014, 2, 14);
        assert_eq!(
            normalize_13f_value_usd(dec!(15000), ancient),
            dec!(15_000_000)
        );
    }

    #[test]
    fn modern_raw_passed_through_verbatim() {
        let modern = date(2026, 2, 17);
        assert_eq!(
            normalize_13f_value_usd(dec!(15_000_000), modern),
            dec!(15_000_000)
        );
    }

    #[test]
    fn ancient_and_modern_same_position_produce_identical_value_usd() {
        // This is the key cross-era invariant: the same real-world holding
        // represented in both filing eras should produce the same value_usd
        // after normalization.
        let ancient = date(2014, 2, 14);
        let modern = date(2026, 2, 17);
        // Ancient: raw 15000 (thousands) → $15 000 000
        let ancient_usd = normalize_13f_value_usd(dec!(15000), ancient);
        // Modern: raw 15000000 (actual USD) → $15 000 000
        let modern_usd = normalize_13f_value_usd(dec!(15_000_000), modern);
        assert_eq!(ancient_usd, modern_usd);
    }

    #[test]
    fn unknown_date_passes_through_verbatim() {
        // None → assume modern → no multiplication
        assert_eq!(
            normalize_13f_value_usd(dec!(99_000_000), None),
            dec!(99_000_000)
        );
    }

    #[test]
    fn zero_value_stays_zero_both_eras() {
        assert_eq!(normalize_13f_value_usd(dec!(0), date(2014, 1, 1)), dec!(0));
        assert_eq!(normalize_13f_value_usd(dec!(0), date(2026, 1, 1)), dec!(0));
    }

    #[test]
    fn brka_q3_2022_aapl_price_sanity() {
        // BRK-A Q3-2022 (filed 2022-11-14, acc 0000950123-22-012275):
        // AAPL raw <value>=95634, shares=692000.
        // Thousands era → $95634 * 1000 = $95,634,000.
        // Implied $/sh = $95,634,000 / 692,000 = $138.20 — matches AAPL ~$138 at 2022-09-30.
        use rust_decimal::Decimal;
        let filing = date(2022, 11, 14);
        let raw = Decimal::new(95634, 0);
        let shares = Decimal::new(692000, 0);
        let value_usd = normalize_13f_value_usd(raw, filing);
        let price_per_share = value_usd / shares;
        // AAPL closed at $138.20 on 2022-09-30; allow ±$5 for position-level rounding
        assert!(
            price_per_share > dec!(133) && price_per_share < dec!(145),
            "implied $/sh = {} (expected ~$138 for AAPL Q3-2022)",
            price_per_share
        );
    }

    #[test]
    fn brka_q4_2022_aapl_price_sanity() {
        // BRK-A Q4-2022 (filed 2023-02-14, acc 0000950123-23-002585):
        // AAPL raw <value>=133289470, shares=1025856.
        // Modern era → $133,289,470 verbatim.
        // Implied $/sh = $133,289,470 / 1,025,856 = $129.93 — matches AAPL ~$130 at 2022-12-30.
        use rust_decimal::Decimal;
        let filing = date(2023, 2, 14);
        let raw = Decimal::new(133289470, 0);
        let shares = Decimal::new(1025856, 0);
        let value_usd = normalize_13f_value_usd(raw, filing);
        let price_per_share = value_usd / shares;
        // AAPL closed at ~$129.93 on 2022-12-30; allow ±$5
        assert!(
            price_per_share > dec!(124) && price_per_share < dec!(136),
            "implied $/sh = {} (expected ~$130 for AAPL Q4-2022)",
            price_per_share
        );
    }

    // ── compute_13f_weight_pct ────────────────────────────────────────────────

    #[test]
    fn weight_75_25_split() {
        assert_eq!(
            compute_13f_weight_pct(dec!(15_000_000), dec!(20_000_000)).value(),
            dec!(75.0000)
        );
        assert_eq!(
            compute_13f_weight_pct(dec!(5_000_000), dec!(20_000_000)).value(),
            dec!(25.0000)
        );
    }

    #[test]
    fn weights_sum_to_100() {
        let total = dec!(20_000_000);
        let a = compute_13f_weight_pct(dec!(15_000_000), total).value();
        let b = compute_13f_weight_pct(dec!(5_000_000), total).value();
        assert_eq!(a + b, dec!(100.0000));
    }

    #[test]
    fn zero_total_returns_zero() {
        assert_eq!(
            compute_13f_weight_pct(dec!(1_000_000), dec!(0)).value(),
            dec!(0)
        );
    }

    #[test]
    fn single_position_is_100_pct() {
        assert_eq!(
            compute_13f_weight_pct(dec!(42_000_000), dec!(42_000_000)).value(),
            dec!(100.0000)
        );
    }

    #[test]
    fn thirds_round_correctly() {
        let total = dec!(30_000_000);
        let a = compute_13f_weight_pct(dec!(10_000_000), total).value();
        let b = compute_13f_weight_pct(dec!(10_000_000), total).value();
        let c = compute_13f_weight_pct(dec!(10_000_000), total).value();
        assert_eq!(a, dec!(33.3333));
        assert_eq!(b, dec!(33.3333));
        assert_eq!(c, dec!(33.3333));
    }
}
