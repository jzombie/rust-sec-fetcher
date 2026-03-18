//! # Portfolio weight percentage normalization
//!
//! Every filing type that produces a portfolio weight percentage must go
//! through this module.  There is **one canonical scale: 0–100** (e.g.
//! `7.7546` means 7.7546%).  The 0–1 fractional scale is never used in this
//! codebase.
//!
//! ## Filing-type mapping
//!
//! | Filing type | XML source | Function | Notes |
//! |-------------|-----------|----------|-------|
//! | N-PORT | `<pctVal>` element | [`normalize_nport_weight_pct`] | SEC reports 0–100 verbatim; function is a pass-through |
//! | 13F-HR | *(no weight field)* | [`compute_13f_weight_pct`] | Derived: `value_usd / portfolio_total × 100` |
//!
//! All future filing types that produce percentage weights **must** add a
//! function here and go through it in their parser.
//!
//! ## SEC references
//!
//! - **Form N-PORT XML Technical Specification** — `<pctVal>` definition:
//!   <https://efts.sec.gov/LATEST/search-index?q=%22pctVal%22>
//!   (Section `invstOrSec/pctVal`: "Percent of net asset value", xs:decimal,
//!   reported on a 0–100 scale per SEC instructions.)
//! - **13F-HR informationTable schema** — no portfolio weight field exists;
//!   the schema only provides `<value>` (position market value):
//!   <https://www.sec.gov/info/edgar/forms/edgarform13f/13fxmltechspec.pdf>

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Normalizes a raw `<pctVal>` element from an N-PORT XML entry to the
/// **canonical 0–100 percentage scale**.
///
/// N-PORT `<pctVal>` is reported by filers already on the 0–100 scale per the
/// SEC N-PORT XML schema (e.g., `7.7546` means 7.7546% of net asset value).
/// This function is the **single non-optional gate** through which every N-PORT
/// percentage value flows.  It does not multiply or divide by 100.
///
/// If the SEC changes the `<pctVal>` scale in a future schema revision, update
/// this function and add a citation (schema version + release date) to the
/// module-level documentation above.
///
/// # Examples
///
/// ```
/// use rust_decimal_macros::dec;
/// use sec_fetcher::normalize::normalize_nport_weight_pct;
///
/// // 7.7546 → 7.7546% — returned as-is (already 0-100)
/// assert_eq!(normalize_nport_weight_pct(dec!(7.7546)), dec!(7.7546));
/// // tiny position
/// assert_eq!(normalize_nport_weight_pct(dec!(0.0074)), dec!(0.0074));
/// // fully concentrated fund
/// assert_eq!(normalize_nport_weight_pct(dec!(100)), dec!(100));
/// ```
pub fn normalize_nport_weight_pct(raw: Decimal) -> Decimal {
    // N-PORT <pctVal> is already on the 0-100 scale per the SEC N-PORT XML
    // technical specification.  No arithmetic conversion required.
    raw
}

/// Computes the portfolio weight for a single 13F-HR holding on the
/// **canonical 0–100 percentage scale**.
///
/// The 13F-HR `informationTable` schema has no portfolio-weight field — weight
/// must be derived from this holding's `value_usd` divided by the sum of all
/// holdings' `value_usd`.  This function is the **single place** in the
/// codebase where that division is performed; do not replicate it.
///
/// Returns `0` when `total_usd` is zero (empty or all-zero portfolio guard).
/// Result is rounded to 4 decimal places.
///
/// # Examples
///
/// ```
/// use rust_decimal_macros::dec;
/// use sec_fetcher::normalize::compute_13f_weight_pct;
///
/// // $15 M out of $20 M total → 75.0000%
/// assert_eq!(compute_13f_weight_pct(dec!(15_000_000), dec!(20_000_000)), dec!(75.0000));
/// // $5 M out of $20 M total → 25.0000%
/// assert_eq!(compute_13f_weight_pct(dec!(5_000_000), dec!(20_000_000)), dec!(25.0000));
/// // Zero total → 0 (guard against divide-by-zero)
/// assert_eq!(compute_13f_weight_pct(dec!(999), dec!(0)), dec!(0));
/// ```
pub fn compute_13f_weight_pct(value_usd: Decimal, total_usd: Decimal) -> Decimal {
    if total_usd.is_zero() {
        dec!(0)
    } else {
        (value_usd / total_usd * dec!(100)).round_dp(4)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    // ── normalize_nport_weight_pct ────────────────────────────────────────────

    #[test]
    fn nport_large_pct_passed_through() {
        assert_eq!(normalize_nport_weight_pct(dec!(7.7546)), dec!(7.7546));
    }

    #[test]
    fn nport_tiny_pct_passed_through() {
        assert_eq!(normalize_nport_weight_pct(dec!(0.007357911161)), dec!(0.007357911161));
    }

    #[test]
    fn nport_zero_pct_passed_through() {
        assert_eq!(normalize_nport_weight_pct(dec!(0)), dec!(0));
    }

    #[test]
    fn nport_100_pct_passed_through() {
        assert_eq!(normalize_nport_weight_pct(dec!(100)), dec!(100));
    }

    // ── compute_13f_weight_pct ────────────────────────────────────────────────

    #[test]
    fn thirteenf_75_25_split() {
        assert_eq!(compute_13f_weight_pct(dec!(15_000_000), dec!(20_000_000)), dec!(75.0000));
        assert_eq!(compute_13f_weight_pct(dec!(5_000_000), dec!(20_000_000)), dec!(25.0000));
    }

    #[test]
    fn thirteenf_weights_sum_to_100() {
        let total = dec!(20_000_000);
        let a = compute_13f_weight_pct(dec!(15_000_000), total);
        let b = compute_13f_weight_pct(dec!(5_000_000), total);
        assert_eq!(a + b, dec!(100.0000));
    }

    #[test]
    fn thirteenf_zero_total_returns_zero() {
        assert_eq!(compute_13f_weight_pct(dec!(1_000_000), dec!(0)), dec!(0));
    }

    #[test]
    fn thirteenf_100_pct_single_position() {
        assert_eq!(compute_13f_weight_pct(dec!(42_000_000), dec!(42_000_000)), dec!(100.0000));
    }

    #[test]
    fn thirteenf_thirds_round_correctly() {
        // Three equal positions, each 33.3333% (rounded to 4 dp)
        let total = dec!(30_000_000);
        let a = compute_13f_weight_pct(dec!(10_000_000), total);
        let b = compute_13f_weight_pct(dec!(10_000_000), total);
        let c = compute_13f_weight_pct(dec!(10_000_000), total);
        assert_eq!(a, dec!(33.3333));
        assert_eq!(b, dec!(33.3333));
        assert_eq!(c, dec!(33.3333));
    }
}
