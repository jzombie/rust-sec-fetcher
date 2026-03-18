//! # `Pct` — a type-safe percentage value
//!
//! SEC EDGAR percentage fields (e.g. `<pctVal>` in N-PORT, portfolio weights
//! derived from 13F holdings) all use the **0–100 scale**: `7.7546` means
//! 7.7546%, never 0.077546.
//!
//! `Pct` makes the scale choice structural — it is impossible to store a
//! percentage without declaring which scale the raw value is on.  Numeric
//! range is **not restricted**: short positions carry negative weights,
//! leveraged fund allocations exceed 100%, and weight deltas are unbounded.
//!
//! ## Constructors
//!
//! | Constructor | Input scale | When to use |
//! |---|---|---|
//! | [`Pct::from_pct`] | already 0–100 | N-PORT `<pctVal>`, most SEC fields |
//! | [`Pct::from_ratio`] | 0–1 fraction | when SEC reports a bare ratio |
//! | [`Pct::ZERO`] | — | default / zero sentinel |

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::fmt;
use std::str::FromStr;

/// A type-safe percentage value on the **0–100 scale**.
///
/// `7.7546` means 7.7546%, not 0.077546.  Scale is the only structural
/// invariant; no bounds constraint is imposed (negative values and values
/// exceeding 100 are both valid — see module-level documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pct(Decimal);

impl Pct {
    /// A `Pct` representing 0% on the 0–100 scale.
    pub const ZERO: Self = Self(Decimal::ZERO);

    /// Constructs a `Pct` from a value **already on the 0–100 scale**.
    ///
    /// `7.7546` → `Pct` representing 7.7546%.  No multiplication or division
    /// is performed.  Use this for all SEC fields that report `<pctVal>` or
    /// similar fields already in 0–100 form.
    ///
    /// Negative values (short positions) and values exceeding 100 (leveraged
    /// funds, gross exposure) are accepted without error.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_decimal_macros::dec;
    /// use sec_fetcher::normalize::Pct;
    ///
    /// assert_eq!(Pct::from_pct(dec!(7.7546)).value(), dec!(7.7546));
    /// assert_eq!(Pct::from_pct(dec!(-3.25)).value(), dec!(-3.25));   // short
    /// assert_eq!(Pct::from_pct(dec!(142.7)).value(), dec!(142.7));   // leveraged
    /// ```
    #[inline]
    pub fn from_pct(value: Decimal) -> Self {
        Self(value)
    }

    /// Constructs a `Pct` from a **0–1 fraction**, multiplying by 100 internally.
    ///
    /// Use this when a computation or field yields a bare ratio (`0.077546`)
    /// that should be stored as a percentage (`7.7546`).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_decimal_macros::dec;
    /// use sec_fetcher::normalize::Pct;
    ///
    /// assert_eq!(Pct::from_ratio(dec!(0.077546)).value(), dec!(7.7546));
    /// assert_eq!(Pct::from_ratio(dec!(0.5)).value(), dec!(50));
    /// ```
    #[inline]
    pub fn from_ratio(ratio: Decimal) -> Self {
        Self(ratio * dec!(100))
    }

    /// Returns the inner value on the **0–100 scale**.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_decimal_macros::dec;
    /// use sec_fetcher::normalize::Pct;
    ///
    /// assert_eq!(Pct::from_pct(dec!(7.7546)).value(), dec!(7.7546));
    /// ```
    #[inline]
    pub fn value(self) -> Decimal {
        self.0
    }

    /// Returns the inner value as a **0–1 fraction** (divides by 100).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_decimal_macros::dec;
    /// use sec_fetcher::normalize::Pct;
    ///
    /// assert_eq!(Pct::from_pct(dec!(7.7546)).as_ratio(), dec!(0.077546));
    /// ```
    #[inline]
    pub fn as_ratio(self) -> Decimal {
        self.0 / dec!(100)
    }
}

/// Formats the percentage using the inner [`Decimal`]'s formatter.
///
/// All format specifiers — width, precision, alignment — are forwarded, so
/// `{:.4}`, `{:>8.2}`, etc. work directly on a `Pct` without calling
/// `.value()` first.
///
/// # Examples
///
/// ```
/// use rust_decimal_macros::dec;
/// use sec_fetcher::normalize::Pct;
///
/// let pct = Pct::from_pct(dec!(7.7546));
/// assert_eq!(format!("{}", pct), "7.7546");
/// assert_eq!(format!("{:.2}", pct), "7.75");
/// assert_eq!(format!("{:>10.4}", pct), "    7.7546");
/// ```
impl fmt::Display for Pct {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Parses a `Pct` from a decimal string.
///
/// No bounds checking is performed.  Used by [`serde_with`]'s `DisplayFromStr`
/// deserializer on fields annotated with `#[serde_as(as = "DisplayFromStr")]`.
impl FromStr for Pct {
    type Err = rust_decimal::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.trim().parse::<Decimal>().map(Self)
    }
}

/// Subtracting two `Pct` values yields a signed [`Decimal`] delta.
///
/// Portfolio weight deltas can be negative (position reduced or fully sold)
/// and are unbounded — the result type is [`Decimal`] rather than a new `Pct`
/// to make this explicit.
impl std::ops::Sub<Pct> for Pct {
    type Output = Decimal;

    #[inline]
    fn sub(self, other: Pct) -> Decimal {
        self.0 - other.0
    }
}

/// Negates a `Pct`, returning a new `Pct` with the sign flipped.
impl std::ops::Neg for Pct {
    type Output = Pct;

    #[inline]
    fn neg(self) -> Pct {
        Pct(-self.0)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn from_pct_zero() {
        assert_eq!(Pct::from_pct(Decimal::ZERO).value(), Decimal::ZERO);
    }

    #[test]
    fn from_pct_typical() {
        assert_eq!(Pct::from_pct(dec!(7.7546)).value(), dec!(7.7546));
    }

    #[test]
    fn from_pct_100() {
        assert_eq!(Pct::from_pct(dec!(100)).value(), dec!(100));
    }

    #[test]
    fn from_pct_negative_short_position() {
        assert_eq!(Pct::from_pct(dec!(-3.25)).value(), dec!(-3.25));
    }

    #[test]
    fn from_pct_leveraged_over_100() {
        assert_eq!(Pct::from_pct(dec!(142.7)).value(), dec!(142.7));
    }

    #[test]
    fn from_ratio_typical() {
        assert_eq!(Pct::from_ratio(dec!(0.077546)).value(), dec!(7.7546));
    }

    #[test]
    fn from_ratio_half() {
        assert_eq!(Pct::from_ratio(dec!(0.5)).value(), dec!(50));
    }

    #[test]
    fn from_ratio_zero() {
        assert_eq!(Pct::from_ratio(Decimal::ZERO).value(), Decimal::ZERO);
    }

    #[test]
    fn as_ratio_roundtrip() {
        let pct = Pct::from_pct(dec!(7.7546));
        assert_eq!(pct.as_ratio(), dec!(0.077546));
    }

    #[test]
    fn const_zero_is_value_zero() {
        assert_eq!(Pct::ZERO.value(), Decimal::ZERO);
    }

    #[test]
    fn display_forwards_precision_specifier() {
        let pct = Pct::from_pct(dec!(7.7546));
        assert_eq!(format!("{:.2}", pct), "7.75");
        assert_eq!(format!("{:.4}", pct), "7.7546");
        assert_eq!(format!("{}", pct), "7.7546");
    }

    #[test]
    fn from_str_typical() {
        let pct: Pct = "7.7546".parse().unwrap();
        assert_eq!(pct.value(), dec!(7.7546));
    }

    #[test]
    fn from_str_negative_accepted() {
        let pct: Pct = "-3.25".parse().unwrap();
        assert_eq!(pct.value(), dec!(-3.25));
    }

    #[test]
    fn from_str_over_100_accepted() {
        let pct: Pct = "142.7".parse().unwrap();
        assert_eq!(pct.value(), dec!(142.7));
    }

    #[test]
    fn from_str_not_a_number_errors() {
        assert!("not_a_number".parse::<Pct>().is_err());
    }

    #[test]
    fn sub_positive_delta() {
        let a = Pct::from_pct(dec!(75));
        let b = Pct::from_pct(dec!(25));
        let delta: Decimal = a - b;
        assert_eq!(delta, dec!(50));
    }

    #[test]
    fn sub_negative_delta() {
        let lo = Pct::from_pct(dec!(25));
        let hi = Pct::from_pct(dec!(75));
        let delta: Decimal = lo - hi;
        assert_eq!(delta, dec!(-50));
    }

    #[test]
    fn neg_flips_sign() {
        let pct = Pct::from_pct(dec!(7.5));
        assert_eq!((-pct).value(), dec!(-7.5));
    }

    #[test]
    fn ordering_consistent_with_inner_decimal() {
        let lo = Pct::from_pct(dec!(10));
        let hi = Pct::from_pct(dec!(90));
        assert!(lo < hi);
        assert!(hi > lo);
        assert_eq!(lo, lo);
    }

    #[test]
    fn copy_semantics() {
        let a = Pct::from_pct(dec!(50));
        let b = a;
        assert_eq!(a.value(), b.value());
    }
}
