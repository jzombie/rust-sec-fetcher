//! Workspace-internal shared types and utilities.
//!
//! Contains fiscal-period parsing, sort-rank helpers, and the canonical set
//! of CSV column names written by the data pipeline.  This crate has no
//! external dependencies and is not published to crates.io.
//!
//! # Fiscal-period sorting
//!
//! The SEC's `fp` field on EDGAR observations uses a variety of tokens to
//! label fiscal periods: `"Q1"`, `"FY"`, `"H1"`, `"SA2"`, `"6M"`, etc.
//! [`parse_period_slot_token`] normalises all of them into a comparable
//! integer rank so that rows can be sorted newest-first without a
//! hardcoded lookup table.

/// Canonical period-slot categories parsed from raw fiscal-period labels.
///
/// This enum is intentionally source-format agnostic: multiple token styles
/// map into the small stable set used by filtering and trend ranking.
///
/// Recognised aliases:
/// - Quarters:     `Q1`..`Q4`
/// - Annual:       `FY`, `ANNUAL`
/// - Semi-annual:  `H1`/`H2`, `HY1`/`HY2`, `SA1`/`SA2`, `S1`/`S2`
/// - Month windows: `3M`, `6M`, `9M`, `12M`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodSlot {
    Q1,
    Q2,
    Q3,
    Q4,
    FY,
    H1,
    H2,
    M3,
    M6,
    M9,
    M12,
}

impl PeriodSlot {
    /// Maps the slot to a normalised quarter-space rank (`1..=4`).
    ///
    /// Higher rank = later in the fiscal year.  Sort descending on this
    /// value to get newest-first row ordering.
    ///
    /// | Rank | Slots |
    /// |------|-------|
    /// | 4    | `Q4`, `FY`, `H2`, `12M` |
    /// | 3    | `Q3`, `9M` |
    /// | 2    | `Q2`, `H1`, `6M` |
    /// | 1    | `Q1`, `3M` |
    pub fn normalized_quarter(self) -> i64 {
        match self {
            PeriodSlot::Q1 | PeriodSlot::M3 => 1,
            PeriodSlot::Q2 | PeriodSlot::H1 | PeriodSlot::M6 => 2,
            PeriodSlot::Q3 | PeriodSlot::M9 => 3,
            PeriodSlot::Q4 | PeriodSlot::FY | PeriodSlot::H2 | PeriodSlot::M12 => 4,
        }
    }
}

/// Extracts a quarter number (1–4) from tokens like `"Q1"`, `"Q3"`.
/// Returns `None` for anything outside that range or without a `Q` prefix.
pub fn parse_quarter_token(s: &str) -> Option<i64> {
    let upper = s.to_ascii_uppercase();
    let pos = upper.find('Q')?;
    let n = upper[pos + 1..]
        .chars()
        .next()
        .and_then(|c| c.to_digit(10))
        .map(i64::from)?;
    if (1..=4).contains(&n) { Some(n) } else { None }
}

/// Parses a raw fiscal-period token into a canonical [`PeriodSlot`].
///
/// ```
/// use sec_fetcher_shared::{PeriodSlot, parse_period_slot};
/// assert_eq!(parse_period_slot("SA1"), Some(PeriodSlot::H1));
/// assert_eq!(parse_period_slot("FY"),  Some(PeriodSlot::FY));
/// assert_eq!(parse_period_slot("Q4"),  Some(PeriodSlot::Q4));
/// ```
pub fn parse_period_slot(s: &str) -> Option<PeriodSlot> {
    let upper = s.trim().to_ascii_uppercase();

    if let Some(q) = parse_quarter_token(&upper) {
        return Some(match q {
            1 => PeriodSlot::Q1,
            2 => PeriodSlot::Q2,
            3 => PeriodSlot::Q3,
            4 => PeriodSlot::Q4,
            _ => return None,
        });
    }

    if upper.contains("FY") || upper.contains("ANNUAL") {
        return Some(PeriodSlot::FY);
    }

    // Semi-annual H1 aliases (must check before H2 to avoid prefix collision)
    if upper.contains("HY1") || upper.contains("H1") || upper.contains("SA1") || upper.contains("S1") {
        return Some(PeriodSlot::H1);
    }
    if upper.contains("HY2") || upper.contains("H2") || upper.contains("SA2") || upper.contains("S2") {
        return Some(PeriodSlot::H2);
    }

    // Month-window aliases (12M first to avoid prefix match on 2M)
    if upper.contains("12M") {
        return Some(PeriodSlot::M12);
    }
    if upper.contains("9M") {
        return Some(PeriodSlot::M9);
    }
    if upper.contains("6M") {
        return Some(PeriodSlot::M6);
    }
    if upper.contains("3M") {
        return Some(PeriodSlot::M3);
    }

    None
}

/// Returns the normalised quarter rank (1–4) for a raw fiscal-period token,
/// or `None` if the token is not recognised.
///
/// Equivalent to `parse_period_slot(s).map(PeriodSlot::normalized_quarter)`.
///
/// This is the primary entry-point for sort-key computation.
///
/// ```
/// use sec_fetcher_shared::parse_period_slot_token;
/// assert_eq!(parse_period_slot_token("FY"),  Some(4));
/// assert_eq!(parse_period_slot_token("Q3"),  Some(3));
/// assert_eq!(parse_period_slot_token("SA1"), Some(2));
/// assert_eq!(parse_period_slot_token("Q4"),  Some(4));
/// assert_eq!(parse_period_slot_token(""),    None);
/// ```
pub fn parse_period_slot_token(s: &str) -> Option<i64> {
    parse_period_slot(s).map(PeriodSlot::normalized_quarter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarters_rank_correctly() {
        assert_eq!(parse_period_slot_token("Q1"), Some(1));
        assert_eq!(parse_period_slot_token("Q2"), Some(2));
        assert_eq!(parse_period_slot_token("Q3"), Some(3));
        assert_eq!(parse_period_slot_token("Q4"), Some(4));
    }

    #[test]
    fn fy_ranks_same_as_q4() {
        assert_eq!(parse_period_slot_token("FY"), Some(4));
    }

    #[test]
    fn semi_annual_aliases() {
        assert_eq!(parse_period_slot_token("H1"),  Some(2));
        assert_eq!(parse_period_slot_token("H2"),  Some(4));
        assert_eq!(parse_period_slot_token("HY1"), Some(2));
        assert_eq!(parse_period_slot_token("HY2"), Some(4));
        assert_eq!(parse_period_slot_token("SA1"), Some(2));
        assert_eq!(parse_period_slot_token("SA2"), Some(4));
        assert_eq!(parse_period_slot_token("S1"),  Some(2));
        assert_eq!(parse_period_slot_token("S2"),  Some(4));
    }

    #[test]
    fn month_window_aliases() {
        assert_eq!(parse_period_slot_token("3M"),  Some(1));
        assert_eq!(parse_period_slot_token("6M"),  Some(2));
        assert_eq!(parse_period_slot_token("9M"),  Some(3));
        assert_eq!(parse_period_slot_token("12M"), Some(4));
    }

    #[test]
    fn unrecognised_returns_none() {
        assert_eq!(parse_period_slot_token(""),   None);
        assert_eq!(parse_period_slot_token("SA"), None);
        assert_eq!(parse_period_slot_token("Q5"), None);
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(parse_period_slot_token("fy"), Some(4));
        assert_eq!(parse_period_slot_token("q2"), Some(2));
        assert_eq!(parse_period_slot_token("sa2"), Some(4));
    }
}

/// A parsed fiscal period — either a specific year+quarter or a bare year.
///
/// Quarter values use the same 1–4 rank as [`PeriodSlot::normalized_quarter`]:
/// Q1 = 1, Q2/H1/6M = 2, Q3/9M = 3, Q4/FY/H2/12M = 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Period {
    YearQuarter { year: i64, quarter: i64 },
    Year { year: i64 },
}

/// Extracts the first 4-digit year (1900–2100) from a string.
///
/// Used by [`parse_period`] to pull the fiscal year out of tokens like
/// `"2024Q3"`, `"2024FY"`, or `"2024"`.
pub fn extract_first_year(s: &str) -> Option<i64> {
    let chars: Vec<char> = s.chars().collect();
    for i in 0..chars.len().saturating_sub(3) {
        if chars[i].is_ascii_digit()
            && chars[i + 1].is_ascii_digit()
            && chars[i + 2].is_ascii_digit()
            && chars[i + 3].is_ascii_digit()
        {
            let year_str: String = chars[i..=i + 3].iter().collect();
            if let Ok(year) = year_str.parse::<i64>() {
                if (1900..=2100).contains(&year) {
                    return Some(year);
                }
            }
        }
    }
    None
}

/// Parses a raw period string into a [`Period`].
///
/// Recognises year-only strings (`"2024"`) and combined year+period tokens
/// (`"2024Q3"`, `"2024FY"`, `"2024H1"`, `"2024 9M"`, etc.).
///
/// Returns `Err` if no 4-digit year can be found.
///
/// ```
/// use sec_fetcher_shared::{Period, parse_period};
/// assert_eq!(parse_period("2024Q3").unwrap(), Period::YearQuarter { year: 2024, quarter: 3 });
/// assert_eq!(parse_period("2024FY").unwrap(), Period::YearQuarter { year: 2024, quarter: 4 });
/// assert_eq!(parse_period("2024").unwrap(),   Period::Year { year: 2024 });
/// ```
pub fn parse_period(period: &str) -> Result<Period, String> {
    let raw = period.trim();
    let upper = raw.to_ascii_uppercase();
    let year = extract_first_year(&upper).ok_or_else(|| {
        format!("Period `{}` is missing year; expected values like 2024Q3, 2024H1, or 2024FY", raw)
    })?;
    if let Some(slot) = parse_period_slot(&upper) {
        let q = slot.normalized_quarter();
        return Ok(Period::YearQuarter { year, quarter: q });
    }
    Ok(Period::Year { year })
}

/// Ordered metadata columns present in every per-symbol US-GAAP CSV file.
///
/// These columns are written by the data pipeline and read by the normaliser
/// and trend analysis layers.  They always appear before the XBRL fact columns
/// (which vary per company) and must be treated as non-fact bookkeeping data.
///
/// The order here is canonical: both the writer and reader rely on it.
pub const US_GAAP_CSV_META_COLUMNS: &[&str] = &[
    "canonical_order",
    "fy",
    "fp",
    "period_end",
    "filed",
    "form",
    "accn",
    "filing_url",
];
