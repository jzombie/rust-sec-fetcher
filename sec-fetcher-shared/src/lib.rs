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
    if upper.contains("HY1")
        || upper.contains("H1")
        || upper.contains("SA1")
        || upper.contains("S1")
    {
        return Some(PeriodSlot::H1);
    }
    if upper.contains("HY2")
        || upper.contains("H2")
        || upper.contains("SA2")
        || upper.contains("S2")
    {
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

/// Normalises a raw SEC `fp` token to its canonical CSV label.
///
/// The only transformation currently applied is **`Q4` → `FY`**: the SEC tags
/// some year-end filings as `"Q4"` (typically via a 10-Q/A) rather than the
/// standard `"FY"` used by annual 10-K filings.  Both map to the same
/// period-slot rank (4) and refer to the same fiscal year-end point, so the
/// canonical form is `"FY"` for consistency in downstream CSV consumers.
///
/// All other tokens are returned unchanged (preserving case as-is from the
/// API, e.g. `"Q1"`, `"H1"`, `"SA2"`).
///
/// ```
/// use sec_fetcher_shared::normalize_fp_label;
/// assert_eq!(normalize_fp_label("Q4"), "FY");
/// assert_eq!(normalize_fp_label("q4"), "FY");
/// assert_eq!(normalize_fp_label("FY"),  "FY");
/// assert_eq!(normalize_fp_label("Q3"),  "Q3");
/// assert_eq!(normalize_fp_label("H1"),  "H1");
/// ```
pub fn normalize_fp_label(fp: &str) -> String {
    if fp.trim().eq_ignore_ascii_case("Q4") {
        "FY".to_string()
    } else {
        fp.to_string()
    }
}

/// Returns the set of candidate ticker symbols that a raw input string might
/// resolve to, normalised to uppercase and with `.`/`-` variants generated.
///
/// EDGAR and data providers use both `.` and `-` as separators in class-share
/// tickers (e.g. `BRK.B` vs `BRK-B`).  This function always returns both
/// forms so callers only need a single lookup rather than separate tries.
///
/// ```
/// use sec_fetcher_shared::normalize_symbol;
/// let c = normalize_symbol("brk.b");
/// assert!(c.contains(&"BRK.B".to_string()));
/// assert!(c.contains(&"BRK-B".to_string()));
/// ```
pub fn normalize_symbol(symbol: &str) -> Vec<String> {
    let upper = symbol.to_ascii_uppercase();
    let dot_to_dash = upper.replace('.', "-");
    let dash_to_dot = upper.replace('-', ".");

    let mut set = std::collections::HashSet::new();
    set.insert(upper);
    set.insert(dot_to_dash);
    set.insert(dash_to_dot);

    let mut out: Vec<String> = set.into_iter().collect();
    out.sort();
    out
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
        assert_eq!(parse_period_slot_token("H1"), Some(2));
        assert_eq!(parse_period_slot_token("H2"), Some(4));
        assert_eq!(parse_period_slot_token("HY1"), Some(2));
        assert_eq!(parse_period_slot_token("HY2"), Some(4));
        assert_eq!(parse_period_slot_token("SA1"), Some(2));
        assert_eq!(parse_period_slot_token("SA2"), Some(4));
        assert_eq!(parse_period_slot_token("S1"), Some(2));
        assert_eq!(parse_period_slot_token("S2"), Some(4));
    }

    #[test]
    fn month_window_aliases() {
        assert_eq!(parse_period_slot_token("3M"), Some(1));
        assert_eq!(parse_period_slot_token("6M"), Some(2));
        assert_eq!(parse_period_slot_token("9M"), Some(3));
        assert_eq!(parse_period_slot_token("12M"), Some(4));
    }

    #[test]
    fn unrecognised_returns_none() {
        assert_eq!(parse_period_slot_token(""), None);
        assert_eq!(parse_period_slot_token("SA"), None);
        assert_eq!(parse_period_slot_token("Q5"), None);
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(parse_period_slot_token("fy"), Some(4));
        assert_eq!(parse_period_slot_token("q2"), Some(2));
        assert_eq!(parse_period_slot_token("sa2"), Some(4));
    }

    #[test]
    fn normalize_fp_label_maps_q4_to_fy() {
        assert_eq!(normalize_fp_label("Q4"), "FY");
        assert_eq!(normalize_fp_label("q4"), "FY");
    }

    #[test]
    fn normalize_fp_label_leaves_other_tokens_unchanged() {
        assert_eq!(normalize_fp_label("FY"), "FY");
        assert_eq!(normalize_fp_label("Q3"), "Q3");
        assert_eq!(normalize_fp_label("H1"), "H1");
        assert_eq!(normalize_fp_label("SA2"), "SA2");
        assert_eq!(normalize_fp_label(""), "");
    }

    #[test]
    fn normalize_symbol_generates_dot_and_dash_variants() {
        let c = normalize_symbol("brk.b");
        assert!(c.contains(&"BRK.B".to_string()));
        assert!(c.contains(&"BRK-B".to_string()));
    }

    #[test]
    fn normalize_symbol_upcases_plain_ticker() {
        let c = normalize_symbol("aapl");
        assert_eq!(c, vec!["AAPL".to_string()]);
    }

    // -- extract_first_year --

    #[test]
    fn test_extract_first_year_from_combined() {
        assert_eq!(extract_first_year("2024Q3"), Some(2024));
    }

    #[test]
    fn test_extract_first_year_from_plain_year() {
        assert_eq!(extract_first_year("2024"), Some(2024));
    }

    #[test]
    fn test_extract_first_year_from_longer_string() {
        assert_eq!(extract_first_year("FY ended 2024-12-31"), Some(2024));
    }

    #[test]
    fn test_extract_first_year_out_of_range_low() {
        assert_eq!(extract_first_year("1899"), None);
    }

    #[test]
    fn test_extract_first_year_out_of_range_high() {
        assert_eq!(extract_first_year("2101"), None);
    }

    #[test]
    fn test_extract_first_year_no_digits() {
        assert_eq!(extract_first_year("hello world"), None);
    }

    #[test]
    fn test_extract_first_year_short_string() {
        assert_eq!(extract_first_year("23"), None);
    }

    #[test]
    fn test_extract_first_year_empty_string() {
        assert_eq!(extract_first_year(""), None);
    }

    // -- parse_period --

    #[test]
    fn test_parse_period_year_only() {
        assert_eq!(parse_period("2024").unwrap(), Period::Year { year: 2024 });
    }

    #[test]
    fn test_parse_period_year_quarter() {
        assert_eq!(
            parse_period("2024Q3").unwrap(),
            Period::YearQuarter {
                year: 2024,
                quarter: 3
            }
        );
    }

    #[test]
    fn test_parse_period_year_fy() {
        assert_eq!(
            parse_period("2024FY").unwrap(),
            Period::YearQuarter {
                year: 2024,
                quarter: 4
            }
        );
    }

    #[test]
    fn test_parse_period_year_h1() {
        assert_eq!(
            parse_period("2024H1").unwrap(),
            Period::YearQuarter {
                year: 2024,
                quarter: 2
            }
        );
    }

    #[test]
    fn test_parse_period_year_9m() {
        assert_eq!(
            parse_period("2024 9M").unwrap(),
            Period::YearQuarter {
                year: 2024,
                quarter: 3
            }
        );
    }

    #[test]
    fn test_parse_period_whitespace_trimmed() {
        assert_eq!(
            parse_period("  2024Q2  ").unwrap(),
            Period::YearQuarter {
                year: 2024,
                quarter: 2
            }
        );
    }

    #[test]
    fn test_parse_period_missing_year() {
        let err = parse_period("Q3").unwrap_err();
        assert!(err.contains("missing year"));
    }

    #[test]
    fn test_parse_period_empty_string() {
        let err = parse_period("").unwrap_err();
        assert!(err.contains("missing year"));
    }

    // -- normalize_symbol --

    #[test]
    fn test_normalize_symbol_dash_to_dot() {
        let c = normalize_symbol("BRK-B");
        assert!(c.contains(&"BRK.B".to_string()));
        assert!(c.contains(&"BRK-B".to_string()));
    }

    #[test]
    fn test_normalize_symbol_no_change_needed() {
        let c = normalize_symbol("AAPL");
        assert_eq!(c, vec!["AAPL".to_string()]);
    }

    #[test]
    fn test_normalize_symbol_dot_and_dash_both_present() {
        let c = normalize_symbol("brk.b");
        assert!(c.contains(&"BRK.B".to_string()));
        assert!(c.contains(&"BRK-B".to_string()));
        // Exactly 2 variants (no duplicates)
        assert_eq!(c.len(), 2);
    }

    // -- normalize_fp_label --

    #[test]
    fn test_normalize_fp_label_q4_case_insensitive() {
        assert_eq!(normalize_fp_label("Q4"), "FY");
        assert_eq!(normalize_fp_label("q4"), "FY");
        assert_eq!(normalize_fp_label("  Q4  "), "FY");
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
            if let Ok(year) = year_str.parse::<i64>()
                && (1900..=2100).contains(&year)
            {
                return Some(year);
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
        format!(
            "Period `{}` is missing year; expected values like 2024Q3, 2024H1, or 2024FY",
            raw
        )
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
///
/// ## Row ordering convention
///
/// Rows are written **newest-first** (reverse chronological).  The primary
/// sort key is `(fy DESC, fp_rank DESC)` where `fp_rank` is the integer
/// returned by [`parse_period_slot_token`] for the `fp` field.
///
/// `canonical_order` is a **0-based physical row index**: the first row has
/// `canonical_order = 0`, the second `1`, and so on.  It encodes the
/// newest-first position assigned at write time and is the bridge between the
/// on-disk row position and the runtime `local_idx` used for global-ID lookups.
/// A mismatch between the stored value and the actual row position indicates
/// that the file has been rewritten or reordered without updating this column.
/// `is_amendment` (index 6): `true` when the winning row for this period came from an
/// amendment filing (i.e. the original `form` value ended with `/A`, e.g. `"10-Q/A"`).
/// The `form` column itself is normalised to the base type (`"10-Q"`).
pub const US_GAAP_CSV_META_COLUMNS: &[&str] = &[
    "canonical_order",
    "fy",
    "fp",
    "period_end",
    "filed",
    "form",
    "is_amendment",
    "accn",
    "filing_url",
];
