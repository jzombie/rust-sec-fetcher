use regex::Regex;
use strum_macros::EnumIter;

/// A standard numbered item of an SEC Form 10-K annual report.
///
/// Variants are ordered exactly as they appear in the filing structure so that
/// [`TenKItem::iter()`] produces them in document order.  This ordering is
/// relied on by the section-extraction algorithm: each item's end boundary is
/// located by the first heading of any *later* item.
///
/// The SEC designator string (e.g. `"1A"`, `"7"`) is available via
/// [`TenKItem::designator`].  The storage key used in `TenKSections`
/// (e.g. `"item_1a"`, `"item_7"`) is returned by [`TenKItem::map_key`].
///
/// # Example
///
/// ```rust
/// use sec_fetcher::enums::TenKItem;
/// use strum::IntoEnumIterator;
/// let designators: Vec<&str> = TenKItem::iter().map(|i| i.designator()).collect();
/// assert_eq!(designators[0], "1");
/// assert_eq!(designators[9], "7");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumIter)]
pub enum TenKItem {
    /// Business
    Item1,
    /// Risk Factors
    Item1A,
    /// Unresolved Staff Comments
    Item1B,
    /// Cybersecurity (added 2023 under Rule 13a-21)
    Item1C,
    /// Properties
    Item2,
    /// Legal Proceedings
    Item3,
    /// Mine Safety Disclosures
    Item4,
    /// Market for Registrant's Common Equity, Related Stockholder Matters
    /// and Issuer Purchases of Equity Securities
    Item5,
    /// Selected Financial Data (requirement removed after 2021)
    Item6,
    /// Management's Discussion and Analysis of Financial Condition
    /// and Results of Operations
    Item7,
    /// Quantitative and Qualitative Disclosures About Market Risk
    Item7A,
    /// Financial Statements and Supplementary Data
    Item8,
    /// Changes in and Disagreements With Accountants on Accounting
    /// and Financial Disclosure
    Item9,
    /// Controls and Procedures
    Item9A,
    /// Other Information
    Item9B,
    /// Disclosure Regarding Foreign Jurisdictions that Prevent Inspections
    /// (added 2021 under the Holding Foreign Companies Accountable Act)
    Item9C,
    /// Directors, Executive Officers and Corporate Governance
    Item10,
    /// Executive Compensation
    Item11,
    /// Security Ownership of Certain Beneficial Owners and Management
    /// and Related Stockholder Matters
    Item12,
    /// Certain Relationships and Related Transactions, and Director Independence
    Item13,
    /// Principal Accountant Fees and Services
    Item14,
    /// Exhibits and Financial Statement Schedules
    Item15,
    /// Form 10-K Summary
    Item16,
}

impl TenKItem {
    /// Returns the SEC-defined designator string for this item.
    ///
    /// Examples: `"1"`, `"1A"`, `"7"`, `"9A"`.
    pub fn designator(&self) -> &'static str {
        match self {
            TenKItem::Item1 => "1",
            TenKItem::Item1A => "1A",
            TenKItem::Item1B => "1B",
            TenKItem::Item1C => "1C",
            TenKItem::Item2 => "2",
            TenKItem::Item3 => "3",
            TenKItem::Item4 => "4",
            TenKItem::Item5 => "5",
            TenKItem::Item6 => "6",
            TenKItem::Item7 => "7",
            TenKItem::Item7A => "7A",
            TenKItem::Item8 => "8",
            TenKItem::Item9 => "9",
            TenKItem::Item9A => "9A",
            TenKItem::Item9B => "9B",
            TenKItem::Item9C => "9C",
            TenKItem::Item10 => "10",
            TenKItem::Item11 => "11",
            TenKItem::Item12 => "12",
            TenKItem::Item13 => "13",
            TenKItem::Item14 => "14",
            TenKItem::Item15 => "15",
            TenKItem::Item16 => "16",
        }
    }

    /// Returns the normalized storage key used in `TenKSections`.
    ///
    /// The key is `item_` followed by the lowercase designator:
    /// `"item_1"`, `"item_1a"`, `"item_7a"`, etc.
    pub fn map_key(&self) -> String {
        format!("item_{}", self.designator().to_ascii_lowercase())
    }

    /// Returns a compiled heading regex anchored to the start of a line.
    ///
    /// Matches this item as it appears in cleaned 10-K plain-text, following
    /// the `adjust_item_patterns` logic from
    /// [edgar-crawler](https://github.com/nlpaueb/edgar-crawler):
    ///
    /// - Alphabetic suffixes (`A`, `B`, `C`) tolerate optional horizontal
    ///   whitespace between the digit and the letter (e.g. `ITEM 9 A`).
    /// - `9A` additionally accepts the legacy `(T)` suffix.
    /// - The heading must be followed by a separator: `.`, `*`, `~`, `-`, `:`,
    ///   whitespace, `(`, en-dash (U+2013), or em-dash (U+2014).
    pub fn item_pattern(&self) -> Regex {
        let des = self.designator().to_ascii_uppercase();
        let pat = if des == "9A" {
            r"9[^\S\r\n]*A(?:\(T\))?".to_string()
        } else if des.ends_with('A') {
            let num = des.trim_end_matches('A');
            format!(r"{}[^\S\r\n]*A", num)
        } else if des.ends_with('B') {
            let num = des.trim_end_matches('B');
            format!(r"{}[^\S\r\n]*B", num)
        } else if des.ends_with('C') {
            let num = des.trim_end_matches('C');
            format!(r"{}[^\S\r\n]*C", num)
        } else {
            des
        };
        // `[.*~\-:\s(\u{2013}\u{2014}]` — ASCII separators plus en/em dashes.
        // Modern iXBRL filings (2018+) often format headings as
        // "Item\u{00a0}1\u{2014}Business" (non-breaking space + em dash), so
        // both Unicode dash characters must be included.
        Regex::new(&format!(
            r"(?im)^[ \t]*ITEMS?\s*{pat}[ \t]*[.*~\-:\s(\u{{2013}}\u{{2014}}]"
        ))
        .expect("bug: bad item regex")
    }
}
