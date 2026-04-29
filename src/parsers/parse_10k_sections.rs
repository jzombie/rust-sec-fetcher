use crate::enums::TenKItem;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock as Lazy;
use strum::IntoEnumIterator;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Body sections shorter than this are discarded as TOC stubs.
const MIN_SECTION_CHARS: usize = 400;

/// Item 7 sections shorter than this trigger the MD&A title-based fallback.
const MIN_ITEM7_CHARS: usize = 2_000;

// ── Static regexes ────────────────────────────────────────────────────────────

/// Strip iXBRL header blocks before handing to html2text; they contain dense
/// machine-readable XBRL manifests that add noise but no prose value.
static IX_HEADER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<ix:header\b[^>]*>.*?</ix:header>").unwrap());

/// Strip remaining HTML / SGML tags (used only on the SGML plain-text path).
static TAG_STRIP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]*>").unwrap());

/// Fix spaced-out ITEM headers produced when span-per-character markup is
/// stripped: `"I T E M   7  A"` → `"ITEM 7A"`.
static SPACED_ITEM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?im)^([ \t]*)(I[ \t]*T[ \t]*E[ \t]*M)([ \t]+)(\d{1,2}(?:[ \t]*[A-C])?)").unwrap()
});

/// Remove standalone page-number lines (bare 1–4-digit numbers, optionally
/// surrounded by dashes).
static PAGE_NUM_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^[ \t]*-{0,5}[ \t]*\d{1,4}[ \t]*-{0,5}[ \t]*$").unwrap());

/// Collapse three or more consecutive blank lines to two.
static MULTI_NL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());

/// Standalone "MANAGEMENT'S DISCUSSION AND ANALYSIS…" heading — used by the
/// Item-7 fallback for filings that incorporate MD&A by reference.
///
/// Matches both ASCII apostrophe `'` (U+0027) and the curly right-quote `'`
/// (U+2019) that html2text emits when it decodes `&#146;`, `&#8217;`, or the
/// raw U+2019 character in iXBRL/HTML source.  The apostrophe and trailing `S`
/// are each independently optional to handle `MANAGEMENT DISCUSSION` variants.
static MDA_TITLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?im)^[ \t]*MANAGEMENT['\u{2019}]?S?[ \t]+DISCUSSION[ \t]+AND[ \t]+ANALYSIS")
        .unwrap()
});

/// End-of-MD&A markers for the Item-7 title-based fallback.
static MDA_END_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?im)^[ \t]*(ITEMS?\s*8\b|QUANTITATIVE\s+AND\s+QUALITATIVE|CONSOLIDATED\s+(STATEMENTS?|INCOME|BALANCE(?:\s+SHEETS?)?|FINANCIAL)|REPORT\s+OF\s+INDEPENDENT|INDEPENDENT\s+REGISTERED\s+PUBLIC)",
    )
    .unwrap()
});

// ── Public types ──────────────────────────────────────────────────────────────

/// Error returned when `html2text` panics on a malformed HTML document.
///
/// html2text has a known byte-index panic bug (text_renderer.rs:2035) on
/// certain EDGAR documents.  This error surfaces through
/// [`extract_sections_from_document`] so callers can decide how to handle it
/// (e.g. write an error row in an audit CSV, fall back to a different
/// document, or skip the filing entirely).
#[derive(Debug)]
pub struct Html2TextPanic;

impl std::fmt::Display for Html2TextPanic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "html2text panicked on malformed document")
    }
}

impl std::error::Error for Html2TextPanic {}

/// All sections extracted from a 10-K filing, keyed by normalized item name.
///
/// Keys use the form `"item_N"` or `"item_NA"` (lowercase, underscore separator):
/// `"item_1"`, `"item_1a"`, `"item_7"`, `"item_7a"`, etc.
///
/// Use [`get`](TenKSections::get) for direct key lookup, or the named
/// convenience accessors [`item1`](TenKSections::item1),
/// [`item7`](TenKSections::item7), etc.
#[derive(Debug, Clone, Default)]
pub struct TenKSections(HashMap<String, String>);

impl TenKSections {
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    /// Retrieve a section by normalized key (e.g. `"item_1"`, `"item_7"`).
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(|s| s.as_str())
    }

    /// All section keys present (in unspecified order).
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(|s| s.as_str())
    }

    /// Iterate over all `(key, text)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    // ── Named convenience accessors ───────────────────────────────────────────

    /// Item 1 — Business
    pub fn item1(&self) -> Option<&str> {
        self.get("item_1")
    }
    /// Item 1A — Risk Factors
    pub fn item1a(&self) -> Option<&str> {
        self.get("item_1a")
    }
    /// Item 2 — Properties
    pub fn item2(&self) -> Option<&str> {
        self.get("item_2")
    }
    /// Item 3 — Legal Proceedings
    pub fn item3(&self) -> Option<&str> {
        self.get("item_3")
    }
    /// Item 7 — Management's Discussion and Analysis
    pub fn item7(&self) -> Option<&str> {
        self.get("item_7")
    }
    /// Item 7A — Quantitative and Qualitative Disclosures About Market Risk
    pub fn item7a(&self) -> Option<&str> {
        self.get("item_7a")
    }
    /// Item 8 — Financial Statements
    pub fn item8(&self) -> Option<&str> {
        self.get("item_8")
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// True when Item 1 and Item 7 are both present and substantial.
    ///
    /// Uses [`MIN_ITEM7_CHARS`] as the floor for Item 7, not the general
    /// [`MIN_SECTION_CHARS`].  Real MD&A bodies are always longer than
    /// `MIN_ITEM7_CHARS`; incorporate-by-reference stubs are typically
    /// 500–1 500 chars and would fail this check, causing the network layer
    /// to keep searching the filing index for a better document.
    pub fn is_adequate(&self) -> bool {
        self.item1().map_or(0, |s| s.len()) >= MIN_SECTION_CHARS
            && self.item7().map_or(0, |s| s.len()) >= MIN_ITEM7_CHARS
    }

    /// For each key, keep whichever value is longer.
    pub(crate) fn merge_with(&mut self, other: Self) {
        for (key, val) in other.0 {
            let entry = self.0.entry(key).or_default();
            if val.len() > entry.len() {
                *entry = val;
            }
        }
    }
}

// ── Public parser entry point ─────────────────────────────────────────────────

/// Parses a 10-K document (HTML, inline iXBRL, or SGML/plain-text) and
/// extracts all numbered sections.
///
/// **Strategy** (ported from [edgar-crawler](https://github.com/nlpaueb/edgar-crawler)):
///
/// 1. **HTML** — convert to plain text with `html2text` (proper entity
///    decoding, block-level newlines, no brittle regexes).
/// 2. **SGML / plain text** — strip tags with a lightweight regex then feed
///    straight into step 3.
/// 3. Normalise the plain text (fix span-shattered "I T E M" headers, remove
///    page-number lines, collapse whitespace).
/// 4. Find all `^ITEM N[.*: ]` anchors in the clean text.
/// 5. For each item, collect every `(start, end)` pair using the ordered item
///    list then pick the **longest** span — the same max-gap trick used by
///    edgar-crawler to skip TOC entries.
/// 6. Positions are tracked monotonically so later items must begin after all
///    earlier items, preventing TOC ordering from poisoning body extraction.
/// 7. If Item 7 is too short (incorporates-by-reference), run a fallback
///    search for the standalone "MANAGEMENT'S DISCUSSION AND ANALYSIS"
///    heading and extract the prose that follows it.
pub fn extract_sections_from_document(raw: &str) -> Result<TenKSections, Html2TextPanic> {
    let text = html_to_plain(raw)?;
    let cleaned = clean_plain_text(&text);
    let mut sections = extract_items(&cleaned);
    apply_item7_fallback(&cleaned, &mut sections);
    Ok(sections)
}

// ── HTML / SGML → plain text ──────────────────────────────────────────────────

/// Convert raw document bytes to a single plain-text string.
///
/// - **HTML / iXBRL** — `html2text` handles entity decoding, block-level line
///   breaks, table layout, and tag stripping uniformly across every EDGAR era.
///   We strip `<ix:header>` blocks first because they embed machine-readable
///   XBRL manifests that appear as dense character noise in plain text.
/// - **SGML / plain-text bundles** — simple tag-strip regex (html2text would
///   mis-parse the SGML meta-tags as HTML content).
///   Calls `html2text` and returns `Err(Html2TextPanic)` if it panics on a
///   malformed document.
///
/// html2text has a known panic bug (byte-index out of bounds in
/// text_renderer.rs) on certain EDGAR documents.
fn html2text_to_plain(html: &str) -> Result<String, Html2TextPanic> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        html2text::config::plain_no_decorate()
            .string_from_read(html.as_bytes(), 1_000_000)
            .unwrap_or_default()
    }))
    .map_err(|_| Html2TextPanic)
}

fn html_to_plain(raw: &str) -> Result<String, Html2TextPanic> {
    let peek = raw[..raw.len().min(2048)].to_ascii_lowercase();
    let is_html = peek.contains("<!doctype")
        || peek.contains("<html")
        || peek.contains("<head")
        || peek.contains("<body");

    if is_html {
        // Remove iXBRL header blocks before handing off to html2text; they
        // contain dense machine-readable metadata with no prose value.
        let no_xbrl = IX_HEADER_RE.replace_all(raw, "").into_owned();
        // Width 1_000_000 prevents html2text from wrapping long lines, which
        // would break the `^` anchors used by the item-heading regex.
        html2text_to_plain(&no_xbrl)
    } else {
        // SGML era: strip every tag with a single O(n) regex; entities are
        // left as-is (they don't affect item-heading detection).
        Ok(TAG_STRIP_RE.replace_all(raw, " ").into_owned())
    }
}

// ── Plain-text normalisation ──────────────────────────────────────────────────

/// Normalise a plain-text document before item extraction.
///
/// Mirrors the pre-processing steps from edgar-crawler's `clean_text`:
/// - Fix spaced-out `I T E M   7  A` headers (span-per-character markup).
/// - Remove bare page-number lines.
/// - Collapse runs of blank lines.
fn clean_plain_text(text: &str) -> String {
    // Fix spaced-out item headers: "I T E M   7  A" → "ITEM 7A"
    // Capture groups: (leading_ws)(I T E M)(sep)(\d{1,2}[A-C]?)
    let fixed = SPACED_ITEM_RE.replace_all(text, |caps: &regex::Captures| {
        let ws = &caps[1];
        let digits_and_suffix = caps[4].replace([' ', '\t'], "");
        format!("{}ITEM {}", ws, digits_and_suffix)
    });

    // Remove standalone page-number lines.
    let no_pages = PAGE_NUM_RE.replace_all(&fixed, "");

    // Collapse 3+ consecutive blank lines to 2.
    MULTI_NL_RE.replace_all(&no_pages, "\n\n").into_owned()
}

// ── Item extraction ───────────────────────────────────────────────────────────

/// Extract all standard 10-K items from already-cleaned plain text.
///
/// Algorithm (edgar-crawler `parse_item` + `get_item_section`):
///
/// 1. For each item in `ITEM_LIST`, find all line-anchored heading matches.
/// 2. For each next-item candidate, build all `(start, end)` pairs and pick
///    the one with the **longest** span.
/// 3. The match must start at or after `last_end` (monotone ordering) so that
///    the TOC — which always appears before the body — cannot win over the
///    real section.
fn extract_items(text: &str) -> TenKSections {
    // Pre-compile all item regexes once.
    let patterns: Vec<(TenKItem, Regex)> = TenKItem::iter()
        .map(|item| (item, item.item_pattern()))
        .collect();

    let mut map: HashMap<String, String> = HashMap::new();
    // `last_end` tracks the byte offset where the previous section ended.
    // Every new section must start at or after this point.
    let mut last_end: usize = 0;

    for (idx, (item, start_re)) in patterns.iter().enumerate() {
        let next_patterns: Vec<&Regex> = patterns[idx + 1..].iter().map(|(_, r)| r).collect();

        if next_patterns.is_empty() {
            // Last item in the list — extract from the first heading after
            // `last_end` to end-of-document.
            if let Some(m) = start_re.find_at(text, last_end) {
                let content = text[m.start()..].trim().to_string();
                if content.len() >= MIN_SECTION_CHARS {
                    map.insert(item.map_key(), content);
                }
            }
            continue;
        }

        // Collect all start-match positions for this item.
        let starts: Vec<usize> = start_re
            .find_iter(text)
            .map(|m| m.start())
            .filter(|&s| s >= last_end)
            .collect();

        if starts.is_empty() {
            continue;
        }

        // Outer loop: try next-item types in order.
        // For the FIRST next-item type that produces ANY (start, end) pair,
        // collect ALL such pairs and pick the longest span among them, then
        // break.  This mirrors edgar-crawler's `if possible_sections_list:
        // break`: we stop looking at further next-item types as soon as the
        // current one yields candidates, but we check EVERY start position
        // before deciding there are no candidates.
        //
        // WRONG order (old): outer=starts, inner=next_patterns — breaks on the
        // first start (TOC entry) before reaching the longer body section.
        // CORRECT order: outer=next_patterns, inner=starts — exhausts all
        // starts per next-item type so the longest span always wins.
        let mut best_start: usize = 0;
        let mut best_end: usize = 0;
        let mut best_len: usize = 0;

        'next_item: for next_re in &next_patterns {
            for &st in &starts {
                if let Some(end_m) = next_re.find_at(text, st + 1) {
                    let span = end_m.start() - st;
                    if span > best_len {
                        best_len = span;
                        best_start = st;
                        best_end = end_m.start();
                    }
                }
            }
            // Once this next-item type produced at least one candidate, stop
            // trying further next-item types (edgar-crawler behaviour).
            if best_len > 0 {
                break 'next_item;
            }
        }

        if best_len == 0 {
            continue;
        }

        let content = text[best_start..best_end].trim().to_string();
        if content.len() >= MIN_SECTION_CHARS {
            map.insert(item.map_key(), content);
            last_end = best_end;
        }
    }

    TenKSections(map)
}

// ── Item-7 fallback ───────────────────────────────────────────────────────────

/// Some filings (e.g. XOM 2006, JPM 2022 Part II) satisfy the SEC requirement
/// by printing a one-sentence "incorporated by reference" stub under the
/// `ITEM 7` heading and embedding the actual MD&A text elsewhere under a
/// standalone "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION…"
/// heading.
///
/// When the extracted Item 7 is shorter than `MIN_ITEM7_CHARS`, scan the text
/// for that standalone heading and use the largest run of prose that follows
/// it (bounded by the next financial-statement section heading).
fn apply_item7_fallback(text: &str, sections: &mut TenKSections) {
    if sections.item7().map_or(0, |s| s.len()) >= MIN_ITEM7_CHARS {
        return;
    }

    let mut best: Option<String> = None;
    let mut best_len: usize = 0;

    for mda_match in MDA_TITLE_RE.find_iter(text) {
        let start = mda_match.start();
        let end = MDA_END_RE
            .find_at(text, start + mda_match.len())
            .map(|m| m.start())
            .unwrap_or(text.len());

        let content = text[start..end].trim().to_string();
        if content.len() > best_len {
            best_len = content.len();
            best = Some(content);
        }
    }

    if best_len >= MIN_ITEM7_CHARS
        && let Some(content) = best
    {
        sections
            .0
            .entry("item_7".to_string())
            .and_modify(|v| {
                if content.len() > v.len() {
                    *v = content.clone();
                }
            })
            .or_insert(content);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── TenKSections unit tests ──────────────────────────────────────────────

    #[test]
    fn empty_sections_has_no_keys() {
        let s = TenKSections::empty();
        assert!(s.keys().next().is_none());
        assert!(s.iter().next().is_none());
    }

    #[test]
    fn get_returns_section_content() {
        let mut map = HashMap::new();
        map.insert("item_1".to_string(), "Business Description".to_string());
        let s = TenKSections(map);
        assert_eq!(s.get("item_1"), Some("Business Description"));
        assert_eq!(s.get("nonexistent"), None);
    }

    #[test]
    fn item_accessors_return_correct_sections() {
        let mut map = HashMap::new();
        map.insert("item_1".to_string(), "Business".to_string());
        map.insert("item_1a".to_string(), "Risk Factors".to_string());
        map.insert("item_2".to_string(), "Properties".to_string());
        map.insert("item_3".to_string(), "Legal".to_string());
        map.insert("item_7".to_string(), "MD&A".to_string());
        map.insert("item_7a".to_string(), "Market Risk".to_string());
        map.insert("item_8".to_string(), "Financials".to_string());
        let s = TenKSections(map);
        assert_eq!(s.item1(), Some("Business"));
        assert_eq!(s.item1a(), Some("Risk Factors"));
        assert_eq!(s.item2(), Some("Properties"));
        assert_eq!(s.item3(), Some("Legal"));
        assert_eq!(s.item7(), Some("MD&A"));
        assert_eq!(s.item7a(), Some("Market Risk"));
        assert_eq!(s.item8(), Some("Financials"));
    }

    #[test]
    fn item_accessors_return_none_when_missing() {
        let s = TenKSections::empty();
        assert!(s.item1().is_none());
        assert!(s.item1a().is_none());
        assert!(s.item2().is_none());
        assert!(s.item3().is_none());
        assert!(s.item7().is_none());
        assert!(s.item7a().is_none());
        assert!(s.item8().is_none());
    }

    #[test]
    fn is_adequate_requires_both_item1_and_item7() {
        let mut map = HashMap::new();
        map.insert("item_1".to_string(), "x".repeat(400));
        map.insert("item_7".to_string(), "y".repeat(2000));
        let s = TenKSections(map);
        assert!(s.is_adequate());
    }

    #[test]
    fn is_adequate_false_when_item1_too_short() {
        let mut map = HashMap::new();
        map.insert("item_1".to_string(), "short".to_string());
        map.insert("item_7".to_string(), "y".repeat(2000));
        let s = TenKSections(map);
        assert!(!s.is_adequate());
    }

    #[test]
    fn is_adequate_false_when_item7_too_short() {
        let mut map = HashMap::new();
        map.insert("item_1".to_string(), "x".repeat(400));
        map.insert("item_7".to_string(), "short".to_string());
        let s = TenKSections(map);
        assert!(!s.is_adequate());
    }

    #[test]
    fn is_adequate_false_when_both_missing() {
        assert!(!TenKSections::empty().is_adequate());
    }

    #[test]
    fn merge_with_keeps_longer_content() {
        let mut map1 = HashMap::new();
        map1.insert("item_1".to_string(), "short".to_string());
        let mut s1 = TenKSections(map1);
        let mut map2 = HashMap::new();
        map2.insert("item_1".to_string(), "longer content here".to_string());
        let s2 = TenKSections(map2);
        s1.merge_with(s2);
        assert_eq!(s1.item1(), Some("longer content here"));
    }

    #[test]
    fn merge_with_adds_new_keys() {
        let mut s1 = TenKSections::empty();
        let mut map2 = HashMap::new();
        map2.insert("item_7".to_string(), "MD&A".to_string());
        let s2 = TenKSections(map2);
        s1.merge_with(s2);
        assert_eq!(s1.item7(), Some("MD&A"));
    }

    #[test]
    fn keys_returns_all_keys() {
        let mut map = HashMap::new();
        map.insert("item_1".to_string(), "a".to_string());
        map.insert("item_7".to_string(), "b".to_string());
        let s = TenKSections(map);
        let mut keys: Vec<&str> = s.keys().collect();
        keys.sort();
        assert_eq!(keys, vec!["item_1", "item_7"]);
    }

    #[test]
    fn iter_returns_key_value_pairs() {
        let mut map = HashMap::new();
        map.insert("item_1".to_string(), "Business".to_string());
        let s = TenKSections(map);
        let pairs: Vec<(&str, &str)> = s.iter().collect();
        assert_eq!(pairs, vec![("item_1", "Business")]);
    }

    #[test]
    fn empty_sections_debug_format() {
        let s = TenKSections::empty();
        let debug = format!("{:?}", s);
        assert!(debug.contains("{}") || debug.contains("TenKSections"));
    }

    // ── Integration tests via SGML path (avoids html2text) ──────────────────

    fn parse_sgml(raw: &str) -> Result<TenKSections, Html2TextPanic> {
        extract_sections_from_document(raw)
    }

    #[test]
    fn sgml_basic_item_extraction() {
        let raw = format!(
            "ITEM 1. BUSINESS\n{}\nITEM 7. MANAGEMENT'S DISCUSSION\n{}\nITEM 8. FINANCIAL STATEMENTS\n{}\n",
            "Our business is great. ".repeat(50),
            "Analysis and discussion. ".repeat(300),
            "Financials. ".repeat(50),
        );
        let s = parse_sgml(&raw).unwrap();
        assert!(s.item1().unwrap().contains("business"));
        assert!(s.item7().unwrap().contains("Analysis"));
        assert!(s.is_adequate());
    }

    #[test]
    fn sgml_spaced_item_header_normalized() {
        let raw = format!(
            "I T E M   1.  BUSINESS\n{}\nI T E M   7.  MANAGEMENT'S DISCUSSION\n{}\nI T E M   8.  FINANCIAL STATEMENTS\n{}\n",
            "Spaced business. ".repeat(50),
            "Spaced analysis. ".repeat(300),
            "Financials. ".repeat(50),
        );
        let s = parse_sgml(&raw).unwrap();
        assert!(s.item1().unwrap().contains("Spaced"));
        assert!(s.item7().unwrap().contains("Spaced"));
    }

    #[test]
    fn sgml_page_numbers_removed() {
        let raw = format!(
            "ITEM 1. BUSINESS\n{}\n  - 42 -  \nITEM 1A. RISK FACTORS\n{}\n1234\nITEM 7. MANAGEMENT'S DISCUSSION\n{}\nITEM 8. FINANCIAL STATEMENTS\n{}\n",
            "Content. ".repeat(50),
            "Risks. ".repeat(58),
            "MD&A. ".repeat(300),
            "Financials. ".repeat(50),
        );
        let s = parse_sgml(&raw).unwrap();
        assert!(!s.item1().unwrap().contains("42"));
        assert!(s.item1a().unwrap().contains("Risks"));
        assert!(s.item7().unwrap().contains("MD&A"));
    }

    #[test]
    fn sgml_short_section_discarded() {
        let raw = format!(
            "ITEM 1. BUSINESS\nshort\nITEM 7. MANAGEMENT'S DISCUSSION\n{}\nITEM 8. FINANCIAL STATEMENTS\n{}\n",
            "MD&A content. ".repeat(300),
            "Financials. ".repeat(50),
        );
        let s = parse_sgml(&raw).unwrap();
        assert!(s.item1().is_none(), "Short section should be discarded");
        assert!(s.item7().is_some());
    }

    #[test]
    fn sgml_multiple_item_types() {
        let raw = format!(
            "ITEM 1. BUSINESS\n{}\nITEM 2. PROPERTIES\n{}\nITEM 3. LEGAL PROCEEDINGS\n{}\nITEM 7. MANAGEMENT'S DISCUSSION\n{}\nITEM 8. FINANCIAL STATEMENTS\n{}\n",
            "Business. ".repeat(50),
            "Properties. ".repeat(50),
            "Legal. ".repeat(58),
            "MD&A. ".repeat(300),
            "Financials. ".repeat(50),
        );
        let s = parse_sgml(&raw).unwrap();
        assert!(s.item1().is_some());
        assert!(s.item2().is_some());
        assert!(s.item3().is_some());
        assert!(s.item7().is_some());
    }

    #[test]
    fn sgml_item7_fallback_via_mda_title() {
        let raw = format!(
            "ITEM 1. BUSINESS\n{}\nITEM 7. MANAGEMENT'S DISCUSSION\nshort stub\nMANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION\n{}\nITEM 8. FINANCIAL STATEMENTS\n{}\n",
            "Business. ".repeat(50),
            "Detailed MD&A content here. ".repeat(300),
            "Financials. ".repeat(50),
        );
        let s = parse_sgml(&raw).unwrap();
        let item7 = s.item7().unwrap();
        assert!(
            item7.contains("Detailed MD&A"),
            "Fallback should find MDA heading"
        );
        assert!(item7.len() >= 2000);
    }

    #[test]
    fn html2text_panic_display_and_error_trait() {
        let err = Html2TextPanic;
        assert_eq!(err.to_string(), "html2text panicked on malformed document");
        let boxed: Box<dyn std::error::Error> = Box::new(Html2TextPanic);
        assert_eq!(
            boxed.to_string(),
            "html2text panicked on malformed document"
        );
    }

    #[test]
    fn tag_strip_regex_works_on_sgml_tags() {
        let input = "<PAGE>1</PAGE>\n<CAPTION>Table</CAPTION>\nText here";
        let result = TAG_STRIP_RE.replace_all(input, " ");
        assert_eq!(result, " 1 \n Table \nText here");
    }
}
