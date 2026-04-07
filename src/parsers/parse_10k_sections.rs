use regex::Regex;
use scraper::{Html, Node};
use std::collections::HashMap;
use std::sync::LazyLock as Lazy;

// ── Heading detection ─────────────────────────────────────────────────────────
//
// A heading token starts with "Item" (case-insensitive), optional whitespace
// including &nbsp; (\u00A0), then the item designator (digits + optional
// trailing letter: `1`, `1A`, `7`, `7A`, `15T`), then a separator.
//
// Separator: any non-alphanumeric character OR end-of-string.
// This guards against partial matches: "Item 10." matches Item 10, not Item 1.

// Matches any item heading. Capture group 1 = raw designator (e.g. "7A").
static ANY_ITEM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^item[\s\u{00A0}]*(\d+[A-Za-z]?)[\s\u{00A0}]*(?:[^0-9A-Za-z]|$)").unwrap()
});

// SGML tag strip for pre-2000 plain-text bundles.
static SGML_TAG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<[^>]{0,400}>").unwrap());

// Sections shorter than this are TOC stubs rather than real bodies.
const MIN_SECTION_CHARS: usize = 400;

// ── Public types ──────────────────────────────────────────────────────────────

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
    pub fn item1(&self) -> Option<&str> { self.get("item_1") }
    /// Item 1A — Risk Factors
    pub fn item1a(&self) -> Option<&str> { self.get("item_1a") }
    /// Item 2 — Properties
    pub fn item2(&self) -> Option<&str> { self.get("item_2") }
    /// Item 3 — Legal Proceedings
    pub fn item3(&self) -> Option<&str> { self.get("item_3") }
    /// Item 7 — Management's Discussion and Analysis
    pub fn item7(&self) -> Option<&str> { self.get("item_7") }
    /// Item 7A — Quantitative and Qualitative Disclosures About Market Risk
    pub fn item7a(&self) -> Option<&str> { self.get("item_7a") }
    /// Item 8 — Financial Statements
    pub fn item8(&self) -> Option<&str> { self.get("item_8") }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// True when Item 1 and Item 7 are both present and substantial.
    pub(crate) fn is_adequate(&self) -> bool {
        self.item1().map_or(0, |s| s.len()) >= MIN_SECTION_CHARS
            && self.item7().map_or(0, |s| s.len()) >= MIN_SECTION_CHARS
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

/// Parses a 10-K document and extracts all its numbered sections.
///
/// - **HTML** — uses `scraper` to walk DOM text nodes directly, avoiding the
///   table-rendering artifacts that defeat line-start regex anchors.
/// - **SGML / plain text** — strips tags with a lightweight regex, then
///   treats each line as a token.
pub fn extract_sections_from_document(raw: &str) -> TenKSections {
    let peek = &raw[..raw.len().min(2048)].to_ascii_lowercase();
    let is_html = peek.contains("<!doctype")
        || peek.contains("<html")
        || peek.contains("<head")
        || peek.contains("<body");

    if is_html {
        extract_sections_from_html(raw)
    } else {
        let stripped = SGML_TAG_RE.replace_all(raw, " ");
        extract_sections_from_plain(&stripped)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Normalize a raw item designator to a map key: `"7A"` → `"item_7a"`.
fn normalize_key(raw_designator: &str) -> String {
    format!("item_{}", raw_designator.to_ascii_lowercase())
}

/// Sort key for item designators: numeric part first, then letter suffix.
/// Ensures `"9"` < `"10"` < `"9A"` < `"10A"`.
fn sort_key(designator: &str) -> (u32, String) {
    let digits: String = designator.chars().take_while(|c| c.is_ascii_digit()).collect();
    let suffix: String = designator.chars().skip_while(|c| c.is_ascii_digit()).collect();
    (digits.parse::<u32>().unwrap_or(0), suffix.to_ascii_lowercase())
}

/// Walk `tokens` and return all extracted sections.
///
/// 1. Find every token that is a heading (matched by `ANY_ITEM_RE`).
/// 2. For each unique designator, apply the max-gap strategy: try every
///    `(start_token, end_token)` pair where the end token is any heading with
///    a strictly greater sort key. The widest gap wins — TOC pairs are always
///    tiny; real bodies span tens of thousands of chars.
/// 3. Discard sections shorter than `MIN_SECTION_CHARS`.
fn extract_all_sections(tokens: &[String], offsets: &[usize], text: &str) -> TenKSections {
    // Map each token index to its designator if it's a heading.
    let heading_positions: Vec<(usize, String)> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            ANY_ITEM_RE
                .captures(t)
                .map(|c| (i, c.get(1).unwrap().as_str().to_ascii_uppercase()))
        })
        .collect();

    if heading_positions.is_empty() {
        return TenKSections::empty();
    }

    // Unique start designators, sorted numerically.
    let mut start_designators: Vec<String> = heading_positions
        .iter()
        .map(|(_, d)| d.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    start_designators.sort_by_key(|d| sort_key(d));

    let mut map: HashMap<String, String> = HashMap::new();

    for start_des in &start_designators {
        let start_sort = sort_key(start_des);

        let start_tokens: Vec<usize> = heading_positions
            .iter()
            .filter(|(_, d)| d == start_des)
            .map(|(i, _)| *i)
            .collect();

        let end_tokens: Vec<usize> = heading_positions
            .iter()
            .filter(|(_, d)| sort_key(d) > start_sort)
            .map(|(i, _)| *i)
            .collect();

        if end_tokens.is_empty() {
            continue;
        }

        let best = start_tokens
            .iter()
            .flat_map(|&si| {
                end_tokens
                    .iter()
                    .filter(move |&&ei| ei > si)
                    .map(move |&ei| (si, ei))
            })
            .max_by_key(|&(si, ei)| offsets[ei].saturating_sub(offsets[si]));

        if let Some((si, ei)) = best {
            let content = text[offsets[si]..offsets[ei]].trim().to_string();
            if content.len() >= MIN_SECTION_CHARS {
                map.insert(normalize_key(start_des), content);
            }
        }
    }

    TenKSections(map)
}

/// HTML path — uses `scraper` to collect DOM text nodes in document order.
fn extract_sections_from_html(raw: &str) -> TenKSections {
    let document = Html::parse_document(raw);

    let mut tokens: Vec<String> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    let mut text = String::new();

    'outer: for node_ref in document.tree.nodes() {
        if let Node::Text(text_node) = node_ref.value() {
            let raw_text = text_node.text.as_ref();
            // Skip content inside <script> and <style>.
            let mut ancestor = node_ref.parent();
            while let Some(a) = ancestor {
                if let Node::Element(el) = a.value() {
                    let name = el.name();
                    if name == "script" || name == "style" {
                        continue 'outer;
                    }
                }
                ancestor = a.parent();
            }
            let trimmed = raw_text.trim();
            if trimmed.is_empty() {
                continue;
            }
            let offset = if text.is_empty() { 0 } else { text.len() + 1 };
            offsets.push(offset);
            tokens.push(trimmed.to_string());
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(trimmed);
        }
    }

    extract_all_sections(&tokens, &offsets, &text)
}

/// SGML/plain-text path — `plain` has already had tags stripped.
///
/// Lines are tokens so that a full heading line `"Item 1. Business"` is one
/// unit and matches the heading regex as a whole.
fn extract_sections_from_plain(plain: &str) -> TenKSections {
    let tokens: Vec<String> = plain
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();
    let mut offsets: Vec<usize> = Vec::new();
    let mut text = String::new();
    for token in &tokens {
        let offset = if text.is_empty() { 0 } else { text.len() + 1 };
        offsets.push(offset);
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(token);
    }
    extract_all_sections(&tokens, &offsets, &text)
}
