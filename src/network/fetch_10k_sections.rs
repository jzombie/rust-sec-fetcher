use crate::enums::Url;
use crate::models::{Cik, CikSubmission};
use crate::network::{SecClient, fetch_cik_submissions, fetch_filing_index};
use regex::Regex;
use scraper::{Html, Node};
use std::collections::HashMap;
use std::error::Error;
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

// Non-prose file extensions — skip in the filing index fallback.
static BINARY_EXT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\.(jpg|jpeg|png|gif|pdf|xlsx|zip|xsd|xml|js|css)$").unwrap()
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
    fn empty() -> Self {
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
    fn is_adequate(&self) -> bool {
        self.item1().map_or(0, |s| s.len()) >= MIN_SECTION_CHARS
            && self.item7().map_or(0, |s| s.len()) >= MIN_SECTION_CHARS
    }

    /// For each key, keep whichever value is longer.
    fn merge_with(&mut self, other: Self) {
        for (key, val) in other.0 {
            let entry = self.0.entry(key).or_default();
            if val.len() > entry.len() {
                *entry = val;
            }
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Fetches and extracts **all sections** from the most recent 10-K filing for
/// the given [`Cik`].
///
/// Returns a [`TenKSections`] map keyed by normalized item name (`"item_1"`,
/// `"item_7"`, `"item_1a"`, …).  Use [`TenKSections::get`],
/// [`TenKSections::item7`], etc. to access individual sections.
///
/// # Example
///
/// ```no_run
/// # use sec_fetcher::network::{SecClient, fetch_cik_by_ticker_symbol, fetch_10k_sections};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&cfg)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
/// let sections = fetch_10k_sections(&client, cik).await?;
/// if let Some(text) = sections.item7() {
///     println!("=== Item 7 ({} chars) ===\n{text}", text.len());
/// }
/// # Ok(()) }
/// ```
pub async fn fetch_10k_sections(
    sec_client: &SecClient,
    cik: Cik,
) -> Result<TenKSections, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(sec_client, cik.clone()).await?;
    let filing = match CikSubmission::most_recent_10k(&submissions) {
        Some(f) => f,
        None => return Ok(TenKSections::empty()),
    };
    fetch_10k_sections_for_filing(sec_client, filing).await
}

/// Extracts all sections from a **specific** 10-K filing you already hold.
///
/// Use this when iterating over historical filings.  Obtain the list with
/// [`fetch_10k_filings`], pick the filing you want, then pass it here.
///
/// [`fetch_10k_filings`]: crate::network::fetch_10k_filings
///
/// # Strategy
///
/// 1. Try the primary document.
/// 2. If Item 1 and Item 7 are not yet substantial, fetch the filing index and
///    try candidate documents in priority order: typed 10-K docs first, then
///    `.htm` files, then `.txt` files.  Stops as soon as both core sections
///    are substantial.
///
/// # Example
///
/// ```no_run
/// # use sec_fetcher::network::{SecClient, fetch_cik_by_ticker_symbol, fetch_10k_filings,
/// #                             fetch_10k_sections_for_filing};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&cfg)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
/// let filings = fetch_10k_filings(&client, cik).await?;
/// for filing in &filings {
///     let date = filing.filing_date.map(|d| d.to_string()).unwrap_or_default();
///     let sections = fetch_10k_sections_for_filing(&client, filing).await?;
///     for (key, text) in sections.iter() {
///         println!("{} {}: {} chars", date, key, text.len());
///     }
/// }
/// # Ok(()) }
/// ```
pub async fn fetch_10k_sections_for_filing(
    sec_client: &SecClient,
    filing: &CikSubmission,
) -> Result<TenKSections, Box<dyn Error>> {
    let old_format = filing.primary_document.is_empty();

    // ── Pass 1: primary document ──────────────────────────────────────────────
    let primary_url = if old_format {
        Url::EdgarArchive(format!(
            "edgar/data/{}/{}.txt",
            filing.cik, filing.accession_number
        ))
        .value()
    } else {
        Url::CikAccessionDocument(
            filing.cik.clone(),
            filing.accession_number.clone(),
            filing.primary_document.clone(),
        )
        .value()
    };

    let mut best = fetch_sections_from_url(sec_client, &primary_url)
        .await
        .unwrap_or_else(|_| TenKSections::empty());

    if best.is_adequate() {
        return Ok(best);
    }

    // ── Pass 2: filing index fallback ─────────────────────────────────────────
    let index = match fetch_filing_index(sec_client, filing).await {
        Ok(idx) => idx,
        Err(_) => return Ok(best),
    };

    // Candidate priority:
    //   Tier 0 — explicitly typed "10-K" / "10-K405" / "10-K/A"
    //   Tier 1 — any .htm / .html
    //   Tier 2 — any .txt
    let mut candidates: Vec<&str> = Vec::new();

    for tier in 0..3u8 {
        for doc in &index.documents {
            let name = doc.name.as_str();
            if name == filing.primary_document.as_str() {
                continue;
            }
            if BINARY_EXT_RE.is_match(name) {
                continue;
            }
            let lower = name.to_ascii_lowercase();
            let doc_type = doc.document_type.to_ascii_uppercase();
            let matches_tier = match tier {
                0 => doc_type == "10-K" || doc_type == "10-K405" || doc_type == "10-K/A",
                1 => lower.ends_with(".htm") || lower.ends_with(".html"),
                2 => lower.ends_with(".txt"),
                _ => false,
            };
            if matches_tier && !candidates.contains(&name) {
                candidates.push(name);
            }
        }
    }

    for name in candidates {
        let url = if old_format {
            Url::EdgarArchive(format!("edgar/data/{}/{}", filing.cik, name)).value()
        } else {
            Url::CikAccessionDocument(
                filing.cik.clone(),
                filing.accession_number.clone(),
                name.to_string(),
            )
            .value()
        };

        if let Ok(sections) = fetch_sections_from_url(sec_client, &url).await {
            best.merge_with(sections);
        }

        if best.is_adequate() {
            break;
        }
    }

    Ok(best)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

async fn fetch_sections_from_url(
    sec_client: &SecClient,
    url: &str,
) -> Result<TenKSections, Box<dyn Error>> {
    let response = sec_client
        .raw_request(reqwest::Method::GET, url, None, None)
        .await?;

    if !response.status().is_success() {
        return Err(format!("HTTP {} for {}", response.status(), url).into());
    }

    let raw = response.text().await?;
    Ok(extract_sections_from_document(&raw))
}

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
///    `(start_token, nearest_end_token)` pair where the end token is any
///    heading with a strictly greater sort key.  The widest gap wins — TOC
///    pairs are always tiny; real bodies span tens of thousands of chars.
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

