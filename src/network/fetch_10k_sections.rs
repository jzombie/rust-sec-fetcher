use crate::enums::Url;
use crate::models::{Cik, CikSubmission};
use crate::network::{SecClient, fetch_cik_submissions, fetch_filing_index};
use regex::Regex;
use std::error::Error;
use std::sync::LazyLock as Lazy;

// ── Section boundary regexes ──────────────────────────────────────────────────
// Start regexes are multiline-anchored ((?m)^) so they only match when
// "Item N" appears at the beginning of a line (modulo leading whitespace).
// This is the critical guard against forward cross-references in the body
// text — e.g. "see Part II, Item 7 of this Form 10-K" — that otherwise
// create spuriously large gaps and defeat the max-gap strategy.
//
// End regexes are NOT anchored: they only need to locate the nearest following
// section boundary, and headings are the only things that can be both on a
// fresh line AND match tightly enough for max-gap to prefer them.

// Item 1 start — anchored to line start.
static ITEM1_START_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?mi)^[ \t]*Item\s*1[.\s]").unwrap());
// Item 1 end — "Item 1A" (post-2004) or "Item 2" (pre-2004, before the SEC
// required a separate Risk Factors section).  "nearest end" semantics in
// extract_section_from_text guarantee that whichever comes first is used.
static ITEM1_END_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?mi)^[ \t]*Item\s*1\s*A\b|^[ \t]*Item\s*2\b").unwrap());

// Item 7 start — anchored to line start.
static ITEM7_START_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?mi)^[ \t]*Item\s*7[.\s]").unwrap());
// Item 7 end — "Item 7A" or "Item 8".  Item 7A was optional before ~2000
// but Item 8 (Financial Statements) is universal across all eras.
static ITEM7_END_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?mi)^[ \t]*Item\s*7\s*A\b|^[ \t]*Item\s*8\b").unwrap());

// Non-prose file extensions — skip these in the filing index fallback.
static BINARY_EXT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\.(jpg|jpeg|png|gif|pdf|xlsx|zip|xsd|xml|js|css)$").unwrap()
});

// SGML structural tags common in pre-2000 EDGAR submissions.
// Stripped but their *text content* is kept.
static SGML_TAG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<[^>]{0,400}>").unwrap());

// A section shorter than this is a TOC stub, not a real body.
const MIN_SECTION_CHARS: usize = 400;

// ── Public types ─────────────────────────────────────────────────────────────

/// The full, untruncated text of Item 1 and Item 7 extracted from a 10-K.
///
/// Both fields are `None` when the section cannot be located in any document
/// of the filing.  All text is plain UTF-8 without HTML markup.
#[derive(Debug, Clone)]
pub struct TenKSections {
    /// **Item 1 — Business.**  The legally-mandated description of the company's
    /// products, brands, markets, distribution channels, and competitive position.
    pub item1: Option<String>,

    /// **Item 7 — Management's Discussion and Analysis.**  Management's narrative
    /// on financial results, supply chain costs, input prices, consumer demand
    /// trends, and forward-looking commentary for the fiscal year.
    pub item7: Option<String>,
}

impl TenKSections {
    fn empty() -> Self {
        Self {
            item1: None,
            item7: None,
        }
    }

    /// Both sections have enough text to be real section bodies (not TOC stubs).
    fn is_adequate(&self) -> bool {
        self.item1.as_ref().map_or(0, |s| s.len()) >= MIN_SECTION_CHARS
            && self.item7.as_ref().map_or(0, |s| s.len()) >= MIN_SECTION_CHARS
    }

    /// Replace each field with `other`'s value whenever `other` is longer.
    fn merge_with(&mut self, other: Self) {
        if other.item1.as_ref().map_or(0, |s| s.len())
            > self.item1.as_ref().map_or(0, |s| s.len())
        {
            self.item1 = other.item1;
        }
        if other.item7.as_ref().map_or(0, |s| s.len())
            > self.item7.as_ref().map_or(0, |s| s.len())
        {
            self.item7 = other.item7;
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Fetches and extracts the full text of **Item 1** (Business) and **Item 7**
/// (MD&A) from the most recent 10-K filing for the given [`Cik`].
///
/// Both sections are returned **without truncation**.
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
/// if let Some(item1) = sections.item1 {
///     println!("=== Item 1 ({} chars) ===\n{item1}", item1.len());
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

/// Extracts Item 1 and Item 7 from a **specific** 10-K filing you already hold.
///
/// Use this when you want to iterate over historical filings rather than just
/// the most recent one.  Obtain the list of all filings with
/// [`fetch_10k_filings`], pick the one you want, then pass it here.
///
/// [`fetch_10k_filings`]: crate::network::fetch_10k_filings
///
/// # Strategy
///
/// 1. Try the filing's `primary_document`.
/// 2. If neither section is found (or both are TOC stubs < 400 chars), fetch
///    the full filing index and try every candidate document in priority order:
///    typed 10-K documents first, then `.htm` files, then `.txt` files.
/// 3. Each candidate's best result is merged in; the first pass that makes both
///    sections adequate terminates the search.
///
/// This handles:
/// - **iXBRL wrapper filings** (2010s+) where `primary_document` is a shell
///   and the narrative body lives in a separate `.htm` file.
/// - **Legacy SGML / plain-text filings** (pre-2000s) where the full 10-K is
///   a single `.txt` submission and `primary_document` points to a header stub.
/// - **HTML entity encoding** — headings like `ITEM&nbsp;7.` are normalised
///   by converting the whole document to plain text before section matching.
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
///     if let Some(item1) = sections.item1 {
///         println!("=== {} Item 1 ({} chars) ===\n{item1}", date, item1.len());
///     }
/// }
/// # Ok(()) }
/// ```
pub async fn fetch_10k_sections_for_filing(
    sec_client: &SecClient,
    filing: &CikSubmission,
) -> Result<TenKSections, Box<dyn Error>> {
    // Pre-2000 EDGAR filings have an empty `primary_document` field in the
    // submissions JSON.  The full SGML bundle is accessible at:
    //   Archives/edgar/data/{CIK}/{accession_formatted}.txt
    // (at the CIK root, NOT inside an accession subdirectory).
    // Modern filings have a named primary document inside the accession dir.
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
    // Handles:
    //   • iXBRL/viewer-wrapper filings where the narrative is in a separate .htm
    //   • Multi-document old bundles where pass 1 only fetched a header stub
    // Fetch the index.  If that call fails, return whatever we already have.
    let index = match fetch_filing_index(sec_client, filing).await {
        Ok(idx) => idx,
        Err(_) => return Ok(best),
    };

    // Build a candidate list in priority order:
    //   Tier 0 — explicitly typed "10-K" / "10-K405" / "10-K/A" documents
    //   Tier 1 — any .htm / .html document
    //   Tier 2 — any .txt document
    // Primary document is skipped (already tried).  Binary assets are skipped.
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
        // Old-format filings: individual document files live at the CIK root,
        // not inside an accession subdirectory.  Use EdgarArchive for them.
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

/// Converts the document to plain text **once** and extracts both sections.
///
/// Two paths depending on document format:
///
/// 1. **HTML** (`<html`, `<!DOCTYPE`, `<head`, `<body` detected anywhere in
///    the first kilobyte) — processed with `html2text`, which resolves HTML
///    entities (`&nbsp;`, `&#160;`, ...), strips tags, and normalises
///    whitespace.  This is the correct path for all filings from ~2000 onward.
///
/// 2. **SGML / plain text** — pre-2000 EDGAR submissions are single large
///    `.txt` bundles wrapping the 10-K in SGML structural tags
///    (`<SEC-DOCUMENT>`, `<DOCUMENT>`, `<TYPE>`, `<TEXT>`, `<PAGE>`, ...).
///    html5ever (used internally by html2text) loses the body content for
///    these files because it places everything before an implicit `<body>`
///    into `<head>`.  Instead we do a lightweight regex tag-strip that
///    preserves every text node verbatim.
fn extract_sections_from_document(raw: &str) -> TenKSections {
    let peek = &raw[..raw.len().min(2048)].to_ascii_lowercase();
    let is_html = peek.contains("<!doctype")
        || peek.contains("<html")
        || peek.contains("<head")
        || peek.contains("<body");

    let text: std::borrow::Cow<str> = if is_html {
        let rendered = html2text::config::plain_no_decorate()
            .string_from_read(raw.as_bytes(), 1_000_000)
            .unwrap_or_default();
        std::borrow::Cow::Owned(rendered)
    } else {
        // Lightweight SGML tag strip: remove every <...> sequence and replace
        // with a single space so adjacent tokens stay separated.
        std::borrow::Cow::Owned(SGML_TAG_RE.replace_all(raw, " ").into_owned())
    };

    TenKSections {
        item1: extract_section_from_text(&text, &ITEM1_START_RE, &ITEM1_END_RE),
        item7: extract_section_from_text(&text, &ITEM7_START_RE, &ITEM7_END_RE),
    }
}

/// Extracts the full, untruncated text of one section from a plain-text
/// document using the max-gap strategy.
///
/// Every `(start_marker, nearest_end_marker)` pair is scored by character
/// distance and the widest gap wins.  TOC pairs are always tiny (< 200 chars);
/// real section bodies span tens-to-hundreds of thousands of characters.
fn extract_section_from_text(text: &str, start_re: &Regex, end_re: &Regex) -> Option<String> {
    let starts: Vec<usize> = start_re.find_iter(text).map(|m| m.start()).collect();
    let ends: Vec<usize> = end_re.find_iter(text).map(|m| m.start()).collect();

    if starts.is_empty() || ends.is_empty() {
        return None;
    }

    let (best_start, best_end) = starts
        .iter()
        .filter_map(|&s| {
            let e = ends.iter().find(|&&pos| pos > s)?;
            Some((s, *e))
        })
        .max_by_key(|(s, e)| e - s)?;

    let section = text[best_start..best_end].trim().to_string();

    if section.len() < 40 {
        return None;
    }

    Some(section)
}
