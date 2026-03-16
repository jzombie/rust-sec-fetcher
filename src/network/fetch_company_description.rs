use crate::enums::Url;
use crate::models::{Cik, CikSubmission};
use crate::network::{fetch_cik_submissions, SecClient};
use once_cell::sync::Lazy;
use regex::Regex;
use std::error::Error;

static ITEM1_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)Item\s*1[.\s]").unwrap());
static ITEM1A_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)Item\s*1\s*A\b").unwrap());

/// Fetches a plain-text business description for the given company from the
/// "Item 1. Business" section of its most recent 10-K filing.
///
/// All three HTTP calls (`submissions/CIK*.json`, the 10-K primary document)
/// are served from the local cache on subsequent runs.
///
/// Returns `None` when no 10-K filing is found in the submission history or
/// when the Item 1 section cannot be located inside the document.
///
/// # Example
///
/// ```no_run
/// # use sec_fetcher::network::{SecClient, fetch_cik_by_ticker_symbol, fetch_company_description};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&cfg)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, "AAPL").await?;
/// if let Some(desc) = fetch_company_description(&client, cik).await? {
///     println!("{desc}");
/// }
/// # Ok(()) }
/// ```
pub async fn fetch_company_description(
    sec_client: &SecClient,
    cik: Cik,
) -> Result<Option<String>, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(sec_client, cik.clone()).await?;

    let filing = match CikSubmission::most_recent_10k(&submissions) {
        Some(f) => f,
        None => return Ok(None),
    };

    let url = Url::CikAccessionDocument(
        cik,
        filing.accession_number.clone(),
        filing.primary_document.clone(),
    )
    .value();

    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;

    if !response.status().is_success() {
        return Err(format!(
            "10-K document returned HTTP {} for {}",
            response.status(),
            url
        )
        .into());
    }

    let html = response.text().await?;
    Ok(parse_item1_business(&html))
}

/// Extracts the first substantive paragraph of the "Item 1. Business" section.
///
/// Strategy: collect all (Item 1, nearest following Item 1A) pairs and take
/// the pair with the largest HTML byte gap between them. The table-of-contents
/// entry has a tiny gap (~100 bytes); the real section spans tens of thousands
/// of bytes, making it unambiguous.
///
/// `html2text` handles entity decoding, tag stripping, and whitespace
/// normalisation. Short heading lines at the start are skipped; the result is
/// truncated at a sentence boundary near 800 characters.
fn parse_item1_business(html: &str) -> Option<String> {
    let item1_positions: Vec<usize> = ITEM1_RE.find_iter(html).map(|m| m.start()).collect();
    let item1a_positions: Vec<usize> = ITEM1A_RE.find_iter(html).map(|m| m.start()).collect();

    if item1a_positions.is_empty() || item1_positions.is_empty() {
        return None;
    }

    // For each "Item 1" occurrence find the nearest "Item 1A" that follows it,
    // then pick the pair whose gap is the largest — that is the real section.
    let (best_start, best_end) = item1_positions
        .iter()
        .filter_map(|&start| {
            let end = item1a_positions.iter().find(|&&pos| pos > start)?;
            Some((start, *end))
        })
        .max_by_key(|(start, end)| end - start)?;

    let section_html = &html[best_start..best_end.min(best_start + 60_000)];

    // html2text handles entity decoding, tag stripping, and whitespace.
    // Width 1_000_000 prevents line-wrapping artifacts.
    let text = html2text::config::plain()
        .string_from_read(section_html.as_bytes(), 1_000_000)
        .ok()?;

    // Skip short lines at the start ("Item 1. Business", sub-headings, etc.).
    // The first line with ≥ 60 chars is the start of actual prose.
    let prose: String = text
        .lines()
        .skip_while(|l| l.trim().len() < 60)
        .take_while(|l| {
            let t = l.trim();
            !t.starts_with("Item ") && !t.starts_with("ITEM ")
        })
        .collect::<Vec<_>>()
        .join(" ");

    if prose.len() < 40 {
        return None;
    }

    // Truncate at ~800 chars on the last sentence boundary that fits.
    if prose.len() <= 800 {
        Some(prose.trim().to_string())
    } else {
        let window = &prose[..800];
        if let Some(pos) = window.rfind(". ") {
            Some(window[..pos + 1].to_string())
        } else {
            Some(format!("{}\u{2026}", window.trim_end()))
        }
    }
}


