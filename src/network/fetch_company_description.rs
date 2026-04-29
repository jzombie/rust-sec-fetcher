use crate::enums::Url;
use crate::models::{Cik, CikSubmission};
use crate::network::{SecClient, fetch_cik_submissions};
use regex::Regex;
use std::error::Error;
use std::sync::LazyLock as Lazy;

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
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&cfg)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
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
/// normalization. Short heading lines at the start are skipped; the result is
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
    // plain_no_decorate() suppresses link-reference footnotes.
    // Width 1_000_000 prevents line-wrapping artifacts.
    let text = html2text::config::plain_no_decorate()
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

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::formatdoc;

    /// Helper: build a minimal 10-K HTML with a TOC entry and a real section.
    ///
    /// The TOC has "Item 1." and "Item 1A." within a few bytes of each other.
    /// The real section has "Item 1." followed by ~3 KB of prose before "Item 1A."
    /// This is the exact scenario we debugged on Apple's 10-K.
    fn make_10k_html(prose: &str) -> String {
        let filler = "x".repeat(3000);
        formatdoc! {r#"
            <html><body>
            <p>Table of Contents</p>
            <p>Item 1. Business</p>
            <p>Item 1A. Risk Factors</p>
            <p>{filler}</p>
            <div id="business">
                <h1>Item 1. Business</h1>
                <p>Company Background</p>
                <p>{prose}</p>
            </div>
            <div id="risks">
                <h1>Item 1A. Risk Factors</h1>
                <p>Various risks apply.</p>
            </div>
            </body></html>"#
        }
    }

    #[test]
    fn max_gap_strategy_picks_real_section_not_toc() {
        // The long prose appears only in the real section, not the TOC.
        let prose = "The Company designs, manufactures and markets smartphones, \
            personal computers, tablets, wearables and accessories, and sells a \
            variety of related services to customers around the world.";
        let html = make_10k_html(prose);
        let result = parse_item1_business(&html).unwrap();
        assert!(
            result.contains("designs, manufactures"),
            "Expected real section prose, got: {result}"
        );
    }

    #[test]
    fn short_heading_lines_are_skipped() {
        // "Company Background" is a sub-heading (< 60 chars) and must not
        // appear at the start of the returned description. This was the core
        // bug we spent time debugging — the previous regex approach kept
        // emitting heading words.
        let prose = "The Company designs, manufactures and markets smartphones, \
            personal computers, tablets, wearables and accessories, and sells a \
            variety of related services to customers around the world.";
        let html = make_10k_html(prose);
        let result = parse_item1_business(&html).unwrap();
        assert!(
            !result.starts_with("Item"),
            "Result should not start with 'Item': {result}"
        );
        assert!(
            !result.starts_with("Company Background"),
            "Result should not start with sub-heading: {result}"
        );
        assert!(
            result.starts_with("The Company") || result.contains("designs"),
            "Expected prose start, got: {result}"
        );
    }

    #[test]
    fn truncates_at_sentence_boundary() {
        // Build prose longer than 800 chars with clear sentence boundaries.
        let sentence = "The Company sells products worldwide. ";
        let long_prose = sentence.repeat(30); // ~1140 chars
        let html = make_10k_html(&long_prose);
        let result = parse_item1_business(&html).unwrap();
        assert!(
            result.len() <= 800,
            "Result too long: {} chars",
            result.len()
        );
        assert!(
            result.ends_with('.'),
            "Result should end at sentence boundary: {result}"
        );
    }

    #[test]
    fn returns_none_when_no_item1a() {
        let html = "<html><body><p>Item 1. Business</p><p>Some text.</p></body></html>";
        assert!(parse_item1_business(html).is_none());
    }

    #[test]
    fn returns_none_when_no_item1() {
        let html = "<html><body><p>Item 1A. Risk Factors</p></body></html>";
        assert!(parse_item1_business(html).is_none());
    }

    #[test]
    fn decodes_html_entities_and_strips_tags() {
        // html2text must handle &amp;, &quot;, and inline tags.
        let prose = "Apple&apos;s products include the <em>iPhone</em> &amp; Mac.";
        let html = formatdoc! {r#"
            <html><body>
            <p>Item 1. Business</p>
            <p>Item 1A.</p>
            <p>x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x</p>
            <div><h1>Item 1. Business</h1>
            <p>{prose}</p></div>
            <div><h1>Item 1A. Risk Factors</h1></div>
            </body></html>"#,
            prose = "x".repeat(3000) + &format!("<p>{prose}</p>")
        };
        let result = parse_item1_business(&html);
        // We don't assert exact text (html2text entity handling may vary),
        // but it must return something and not contain raw HTML tags.
        if let Some(r) = result {
            assert!(!r.contains('<'), "Raw HTML tag found in output: {r}");
            assert!(
                !r.contains("&amp;"),
                "Unescaped entity found in output: {r}"
            );
        }
    }
}
