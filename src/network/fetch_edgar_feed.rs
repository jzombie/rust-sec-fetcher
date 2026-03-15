use crate::enums::Url;
use crate::models::{Cik, FeedEntry};
use crate::network::SecClient;
use chrono::NaiveDate;
use once_cell::sync::Lazy;
use quick_xml::events::Event;
use quick_xml::Reader;
use regex::Regex;
use std::error::Error;

// ---------------------------------------------------------------------------
// Compiled regexes
// ---------------------------------------------------------------------------

/// Extracts the CIK from an EDGAR archive URL.
/// e.g. "/Archives/edgar/data/1863990/..." → "1863990"
static CIK_FROM_URL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"/Archives/edgar/data/(\d+)/").unwrap());

/// Extracts company name from the Atom feed title.
/// e.g. "8-K - MultiSensor AI Holdings, Inc. (0001863990) (Filer)" → "MultiSensor AI Holdings, Inc."
static COMPANY_FROM_TITLE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[^-]+ - (.+?)\s*\(\d+\)").unwrap());

/// Extracts the filing date from the decoded summary text.
/// e.g. "Filed: 2026-03-13 AccNo: ..." → "2026-03-13"
static FILED_DATE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"Filed:\s*(\d{4}-\d{2}-\d{2})").unwrap());

/// Extracts 8-K item codes from the decoded summary text.
/// Matches both "Item 1.01:" and legacy "Item 5:" (pre-2004 integer form).
static ITEM_CODE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"Item (\d+(?:\.\d+)?):").unwrap());

/// Strips HTML tags from a string.
static HTML_TAG: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]*>").unwrap());

// ---------------------------------------------------------------------------
// Atom XML parser
// ---------------------------------------------------------------------------

/// Accumulates raw fields from one `<entry>` block before converting to a
/// `FeedEntry`. Using a plain struct avoids nested borrows with quick-xml.
#[derive(Default)]
struct PartialEntry {
    title: String,
    updated: String,
    id: String,
    summary: String,
    link_href: String,
    form_type: String,
}

impl PartialEntry {
    fn build(self) -> Option<FeedEntry> {
        // Accession number lives after "accession-number=" in the <id> URN:
        // "urn:tag:sec.gov,2008:accession-number=0001104659-26-027766"
        let accession_number = self
            .id
            .split("accession-number=")
            .nth(1)
            .unwrap_or("")
            .to_string();
        if accession_number.is_empty() {
            return None;
        }

        let cik = CIK_FROM_URL
            .captures(&self.link_href)
            .and_then(|c| c.get(1))
            .and_then(|m| m.as_str().parse::<u64>().ok())
            .and_then(|n| Cik::from_u64(n).ok());

        let company_name = COMPANY_FROM_TITLE
            .captures(&self.title)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_else(|| self.title.clone());

        // The summary contains HTML-encoded HTML. Unescape the XML entities
        // first (done by quick-xml's unescape()), then strip the HTML tags.
        let summary_plain = HTML_TAG.replace_all(&self.summary, " ").to_string();

        let filing_date = FILED_DATE
            .captures(&summary_plain)
            .and_then(|c| c.get(1))
            .and_then(|m| NaiveDate::parse_from_str(m.as_str(), "%Y-%m-%d").ok());

        let items: Vec<String> = ITEM_CODE
            .captures_iter(&summary_plain)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect();

        Some(FeedEntry {
            accession_number,
            cik,
            company_name,
            form_type: self.form_type,
            filing_date,
            filing_href: self.link_href,
            updated: self.updated,
            items,
        })
    }
}

/// Parses an EDGAR filing Atom feed XML string into a list of [`FeedEntry`] items.
///
/// Works for both the global "current filings" feed (firehose) and the
/// per-company feed. Both feeds share the same `<entry>` structure for the
/// fields this parser reads: `<title>`, `<id>`, `<link>`, `<updated>`,
/// `<summary>`, and `<category>`.
pub fn parse_edgar_atom_feed(xml: &str) -> Result<Vec<FeedEntry>, Box<dyn Error>> {
    let mut reader = Reader::from_str(xml);
    let mut entries: Vec<FeedEntry> = Vec::new();
    let mut current: Option<PartialEntry> = None;
    let mut current_field: Option<&'static str> = None;

    while let Ok(event) = reader.read_event() {
        match event {
            Event::Start(ref e) => match e.name().as_ref() {
                b"entry" => {
                    current = Some(PartialEntry::default());
                    current_field = None;
                }
                b"title" if current.is_some() => current_field = Some("title"),
                b"updated" if current.is_some() => current_field = Some("updated"),
                b"id" if current.is_some() => current_field = Some("id"),
                b"summary" if current.is_some() => current_field = Some("summary"),
                b"link" if current.is_some() => {
                    if let Some(attr) = e
                        .attributes()
                        .find(|a| a.as_ref().map_or(false, |a| a.key.as_ref() == b"href"))
                    {
                        if let Some(ref mut p) = current {
                            p.link_href = attr?.unescape_value()?.to_string();
                        }
                    }
                }
                b"category" if current.is_some() => {
                    if let Some(attr) = e
                        .attributes()
                        .find(|a| a.as_ref().map_or(false, |a| a.key.as_ref() == b"term"))
                    {
                        if let Some(ref mut p) = current {
                            if p.form_type.is_empty() {
                                p.form_type = attr?.unescape_value()?.to_string();
                            }
                        }
                    }
                }
                _ => {}
            },

            // Self-closing <link .../> and <category .../> come as Empty events.
            Event::Empty(ref e) => match e.name().as_ref() {
                b"link" if current.is_some() => {
                    if let Some(attr) = e
                        .attributes()
                        .find(|a| a.as_ref().map_or(false, |a| a.key.as_ref() == b"href"))
                    {
                        if let Some(ref mut p) = current {
                            p.link_href = attr?.unescape_value()?.to_string();
                        }
                    }
                }
                b"category" if current.is_some() => {
                    if let Some(attr) = e
                        .attributes()
                        .find(|a| a.as_ref().map_or(false, |a| a.key.as_ref() == b"term"))
                    {
                        if let Some(ref mut p) = current {
                            if p.form_type.is_empty() {
                                p.form_type = attr?.unescape_value()?.to_string();
                            }
                        }
                    }
                }
                _ => {}
            },

            Event::Text(ref e) => {
                if let (Some(field), Some(ref mut p)) = (current_field.take(), &mut current) {
                    let text = e.unescape()?.to_string();
                    match field {
                        "title" => p.title = text,
                        "updated" => p.updated = text,
                        "id" => p.id = text,
                        "summary" => p.summary = text,
                        _ => {}
                    }
                }
            }

            Event::End(ref e) => {
                if e.name().as_ref() == b"entry" {
                    if let Some(p) = current.take() {
                        if let Some(entry) = p.build() {
                            entries.push(entry);
                        }
                    }
                } else {
                    // Reset field tracker on any closing tag; leaf elements
                    // (title, id, etc.) have no nested children so this is safe.
                    current_field = None;
                }
            }

            Event::Eof => break,
            _ => {}
        }
    }

    Ok(entries)
}

// ---------------------------------------------------------------------------
// Public fetch function
// ---------------------------------------------------------------------------

/// Fetches and parses the EDGAR filing Atom feed, returning a list of
/// [`FeedEntry`] items ordered **newest-first**.
///
/// # Parameters
///
/// - `form_type` — SEC form type filter, e.g. `"8-K"`, `"10-K"`, `"4"`.
///   Pass `""` to receive **all** form types (the full firehose).
/// - `count` — Number of entries to return. EDGAR caps this at **40**.
///
/// # Delta / event-driven pipeline
///
/// ```text
/// first poll:
///   entries = fetch_edgar_feed(&client, "8-K", 40).await?
///   high_water = entries[0].updated          // newest entry
///   process all entries
///
/// subsequent polls:
///   entries = fetch_edgar_feed(&client, "8-K", 40).await?
///   for entry in entries {
///       if entry.updated <= high_water { break; }  // stop at known territory
///       process(entry);
///   }
///   high_water = entries[0].updated          // advance the mark
/// ```
///
/// This gives you exactly the delta — only filings accepted since the last poll.
pub async fn fetch_edgar_feed(
    client: &SecClient,
    form_type: &str,
    count: usize,
) -> Result<Vec<FeedEntry>, Box<dyn Error>> {
    let url = Url::EdgarCurrentFeed {
        form_type: form_type.to_string(),
        count,
    }
    .value();

    let response = client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;

    let xml = response.text().await?;
    parse_edgar_atom_feed(&xml)
}
