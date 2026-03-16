use crate::enums::Url;
use crate::models::{Cik, FeedEntry};
use crate::network::SecClient;
use chrono::{DateTime, FixedOffset, NaiveDate};
use futures::future::join_all;
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
/// e.g. "NPORT-P - Crossmark ETF Trust (0002062986) (Filer)" → "Crossmark ETF Trust"
/// Uses `.*?` (lazy) so it stops at the FIRST " - " separator, correctly
/// handling form types that contain hyphens (8-K, 8-K/A, NPORT-P, SC 13G, …).
static COMPANY_FROM_TITLE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^.*? - (.+?)\s*\(\d+\)").unwrap());

/// Extracts the filing date from the decoded summary text.
/// e.g. "Filed: 2026-03-13 AccNo: ..." → "2026-03-13"
static FILED_DATE: Lazy<Regex> = Lazy::new(|| Regex::new(r"Filed:\s*(\d{4}-\d{2}-\d{2})").unwrap());

/// Extracts 8-K item codes from the decoded summary text.
/// Matches both "Item 1.01:" and legacy "Item 5:" (pre-2004 integer form).
static ITEM_CODE: Lazy<Regex> = Lazy::new(|| Regex::new(r"Item (\d+(?:\.\d+)?):").unwrap());

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

        let updated = DateTime::parse_from_rfc3339(&self.updated).ok()?;

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
            updated,
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
/// # What is the EDGAR Atom feed?
///
/// EDGAR's real-time Atom feed (`https://efts.sec.gov/LATEST/search-index?q=...`)
/// delivers newly accepted filings within seconds of EDGAR processing them,
/// making it the fastest way to detect new disclosures without polling
/// individual company submission endpoints.  The feed is updated continuously
/// throughout the trading day and covers all filing types from all registrants.
///
/// Key properties of the feed:
/// - **Not paginated in the traditional sense** — each request returns the
///   most recent `count` entries (max 40) as of the moment of the request.
/// - **No guarantee of exactly-once delivery** — amendments, corrections, or
///   EDGAR reprocessing may cause a filing to reappear.
/// - The `updated` timestamp on each entry is the SEC's acceptance time, which
///   is the authoritative ordering key for building a high-water-mark delta.
/// - Use `form_type = ""` to receive all form types (full firehose), or narrow
///   to a specific form (e.g. `"8-K"`, `"NPORT-P"`).
///
/// For historical back-fill use [`fetch_edgar_master_index`] instead.
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
///
/// [`fetch_edgar_master_index`]: crate::network::fetch_edgar_master_index
pub async fn fetch_edgar_feed(
    client: &SecClient,
    form_type: &str,
    count: usize,
) -> Result<Vec<FeedEntry>, Box<dyn Error>> {
    fetch_edgar_feed_page(client, form_type, count, "").await
}

/// Fetches a page of the EDGAR Atom feed starting *before* the given
/// acceptance timestamp, enabling backward pagination.
///
/// EDGAR caps `count` at 40 entries per request. To walk backwards through
/// older filings, take the `updated` field of the **oldest** (last) entry in
/// the current batch, convert it to `"YYYYMMDDHHmmss"` format, and pass it
/// as `before` on the next call.
///
/// # Example (walking backwards, two pages)
///
/// ```no_run
/// # use sec_fetcher::network::{SecClient, fetch_edgar_feed, fetch_edgar_feed_page};
/// # async fn example(client: &SecClient) -> Result<(), Box<dyn std::error::Error>> {
/// // Page 1 — most recent 40 entries
/// let page1 = fetch_edgar_feed(client, "", 40).await?;
/// let oldest = page1.last().map(|e| e.updated_as_dateb()).unwrap_or_default();
///
/// // Page 2 — 40 entries before the oldest entry of page 1
/// let page2 = fetch_edgar_feed_page(client, "", 40, &oldest).await?;
/// # Ok(())
/// # }
/// ```
pub async fn fetch_edgar_feed_page(
    client: &SecClient,
    form_type: &str,
    count: usize,
    before: &str,
) -> Result<Vec<FeedEntry>, Box<dyn Error>> {
    let url = Url::EdgarCurrentFeed {
        form_type: form_type.to_string(),
        count,
        before: before.to_string(),
    }
    .value();

    let response = client
        .raw_request_live(reqwest::Method::GET, &url, None)
        .await?;

    let xml = response.text().await?;
    parse_edgar_atom_feed(&xml)
}

// ---------------------------------------------------------------------------
// Multi-page fetch + delta
// ---------------------------------------------------------------------------

/// The result of a [`fetch_edgar_feed_since`] call.
///
/// `entries` contains only the filings that are *strictly newer* than the
/// `since` timestamp you passed in — i.e. the new work since your last poll.
///
/// `high_water` is the timestamp of the absolute newest entry the feed
/// returned, regardless of the `since` filter. Save this value and pass it as
/// `since` on the next call — you will then receive only filings that arrived
/// after that point, with no gaps and no duplicates.
#[derive(Debug)]
pub struct FeedDelta {
    /// New entries, newest-first. Empty when nothing has been filed since the
    /// supplied mark.
    pub entries: Vec<FeedEntry>,
    /// Timestamp of the newest entry across all fetched pages. Use this as
    /// `since` on your next call to receive only what is filed after this
    /// moment. `None` only when the feed returned no entries at all.
    pub high_water: Option<DateTime<FixedOffset>>,
}

/// EDGAR caps every feed response at this many entries per page.
pub const EDGAR_PAGE_SIZE: usize = 40;

/// Fetches pages of the EDGAR Atom feed backwards in time until `since` is
/// covered, then returns only the entries that are strictly newer than `since`.
///
/// **How pagination works:**
/// The EDGAR feed always serves the *newest* entries first. This function
/// fetches page 1 (the 40 most recent), then page 2 (the next 40 older), and
/// so on. It stops as soon as the oldest entry on a page is at or before
/// `since`, because at that point everything newer is already in memory.
///
/// **Parameters:**
/// - `form_type` — e.g. `"8-K"`. Pass `""` for the combined firehose.
/// - `since` — the `high_water` from your previous [`FeedDelta`]. Pass `None`
///   on the very first call to receive everything (up to `max_pages` pages).
/// - `max_pages` — hard upper bound on requests. Use this to limit cost when
///   catching up after a long gap. Each page is [`EDGAR_PAGE_SIZE`] entries.
pub async fn fetch_edgar_feed_since(
    client: &SecClient,
    form_type: &str,
    since: Option<DateTime<FixedOffset>>,
    max_pages: usize,
) -> Result<FeedDelta, Box<dyn Error>> {
    let max_pages = max_pages.max(1);
    let mut all: Vec<FeedEntry> = Vec::new();
    let mut before = String::new();

    for _ in 0..max_pages {
        let page = fetch_edgar_feed_page(client, form_type, EDGAR_PAGE_SIZE, &before).await?;
        if page.is_empty() {
            break;
        }
        let covered = since
            .and_then(|s| page.last().map(|e| e.updated <= s))
            .unwrap_or(false);
        before = page
            .last()
            .map(|e| e.updated_as_dateb())
            .unwrap_or_default();
        all.extend(page);
        if covered {
            break;
        }
    }

    // High-water is the newest entry seen, before any delta filter.
    let high_water = all.first().map(|e| e.updated);

    // Keep only entries strictly newer than `since`.
    let entries = match since {
        None => all,
        Some(s) => all.into_iter().take_while(|e| e.updated > s).collect(),
    };

    Ok(FeedDelta {
        entries,
        high_water,
    })
}

/// Fetches multiple form types in parallel and merges the results into a
/// single [`FeedDelta`] sorted newest-first.
///
/// This is the main entry point for polling. It fans out one
/// [`fetch_edgar_feed_since`] call per form type concurrently, then merges
/// and re-sorts all results so the caller sees a single unified stream.
///
/// - `form_types` — e.g. `&["8-K", "NPORT-P"]`. Pass `&[""]` for the
///   combined firehose with no type filter.
/// - `since` — pass the `high_water` from the previous [`FeedDelta`], or
///   `None` to start fresh.
/// - `max_pages` — forwarded to each per-type fetch; acts as a per-type cap.
///
/// The returned `high_water` is the maximum timestamp seen across all types.
/// Pass it as `since` on the next call to pick up exactly where you left off.
pub async fn fetch_edgar_feeds_since(
    client: &SecClient,
    form_types: &[&str],
    since: Option<DateTime<FixedOffset>>,
    max_pages: usize,
) -> Result<FeedDelta, Box<dyn std::error::Error>> {
    let handles: Vec<_> = form_types
        .iter()
        .map(|&ft| fetch_edgar_feed_since(client, ft, since, max_pages))
        .collect();

    let results = join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;

    let mut entries: Vec<FeedEntry> = results
        .iter()
        .flat_map(|d| d.entries.iter().cloned())
        .collect();
    entries.sort_by(|a, b| b.updated.cmp(&a.updated));

    let high_water = results.iter().filter_map(|d| d.high_water).max();

    Ok(FeedDelta {
        entries,
        high_water,
    })
}
