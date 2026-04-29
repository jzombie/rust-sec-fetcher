use crate::enums::Url;
use crate::models::{Cik, FeedEntry};
use crate::network::SecClient;
use chrono::{DateTime, FixedOffset, NaiveDate};
use futures::future::join_all;
use quick_xml::Reader;
use quick_xml::events::Event;
use regex::Regex;
use std::error::Error;
use std::sync::LazyLock as Lazy;

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
                        .find(|a| a.as_ref().is_ok_and(|a| a.key.as_ref() == b"href"))
                        && let Some(ref mut p) = current
                    {
                        p.link_href = attr?.unescape_value()?.to_string();
                    }
                }
                b"category" if current.is_some() => {
                    if let Some(attr) = e
                        .attributes()
                        .find(|a| a.as_ref().is_ok_and(|a| a.key.as_ref() == b"term"))
                        && let Some(ref mut p) = current
                        && p.form_type.is_empty()
                    {
                        p.form_type = attr?.unescape_value()?.to_string();
                    }
                }
                _ => {}
            },

            // Self-closing <link .../> and <category .../> come as Empty events.
            Event::Empty(ref e) => match e.name().as_ref() {
                b"link" if current.is_some() => {
                    if let Some(attr) = e
                        .attributes()
                        .find(|a| a.as_ref().is_ok_and(|a| a.key.as_ref() == b"href"))
                        && let Some(ref mut p) = current
                    {
                        p.link_href = attr?.unescape_value()?.to_string();
                    }
                }
                b"category" if current.is_some() => {
                    if let Some(attr) = e
                        .attributes()
                        .find(|a| a.as_ref().is_ok_and(|a| a.key.as_ref() == b"term"))
                        && let Some(ref mut p) = current
                        && p.form_type.is_empty()
                    {
                        p.form_type = attr?.unescape_value()?.to_string();
                    }
                }
                _ => {}
            },

            Event::Text(e) => {
                if let Some(p) = &mut current {
                    let text = e.decode()?.to_string();
                    match current_field {
                        Some("summary") => {
                            // Summary may span multiple text nodes if embedded HTML
                            // (e.g. <br/>) splits the content. Accumulate all
                            // fragments rather than overwriting with only the first.
                            p.summary.push_str(&text);
                        }
                        Some(field) => {
                            // These elements always contain a single text node, so
                            // consume the field tracker to ignore any stray text.
                            current_field = None;
                            match field {
                                "title" => p.title = text,
                                "updated" => p.updated = text,
                                "id" => p.id = text,
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }
            }

            Event::End(ref e) => {
                if e.name().as_ref() == b"entry" {
                    if let Some(p) = current.take()
                        && let Some(entry) = p.build()
                    {
                        entries.push(entry);
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
/// [`fetch_edgar_master_index`]: crate::network::fetch_edgar_master_index()
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

    // Ensure we get the latest data from EDGAR rather than a cached copy
    let response = client
        .raw_request_nocache(reqwest::Method::GET, &url, None)
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
    entries.sort_by_key(|b| std::cmp::Reverse(b.updated));

    let high_water = results.iter().filter_map(|d| d.high_water).max();

    Ok(FeedDelta {
        entries,
        high_water,
    })
}

// ---------------------------------------------------------------------------
// Tests for parse_edgar_atom_feed
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Build an Atom `<entry>` element (without the surrounding `<feed>`).
    fn entry_xml(
        title: &str,
        id: &str,
        updated: &str,
        link_href: &str,
        summary: &str,
        category_term: &str,
    ) -> String {
        // TODO: Use indoc! for formatting
        format!(
            r#"    <entry>
      <title>{title}</title>
      <id>{id}</id>
      <updated>{updated}</updated>
      <link href="{link_href}" rel="alternate" type="text/html"/>
      <category term="{category_term}" label="{category_term}"/>
      <summary>{summary}</summary>
    </entry>"#,
        )
    }

    /// Build a complete Atom feed XML with the given entry XML strings.
    fn feed_xml(entries: &[&str]) -> String {
        let body = entries.join("\n");
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
{body}
</feed>"#,
        )
    }

    /// Convenience: single-entry feed for a standard Apple 8-K.
    fn aapl_feed() -> String {
        // This test helper uses a plain-text summary for simplicity.  The
        // parser accumulates all Text nodes within <summary>, so it handles
        // both plain-text and HTML-containing summaries correctly.
        let entry = entry_xml(
            "8-K - Apple Inc. (0000320193) (Filer)",
            "urn:tag:sec.gov,2008:accession-number=0000320193-24-000001",
            "2024-06-15T10:30:00-04:00",
            "/Archives/edgar/data/320193/0000320193-24-000001-index.htm",
            "Filed: 2024-06-15 AccNo: 0000320193-24-000001 Item 2.02: Results Item 9.01: Financial Statements",
            "8-K",
        );
        feed_xml(&[&entry])
    }

    #[test]
    fn parse_empty_feed_returns_no_entries() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"#;
        let entries = parse_edgar_atom_feed(xml).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn parse_single_entry() {
        let xml = aapl_feed();
        let entries = parse_edgar_atom_feed(&xml).unwrap();
        assert_eq!(entries.len(), 1);

        let e = &entries[0];
        assert_eq!(e.accession_number, "0000320193-24-000001");
        assert_eq!(e.form_type, "8-K");
        assert_eq!(e.company_name, "Apple Inc.");
        assert_eq!(
            e.filing_href,
            "/Archives/edgar/data/320193/0000320193-24-000001-index.htm"
        );
        assert!(e.cik.is_some());
        assert_eq!(e.cik.as_ref().unwrap().value, 320193);
        assert_eq!(
            e.filing_date,
            Some(NaiveDate::from_ymd_opt(2024, 6, 15).unwrap())
        );
        assert_eq!(
            e.updated,
            DateTime::parse_from_rfc3339("2024-06-15T10:30:00-04:00").unwrap()
        );
        assert_eq!(e.items, vec!["2.02", "9.01"]);
    }

    #[test]
    fn summary_with_embedded_html_accumulates_all_text_nodes() {
        // Build an entry whose <summary> contains <br/> elements that would
        // split the text into multiple quick-xml Text events.  The parser
        // must accumulate them rather than capturing only the first fragment.
        // We verify through `items` (parsed from the accumulated summary) and
        // `filing_date` (parsed from the first fragment) to confirm both early
        // and late text nodes are correctly captured.
        let xml = feed_xml(&[&format!(
            r#"    <entry>
      <title>8-K - Apple Inc. (0000320193) (Filer)</title>
      <id>urn:tag:sec.gov,2008:accession-number=0000320193-24-000001</id>
      <updated>2024-06-15T10:30:00-04:00</updated>
      <link href="/Archives/edgar/data/320193/0000320193-24-000001-index.htm" rel="alternate" type="text/html"/>
      <category term="8-K" label="8-K"/>
      <summary>Filed: 2024-06-15 AccNo: 0000320193-24-000001<br/>Item 2.02: Results<br/>Item 9.01: Financial Statements</summary>
    </entry>"#,
        )]);
        let entries = parse_edgar_atom_feed(&xml).unwrap();
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.accession_number, "0000320193-24-000001");
        // `filing_date` is parsed from the first text fragment ("Filed: 2024-06-15...") —
        // this confirms the early text node is captured.
        assert_eq!(
            e.filing_date,
            Some(NaiveDate::from_ymd_opt(2024, 6, 15).unwrap())
        );
        // `items` are parsed from the later fragments ("Item 2.02:...", "Item 9.01:...") —
        // this confirms the text nodes after <br/> are also accumulated.
        assert_eq!(e.items, vec!["2.02", "9.01"]);
    }

    #[test]
    fn parse_multiple_entries() {
        let e1 = entry_xml(
            "10-Q - Apple Inc. (0000320193) (Filer)",
            "urn:tag:sec.gov,2008:accession-number=0000320193-24-000002",
            "2024-07-20T09:00:00-04:00",
            "/Archives/edgar/data/320193/0000320193-24-000002-index.htm",
            "Filed: 2024-07-20 AccNo: 0000320193-24-000002",
            "10-Q",
        );
        let e2 = entry_xml(
            "8-K - Microsoft Corporation (0000789019) (Filer)",
            "urn:tag:sec.gov,2008:accession-number=0000789019-24-000001",
            "2024-07-19T14:00:00-04:00",
            "/Archives/edgar/data/789019/0000789019-24-000001-index.htm",
            "Filed: 2024-07-19 AccNo: 0000789019-24-000001 Item 5.02: Departure Item 9.01: Financial Statements",
            "8-K",
        );
        let xml = feed_xml(&[&e1, &e2]);
        let entries = parse_edgar_atom_feed(&xml).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].company_name, "Apple Inc.");
        assert_eq!(entries[1].company_name, "Microsoft Corporation");
    }

    #[test]
    fn form_type_and_company_name_vary_with_feed_content() {
        let entry = entry_xml(
            "NPORT-P - Vanguard Index Funds (0001234567) (Filer)",
            "urn:tag:sec.gov,2008:accession-number=0001234567-24-000003",
            "2024-08-01T12:00:00-04:00",
            "/Archives/edgar/data/1234567/0001234567-24-000003-index.htm",
            "Filed: 2024-08-01 AccNo: 0001234567-24-000003",
            "NPORT-P",
        );
        let xml = feed_xml(&[&entry]);
        let entries = parse_edgar_atom_feed(&xml).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].form_type, "NPORT-P");
        assert_eq!(entries[0].company_name, "Vanguard Index Funds");
    }

    #[test]
    fn parses_pre_2004_legacy_item_format() {
        // Pre-2004 items use plain integers like "Item 5:" instead of "Item 5.02:"
        let entry = entry_xml(
            "8-K - Apple Inc. (0000320193) (Filer)",
            "urn:tag:sec.gov,2008:accession-number=0000320193-99-000001",
            "1999-03-15T11:00:00-05:00",
            "/Archives/edgar/data/320193/0000320193-99-000001-index.htm",
            "Filed: 1999-03-15 AccNo: 0000320193-99-000001 Item 5: Old format Item 7: Financials",
            "8-K",
        );
        let xml = feed_xml(&[&entry]);
        let entries = parse_edgar_atom_feed(&xml).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].items, vec!["5", "7"]);
    }

    #[test]
    fn handles_entry_without_items() {
        let entry = entry_xml(
            "10-K - Apple Inc. (0000320193) (Filer)",
            "urn:tag:sec.gov,2008:accession-number=0000320193-24-000004",
            "2024-10-31T16:00:00-04:00",
            "/Archives/edgar/data/320193/0000320193-24-000004-index.htm",
            "Filed: 2024-10-31 AccNo: 0000320193-24-000004",
            "10-K",
        );
        let xml = feed_xml(&[&entry]);
        let entries = parse_edgar_atom_feed(&xml).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].items.is_empty());
    }

    #[test]
    fn handles_entry_without_filing_date() {
        let entry = entry_xml(
            "4 - Cook Timothy D (0000320193) (Filer)",
            "urn:tag:sec.gov,2008:accession-number=0000320193-24-000005",
            "2024-11-01T09:00:00-04:00",
            "/Archives/edgar/data/320193/0000320193-24-000005-index.htm",
            "AccNo: 0000320193-24-000005", // No "Filed:" prefix
            "4",
        );
        let xml = feed_xml(&[&entry]);
        let entries = parse_edgar_atom_feed(&xml).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].filing_date.is_none());
    }

    #[test]
    fn fallback_company_name_when_regex_does_not_match() {
        // Title that doesn't match the COMPANY_FROM_TITLE regex
        let entry = entry_xml(
            "Some Nonstandard Title (No Filer)",
            "urn:tag:sec.gov,2008:accession-number=0000320193-24-000006",
            "2024-12-01T08:00:00-05:00",
            "/Archives/edgar/data/320193/0000320193-24-000006-index.htm",
            "Filed: 2024-12-01 AccNo: 0000320193-24-000006",
            "8-K",
        );
        let xml = feed_xml(&[&entry]);
        let entries = parse_edgar_atom_feed(&xml).unwrap();
        assert_eq!(entries.len(), 1);
        // Falls back to the full title string
        assert_eq!(entries[0].company_name, "Some Nonstandard Title (No Filer)");
    }

    #[test]
    fn missing_cik_when_url_has_no_archive_pattern() {
        // href without /Archives/edgar/data/NNNN/ pattern
        let entry = entry_xml(
            "8-K - Test Company (0000123456) (Filer)",
            "urn:tag:sec.gov,2008:accession-number=0000123456-24-000007",
            "2024-12-15T10:00:00-05:00",
            "https://efts.sec.gov/LATEST/some-other-path",
            "Filed: 2024-12-15 AccNo: 0000123456-24-000007",
            "8-K",
        );
        let xml = feed_xml(&[&entry]);
        let entries = parse_edgar_atom_feed(&xml).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0].cik.is_none(),
            "CIK should be None when URL lacks archive pattern"
        );
    }
}
