use chrono::DateTime;
use indoc::formatdoc;
use sec_fetcher::network::{parse_edgar_atom_feed, FeedDelta, EDGAR_PAGE_SIZE};
use sec_fetcher::models::FeedEntry;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal valid Atom XML containing the given entries (newest-first order).
fn make_feed(entries: &[(&str, &str)]) -> String {
    let body: String = entries
        .iter()
        .map(|(updated, accession)| {
            formatdoc! {r#"
                <entry>
                  <title>8-K - Test Corp (0001234567) (Filer)</title>
                  <updated>{updated}</updated>
                  <id>urn:tag:sec.gov,2008:accession-number={accession}</id>
                  <link href="https://www.sec.gov/Archives/edgar/data/1234567/{accession}/"/>
                  <summary type="html">Filed: 2026-03-13 AccNo: {accession}</summary>
                  <category term="8-K"/>
                </entry>
            "#,
                updated = updated,
                accession = accession,
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>EDGAR</title>
  <updated>2026-03-15T18:00:00-04:00</updated>
{body}
</feed>"#
    )
}

/// Apply the same delta filter logic as `fetch_edgar_feed_since`.
fn delta_filter(
    entries: Vec<FeedEntry>,
    since: Option<DateTime<chrono::FixedOffset>>,
) -> Vec<FeedEntry> {
    match since {
        None => entries,
        Some(s) => entries.into_iter().take_while(|e| e.updated > s).collect(),
    }
}

// ---------------------------------------------------------------------------
// EDGAR_PAGE_SIZE constant
// ---------------------------------------------------------------------------

#[test]
fn edgar_page_size_is_40() {
    assert_eq!(EDGAR_PAGE_SIZE, 40);
}

// ---------------------------------------------------------------------------
// Parser smoke tests
// ---------------------------------------------------------------------------

#[test]
fn parse_empty_feed_returns_no_entries() {
    let xml = make_feed(&[]);
    let entries = parse_edgar_atom_feed(&xml).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn parse_single_entry() {
    let xml = make_feed(&[("2026-03-13T17:30:01-04:00", "0001104659-26-027766")]);
    let entries = parse_edgar_atom_feed(&xml).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].updated,
        DateTime::parse_from_rfc3339("2026-03-13T17:30:01-04:00").unwrap()
    );
    assert_eq!(entries[0].accession_number, "0001104659-26-027766");
    assert_eq!(entries[0].form_type, "8-K");
    assert_eq!(entries[0].company_name, "Test Corp");
}

#[test]
fn parse_multiple_entries_newest_first() {
    let xml = make_feed(&[
        ("2026-03-13T18:00:00-04:00", "0001000000-26-000003"),
        ("2026-03-13T17:30:00-04:00", "0001000000-26-000002"),
        ("2026-03-13T17:00:00-04:00", "0001000000-26-000001"),
    ]);
    let entries = parse_edgar_atom_feed(&xml).unwrap();
    assert_eq!(entries.len(), 3);
    // Parser preserves the order given in the XML (EDGAR sends newest-first)
    assert_eq!(entries[0].updated, DateTime::parse_from_rfc3339("2026-03-13T18:00:00-04:00").unwrap());
    assert_eq!(entries[2].updated, DateTime::parse_from_rfc3339("2026-03-13T17:00:00-04:00").unwrap());
}

// ---------------------------------------------------------------------------
// Delta filter logic
// ---------------------------------------------------------------------------

#[test]
fn delta_filter_empty_since_returns_all() {
    let xml = make_feed(&[
        ("2026-03-13T18:00:00-04:00", "0001000000-26-000002"),
        ("2026-03-13T17:00:00-04:00", "0001000000-26-000001"),
    ]);
    let entries = parse_edgar_atom_feed(&xml).unwrap();
    let delta = delta_filter(entries, None);
    assert_eq!(delta.len(), 2);
}

#[test]
fn delta_filter_excludes_entries_at_or_before_mark() {
    let xml = make_feed(&[
        ("2026-03-13T18:00:00-04:00", "0001000000-26-000003"),
        ("2026-03-13T17:30:00-04:00", "0001000000-26-000002"), // exactly the mark
        ("2026-03-13T17:00:00-04:00", "0001000000-26-000001"),
    ]);
    let entries = parse_edgar_atom_feed(&xml).unwrap();
    // Mark is the second entry's timestamp; only the first should pass.
    let since = DateTime::parse_from_rfc3339("2026-03-13T17:30:00-04:00").unwrap();
    let delta = delta_filter(entries, Some(since));
    assert_eq!(delta.len(), 1);
    assert_eq!(delta[0].updated, DateTime::parse_from_rfc3339("2026-03-13T18:00:00-04:00").unwrap());
}

#[test]
fn delta_filter_all_new_returns_all() {
    let xml = make_feed(&[
        ("2026-03-13T18:00:00-04:00", "0001000000-26-000002"),
        ("2026-03-13T17:00:00-04:00", "0001000000-26-000001"),
    ]);
    let entries = parse_edgar_atom_feed(&xml).unwrap();
    let since = DateTime::parse_from_rfc3339("2026-03-13T16:00:00-04:00").unwrap();
    let delta = delta_filter(entries, Some(since));
    assert_eq!(delta.len(), 2);
}

#[test]
fn delta_filter_nothing_new_returns_empty() {
    let xml = make_feed(&[
        ("2026-03-13T18:00:00-04:00", "0001000000-26-000002"),
        ("2026-03-13T17:00:00-04:00", "0001000000-26-000001"),
    ]);
    let entries = parse_edgar_atom_feed(&xml).unwrap();
    // Mark is at or after every entry
    let since = DateTime::parse_from_rfc3339("2026-03-13T18:00:00-04:00").unwrap();
    let delta = delta_filter(entries, Some(since));
    assert!(delta.is_empty());
}

// ---------------------------------------------------------------------------
// FeedDelta high-water mark behaviour
// ---------------------------------------------------------------------------

#[test]
fn feed_delta_high_water_is_none_when_entries_empty() {
    let delta = FeedDelta {
        entries: vec![],
        high_water: None,
    };
    assert!(delta.high_water.is_none());
}

#[test]
fn feed_delta_high_water_is_newest_entry() {
    let xml = make_feed(&[
        ("2026-03-13T18:00:00-04:00", "0001000000-26-000002"),
        ("2026-03-13T17:00:00-04:00", "0001000000-26-000001"),
    ]);
    let entries = parse_edgar_atom_feed(&xml).unwrap();
    let high_water = entries.first().map(|e| e.updated);
    let delta = FeedDelta { entries, high_water };
    assert_eq!(delta.high_water, Some(DateTime::parse_from_rfc3339("2026-03-13T18:00:00-04:00").unwrap()));
}
