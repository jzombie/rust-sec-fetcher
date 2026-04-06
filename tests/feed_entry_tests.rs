/// Unit tests for [`sec_fetcher::models::FeedEntry`].
use chrono::DateTime;
use sec_fetcher::models::FeedEntry;

fn make_entry(form_type: &str, items: Vec<&str>) -> FeedEntry {
    FeedEntry {
        accession_number: "0001104659-26-027766".to_string(),
        cik: None,
        company_name: "Test Corp".to_string(),
        form_type: form_type.to_string(),
        filing_date: None,
        filing_href: "https://www.sec.gov/cgi-bin/browse-edgar".to_string(),
        updated: DateTime::parse_from_rfc3339("2026-01-15T13:45:00+00:00").unwrap(),
        items: items.into_iter().map(|s| s.to_string()).collect(),
    }
}

// ── is_earnings_release ───────────────────────────────────────────────────────

#[test]
fn is_earnings_release_with_item_2_02() {
    let entry = make_entry("8-K", vec!["2.02", "9.01"]);
    assert!(entry.is_earnings_release());
}

#[test]
fn is_earnings_release_with_legacy_item_12() {
    let entry = make_entry("8-K", vec!["12"]);
    assert!(entry.is_earnings_release());
}

#[test]
fn not_earnings_release_without_matching_items() {
    let entry = make_entry("8-K", vec!["1.01", "9.01"]);
    assert!(!entry.is_earnings_release());
}

#[test]
fn not_earnings_release_when_no_items() {
    let entry = make_entry("8-K", vec![]);
    assert!(!entry.is_earnings_release());
}

#[test]
fn not_earnings_release_for_non_8k() {
    let entry = make_entry("10-K", vec!["2.02"]);
    // is_earnings_release checks items only, not form_type
    assert!(entry.is_earnings_release());
}

// ── is_mid_quarter_event ──────────────────────────────────────────────────────

#[test]
fn is_mid_quarter_event_for_8k_with_non_earnings_items() {
    let entry = make_entry("8-K", vec!["1.01", "9.01"]);
    assert!(entry.is_mid_quarter_event());
}

#[test]
fn not_mid_quarter_if_earnings_release() {
    let entry = make_entry("8-K", vec!["2.02", "9.01"]);
    assert!(!entry.is_mid_quarter_event());
}

#[test]
fn not_mid_quarter_if_not_8k() {
    let entry = make_entry("10-K", vec!["1.01"]);
    assert!(!entry.is_mid_quarter_event());
}

#[test]
fn not_mid_quarter_if_only_9_01_item() {
    // Only 9.01 (exhibits) — no meaningful item other than the exhibit attachment
    let entry = make_entry("8-K", vec!["9.01"]);
    assert!(!entry.is_mid_quarter_event());
}

#[test]
fn not_mid_quarter_if_no_items_at_all() {
    let entry = make_entry("8-K", vec![]);
    assert!(!entry.is_mid_quarter_event());
}

// ── updated_as_dateb ──────────────────────────────────────────────────────────

#[test]
fn updated_as_dateb_formats_correctly() {
    let entry = make_entry("8-K", vec![]);
    // updated = 2026-01-15T13:45:00+00:00
    assert_eq!(entry.updated_as_dateb(), "20260115134500");
}

#[test]
fn updated_as_dateb_different_time() {
    let mut entry = make_entry("8-K", vec![]);
    entry.updated = DateTime::parse_from_rfc3339("2025-03-07T09:05:30+00:00").unwrap();
    assert_eq!(entry.updated_as_dateb(), "20250307090530");
}
