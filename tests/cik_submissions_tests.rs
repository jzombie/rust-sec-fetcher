// Fixture: real download from SEC EDGAR
//   Apple Inc. (CIK 0000320193) — AAPL_submissions.json
//   Snapshot as of March 2026: 1006 recent filings
//
// Assertions are anchored to known accession numbers, not positional indices,
// so they remain correct as new filings are added to the live EDGAR feed.

use chrono::NaiveDate;
use sec_fetcher::models::{Cik, CikSubmission};
use sec_fetcher::network::parse_cik_submissions_json;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn load_fixture(name: &str) -> Value {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(name);
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
    serde_json::from_str(&raw).expect("fixture is not valid JSON")
}

fn aapl_cik() -> Cik {
    Cik::from_u64(320193).unwrap()
}

fn aapl_submissions() -> Vec<CikSubmission> {
    let data = load_fixture("AAPL_submissions.json");
    parse_cik_submissions_json(&data, aapl_cik())
}

/// Find a submission by its exact accession number or panic with a clear message.
fn by_accession<'a>(subs: &'a [CikSubmission], acc: &str) -> &'a CikSubmission {
    subs.iter()
        .find(|s| s.accession_number.to_string() == acc)
        .unwrap_or_else(|| panic!("accession {} not found in fixture", acc))
}

// ── count and ordering ────────────────────────────────────────────────────────

#[test]
fn aapl_has_over_1000_recent_submissions() {
    // The March 2026 snapshot had 1006 filings; use a lower bound so
    // re-downloading a newer snapshot never breaks this test.
    assert!(aapl_submissions().len() >= 1000);
}

#[test]
fn submissions_are_newest_first() {
    let subs = aapl_submissions();
    // First entry must be strictly newer than the last.
    assert!(subs[0].filing_date > subs[subs.len() - 1].filing_date);
}

// ── field parsing ─────────────────────────────────────────────────────────────

#[test]
fn entity_type_propagated_to_every_submission() {
    let subs = aapl_submissions();
    assert!(subs
        .iter()
        .all(|s| s.entity_type.as_deref() == Some("operating")));
}

#[test]
fn cik_propagated_to_every_submission() {
    let subs = aapl_submissions();
    assert!(subs.iter().all(|s| s.cik.to_string() == "0000320193"));
}

#[test]
fn filing_date_parses_correctly() {
    let subs = aapl_submissions();
    // FY2025 10-K — real accession confirmed on EDGAR
    let ten_k = by_accession(&subs, "0000320193-25-000079");
    assert_eq!(ten_k.filing_date, NaiveDate::from_ymd_opt(2025, 10, 31));
}

#[test]
fn primary_document_parses_correctly() {
    let subs = aapl_submissions();
    let ten_k = by_accession(&subs, "0000320193-25-000079");
    assert_eq!(ten_k.primary_document, "aapl-20250927.htm");
}

#[test]
fn accession_number_round_trips() {
    let subs = aapl_submissions();
    // FY2025 10-K
    let ten_k = by_accession(&subs, "0000320193-25-000079");
    assert_eq!(ten_k.accession_number.to_string(), "0000320193-25-000079");
}

#[test]
fn items_split_correctly_on_comma() {
    let subs = aapl_submissions();
    // Jan 29 2026 earnings 8-K: raw items field is "2.02,9.01"
    let s = by_accession(&subs, "0000320193-26-000005");
    assert_eq!(s.items, vec!["2.02", "9.01"]);
}

#[test]
fn items_empty_for_10k() {
    let subs = aapl_submissions();
    let ten_k = by_accession(&subs, "0000320193-25-000079");
    assert!(ten_k.items.is_empty());
}

// ── form filtering ─────────────────────────────────────────────────────────────

#[test]
fn by_form_8k_returns_many() {
    let subs = aapl_submissions();
    // 106 in the March 2026 snapshot; use a stable lower bound.
    assert!(CikSubmission::by_form(&subs, "8-K").len() >= 50);
}

#[test]
fn by_form_10k_returns_multiple() {
    let subs = aapl_submissions();
    // 11 in the March 2026 snapshot.
    assert!(CikSubmission::by_form(&subs, "10-K").len() >= 5);
}

#[test]
fn by_form_10q_returns_multiple() {
    let subs = aapl_submissions();
    // 33 in the March 2026 snapshot.
    assert!(CikSubmission::by_form(&subs, "10-Q").len() >= 10);
}

#[test]
fn most_recent_10k_is_fy2025() {
    let subs = aapl_submissions();
    // AAPL FY ends in September; FY2025 10-K filed 2025-10-31.
    let ten_k = CikSubmission::most_recent_10k(&subs).unwrap();
    assert_eq!(ten_k.accession_number.to_string(), "0000320193-25-000079");
    assert_eq!(ten_k.filing_date, NaiveDate::from_ymd_opt(2025, 10, 31));
}

// ── 8-K semantic helpers ──────────────────────────────────────────────────────

#[test]
fn earnings_release_detected_by_items_202() {
    let subs = aapl_submissions();
    // Jan 29 2026 8-K, items "2.02,9.01" — earnings release.
    let s = by_accession(&subs, "0000320193-26-000005");
    assert!(s.is_earnings_release());
}

#[test]
fn non_earnings_8k_is_not_earnings_release() {
    let subs = aapl_submissions();
    // Jan 2 2026 8-K, items "5.02" (officer change) — not an earnings release.
    let s = by_accession(&subs, "0001140361-26-000199");
    assert!(!s.is_earnings_release());
}

#[test]
fn non_earnings_8k_is_mid_quarter_event() {
    let subs = aapl_submissions();
    // Jan 2 2026 8-K, items "5.02" (officer change).
    let s = by_accession(&subs, "0001140361-26-000199");
    assert!(s.is_mid_quarter_event());
}
