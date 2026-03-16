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
    let data = load_fixture("CIK0000320193_submissions.json");
    parse_cik_submissions_json(&data, aapl_cik())
}

// ── count and ordering ────────────────────────────────────────────────────────

#[test]
fn parses_all_six_submissions() {
    assert_eq!(aapl_submissions().len(), 6);
}

#[test]
fn submissions_are_newest_first() {
    let subs = aapl_submissions();
    // The Jan 2025 8-K should come before the Nov 2024 10-K
    assert!(subs[0].filing_date > subs[2].filing_date);
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
fn filing_dates_parse_correctly() {
    let subs = aapl_submissions();
    assert_eq!(subs[0].filing_date, NaiveDate::from_ymd_opt(2025, 1, 30));
    assert_eq!(subs[2].filing_date, NaiveDate::from_ymd_opt(2024, 11, 1));
    assert_eq!(subs[4].filing_date, NaiveDate::from_ymd_opt(2024, 8, 2));
}

#[test]
fn accession_number_round_trips() {
    let subs = aapl_submissions();
    // Formatted form: 0000320193-25-000008
    assert_eq!(subs[0].accession_number.to_string(), "0000320193-25-000008");
}

#[test]
fn items_split_correctly_on_comma() {
    let subs = aapl_submissions();
    // First 8-K: "2.02,9.01"
    let first_8k = subs.iter().find(|s| s.form == "8-K").unwrap();
    assert_eq!(first_8k.items, vec!["2.02", "9.01"]);
}

#[test]
fn items_empty_for_10k() {
    let subs = aapl_submissions();
    let ten_k = subs.iter().find(|s| s.form == "10-K").unwrap();
    assert!(ten_k.items.is_empty());
}

// ── form filtering ─────────────────────────────────────────────────────────────

#[test]
fn by_form_8k_returns_four() {
    let subs = aapl_submissions();
    assert_eq!(CikSubmission::by_form(&subs, "8-K").len(), 4);
}

#[test]
fn by_form_10k_returns_one() {
    let subs = aapl_submissions();
    assert_eq!(CikSubmission::by_form(&subs, "10-K").len(), 1);
}

#[test]
fn by_form_10q_returns_one() {
    let subs = aapl_submissions();
    assert_eq!(CikSubmission::by_form(&subs, "10-Q").len(), 1);
}

#[test]
fn most_recent_10k_is_nov_2024() {
    let subs = aapl_submissions();
    let ten_k = CikSubmission::most_recent_10k(&subs).unwrap();
    assert_eq!(ten_k.filing_date, NaiveDate::from_ymd_opt(2024, 11, 1));
}

// ── 8-K semantic helpers ──────────────────────────────────────────────────────

#[test]
fn earnings_release_detected_by_items_202() {
    let subs = aapl_submissions();
    let earnings: Vec<_> = subs.iter().filter(|s| s.is_earnings_release()).collect();
    // Fixture has 3 filings with "2.02" — indices 0, 3, 5
    assert_eq!(earnings.len(), 3);
}

#[test]
fn non_earnings_8k_is_mid_quarter_event() {
    let subs = aapl_submissions();
    // Index 1: form 8-K, items "5.02,9.01" (no 2.02)
    let sub = subs
        .iter()
        .find(|s| s.items.contains(&"5.02".to_string()))
        .unwrap();
    assert!(!sub.is_earnings_release());
    assert!(sub.is_mid_quarter_event());
}
