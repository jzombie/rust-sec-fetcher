// Fixtures are full downloads from SEC EDGAR (not filtered or abbreviated):
//
//   Reddit, Inc. (CIK 0001713445) — RDDT_submissions.json
//     March 2026 snapshot: 405 recent filings
//     Includes S-1 (IPO 2024-02-22), S-1/A ×3, SC 13D (2024-05-03),
//     SC 13D/A (2024-08-22), DEF 14A (2025-04-28)
//
//   Boeing Co (CIK 0000012927) — BA_submissions.json
//     March 2026 snapshot: 1000 recent filings
//     Includes S-3 (2024-10-15)

use chrono::NaiveDate;
use flate2::read::GzDecoder;
use sec_fetcher::types::{Cik, CikSubmission};
use sec_fetcher::network::parse_cik_submissions_json;
use serde_json::Value;
use std::fs::File;
use std::path::PathBuf;

/// Load a fixture by its logical name (e.g. `"RDDT_submissions.json"`).
/// The file is stored on disk as `{name}.gz` and decompressed in memory.
/// Run `cargo run --example refresh_test_fixtures` to update the fixtures.
fn load_fixture(name: &str) -> Value {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(format!("{}.gz", name));
    let file = File::open(&path).unwrap_or_else(|_| {
        panic!(
            "missing fixture: {} (run `cargo run --example refresh_test_fixtures`)",
            path.display()
        )
    });
    serde_json::from_reader(GzDecoder::new(file)).expect("fixture is not valid JSON")
}

/// Reddit, Inc. — full EDGAR download, CIK 1713445 (ticker: RDDT).
fn reddit_submissions() -> Vec<CikSubmission> {
    let data = load_fixture("RDDT_submissions.json");
    parse_cik_submissions_json(&data, Cik::from_u64(1713445).unwrap())
}

/// Boeing Co — full EDGAR download, CIK 12927 (ticker: BA).
fn boeing_submissions() -> Vec<CikSubmission> {
    let data = load_fixture("BA_submissions.json");
    parse_cik_submissions_json(&data, Cik::from_u64(12927).unwrap())
}

// ── fixture sanity ────────────────────────────────────────────────────────────

#[test]
fn rddt_has_over_400_recent_submissions() {
    assert!(reddit_submissions().len() >= 400);
}

#[test]
fn rddt_fixture_submissions_are_newest_first() {
    let subs = reddit_submissions();
    // DEF 14A (2025-04-28) must precede S-1 (2024-02-22)
    let def14a_date = subs
        .iter()
        .find(|s| s.form == "DEF 14A")
        .unwrap()
        .filing_date;
    let s1_date = subs.iter().find(|s| s.form == "S-1").unwrap().filing_date;
    assert!(def14a_date > s1_date);
}

#[test]
fn ba_has_1000_recent_submissions() {
    // EDGAR caps the "recent" window at 1000 for large filers like Boeing.
    assert_eq!(boeing_submissions().len(), 1000);
}

// ── S-1 (Reddit, real IPO filing 2024-02-22) ─────────────────────────────────

#[test]
fn s1_initial_filing_date_and_accession() {
    let subs = reddit_submissions();
    let s1: Vec<_> = CikSubmission::by_form(&subs, "S-1");
    assert_eq!(s1.len(), 1);
    assert_eq!(s1[0].filing_date, NaiveDate::from_ymd_opt(2024, 2, 22));
    assert_eq!(s1[0].accession_number.to_string(), "0001628280-24-006294");
    assert_eq!(s1[0].primary_document, "reddits-1q423.htm");
}

#[test]
fn s1_has_three_amendments() {
    let subs = reddit_submissions();
    let amendments: Vec<_> = CikSubmission::by_form(&subs, "S-1/A");
    assert_eq!(amendments.len(), 3);
    // Most recent amendment first
    assert_eq!(
        amendments[0].filing_date,
        NaiveDate::from_ymd_opt(2024, 3, 19)
    );
    assert_eq!(
        amendments[0].accession_number.to_string(),
        "0001628280-24-011789"
    );
}

#[test]
fn s1_combined_newest_amendment_precedes_initial() {
    let subs = reddit_submissions();
    let mut combined: Vec<_> = CikSubmission::by_form(&subs, "S-1")
        .into_iter()
        .chain(CikSubmission::by_form(&subs, "S-1/A"))
        .collect();
    combined.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
    // 3 amendments + 1 initial = 4 total; last entry is the S-1
    assert_eq!(combined.len(), 4);
    assert_eq!(combined.last().unwrap().form, "S-1");
    assert_eq!(
        combined.last().unwrap().filing_date,
        NaiveDate::from_ymd_opt(2024, 2, 22)
    );
}

#[test]
fn s1_filter_does_not_include_s3_or_sc13d() {
    let subs = reddit_submissions();
    let s1_forms: Vec<_> = CikSubmission::by_form(&subs, "S-1");
    assert!(s1_forms.iter().all(|s| s.form == "S-1"));
    assert_eq!(CikSubmission::by_form(&subs, "S-3").len(), 0);
}

// ── S-3 (Boeing, real shelf registration 2024-10-15) ─────────────────────────

#[test]
fn s3_filing_date_and_accession() {
    let subs = boeing_submissions();
    let s3: Vec<_> = CikSubmission::by_form(&subs, "S-3");
    assert_eq!(s3.len(), 1);
    assert_eq!(s3[0].filing_date, NaiveDate::from_ymd_opt(2024, 10, 15));
    assert_eq!(s3[0].accession_number.to_string(), "0001193125-24-237177");
    assert_eq!(s3[0].primary_document, "d871346ds3.htm");
}

#[test]
fn s3_cik_is_boeing() {
    let subs = boeing_submissions();
    let s3 = CikSubmission::by_form(&subs, "S-3");
    assert_eq!(s3[0].cik, Cik::from_u64(12927).unwrap());
}

// ── DEF 14A (Reddit, real proxy 2025-04-28) ───────────────────────────────────

#[test]
fn def14a_filing_date_and_accession() {
    let subs = reddit_submissions();
    let proxies: Vec<_> = CikSubmission::by_form(&subs, "DEF 14A");
    assert_eq!(proxies.len(), 1);
    assert_eq!(proxies[0].filing_date, NaiveDate::from_ymd_opt(2025, 4, 28));
    assert_eq!(
        proxies[0].accession_number.to_string(),
        "0001713445-25-000092"
    );
    assert_eq!(proxies[0].primary_document, "rddt-20250428.htm");
}

#[test]
fn def14a_does_not_include_s1_or_sc13d() {
    let subs = reddit_submissions();
    let proxies: Vec<_> = CikSubmission::by_form(&subs, "DEF 14A");
    assert!(proxies.iter().all(|s| s.form == "DEF 14A"));
}

// ── SC 13D (Reddit, real activist filing 2024-05-03) ─────────────────────────

#[test]
fn sc13d_initial_filing_date_and_accession() {
    let subs = reddit_submissions();
    let filings: Vec<_> = CikSubmission::by_form(&subs, "SC 13D");
    assert_eq!(filings.len(), 1);
    assert_eq!(filings[0].filing_date, NaiveDate::from_ymd_opt(2024, 5, 3));
    assert_eq!(
        filings[0].accession_number.to_string(),
        "0001193125-24-130868"
    );
    assert_eq!(filings[0].primary_document, "d804372dsc13d.htm");
}

#[test]
fn sc13d_amendment_date_and_accession() {
    let subs = reddit_submissions();
    let amendments: Vec<_> = CikSubmission::by_form(&subs, "SC 13D/A");
    assert_eq!(amendments.len(), 1);
    assert_eq!(
        amendments[0].filing_date,
        NaiveDate::from_ymd_opt(2024, 8, 22)
    );
    assert_eq!(
        amendments[0].accession_number.to_string(),
        "0001193125-24-205536"
    );
}

#[test]
fn sc13d_combined_amendment_precedes_initial() {
    let subs = reddit_submissions();
    let mut combined: Vec<_> = CikSubmission::by_form(&subs, "SC 13D")
        .into_iter()
        .chain(CikSubmission::by_form(&subs, "SC 13D/A"))
        .collect();
    combined.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0].form, "SC 13D/A");
    assert_eq!(combined[1].form, "SC 13D");
}
