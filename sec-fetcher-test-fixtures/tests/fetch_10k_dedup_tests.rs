// Regression tests: `fetch_10k_filings` must not return duplicate accession
// numbers after merging filings from a primary CIK and its co-registrant
// subsidiaries.
//
// AEP (American Electric Power Co Inc, CIK 4904) co-files its annual 10-K
// with seven operating subsidiaries.  Each subsidiary's submissions JSON lists
// the same 10-K accession numbers as the primary registrant.  Before the fix,
// `fetch_10k_filings` (which calls `fetch_all_entity_submissions` to combine
// primary + subsidiaries) could return the same accession up to eight times,
// producing duplicate rows in the audit CSV.
//
// The tests here replicate the exact merge path offline using real EDGAR
// fixture files — no network calls, no synthetic data.

use sec_fetcher::models::{Cik, CikSubmission};
use sec_fetcher::network::{merge_10k_submissions, parse_cik_submissions_json};
use std::collections::HashSet;

mod common;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Load and parse the submissions fixture for a single CIK.
fn load_submissions(fixture_name: &str, cik_value: u64) -> Vec<CikSubmission> {
    let data = common::fixture_json(fixture_name);
    let cik = Cik::from_u64(cik_value).unwrap();
    parse_cik_submissions_json(&data, cik)
}

/// Load `AEP_related_ciks.json` and return the numeric CIK values.
fn load_related_cik_values() -> Vec<u64> {
    let data = common::fixture_json("AEP_related_ciks.json");
    let strings: Vec<String> = serde_json::from_value(data).unwrap();
    strings
        .iter()
        .map(|s| s.trim_start_matches('0').parse::<u64>().unwrap())
        .collect()
}

/// Derive the subsidiary fixture filename from a raw CIK value.
/// Mirrors the names produced by `refresh-test-fixtures`.
fn subsidiary_fixture_name(cik: u64) -> String {
    format!("AEP_subsidiary_{}_submissions.json", cik)
}

/// Combine primary AEP submissions with every co-registrant subsidiary,
/// replicating what `fetch_all_entity_submissions` does at runtime.
fn all_aep_submissions() -> Vec<CikSubmission> {
    let mut all = load_submissions("AEP_submissions.json", 4904);
    for cik in load_related_cik_values() {
        let name = subsidiary_fixture_name(cik);
        let mut subs = load_submissions(&name, cik);
        all.append(&mut subs);
    }
    all.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));
    all
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Bug trigger: without dedup, the combined submission list must contain at
/// least one accession number that appears more than once (same 10-K filed by
/// the primary CIK and one or more subsidiaries).
///
/// If this fails, the fixture no longer exercises the real duplicate code path
/// and should be refreshed via `cargo run --bin refresh-test-fixtures`.
#[test]
fn combined_aep_submissions_contain_duplicate_10k_accessions() {
    let all = all_aep_submissions();

    // Build the raw (un-deduped) merged list — intentionally bypass
    // merge_10k_submissions so we can confirm the input data actually contains
    // duplicates before the library fix is applied.
    let mut raw: Vec<CikSubmission> = CikSubmission::by_form(&all, "10-K")
        .into_iter()
        .cloned()
        .collect();
    let mut k405: Vec<CikSubmission> = CikSubmission::by_form(&all, "10-K405")
        .into_iter()
        .cloned()
        .collect();
    raw.append(&mut k405);

    let mut seen: HashSet<String> = HashSet::new();
    let has_duplicates = raw
        .iter()
        .any(|s| !seen.insert(s.accession_number.to_string()));

    assert!(
        has_duplicates,
        "Expected at least one duplicate accession number in the undeduped merge \
         of AEP primary + {} subsidiaries, but found none. \
         Refresh fixtures with `cargo run --bin refresh-test-fixtures`.",
        load_related_cik_values().len()
    );
}

/// Fix verification: `merge_10k_submissions` (the library function called by
/// `fetch_10k_filings`) must return each accession number exactly once
/// regardless of how many co-registrant CIKs list it.
#[test]
fn fetch_10k_filings_dedup_eliminates_co_registrant_duplicates() {
    let all = all_aep_submissions();

    // Call the actual library function — not a reimplementation.
    let results = merge_10k_submissions(&all);

    let mut seen: HashSet<String> = HashSet::new();
    for f in &results {
        let acc = f.accession_number.to_string();
        assert!(
            seen.insert(acc.clone()),
            "duplicate accession {} in merge_10k_submissions output for AEP + subsidiaries",
            acc
        );
    }
}

/// Sanity: `merge_10k_submissions` returns a non-empty list sorted newest-first.
#[test]
fn aep_has_10k_filings_sorted_newest_first() {
    let all = all_aep_submissions();
    let results = merge_10k_submissions(&all);

    assert!(
        !results.is_empty(),
        "no 10-K or 10-K405 filings found for AEP in fixture"
    );
    for window in results.windows(2) {
        assert!(
            window[0].filing_date >= window[1].filing_date,
            "filings not sorted newest-first: {:?} precedes {:?}",
            window[0].filing_date,
            window[1].filing_date
        );
    }
}
