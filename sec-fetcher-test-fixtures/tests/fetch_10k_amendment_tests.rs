// Regression tests: `collect_10k_filings` must include 10-K/A (amended
// annual report) filings alongside the original 10-K.
//
// GPC (Genuine Parts Co, CIK 0000040987) filed a 10-K/A on 2019-08-09
// amending its FY2018 annual report (original 10-K filed 2019-02-25).
// Both appear in the "recent" block of the submissions JSON, so the fixture
// exercises the amendment code path without fetching additional pages.
// Before the fix, amendments were silently dropped because
// `collect_10k_filings` only collected "10-K" and "10-K405".

use sec_fetcher::enums::FormType;
use sec_fetcher::models::{Cik, CikSubmission};
use sec_fetcher::network::{collect_10k_filings, parse_cik_submissions_json};

mod common;

fn gpc_submissions() -> Vec<CikSubmission> {
    let data = common::fixture_json("GPC_submissions.json");
    parse_cik_submissions_json(&data, Cik::from_u64(40987).unwrap())
}

/// The GPC fixture must contain at least one 10-K/A filing.
/// If this fails, the fixture no longer exercises the amendment path and
/// should be refreshed via `cargo run --bin refresh-test-fixtures`.
#[test]
fn gpc_fixture_contains_10k_amendment() {
    let subs = gpc_submissions();
    let amendments: Vec<_> = CikSubmission::by_form(&subs, FormType::TenKA.as_edgar_str());
    assert!(
        !amendments.is_empty(),
        "GPC fixture has no 10-K/A filings — refresh with `cargo run --bin refresh-test-fixtures`"
    );
}

/// `collect_10k_filings` must include 10-K/A filings in its output.
#[test]
fn collect_10k_filings_includes_amendments() {
    let subs = gpc_submissions();
    let merged = collect_10k_filings(&subs);
    let has_amendment = merged.iter().any(|s| s.form_type() == FormType::TenKA);
    assert!(
        has_amendment,
        "collect_10k_filings dropped all 10-K/A filings for GPC"
    );
}

/// Both the original 10-K and its amendment must appear as separate rows —
/// they have different accession numbers and different document content.
#[test]
fn original_and_amendment_are_both_present() {
    let subs = gpc_submissions();
    let merged = collect_10k_filings(&subs);

    // GPC FY2018: original 10-K filed 2019-02-25, amendment filed 2019-08-09.
    let original_acc = "0000040987-19-000015";
    let amendment_acc = "0000040987-19-000042";

    let has_original = merged
        .iter()
        .any(|s| s.accession_number.to_string() == original_acc);
    let has_amendment = merged
        .iter()
        .any(|s| s.accession_number.to_string() == amendment_acc);

    assert!(
        has_original,
        "GPC FY2018 original 10-K ({}) missing from merge output",
        original_acc
    );
    assert!(
        has_amendment,
        "GPC FY2018 10-K/A ({}) missing from merge output",
        amendment_acc
    );
}

/// The merged list must still be sorted newest-first when amendments are present.
#[test]
fn annual_report_filings_sorted_newest_first_with_amendments() {
    let subs = gpc_submissions();
    let merged = collect_10k_filings(&subs);

    assert!(!merged.is_empty(), "no 10-K* filings found for GPC");
    for window in merged.windows(2) {
        assert!(
            window[0].filing_date >= window[1].filing_date,
            "filings not sorted newest-first: {:?} precedes {:?}",
            window[0].filing_date,
            window[1].filing_date
        );
    }
}
