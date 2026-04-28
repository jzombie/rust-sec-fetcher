// Integration tests for the `FormType` enum.
//
// # Test groups
//
// 1. **Unit tests** (pure, no I/O) — round-trip, case-insensitivity, Display
// 2. **Fixture tests** — parse form types from the stored AAPL submissions
//    fixture and verify expected types are present
//
// Run with:
//     cargo test --test form_type_tests
//
// To verify coverage against live EDGAR data, use the dedicated binary:
//     cargo run --bin check_form_type_coverage

use sec_fetcher::enums::FormType;
use sec_fetcher::models::{Cik, CikSubmission};
use sec_fetcher::network::parse_cik_submissions_json;
use std::collections::HashSet;
use strum::IntoEnumIterator;

mod common;

fn load_fixture(name: &str) -> serde_json::Value {
    common::fixture_json(name)
}

fn aapl_cik() -> Cik {
    Cik::from_u64(320193).unwrap()
}

fn aapl_submissions() -> Vec<CikSubmission> {
    let data = load_fixture("AAPL_submissions.json");
    parse_cik_submissions_json(&data, aapl_cik())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[test]
fn all_named_variants_round_trip() {
    for ft in FormType::iter() {
        let edgar_str = ft.to_string();
        let parsed: FormType = edgar_str.parse().unwrap();
        assert_eq!(
            parsed, ft,
            "round-trip failed: \"{}\" parsed back as {:?}",
            edgar_str, parsed
        );
    }
}

#[test]
fn parsing_is_case_insensitive() {
    assert_eq!("10-k".parse::<FormType>().unwrap(), FormType::TenK);
    assert_eq!("8-k".parse::<FormType>().unwrap(), FormType::EightK);
    assert_eq!("def 14a".parse::<FormType>().unwrap(), FormType::Def14A);
    assert_eq!(
        "schedule 13g/a".parse::<FormType>().unwrap(),
        FormType::Sc13GA
    );
    assert_eq!("nport-p".parse::<FormType>().unwrap(), FormType::NportP);
}

#[test]
fn unknown_type_becomes_other() {
    let ft: FormType = "WEIRD-FORM-XYZ".parse().unwrap();
    assert_eq!(ft, FormType::Other("WEIRD-FORM-XYZ".to_string()));
}

#[test]
fn other_also_preserves_original_case() {
    // Unknown types are stored verbatim (not uppercased).
    let ft: FormType = "Weird-Form".parse().unwrap();
    assert_eq!(ft, FormType::Other("Weird-Form".to_string()));
}

// ── Fixture tests ─────────────────────────────────────────────────────────────

/// Parse the AAPL submissions fixture and verify expected form types are
/// present.  Apple is representative for: 8-K, 10-K, 10-Q, DEF 14A, Form 4,
/// and S-3 (shelf registration).
#[test]
fn aapl_fixture_contains_expected_form_types() {
    let subs = aapl_submissions();
    let present: HashSet<FormType> = subs.iter().map(|s| s.form_type()).collect();

    let expected = [
        FormType::EightK,
        FormType::TenK,
        FormType::TenQ,
        FormType::Def14A,
        FormType::Form4,
    ];
    for ft in &expected {
        assert!(
            present.contains(ft),
            "Expected FormType::{:?} (\"{}\") in AAPL submissions but it was absent",
            ft,
            ft
        );
    }

    // Apple is an operating company — it must never appear in investment-company
    // specific filing types.  This verifies `present` is not a superset that
    // contains everything (which would make the positive assertions above trivially pass).
    assert!(
        !present.contains(&FormType::NportP),
        "AAPL should not have NPORT-P filings (that form is only for investment companies)"
    );
}

/// The raw `form` field on each `CikSubmission` must parse to a `FormType`
/// that round-trips back to the same string (case-insensitively).
#[test]
fn aapl_fixture_form_type_round_trips() {
    let subs = aapl_submissions();
    for s in &subs {
        let ft = s.form_type();
        assert!(
            s.form.eq_ignore_ascii_case(ft.as_edgar_str()),
            "Round-trip failed for AAPL submission: raw=\"{}\" → FormType={:?} → \"{}\"",
            s.form,
            ft,
            ft.as_edgar_str()
        );
    }
}
