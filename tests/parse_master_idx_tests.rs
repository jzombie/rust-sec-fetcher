//! Integration tests for [`sec_fetcher::parsers::parse_master_idx`].
//!
//! Real-data tests load the Q4 2025 EDGAR full-index master.idx (Oct–Dec 2025,
//! ~275 000 filings).  Known anchor entries used as test targets:
//!
//! | CIK    | Company        | Form | Filed      | Accession            |
//! |--------|----------------|------|------------|----------------------|
//! | 320193 | Apple Inc.     | 10-K | 2025-10-31 | 0000320193-25-000079 |
//! | 320193 | Apple Inc.     | 8-K  | 2025-10-30 | 0000320193-25-000077 |
//! | 789019 | MICROSOFT CORP | 10-Q | 2025-10-29 | 0001193125-25-256321 |
//!
//! Run `cargo run --bin refresh-test-fixtures` to recreate the fixture file.

use chrono::NaiveDate;
use flate2::read::GzDecoder;
use sec_fetcher::enums::FormType;
use sec_fetcher::models::MasterIndexEntry;
use sec_fetcher::parsers::parse_master_idx;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

// ── Fixture loader ────────────────────────────────────────────────────────────

fn try_load_text_fixture(name: &str) -> Option<String> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(format!("{}.gz", name));
    let file = File::open(&path).ok()?;
    let mut decoder = GzDecoder::new(file);
    let mut text = String::new();
    decoder.read_to_string(&mut text).ok()?;
    Some(text)
}

fn q4_2025() -> Vec<MasterIndexEntry> {
    let text = try_load_text_fixture("master_idx_2025_Q4.idx")
        .expect("run `cargo run --bin refresh-test-fixtures`");
    parse_master_idx(&text).unwrap()
}

/// Find entry whose filename contains the accession number, or panic.
fn by_accession<'a>(entries: &'a [MasterIndexEntry], accession: &str) -> &'a MasterIndexEntry {
    entries
        .iter()
        .find(|e| e.filename.contains(accession))
        .unwrap_or_else(|| panic!("entry for accession '{}' not found in fixture", accession))
}

// ── Real-data tests ───────────────────────────────────────────────────────────

#[test]
fn q4_2025_has_many_filings() {
    // A full calendar quarter routinely yields >100 000 filings.
    assert!(
        q4_2025().len() > 100_000,
        "Q4 2025 master.idx must contain at least 100 000 entries"
    );
}

#[test]
fn aapl_10k_fields_are_exact() {
    let entries = q4_2025();
    let e = by_accession(&entries, "0000320193-25-000079");
    assert_eq!(e.cik, "320193");
    assert_eq!(e.company_name, "Apple Inc.");
    assert_eq!(e.form_type, "10-K");
    assert_eq!(e.date_filed, NaiveDate::from_ymd_opt(2025, 10, 31).unwrap());
    assert_eq!(e.filename, "edgar/data/320193/0000320193-25-000079.txt");
}

#[test]
fn aapl_8k_fields_are_exact() {
    let entries = q4_2025();
    let e = by_accession(&entries, "0000320193-25-000077");
    assert_eq!(e.cik, "320193");
    assert_eq!(e.company_name, "Apple Inc.");
    assert_eq!(e.form_type, "8-K");
    assert_eq!(e.date_filed, NaiveDate::from_ymd_opt(2025, 10, 30).unwrap());
    assert_eq!(e.filename, "edgar/data/320193/0000320193-25-000077.txt");
}

#[test]
fn msft_10q_fields_are_exact() {
    let entries = q4_2025();
    let e = by_accession(&entries, "0001193125-25-256321");
    assert_eq!(e.cik, "789019");
    assert_eq!(e.company_name, "MICROSOFT CORP");
    assert_eq!(e.form_type, "10-Q");
    assert_eq!(e.date_filed, NaiveDate::from_ymd_opt(2025, 10, 29).unwrap());
    assert_eq!(e.filename, "edgar/data/789019/0001193125-25-256321.txt");
}

/// Every field of the AAPL and MSFT entries must differ.  If the parser
/// confused rows or bled a field forward, one of these `assert_ne!` checks
/// will catch it against real, independently-verified values.
#[test]
fn aapl_and_msft_entries_are_independent() {
    let entries = q4_2025();
    let aapl = by_accession(&entries, "0000320193-25-000079");
    let msft = by_accession(&entries, "0001193125-25-256321");

    assert_ne!(aapl.cik, msft.cik); // "320193" vs "789019"
    assert_ne!(aapl.company_name, msft.company_name); // "Apple Inc." vs "MICROSOFT CORP"
    assert_ne!(aapl.form_type, msft.form_type); // "10-K" vs "10-Q"
    assert_ne!(aapl.date_filed, msft.date_filed); // 2025-10-31 vs 2025-10-29
    assert_ne!(aapl.filename, msft.filename);

    // Cross-contamination: Apple's CIK must not appear in MSFT's filename.
    assert!(
        !msft.filename.contains("320193"),
        "MSFT filename '{}' must not contain Apple's CIK",
        msft.filename
    );
    assert!(
        !aapl.filename.contains("789019"),
        "AAPL filename '{}' must not contain MSFT's CIK",
        aapl.filename
    );
}

#[test]
fn aapl_10k_url_is_exact() {
    let entries = q4_2025();
    let e = by_accession(&entries, "0000320193-25-000079");
    assert_eq!(
        e.as_url(),
        "https://www.sec.gov/Archives/edgar/data/320193/0000320193-25-000079.txt"
    );
    // Must not contain MSFT's CIK.
    assert!(!e.as_url().contains("789019"));
}

#[test]
fn msft_10q_url_is_exact() {
    let entries = q4_2025();
    let e = by_accession(&entries, "0001193125-25-256321");
    assert_eq!(
        e.as_url(),
        "https://www.sec.gov/Archives/edgar/data/789019/0001193125-25-256321.txt"
    );
    // Must not contain Apple's CIK.
    assert!(!e.as_url().contains("320193"));
}

#[test]
fn form_type_method_returns_correct_enum_variants() {
    let entries = q4_2025();
    assert_eq!(
        by_accession(&entries, "0000320193-25-000079").form_type(),
        FormType::TenK
    );
    assert_eq!(
        by_accession(&entries, "0000320193-25-000077").form_type(),
        FormType::EightK
    );
    assert_eq!(
        by_accession(&entries, "0001193125-25-256321").form_type(),
        FormType::TenQ
    );
}

// ── Edge-case tests (inline data — exercises parser robustness) ───────────────

const IDX_WITH_BAD_ROWS: &str = "\
Header line
--------------------------------------------------------------------------------
1000045|Good Corp|4|2026-01-12|edgar/data/1000045/file.txt
BADROW_NO_PIPES
1000046|Another Co|10-K|not-a-date|edgar/data/1000046/file.txt
1000047|Final Co|8-K|2026-01-15|edgar/data/1000047/file.txt
";

#[test]
fn skips_malformed_rows_and_bad_dates() {
    let entries = parse_master_idx(IDX_WITH_BAD_ROWS).unwrap();
    // Only 2 valid rows survive: the no-pipe row and bad-date row are both skipped.
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].cik, "1000045");
    assert_eq!(entries[1].cik, "1000047");
    assert!(
        !entries.iter().any(|e| e.cik == "1000046"),
        "CIK 1000046 has an invalid date and must be completely absent"
    );
    assert!(
        !entries.iter().any(|e| e.cik.contains("BADROW")),
        "BADROW_NO_PIPES must not produce any entry"
    );
}

#[test]
fn empty_string_returns_empty_vec() {
    assert!(parse_master_idx("").unwrap().is_empty());
}

#[test]
fn header_only_returns_empty_vec() {
    let input = "Header\n--------------------------------------------------------------------------------\n";
    assert!(parse_master_idx(input).unwrap().is_empty());
}
