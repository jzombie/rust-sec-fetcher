use chrono::NaiveDate;
/// Unit tests for [`sec_fetcher::parsers::parse_master_idx`].
use sec_fetcher::parsers::parse_master_idx;

const SAMPLE_IDX: &str = "\
Full-Index Header Line 1
Full-Index Header Line 2
CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
1000045|OLD MARKET CAPITAL Corp|4|2026-01-12|edgar/data/1000045/0001000045-26-000001.txt
320193|Apple Inc.|10-K|2025-10-31|edgar/data/320193/0000320193-25-000123.txt
789019|Microsoft Corp|8-K|2025-11-05|edgar/data/789019/0000789019-25-000456.txt
";

const IDX_WITH_BAD_ROWS: &str = "\
Header line
--------------------------------------------------------------------------------
1000045|Good Corp|4|2026-01-12|edgar/data/1000045/file.txt
BADROW_NO_PIPES
1000046|Another Co|10-K|not-a-date|edgar/data/1000046/file.txt
1000047|Final Co|8-K|2026-01-15|edgar/data/1000047/file.txt
";

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn parses_valid_entries() {
    let entries = parse_master_idx(SAMPLE_IDX).unwrap();
    assert_eq!(entries.len(), 3);
}

#[test]
fn first_entry_fields_correct() {
    let entries = parse_master_idx(SAMPLE_IDX).unwrap();
    let e = &entries[0];
    assert_eq!(e.cik, "1000045");
    assert_eq!(e.company_name, "OLD MARKET CAPITAL Corp");
    assert_eq!(e.form_type, "4");
    assert_eq!(e.date_filed, NaiveDate::from_ymd_opt(2026, 1, 12).unwrap());
    assert_eq!(e.filename, "edgar/data/1000045/0001000045-26-000001.txt");
}

#[test]
fn second_entry_is_apple_10k() {
    let entries = parse_master_idx(SAMPLE_IDX).unwrap();
    let e = &entries[1];
    assert_eq!(e.cik, "320193");
    assert_eq!(e.company_name, "Apple Inc.");
    assert_eq!(e.form_type, "10-K");
    assert_eq!(e.date_filed, NaiveDate::from_ymd_opt(2025, 10, 31).unwrap());
}

#[test]
fn skips_malformed_rows_and_bad_dates() {
    let entries = parse_master_idx(IDX_WITH_BAD_ROWS).unwrap();
    // Only 2 valid rows: the bad-pipe row and bad-date row are skipped
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].cik, "1000045");
    assert_eq!(entries[1].cik, "1000047");
}

#[test]
fn empty_string_returns_empty_vec() {
    let entries = parse_master_idx("").unwrap();
    assert!(entries.is_empty());
}

#[test]
fn no_data_rows_after_header() {
    let input = "\
Header
--------------------------------------------------------------------------------
";
    let entries = parse_master_idx(input).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn as_url_produces_correct_edgar_url() {
    let entries = parse_master_idx(SAMPLE_IDX).unwrap();
    let url = entries[0].as_url();
    assert_eq!(
        url,
        "https://www.sec.gov/Archives/edgar/data/1000045/0001000045-26-000001.txt"
    );
}

#[test]
fn form_type_method_parses_known_types() {
    use sec_fetcher::enums::FormType;
    let entries = parse_master_idx(SAMPLE_IDX).unwrap();
    assert_eq!(entries[0].form_type(), FormType::Form4);
    assert_eq!(entries[1].form_type(), FormType::TenK);
    assert_eq!(entries[2].form_type(), FormType::EightK);
}
