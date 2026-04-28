use sec_fetcher::models::{Cik, CompanyProfile};
use chrono::NaiveDate;

fn make_profile(owner_org: Option<&str>, fiscal_year_end: Option<&str>) -> CompanyProfile {
    CompanyProfile {
        cik: Cik::from_u64(320193).unwrap(),
        name: "Apple Inc.".into(),
        entity_type: None,
        sic: None,
        sic_description: None,
        owner_org: owner_org.map(String::from),
        tickers: vec![],
        exchanges: vec![],
        category: None,
        state_of_incorporation: None,
        fiscal_year_end: fiscal_year_end.map(String::from),
        website: None,
        investor_website: None,
        phone: None,
        description: None,
    }
}

#[test]
fn test_sector_with_numeric_prefix() {
    let p = make_profile(Some("06 Technology"), None);
    assert_eq!(p.sector(), Some("Technology"));
}

#[test]
fn test_sector_with_multi_digit_prefix() {
    let p = make_profile(Some("12 Finance & Insurance"), None);
    assert_eq!(p.sector(), Some("Finance & Insurance"));
}

#[test]
fn test_sector_no_prefix() {
    let p = make_profile(Some("Technology"), None);
    assert_eq!(p.sector(), Some("Technology"));
}

#[test]
fn test_sector_none() {
    let p = make_profile(None, None);
    assert_eq!(p.sector(), None);
}

#[test]
fn test_sector_empty_string() {
    let p = make_profile(Some(""), None);
    assert_eq!(p.sector(), Some(""));
}

#[test]
fn test_fiscal_year_end_date_valid() {
    let p = make_profile(None, Some("0926"));
    let d = p.fiscal_year_end_date().unwrap();
    assert_eq!(d, NaiveDate::from_ymd_opt(2000, 9, 26).unwrap());
}

#[test]
fn test_fiscal_year_end_date_january() {
    let p = make_profile(None, Some("0115"));
    let d = p.fiscal_year_end_date().unwrap();
    assert_eq!(d, NaiveDate::from_ymd_opt(2000, 1, 15).unwrap());
}

#[test]
fn test_fiscal_year_end_date_none() {
    let p = make_profile(None, None);
    assert!(p.fiscal_year_end_date().is_none());
}

#[test]
fn test_fiscal_year_end_date_invalid_length() {
    let p = make_profile(None, Some("123"));
    assert!(p.fiscal_year_end_date().is_none());
}

#[test]
fn test_fiscal_year_end_date_non_numeric() {
    let p = make_profile(None, Some("ab12"));
    assert!(p.fiscal_year_end_date().is_none());
}

#[test]
fn test_fiscal_year_end_date_invalid_month() {
    let p = make_profile(None, Some("1326"));
    assert!(p.fiscal_year_end_date().is_none());
}

#[test]
fn test_fiscal_year_end_date_invalid_day() {
    let p = make_profile(None, Some("0932"));
    assert!(p.fiscal_year_end_date().is_none());
}
