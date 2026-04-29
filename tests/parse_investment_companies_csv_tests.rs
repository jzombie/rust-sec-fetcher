/// Unit tests for [`sec_fetcher::parsers::parse_investment_companies_csv`].
use bytes::Bytes;
use sec_fetcher::parsers::parse_investment_companies_csv;

#[test]
fn test_parse_empty_csv() {
    let csv = b"CIK Number,Entity Name,Class Ticker\n";
    let companies = parse_investment_companies_csv(Bytes::from_static(csv)).unwrap();
    assert!(companies.is_empty());
}

#[test]
fn test_parse_single_record() {
    let csv = b"CIK Number,Entity Name,Class Ticker\n0001234567,Test Fund,VFINX\n";
    let companies = parse_investment_companies_csv(Bytes::from_static(csv)).unwrap();
    assert_eq!(companies.len(), 1);
    assert_eq!(companies[0].cik_number.as_deref(), Some("0001234567"));
    assert_eq!(companies[0].entity_name.as_deref(), Some("Test Fund"));
    assert_eq!(companies[0].class_ticker.as_deref(), Some("VFINX"));
}

#[test]
fn test_parse_multiple_records() {
    let csv = b"CIK Number,Entity Name,Class Ticker\n0001234567,Fidelity Balanced Fund,VFGAX\n0007654321,Vanguard Total Intl Fund,VTIAX\n0009999999,Fidelity Fund,FFIDX\n";
    let companies = parse_investment_companies_csv(Bytes::from_static(csv)).unwrap();
    assert_eq!(companies.len(), 3);
    assert_eq!(companies[0].class_ticker.as_deref(), Some("VFGAX"));
    assert_eq!(companies[1].class_ticker.as_deref(), Some("VTIAX"));
    assert_eq!(companies[2].class_ticker.as_deref(), Some("FFIDX"));
}

#[test]
fn test_parse_csv_with_optional_fields_empty() {
    let csv = b"CIK Number,Entity Name,Class Ticker,Series ID\n0001234567,Fidelity Fund,FFIDX,\n";
    let companies = parse_investment_companies_csv(Bytes::from_static(csv)).unwrap();
    assert_eq!(companies.len(), 1);
    // Empty optional field should be None
    assert!(companies[0].series_id.is_none());
}

#[test]
fn test_parse_csv_trailing_newline() {
    let csv = b"CIK Number,Entity Name,Class Ticker\n0001234567,Test Fund,VFINX\n";
    let companies = parse_investment_companies_csv(Bytes::from_static(csv)).unwrap();
    assert_eq!(companies.len(), 1);
}

#[test]
fn test_parse_csv_with_utf8_content() {
    let csv = "CIK Number,Entity Name,Class Ticker\n0001234567,Fidelity® Fund,FFIDX\n".as_bytes();
    let companies = parse_investment_companies_csv(Bytes::from_static(csv)).unwrap();
    assert_eq!(companies.len(), 1);
    assert!(companies[0].entity_name.as_deref().unwrap().contains("®"));
}
