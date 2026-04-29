/// Unit tests for [`sec_fetcher::models::InvestmentCompany`].
use sec_fetcher::models::{Cik, InvestmentCompany, TickerSymbol};

fn make_company(cik: &str, ticker: &str) -> InvestmentCompany {
    InvestmentCompany {
        reporting_file_number: None,
        cik_number: Some(cik.to_string()),
        entity_name: Some("Test Fund".to_string()),
        entity_org_type: None,
        series_id: None,
        series_name: None,
        class_id: None,
        class_name: None,
        class_ticker: Some(ticker.to_string()),
        address_1: None,
        address_2: None,
        city: None,
        state: None,
        zip_code: None,
    }
}

#[test]
fn test_get_fund_cik_found() {
    let companies = vec![
        make_company("0001234567", "VFINX"),
        make_company("0007654321", "FXAIX"),
    ];
    let cik = InvestmentCompany::get_fund_cik_by_ticker_symbol(
        &companies,
        &TickerSymbol::new("VFINX"),
    )
    .unwrap();
    assert_eq!(cik, Cik::from_u64(1234567).unwrap());
}

#[test]
fn test_get_fund_cik_not_found() {
    let companies = vec![make_company("0001234567", "VFINX")];
    let result = InvestmentCompany::get_fund_cik_by_ticker_symbol(
        &companies,
        &TickerSymbol::new("UNKNOWN"),
    );
    assert!(result.is_err());
}

#[test]
fn test_get_fund_cik_case_insensitive() {
    let companies = vec![make_company("0001234567", "VFINX")];
    let cik = InvestmentCompany::get_fund_cik_by_ticker_symbol(
        &companies,
        &TickerSymbol::new("vfinx"),
    )
    .unwrap();
    assert_eq!(cik, Cik::from_u64(1234567).unwrap());
}

#[test]
fn test_get_fund_cik_empty_companies() {
    let companies: Vec<InvestmentCompany> = vec![];
    let result = InvestmentCompany::get_fund_cik_by_ticker_symbol(
        &companies,
        &TickerSymbol::new("VFINX"),
    );
    assert!(result.is_err());
}

#[test]
fn test_get_fund_cik_missing_cik_number() {
    let mut company = make_company("0001234567", "VFINX");
    company.cik_number = None;
    let companies = vec![company];
    let result = InvestmentCompany::get_fund_cik_by_ticker_symbol(
        &companies,
        &TickerSymbol::new("VFINX"),
    );
    assert!(result.is_err());
}

#[test]
fn test_get_fund_cik_invalid_cik_format() {
    let mut company = make_company("invalid", "VFINX");
    company.cik_number = Some("not-a-number".to_string());
    let companies = vec![company];
    let result = InvestmentCompany::get_fund_cik_by_ticker_symbol(
        &companies,
        &TickerSymbol::new("VFINX"),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("CIK"));
}

#[test]
fn test_get_fund_cik_deserialize_from_csv() {
    let csv_data = b"CIK Number,Entity Name,Class Ticker\n0001234567,Test Fund,VFGAX\n0007654321,Other Fund,VTIAX";
    use bytes::Bytes;
    use sec_fetcher::parsers::parse_investment_companies_csv;
    let companies = parse_investment_companies_csv(Bytes::from_static(csv_data)).unwrap();
    assert_eq!(companies.len(), 2);
    assert_eq!(companies[0].cik_number.as_deref(), Some("0001234567"));
    assert_eq!(companies[0].class_ticker.as_deref(), Some("VFGAX"));
    assert_eq!(companies[1].class_ticker.as_deref(), Some("VTIAX"));
}
