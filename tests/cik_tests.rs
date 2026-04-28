use sec_fetcher::enums::TickerOrigin;
use sec_fetcher::models::{AccessionNumber, Cik, CikError, Ticker, TickerSymbol};

#[test]
fn test_cik_to_string() {
    assert_eq!(Cik::from_u64(12345).unwrap().to_string(), "0000012345");
    assert_eq!(Cik::from_u64(0).unwrap().to_string(), "0000000000");
    assert_eq!(Cik::from_u64(9876543210).unwrap().to_string(), "9876543210");
}

#[test]
fn test_cik_from_str() {
    assert_eq!(Cik::from_str("0000012345").unwrap().to_u64(), 12345);
    assert_eq!(Cik::from_str("0000000000").unwrap().to_u64(), 0);
    assert_eq!(Cik::from_str("9876543210").unwrap().to_u64(), 9876543210);
}

#[test]
fn test_cik_from_str_invalid() {
    assert!(matches!(
        Cik::from_str("invalid"),
        Err(CikError::ParseError(_))
    ));
    assert!(matches!(
        Cik::from_str("12345678901"),
        Err(CikError::InvalidLength)
    )); // More than 10 digits
    assert!(matches!(Cik::from_str(""), Err(CikError::ParseError(_)))); // Empty string
    assert!(matches!(
        Cik::from_str(" 12345"),
        Err(CikError::ParseError(_))
    )); // Leading space
}

#[test]
fn test_cik_from_u64_invalid() {
    assert!(matches!(
        Cik::from_u64(10000000000),
        Err(CikError::InvalidLength)
    )); // More than 10 digits
}

#[test]
fn test_from_accession_number() {
    {
        let accession = AccessionNumber::from_str("0001234567-23-000045").unwrap();
        let cik = Cik::from_accession_number(&accession);
        assert_eq!(accession.cik, cik);
    }

    {
        let accession = AccessionNumber::from_str("0009876543-99-123456").unwrap();
        let cik = Cik::from_accession_number(&accession);
        assert_eq!(accession.cik, cik);
    }
}

fn make_ticker(symbol: &str, cik: u64, origin: TickerOrigin) -> Ticker {
    Ticker {
        symbol: TickerSymbol::new(symbol),
        cik: Cik::from_u64(cik).unwrap(),
        company_name: String::new(),
        origin,
    }
}

// Looking up a primary listing returns its own CIK directly.
#[test]
fn test_cik_lookup_primary_listing() {
    let tickers = vec![make_ticker("MS", 895421, TickerOrigin::PrimaryListing)];
    let cik = Cik::get_company_cik_by_ticker_symbol(&tickers, &TickerSymbol::new("MS")).unwrap();
    assert_eq!(cik, Cik::from_u64(895421).unwrap());
}

// A derived instrument that shares a CIK with a primary listing resolves to
// the primary listing's CIK (which is identical — this locks in the guarantee).
#[test]
fn test_cik_lookup_derived_instrument_resolves_to_primary() {
    let tickers = vec![
        make_ticker("MS", 895421, TickerOrigin::PrimaryListing),
        make_ticker("MS-PA", 895421, TickerOrigin::DerivedInstrument),
    ];
    let primary_cik =
        Cik::get_company_cik_by_ticker_symbol(&tickers, &TickerSymbol::new("MS")).unwrap();
    let derived_cik =
        Cik::get_company_cik_by_ticker_symbol(&tickers, &TickerSymbol::new("MS-PA")).unwrap();
    assert_eq!(primary_cik, derived_cik);
}

// A derived instrument whose CIK has no matching primary listing falls back
// to its own CIK rather than erroring (e.g. an ADR with a standalone CIK).
#[test]
fn test_cik_lookup_derived_instrument_fallback_no_primary() {
    let tickers = vec![make_ticker(
        "FOO-WT",
        999999,
        TickerOrigin::DerivedInstrument,
    )];
    let cik =
        Cik::get_company_cik_by_ticker_symbol(&tickers, &TickerSymbol::new("FOO-WT")).unwrap();
    assert_eq!(cik, Cik::from_u64(999999).unwrap());
}

// ============================================================================
// TickerOrigin enum
// ============================================================================

#[test]
fn test_ticker_origin_variants() {
    use sec_fetcher::enums::TickerOrigin;
    // Verify all three variants exist and are distinct
    match TickerOrigin::PrimaryListing {
        TickerOrigin::PrimaryListing => {} // ok
        _ => panic!("Expected PrimaryListing"),
    }
    match TickerOrigin::DerivedInstrument {
        TickerOrigin::DerivedInstrument => {} // ok
        _ => panic!("Expected DerivedInstrument"),
    }
    match TickerOrigin::InvestmentCompany {
        TickerOrigin::InvestmentCompany => {} // ok
        _ => panic!("Expected InvestmentCompany"),
    }
}

#[test]
fn test_ticker_origin_display() {
    use sec_fetcher::enums::TickerOrigin;
    assert_eq!(
        TickerOrigin::PrimaryListing.to_string(),
        "PrimaryListing"
    );
    assert_eq!(
        TickerOrigin::DerivedInstrument.to_string(),
        "DerivedInstrument"
    );
    assert_eq!(
        TickerOrigin::InvestmentCompany.to_string(),
        "InvestmentCompany"
    );
}

#[test]
fn test_ticker_origin_debug() {
    use sec_fetcher::enums::TickerOrigin;
    let dbg = format!("{:?}", TickerOrigin::PrimaryListing);
    assert_eq!(dbg, "PrimaryListing");
}

#[test]
fn test_ticker_origin_equality() {
    use sec_fetcher::enums::TickerOrigin;
    assert_eq!(
        TickerOrigin::PrimaryListing,
        TickerOrigin::PrimaryListing
    );
    assert_ne!(
        TickerOrigin::PrimaryListing,
        TickerOrigin::DerivedInstrument
    );
}

#[test]
fn test_ticker_origin_serialize_deserialize() {
    use sec_fetcher::enums::TickerOrigin;
    let json = serde_json::to_string(&TickerOrigin::PrimaryListing).unwrap();
    assert_eq!(json, "\"PrimaryListing\"");
    let deserialized: TickerOrigin = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, TickerOrigin::PrimaryListing);
}

#[test]
fn test_ticker_origin_from_str() {
    use sec_fetcher::enums::TickerOrigin;
    use std::str::FromStr;
    let origin = TickerOrigin::from_str("PrimaryListing").unwrap();
    assert_eq!(origin, TickerOrigin::PrimaryListing);
    let origin = TickerOrigin::from_str("DerivedInstrument").unwrap();
    assert_eq!(origin, TickerOrigin::DerivedInstrument);
    let origin = TickerOrigin::from_str("InvestmentCompany").unwrap();
    assert_eq!(origin, TickerOrigin::InvestmentCompany);
}

// Lookup is case-insensitive via normalization.
#[test]
fn test_cik_lookup_case_insensitive() {
    let tickers = vec![make_ticker("AAPL", 320193, TickerOrigin::PrimaryListing)];
    assert_eq!(
        Cik::get_company_cik_by_ticker_symbol(&tickers, &TickerSymbol::new("aapl")).unwrap(),
        Cik::get_company_cik_by_ticker_symbol(&tickers, &TickerSymbol::new("AAPL")).unwrap(),
    );
}

// Unknown ticker returns an error.
#[test]
fn test_cik_lookup_not_found() {
    let tickers = vec![make_ticker("AAPL", 320193, TickerOrigin::PrimaryListing)];
    assert!(Cik::get_company_cik_by_ticker_symbol(&tickers, &TickerSymbol::new("NOPE")).is_err());
}
