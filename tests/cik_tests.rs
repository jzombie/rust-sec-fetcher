use sec_fetcher::models::{Cik, CikError, AccessionNumber};

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
    assert!(matches!(Cik::from_str("invalid"), Err(CikError::ParseError(_))));
    assert!(matches!(Cik::from_str("12345678901"), Err(CikError::InvalidLength))); // More than 10 digits
    assert!(matches!(Cik::from_str(""), Err(CikError::ParseError(_)))); // Empty string
    assert!(matches!(Cik::from_str(" 12345"), Err(CikError::ParseError(_)))); // Leading space
}

#[test]
fn test_cik_from_u64_invalid() {
    assert!(matches!(Cik::from_u64(10000000000), Err(CikError::InvalidLength))); // More than 10 digits
}

#[test]
fn test_from_accession_number() {
    {
        let accession = AccessionNumber::from_str("0001234567-23-000045").unwrap();
        let cik = Cik::from_accession_number(&accession);
        assert_eq!(accession.cik.to_u64(), cik.to_u64());
    }
 
    {
        let accession = AccessionNumber::from_str("0009876543-99-123456").unwrap();
        let cik = Cik::from_accession_number(&accession);
        assert_eq!(accession.cik.to_u64(), cik.to_u64());        
    }
}
