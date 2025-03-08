use sec_fetcher::models::{AccessionNumber, AccessionNumberError};

#[test]
fn test_accession_number_from_str_valid() {
    let accession = AccessionNumber::from_str("0001234567-23-000045").unwrap();
    assert_eq!(accession.cik, 1234567);
    assert_eq!(accession.year, 23);
    assert_eq!(accession.sequence, 45);

    let accession = AccessionNumber::from_str("0009876543-99-123456").unwrap();
    assert_eq!(accession.cik, 9876543);
    assert_eq!(accession.year, 99);
    assert_eq!(accession.sequence, 123456);
}

#[test]
fn test_accession_number_from_str_without_dashes() {
    let accession = AccessionNumber::from_str("000123456723000045").unwrap();
    assert_eq!(accession.to_string(), "0001234567-23-000045");

    let accession = AccessionNumber::from_str("987654321099123456").unwrap();
    assert_eq!(accession.to_string(), "9876543210-99-123456");
}

#[test]
fn test_accession_number_to_string() {
    let accession = AccessionNumber::from_parts(1234567, 23, 45).unwrap();
    assert_eq!(accession.to_string(), "0001234567-23-000045");

    let accession = AccessionNumber::from_parts(9876543210, 99, 123456).unwrap();
    assert_eq!(accession.to_string(), "9876543210-99-123456");
}

#[test]
fn test_accession_number_invalid_length() {
    // Extra digit in the year
    assert!(matches!(
        AccessionNumber::from_str("0001234567-233-000045"),
        Err(AccessionNumberError::InvalidLength)
    ));

    // Extra digit in sequence
    assert!(matches!(
        AccessionNumber::from_str("0001234567-23-0000456"),
        Err(AccessionNumberError::InvalidLength)
    ));

    // Empty string
    assert!(matches!(
        AccessionNumber::from_str(""),
        Err(AccessionNumberError::InvalidLength)
    ));

    // Too short
    assert!(matches!(
        AccessionNumber::from_str("123"),
        Err(AccessionNumberError::InvalidLength)
    ));
}

#[test]
fn test_accession_number_parse_error() {
    // CIK contains non-numeric characters
    assert!(matches!(
        AccessionNumber::from_str("abcdefghij-23-000045"),
        Err(AccessionNumberError::ParseError(_))
    ));

    // Year contains non-numeric characters
    assert!(matches!(
        AccessionNumber::from_str("0001234567-xx-000045"),
        Err(AccessionNumberError::ParseError(_))
    ));

    // Sequence contains non-numeric characters
    assert!(matches!(
        AccessionNumber::from_str("0001234567-23-xxxxxx"),
        Err(AccessionNumberError::ParseError(_))
    ));
}

#[test]
fn test_accession_number_from_parts_invalid() {
    assert!(matches!(
        AccessionNumber::from_parts(10000000000, 23, 45),
        Err(AccessionNumberError::InvalidLength)
    ));

    assert!(matches!(
        AccessionNumber::from_parts(1234567, 100, 45),
        Err(AccessionNumberError::InvalidLength)
    ));

    assert!(matches!(
        AccessionNumber::from_parts(1234567, 23, 1000000),
        Err(AccessionNumberError::InvalidLength)
    ));
}
