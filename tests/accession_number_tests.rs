use sec_fetcher::models::{AccessionNumber, AccessionNumberError, CikError};

#[test]
fn test_accession_number_from_str_valid() {
    let accession = AccessionNumber::from_str("0001234567-23-000045").unwrap();
    assert_eq!(accession.cik.to_u64(), 1234567);
    assert_eq!(accession.year, 23);
    assert_eq!(accession.sequence, 45);

    let accession = AccessionNumber::from_str("0009876543-99-123456").unwrap();
    assert_eq!(accession.cik.to_u64(), 9876543);
    assert_eq!(accession.year, 99);
    assert_eq!(accession.sequence, 123456);
}

#[test]
fn test_accession_number_display_format() {
    let accession = AccessionNumber::from_parts(1234567, 23, 45).unwrap();
    assert_eq!(format!("{}", accession), "0001234567-23-000045");
}

#[test]
fn test_accession_number_error_display() {
    let err = AccessionNumberError::InvalidLength;
    assert_eq!(err.to_string(), "Accession number must be 18 digits (XXXXXXXXXX-YY-NNNNNN)");

    let parse_err = "bad".parse::<u32>().unwrap_err();
    let err = AccessionNumberError::ParseError(parse_err);
    let msg = err.to_string();
    assert!(msg.contains("Failed to parse"), "ParseError display should mention parse failure: {}", msg);

    let err = AccessionNumberError::CikError(CikError::InvalidLength);
    let msg = err.to_string();
    assert!(msg.contains("CIK"), "CikError display should mention CIK: {}", msg);
}

#[test]
fn test_accession_number_from_parts_max_values() {
    // Boundary: max CIK (9,999,999,999), max year (99), max sequence (999,999)
    let accession = AccessionNumber::from_parts(9999999999, 99, 999999).unwrap();
    assert_eq!(accession.to_string(), "9999999999-99-999999");
    assert_eq!(accession.to_unformatted_string(), "999999999999999999");
}

#[test]
fn test_accession_number_from_parts_zero_values() {
    // Boundary: all zeros
    let accession = AccessionNumber::from_parts(0, 0, 0).unwrap();
    assert_eq!(accession.to_string(), "0000000000-00-000000");
    assert_eq!(accession.to_unformatted_string(), "000000000000000000");
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
    assert_eq!(accession.to_unformatted_string(), "000123456723000045");

    let accession = AccessionNumber::from_parts(9876543210, 99, 123456).unwrap();
    assert_eq!(accession.to_string(), "9876543210-99-123456");
    assert_eq!(accession.to_unformatted_string(), "987654321099123456");
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
