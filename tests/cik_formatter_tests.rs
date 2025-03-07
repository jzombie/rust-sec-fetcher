use sec_fetcher::transformers::{cik_to_string, cik_to_u64};

#[test]
fn test_cik_to_string() {
    assert_eq!(cik_to_string(12345), "0000012345");
    assert_eq!(cik_to_string(0), "0000000000");
    assert_eq!(cik_to_string(9876543210), "9876543210");
}

#[test]
fn test_cik_to_u64() {
    assert_eq!(cik_to_u64("0000012345").unwrap(), 12345);
    assert_eq!(cik_to_u64("0000000000").unwrap(), 0);
    assert_eq!(cik_to_u64("9876543210").unwrap(), 9876543210);
}

#[test]
fn test_parse_cik_invalid() {
    assert!(cik_to_u64("invalid").is_err());
    assert!(cik_to_u64("12345678901").is_err()); // More than 10 digits
    assert!(cik_to_u64("").is_err()); // Empty string
    assert!(cik_to_u64(" 12345").is_err()); // Leading space
}
