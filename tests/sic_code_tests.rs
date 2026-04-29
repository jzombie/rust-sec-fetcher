use sec_fetcher::models::SicCode;

#[test]
fn test_office_short_strips_prefix() {
    let s = SicCode {
        code: 3571,
        description: "ELECTRONIC COMPUTERS".into(),
        office: "Office of Technology".into(),
    };
    assert_eq!(s.office_short(), "Technology");
}

#[test]
fn test_office_short_no_prefix() {
    let s = SicCode {
        code: 9999,
        description: "UNKNOWN".into(),
        office: "Direct".into(),
    };
    assert_eq!(s.office_short(), "Direct");
}

#[test]
fn test_office_short_empty_string() {
    let s = SicCode {
        code: 0,
        description: "NONE".into(),
        office: String::new(),
    };
    assert_eq!(s.office_short(), "");
}

#[test]
fn test_office_short_office_of_with_extra_spaces() {
    let s = SicCode {
        code: 1234,
        description: "SAMPLE".into(),
        office: "Office of   Industrial Applications".into(),
    };
    // strip_prefix only strips the exact prefix, so "Office of   " doesn't match "Office of "
    // It just strips "Office of " and leaves the rest including extra spaces
    assert_eq!(s.office_short(), "  Industrial Applications");
}

#[test]
fn test_sic_code_debug() {
    let s = SicCode {
        code: 3571,
        description: "ELECTRONIC COMPUTERS".into(),
        office: "Office of Technology".into(),
    };
    let debug = format!("{:?}", s);
    assert!(debug.contains("ELECTRONIC COMPUTERS"));
    assert!(debug.contains("Office of Technology"));
    assert!(debug.contains("3571"));
}
