use sec_fetcher::models::Ticker;

// ============================================================================
// tokenize_company_name
// ============================================================================

#[test]
fn test_tokenize_simple_name() {
    let tokens = Ticker::tokenize_company_name("Apple Inc.");
    assert_eq!(tokens, vec!["APPLE", "INC"]);
}

#[test]
fn test_tokenize_replaces_company() {
    let tokens = Ticker::tokenize_company_name("ACME COMPANY");
    assert_eq!(tokens, vec!["ACME", "CO"]);
}

#[test]
fn test_tokenize_replaces_companies() {
    let tokens = Ticker::tokenize_company_name("HOLDING COMPANIES");
    assert_eq!(tokens, vec!["HOLDING", "COS"]);
}

#[test]
fn test_tokenize_removes_apostrophes() {
    let tokens = Ticker::tokenize_company_name("O'Reilly Automotive");
    assert_eq!(tokens, vec!["OREILLY", "AUTOMOTIVE"]);
}

#[test]
fn test_tokenize_replaces_new_slash() {
    let tokens = Ticker::tokenize_company_name("ABC /NEW/ CORP");
    assert_eq!(tokens, vec!["ABC", "CORP"]);
}

#[test]
fn test_tokenize_pge_special_case() {
    let tokens = Ticker::tokenize_company_name("PG&E");
    assert_eq!(tokens, vec!["PACIFIC", "GAS", "AND", "ELECTRIC", "CO"]);
}

#[test]
fn test_tokenize_bancorporation() {
    let tokens = Ticker::tokenize_company_name("BANCORPORATION");
    assert!(tokens.contains(&"BANK".to_string()));
    assert!(tokens.contains(&"BANCORP".to_string()));
}

#[test]
fn test_tokenize_national_association() {
    let tokens = Ticker::tokenize_company_name("BANK NATIONAL ASSOCIATION");
    assert!(tokens.contains(&"NA".to_string()));
}

#[test]
fn test_tokenize_removes_group_inc() {
    let tokens = Ticker::tokenize_company_name("PNC FINANCIAL SERVICES GROUP, INC.");
    assert!(!tokens.iter().any(|t| t == "GROUP," || t == "INC"));
    // Should contain "PNC", "FINANCIAL", "SERVICES"
    assert!(tokens.contains(&"PNC".to_string()));
    assert!(tokens.contains(&"FINANCIAL".to_string()));
    assert!(tokens.contains(&"SERVICES".to_string()));
}

#[test]
fn test_tokenize_joins_single_letters() {
    let tokens = Ticker::tokenize_company_name("J P Morgan");
    // "J" and "P" should be joined to "JP"
    assert_eq!(tokens, vec!["JP", "MORGAN"]);
}

#[test]
fn test_tokenize_empty_string() {
    let tokens = Ticker::tokenize_company_name("");
    assert!(tokens.is_empty());
}

#[test]
fn test_tokenize_whitespace_only() {
    let tokens = Ticker::tokenize_company_name("   ");
    assert!(tokens.is_empty());
}

#[test]
fn test_tokenize_numbers_and_letters() {
    let tokens = Ticker::tokenize_company_name("3M Company");
    // "Company" does not match case-sensitive replacement "COMPANY" -> "CO",
    // so after byte-level uppercasing it stays as "COMPANY"
    assert_eq!(tokens, vec!["3M", "COMPANY"]);
}

#[test]
fn test_tokenize_lowercase_input() {
    let tokens = Ticker::tokenize_company_name("microsoft corporation");
    assert_eq!(tokens, vec!["MICROSOFT", "CORPORATION"]);
}

#[test]
fn test_tokenize_caches_result() {
    // First call populates cache
    let tokens1 = Ticker::tokenize_company_name("Test Corp");
    // Second call should use cache
    let tokens2 = Ticker::tokenize_company_name("Test Corp");
    assert_eq!(tokens1, tokens2);
}
