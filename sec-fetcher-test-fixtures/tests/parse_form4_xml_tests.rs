//! Integration tests for [`sec_fetcher::parsers::parse_form4_xml`].
//!
//! Real-data tests load Arthur D. Levinson's Form 4 for Apple Inc. (filed
//! 2026-02-26, accession 0001214128-26-000004).  The filing contains exactly
//! two transactions:
//!
//! | Row | Table         | Security              | Code | Shares | A/D | After     |
//! |-----|---------------|-----------------------|------|--------|-----|-----------|
//! | 0   | nonDerivative | Common Stock          | G    | 1 113  | D   | 4 069 576 |
//! | 1   | derivative    | Restricted Stock Unit | A    | 1 011  | A   | 1 011     |
//!
//! Run `cargo run --bin refresh-test-fixtures` to recreate the fixture file.

use chrono::NaiveDate;
use indoc::indoc;
use rust_decimal_macros::dec;
use sec_fetcher::models::Form4Transaction;
use sec_fetcher::parsers::parse_form4_xml;

mod common;

fn load_text_fixture(name: &str) -> String {
    common::fixture_string(name)
}

// ── Real-data tests (AAPL_form4_levinson.xml) ────────────────────────────────
//
// Arthur D. Levinson (CIK 0001214128) is a director of Apple Inc. (CIK 0000320193).

fn levinson_form4() -> Vec<Form4Transaction> {
    let xml = load_text_fixture("AAPL_form4_levinson.xml");
    let filing_date = NaiveDate::from_ymd_opt(2026, 2, 26).unwrap();
    parse_form4_xml(&xml, Some(filing_date)).unwrap()
}

#[test]
fn levinson_two_transactions_total() {
    assert_eq!(levinson_form4().len(), 2);
}

#[test]
fn levinson_sorted_newest_first() {
    let txns = levinson_form4();
    // G (gift, nonDerivative) was 2026-02-26; RSU grant (derivative) was 2026-02-24.
    assert_eq!(
        txns[0].transaction_date,
        Some(NaiveDate::from_ymd_opt(2026, 2, 26).unwrap())
    );
    assert_eq!(
        txns[1].transaction_date,
        Some(NaiveDate::from_ymd_opt(2026, 2, 24).unwrap())
    );
    assert!(
        txns[0].transaction_date > txns[1].transaction_date,
        "results must be ordered newest-first"
    );
}

#[test]
fn levinson_reporter_identity_on_both_rows() {
    for txn in &levinson_form4() {
        assert_eq!(txn.filer_name, "LEVINSON ARTHUR D");
        assert_eq!(txn.filer_cik, "0001214128");
        assert!(
            txn.is_director,
            "Levinson is flagged as a director in the XML"
        );
        assert!(!txn.is_officer, "Levinson is not an officer");
        assert!(
            txn.officer_title.is_none(),
            "no officer_title when isOfficer=0"
        );
        assert!(!txn.is_ten_pct_owner);
    }
}

#[test]
fn levinson_reporter_cik_is_not_issuer_cik() {
    // Reporter (Levinson, 0001214128) must be distinct from Issuer (Apple, 0000320193).
    for txn in &levinson_form4() {
        assert_ne!(
            txn.filer_cik, "0000320193",
            "filer_cik must be the reporter's CIK, not Apple's issuer CIK"
        );
        assert_eq!(txn.filer_cik, "0001214128");
    }
}

#[test]
fn levinson_nonderivative_row_fields_exact() {
    let txns = levinson_form4();
    // Row 0 (newest, 2026-02-26): Common Stock gift.
    let row = &txns[0];
    assert_eq!(row.security_title, "Common Stock");
    assert_eq!(row.transaction_code, "G");
    assert_eq!(row.shares, dec!(1113));
    assert!(
        row.price_per_share.is_none(),
        "gift at price=0 must produce None, not Some(0)"
    );
    assert_eq!(row.acquired_disposed, "D");
    assert_eq!(row.shares_owned_after, Some(dec!(4069576)));
    assert!(
        !row.is_derivative,
        "nonDerivativeTransaction must have is_derivative=false"
    );
}

#[test]
fn levinson_derivative_row_fields_exact() {
    let txns = levinson_form4();
    // Row 1 (older, 2026-02-24): RSU grant.
    let row = &txns[1];
    assert_eq!(row.security_title, "Restricted Stock Unit");
    assert_eq!(row.transaction_code, "A");
    assert_eq!(row.shares, dec!(1011));
    assert!(
        row.price_per_share.is_none(),
        "RSU grant at price=0 must produce None"
    );
    assert_eq!(row.acquired_disposed, "A");
    assert_eq!(row.shares_owned_after, Some(dec!(1011)));
    assert!(
        row.is_derivative,
        "derivativeTransaction must have is_derivative=true"
    );
}

#[test]
fn levinson_filing_date_propagated_to_both_rows() {
    let expected = NaiveDate::from_ymd_opt(2026, 2, 26).unwrap();
    for txn in &levinson_form4() {
        assert_eq!(txn.filing_date, Some(expected));
    }
}

/// State isolation: every meaningful field differs between the two rows in the
/// real fixture.  If the parser bleeds state from one row into the next, the
/// `assert_ne!` checks here will catch it against verifiable real values.
#[test]
fn levinson_state_does_not_bleed_between_rows() {
    let txns = levinson_form4();
    let nd = &txns[0]; // nonDerivative: "Common Stock", G, 1113 sh, D, 4069576 after, 2026-02-26
    let d = &txns[1]; // derivative:    "Restricted Stock Unit", A, 1011 sh, A, 1011 after, 2026-02-24

    assert_ne!(nd.security_title, d.security_title); // "Common Stock" vs "Restricted Stock Unit"
    assert_ne!(nd.transaction_code, d.transaction_code); // "G" vs "A"
    assert_ne!(nd.shares, d.shares); // dec!(1113) vs dec!(1011)
    assert_ne!(nd.acquired_disposed, d.acquired_disposed); // "D" vs "A"
    assert_ne!(nd.shares_owned_after, d.shares_owned_after); // Some(4069576) vs Some(1011)
    assert_ne!(nd.is_derivative, d.is_derivative); // false vs true
    assert_ne!(nd.transaction_date, d.transaction_date); // 2026-02-26 vs 2026-02-24
}

// ── Edge-case tests (minimal inline XML — no real-world equivalent) ───────────

#[test]
fn filing_date_none_is_stored_as_none() {
    let xml = load_text_fixture("AAPL_form4_levinson.xml");
    let result = parse_form4_xml(&xml, None).unwrap();
    assert_eq!(result.len(), 2);
    for txn in &result {
        assert_eq!(txn.filing_date, None);
    }
}

#[test]
fn empty_xml_returns_empty_vec() {
    let xml = r#"<?xml version="1.0"?><ownershipDocument></ownershipDocument>"#;
    assert!(parse_form4_xml(xml, None).unwrap().is_empty());
}

#[test]
fn invalid_transaction_date_parses_to_none() {
    // No real EDGAR Form 4 has a malformed date, so we exercise this path inline.
    let xml = indoc! {r#"
        <?xml version="1.0"?><ownershipDocument>
          <reportingOwner>
            <reportingOwnerId>
              <rptOwnerCik>0001</rptOwnerCik>
              <rptOwnerName>Test User</rptOwnerName>
            </reportingOwnerId>
            <reportingOwnerRelationship>
              <isDirector>0</isDirector><isOfficer>0</isOfficer><is10PercentOwner>0</is10PercentOwner>
            </reportingOwnerRelationship>
          </reportingOwner>
          <nonDerivativeTable>
            <nonDerivativeTransaction>
              <securityTitle><value>Common Stock</value></securityTitle>
              <transactionDate><value>not-a-date</value></transactionDate>
              <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
              <transactionAmounts>
                <transactionShares><value>100</value></transactionShares>
                <transactionPricePerShare><value>50</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
              </transactionAmounts>
            </nonDerivativeTransaction>
          </nonDerivativeTable>
        </ownershipDocument>
    "#};
    let result = parse_form4_xml(xml, None).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result[0].transaction_date.is_none());
}

/// A positive price must round-trip as `Some(exact_value)`, not `None`.
/// Uses a minimal inline fixture because the real Levinson Form 4 has price=0
/// on both transactions (a charitable gift and an RSU grant).
#[test]
fn positive_price_round_trips_as_some_exact_value() {
    let xml = indoc! {r#"
        <?xml version="1.0"?><ownershipDocument>
          <reportingOwner>
            <reportingOwnerId>
              <rptOwnerCik>0001111111</rptOwnerCik>
              <rptOwnerName>Seller Person</rptOwnerName>
            </reportingOwnerId>
            <reportingOwnerRelationship>
              <isDirector>0</isDirector><isOfficer>1</isOfficer>
              <officerTitle>CFO</officerTitle><is10PercentOwner>0</is10PercentOwner>
            </reportingOwnerRelationship>
          </reportingOwner>
          <nonDerivativeTable>
            <nonDerivativeTransaction>
              <securityTitle><value>Common Stock</value></securityTitle>
              <transactionDate><value>2025-06-15</value></transactionDate>
              <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
              <transactionAmounts>
                <transactionShares><value>500</value></transactionShares>
                <transactionPricePerShare><value>189.47</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
              </transactionAmounts>
            </nonDerivativeTransaction>
          </nonDerivativeTable>
        </ownershipDocument>
    "#};
    let result = parse_form4_xml(xml, None).unwrap();
    assert_eq!(result.len(), 1);
    let txn = &result[0];
    assert_eq!(txn.price_per_share, Some(dec!(189.47)));
    assert_ne!(
        txn.price_per_share, None,
        "positive price must not become None"
    );
    assert_ne!(txn.price_per_share, Some(dec!(0)));
    // Shares must be the shares field value, not contaminated by the price.
    assert_eq!(txn.shares, dec!(500));
    assert_ne!(txn.shares, dec!(189));
    // The XML above has isOfficer=1 / officerTitle=CFO — assert those too so a
    // parser that always returns is_officer=false / officer_title=None is caught.
    assert!(txn.is_officer, "isOfficer=1 in XML must parse to true");
    assert!(!txn.is_director, "isDirector=0 in XML must parse to false");
    assert_eq!(
        txn.officer_title.as_deref(),
        Some("CFO"),
        "officerTitle must be Some(\"CFO\") when isOfficer=1"
    );
    assert_ne!(
        txn.officer_title, None,
        "officer_title must not be None when isOfficer=1"
    );
}

/// `is10PercentOwner=1` must parse to `is_ten_pct_owner=true`.
/// The real Levinson fixture always has 0 here, so we use a minimal inline fixture.
#[test]
fn is_ten_pct_owner_true_parsed_from_xml() {
    let xml = indoc! {r#"
        <?xml version="1.0"?><ownershipDocument>
          <reportingOwner>
            <reportingOwnerId>
              <rptOwnerCik>0005555555</rptOwnerCik>
              <rptOwnerName>Large Holder</rptOwnerName>
            </reportingOwnerId>
            <reportingOwnerRelationship>
              <isDirector>0</isDirector><isOfficer>0</isOfficer><is10PercentOwner>1</is10PercentOwner>
            </reportingOwnerRelationship>
          </reportingOwner>
          <nonDerivativeTable>
            <nonDerivativeTransaction>
              <securityTitle><value>Common Stock</value></securityTitle>
              <transactionDate><value>2025-01-10</value></transactionDate>
              <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
              <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
                <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
              </transactionAmounts>
            </nonDerivativeTransaction>
          </nonDerivativeTable>
        </ownershipDocument>
    "#};
    let result = parse_form4_xml(xml, None).unwrap();
    assert_eq!(result.len(), 1);
    let txn = &result[0];
    assert!(
        txn.is_ten_pct_owner,
        "is10PercentOwner=1 in XML must parse to is_ten_pct_owner=true"
    );
    assert!(!txn.is_director);
    assert!(!txn.is_officer);
    assert!(txn.officer_title.is_none());
    // Cross-check: the inline fixture explicitly has is10PercentOwner=1, so true is expected.
    assert!(
        txn.is_ten_pct_owner,
        "sanity: just confirmed the inline fixture gives true"
    );
}
