/// Unit tests for [`sec_fetcher::parsers::parse_form4_xml`].
///
/// Uses inline XML fixtures that mirror the SEC Form 4 schema.  Each fixture
/// is intentionally small so expected values can be verified by hand.
use chrono::NaiveDate;
use rust_decimal_macros::dec;
use sec_fetcher::parsers::parse_form4_xml;

// ── Fixtures ──────────────────────────────────────────────────────────────────

const SIMPLE_NONDERIVATIVE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>Apple Inc.</issuerName>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001214156</rptOwnerCik>
      <rptOwnerName>Cook Timothy D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <officerTitle>CEO</officerTitle>
      <is10PercentOwner>0</is10PercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-11-01</value></transactionDate>
      <transactionCoding>
        <transactionCode>S</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>100000</value></transactionShares>
        <transactionPricePerShare><value>222.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>3300000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"#;

const TWO_TRANSACTIONS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001111111</rptOwnerCik>
      <rptOwnerName>Doe Jane</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>0</isOfficer>
      <is10PercentOwner>0</is10PercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-03-15</value></transactionDate>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>100.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>25000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-01-10</value></transactionDate>
      <transactionCoding>
        <transactionCode>A</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>20000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"#;

const DERIVATIVE_TRANSACTION: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0002222222</rptOwnerCik>
      <rptOwnerName>Smith Bob</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <officerTitle>CFO</officerTitle>
      <is10PercentOwner>0</is10PercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <derivativeTable>
    <derivativeTransaction>
      <securityTitle><value>Stock Option</value></securityTitle>
      <transactionDate><value>2024-06-01</value></transactionDate>
      <transactionCoding>
        <transactionCode>M</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>50000</value></transactionShares>
        <transactionPricePerShare><value>45.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>"#;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn parses_single_nonderivative_transaction() {
    let result = parse_form4_xml(
        SIMPLE_NONDERIVATIVE,
        Some(NaiveDate::from_ymd_opt(2024, 11, 5).unwrap()),
    )
    .unwrap();
    assert_eq!(result.len(), 1);

    let txn = &result[0];
    assert_eq!(txn.filer_name, "Cook Timothy D");
    assert_eq!(txn.filer_cik, "0001214156");
    assert!(!txn.is_director);
    assert!(txn.is_officer);
    assert_eq!(txn.officer_title.as_deref(), Some("CEO"));
    assert!(!txn.is_ten_pct_owner);
    assert_eq!(txn.security_title, "Common Stock");
    assert_eq!(
        txn.transaction_date,
        Some(NaiveDate::from_ymd_opt(2024, 11, 1).unwrap())
    );
    assert_eq!(txn.transaction_code, "S");
    assert_eq!(txn.shares, dec!(100000));
    assert_eq!(txn.price_per_share, Some(dec!(222.50)));
    assert_eq!(txn.acquired_disposed, "D");
    assert_eq!(txn.shares_owned_after, Some(dec!(3300000)));
    assert!(!txn.is_derivative);
}

#[test]
fn preserves_filing_date() {
    let filing_date = NaiveDate::from_ymd_opt(2024, 11, 5).unwrap();
    let result = parse_form4_xml(SIMPLE_NONDERIVATIVE, Some(filing_date)).unwrap();
    assert_eq!(result[0].filing_date, Some(filing_date));
}

#[test]
fn filing_date_none_is_allowed() {
    let result = parse_form4_xml(SIMPLE_NONDERIVATIVE, None).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].filing_date, None);
}

#[test]
fn two_transactions_sorted_newest_first() {
    let result = parse_form4_xml(TWO_TRANSACTIONS, None).unwrap();
    assert_eq!(result.len(), 2);
    // March should come before January
    assert_eq!(
        result[0].transaction_date,
        Some(NaiveDate::from_ymd_opt(2024, 3, 15).unwrap())
    );
    assert_eq!(
        result[1].transaction_date,
        Some(NaiveDate::from_ymd_opt(2024, 1, 10).unwrap())
    );
}

#[test]
fn two_transactions_filer_identity_consistent() {
    let result = parse_form4_xml(TWO_TRANSACTIONS, None).unwrap();
    for txn in &result {
        assert_eq!(txn.filer_name, "Doe Jane");
        assert_eq!(txn.filer_cik, "0001111111");
        assert!(txn.is_director);
        assert!(!txn.is_officer);
        assert!(txn.officer_title.is_none());
    }
}

#[test]
fn zero_price_treated_as_none() {
    let result = parse_form4_xml(TWO_TRANSACTIONS, None).unwrap();
    // The older transaction (Award, price=0) should have no price_per_share
    let award = result.iter().find(|t| t.transaction_code == "A").unwrap();
    assert!(award.price_per_share.is_none());
}

#[test]
fn derivative_transaction_flagged() {
    let result = parse_form4_xml(DERIVATIVE_TRANSACTION, None).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result[0].is_derivative);
    assert_eq!(result[0].security_title, "Stock Option");
    assert_eq!(result[0].transaction_code, "M");
}

#[test]
fn empty_xml_returns_empty_vec() {
    let xml = r#"<?xml version="1.0"?><ownershipDocument></ownershipDocument>"#;
    let result = parse_form4_xml(xml, None).unwrap();
    assert!(result.is_empty());
}

#[test]
fn invalid_date_is_skipped_gracefully() {
    let xml = r#"<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001</rptOwnerCik>
      <rptOwnerName>Test User</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>0</isOfficer>
      <is10PercentOwner>0</is10PercentOwner>
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
</ownershipDocument>"#;
    let result = parse_form4_xml(xml, None).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result[0].transaction_date.is_none());
}
