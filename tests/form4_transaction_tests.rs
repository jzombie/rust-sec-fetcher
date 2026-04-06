/// Unit tests for [`sec_fetcher::models::Form4Transaction::code_description`].
use chrono::NaiveDate;
use rust_decimal_macros::dec;
use sec_fetcher::models::Form4Transaction;

fn txn(code: &str) -> Form4Transaction {
    Form4Transaction {
        filer_name: "Test User".to_string(),
        filer_cik: "0000000001".to_string(),
        is_director: false,
        is_officer: true,
        officer_title: Some("CEO".to_string()),
        is_ten_pct_owner: false,
        filing_date: Some(NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()),
        security_title: "Common Stock".to_string(),
        transaction_date: Some(NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()),
        transaction_code: code.to_string(),
        shares: dec!(1000),
        price_per_share: Some(dec!(150.00)),
        acquired_disposed: "A".to_string(),
        shares_owned_after: Some(dec!(10000)),
        is_derivative: false,
    }
}

#[test]
fn code_p_is_purchase() {
    assert_eq!(txn("P").code_description(), "Purchase");
}

#[test]
fn code_s_is_sale() {
    assert_eq!(txn("S").code_description(), "Sale");
}

#[test]
fn code_a_is_award_grant() {
    assert_eq!(txn("A").code_description(), "Award/Grant");
}

#[test]
fn code_f_is_tax_withholding() {
    assert_eq!(txn("F").code_description(), "Tax withholding");
}

#[test]
fn code_m_is_option_exercise() {
    assert_eq!(txn("M").code_description(), "Option exercise");
}

#[test]
fn code_x_is_itm_exercise() {
    assert_eq!(txn("X").code_description(), "In-the-money exercise");
}

#[test]
fn code_g_is_gift() {
    assert_eq!(txn("G").code_description(), "Gift");
}

#[test]
fn code_d_is_disposition_to_issuer() {
    assert_eq!(txn("D").code_description(), "Disposition to issuer");
}

#[test]
fn code_v_is_voluntary_report() {
    assert_eq!(txn("V").code_description(), "Voluntary report");
}

#[test]
fn code_j_is_other() {
    assert_eq!(txn("J").code_description(), "Other");
}

#[test]
fn unknown_code_is_generic_transaction() {
    assert_eq!(txn("Z").code_description(), "Transaction");
    assert_eq!(txn("").code_description(), "Transaction");
}
