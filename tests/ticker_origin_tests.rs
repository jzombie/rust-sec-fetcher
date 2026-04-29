/// Unit tests for [`sec_fetcher::enums::TickerOrigin`].
use sec_fetcher::enums::TickerOrigin;

#[test]
fn test_ticker_origin_display_primary_listing() {
    let origin = TickerOrigin::PrimaryListing;
    assert_eq!(format!("{}", origin), "PrimaryListing");
}

#[test]
fn test_ticker_origin_display_derived_instrument() {
    let origin = TickerOrigin::DerivedInstrument;
    assert_eq!(format!("{}", origin), "DerivedInstrument");
}

#[test]
fn test_ticker_origin_display_investment_company() {
    let origin = TickerOrigin::InvestmentCompany;
    assert_eq!(format!("{}", origin), "InvestmentCompany");
}

#[test]
fn test_ticker_origin_debug() {
    let origin = TickerOrigin::PrimaryListing;
    let debug = format!("{:?}", origin);
    assert_eq!(debug, "PrimaryListing");
}

#[test]
fn test_ticker_origin_equality() {
    assert_eq!(TickerOrigin::PrimaryListing, TickerOrigin::PrimaryListing);
    assert_ne!(
        TickerOrigin::PrimaryListing,
        TickerOrigin::DerivedInstrument
    );
}

#[test]
fn test_ticker_origin_clone() {
    let a = TickerOrigin::InvestmentCompany;
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_ticker_origin_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(TickerOrigin::PrimaryListing);
    set.insert(TickerOrigin::PrimaryListing);
    assert_eq!(set.len(), 1, "Duplicate insert should be deduplicated");
}
