/// Unit tests for [`sec_fetcher::enums::CacheNamespacePrefix`].
use sec_fetcher::enums::CacheNamespacePrefix;

#[test]
fn test_latest_funds_year_value() {
    let prefix = CacheNamespacePrefix::LatestFundsYear;
    let val = prefix.value();
    assert_eq!(
        val,
        b"network::fetch_investment_company_series_and_class_dataset::latest_funds_year"
    );
}

#[test]
fn test_company_ticker_fuzzy_match_value() {
    let prefix = CacheNamespacePrefix::CompanyTickerFuzzyMatch;
    let val = prefix.value();
    assert_eq!(val, b"CompanyTicker::get_by_fuzzy_matched_name");
}
