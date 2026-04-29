/// Unit tests for [`sec_fetcher::models::Ticker::get_by_fuzzy_matched_name`].
///
/// These tests exercise the fuzzy matching algorithm without a preprocessor
/// cache (passing `None` for the cache parameter).
use sec_fetcher::enums::TickerOrigin;
use sec_fetcher::models::{Cik, Ticker, TickerSymbol};

fn make_ticker(symbol: &str, cik: u64, name: &str, origin: TickerOrigin) -> Ticker {
    Ticker {
        cik: Cik::from_u64(cik).unwrap(),
        symbol: TickerSymbol::new(symbol),
        company_name: name.to_string(),
        origin,
    }
}

fn apple_ticker() -> Ticker {
    make_ticker("AAPL", 320193, "Apple Inc.", TickerOrigin::PrimaryListing)
}

fn microsoft_ticker() -> Ticker {
    make_ticker(
        "MSFT",
        789019,
        "Microsoft Corporation",
        TickerOrigin::PrimaryListing,
    )
}

fn google_ticker() -> Ticker {
    make_ticker(
        "GOOGL",
        1652044,
        "Alphabet Inc.",
        TickerOrigin::PrimaryListing,
    )
}

#[test]
fn test_fuzzy_match_exact_name() {
    let tickers = vec![apple_ticker(), microsoft_ticker(), google_ticker()];
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "Apple Inc.", None);
    assert!(result.is_some());
    assert_eq!(result.unwrap().symbol.to_string(), "AAPL");
}

#[test]
fn test_fuzzy_match_partial_name() {
    let tickers = vec![apple_ticker(), microsoft_ticker(), google_ticker()];
    // "Apple Inc" hits 2/2 tokens (100%) — well above the 60% threshold
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "Apple Inc", None);
    assert!(result.is_some(), "Should match Apple Inc to Apple Inc.");
    assert_eq!(result.unwrap().symbol.to_string(), "AAPL");
}

#[test]
fn test_fuzzy_match_no_match_returns_none() {
    let tickers = vec![apple_ticker(), microsoft_ticker()];
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "NonExistentCompanyXYZ", None);
    assert!(
        result.is_none(),
        "Should return None for non-matching query"
    );
}

#[test]
fn test_fuzzy_match_empty_tickers() {
    let tickers: Vec<Ticker> = vec![];
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "Apple", None);
    assert!(result.is_none(), "Empty ticker list should return None");
}

#[test]
fn test_fuzzy_match_empty_query() {
    let tickers = vec![apple_ticker()];
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "", None);
    assert!(result.is_none(), "Empty query should return None");
}

#[test]
fn test_fuzzy_match_preferred_over_derived() {
    let primary = apple_ticker();
    let derived = Ticker {
        cik: Cik::from_u64(320193).unwrap(),
        symbol: TickerSymbol::new("AAPL-WT"),
        company_name: "Apple Inc.".to_string(),
        origin: TickerOrigin::DerivedInstrument,
    };
    let tickers = vec![derived.clone(), primary];
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "Apple Inc.", None);
    assert!(result.is_some());
    // The primary listing should score higher due to common-stock boost
    assert_eq!(result.unwrap().origin, TickerOrigin::PrimaryListing);
}

#[test]
fn test_fuzzy_match_company_name_with_comma() {
    let pnc = make_ticker(
        "PNC",
        713676,
        "PNC FINANCIAL SERVICES GROUP, INC.",
        TickerOrigin::PrimaryListing,
    );
    let tickers = vec![pnc];
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "PNC Financial Services", None);
    assert!(
        result.is_some(),
        "Should match PNC despite 'GROUP, INC.' being filtered"
    );
    assert_eq!(result.unwrap().symbol.to_string(), "PNC");
}

#[test]
fn test_fuzzy_match_company_abbreviation() {
    let tickers = vec![
        make_ticker(
            "JPM",
            19617,
            "JPMORGAN CHASE & CO",
            TickerOrigin::PrimaryListing,
        ),
        make_ticker(
            "BAC",
            70858,
            "BANK OF AMERICA CORP",
            TickerOrigin::PrimaryListing,
        ),
    ];
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "JPMorgan Chase", None);
    assert!(result.is_some(), "Should match JPMorgan Chase & Co");
    assert_eq!(result.unwrap().symbol.to_string(), "JPM");
}

#[test]
fn test_fuzzy_match_matches_correct_company_from_multiple() {
    let tickers = vec![
        make_ticker("AAPL", 320193, "Apple Inc.", TickerOrigin::PrimaryListing),
        make_ticker(
            "APLE",
            1608676,
            "Apple Hospitality REIT Inc.",
            TickerOrigin::PrimaryListing,
        ),
        make_ticker(
            "APPN",
            1649801,
            "Appian Corporation",
            TickerOrigin::PrimaryListing,
        ),
    ];
    // "Apple Inc" gives 2/2 (100%) vs Apple Inc., 2/4 (50%) vs Apple Hospitality => threshold 0.6
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "Apple Inc", None);
    assert!(result.is_some());
    // Apple Inc. should score higher than Apple Hospitality for query "Apple Inc"
    assert_eq!(result.unwrap().symbol.to_string(), "AAPL");
}

#[test]
fn test_fuzzy_match_national_association_normalization() {
    let tickers = vec![make_ticker(
        "BAC",
        70858,
        "BANK OF AMERICA CORP NATIONAL ASSOCIATION",
        TickerOrigin::PrimaryListing,
    )];
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "Bank of America NA", None);
    assert!(result.is_some(), "Should match with NA abbreviation");
}

#[test]
fn test_fuzzy_match_single_letter_joins() {
    let tickers = vec![make_ticker(
        "ABC",
        12345,
        "A B C Company",
        TickerOrigin::PrimaryListing,
    )];
    let result = Ticker::get_by_fuzzy_matched_name(&tickers, "ABC Company", None);
    assert!(
        result.is_some(),
        "Should match where single letters join together"
    );
    assert_eq!(result.unwrap().symbol.to_string(), "ABC");
}
