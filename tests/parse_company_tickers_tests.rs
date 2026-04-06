/// Unit tests for [`sec_fetcher::parsers::parse_company_tickers_json`] and
/// [`sec_fetcher::parsers::parse_ticker_txt`].
use sec_fetcher::enums::TickerOrigin;
use sec_fetcher::parsers::{parse_company_tickers_json, parse_ticker_txt};
use serde_json::json;

// ── parse_company_tickers_json ────────────────────────────────────────────────

#[test]
fn parses_single_entry() {
    let data = json!({
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    });
    let tickers = parse_company_tickers_json(&data).unwrap();
    assert_eq!(tickers.len(), 1);
    let t = &tickers[0];
    assert_eq!(t.cik.value, 320193);
    assert_eq!(t.symbol.as_str(), "AAPL");
    assert_eq!(t.company_name, "Apple Inc.");
    assert_eq!(t.origin, TickerOrigin::PrimaryListing);
}

#[test]
fn parses_multiple_entries() {
    let data = json!({
        "0": {"cik_str": 320193,  "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019,  "ticker": "MSFT", "title": "Microsoft Corporation"},
        "2": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA Corporation"}
    });
    let tickers = parse_company_tickers_json(&data).unwrap();
    assert_eq!(tickers.len(), 3);
}

#[test]
fn trims_whitespace_from_title() {
    let data = json!({
        "0": {"cik_str": 1, "ticker": "X", "title": "  Padded Name  "}
    });
    let tickers = parse_company_tickers_json(&data).unwrap();
    assert_eq!(tickers[0].company_name, "Padded Name");
}

#[test]
fn empty_object_returns_empty_vec() {
    let data = json!({});
    let tickers = parse_company_tickers_json(&data).unwrap();
    assert!(tickers.is_empty());
}

#[test]
fn non_object_input_returns_empty_vec() {
    let data = json!([1, 2, 3]);
    let tickers = parse_company_tickers_json(&data).unwrap();
    assert!(tickers.is_empty());
}

// ── parse_ticker_txt ──────────────────────────────────────────────────────────

#[test]
fn parses_tab_separated_lines() {
    let text = "aapl\t320193\nmsft\t789019\n";
    let tickers = parse_ticker_txt(text);
    assert_eq!(tickers.len(), 2);
    // TickerSymbol normalizes to uppercase on construction
    assert_eq!(tickers[0].symbol.as_str(), "AAPL");
    assert_eq!(tickers[0].cik.value, 320193);
    assert_eq!(tickers[0].origin, TickerOrigin::DerivedInstrument);
    assert_eq!(tickers[0].company_name, "");
}

#[test]
fn skips_empty_lines() {
    let text = "aapl\t320193\n\n\nmsft\t789019\n";
    let tickers = parse_ticker_txt(text);
    assert_eq!(tickers.len(), 2);
}

#[test]
fn skips_malformed_lines_no_tab() {
    let text = "AAPL320193\nmsft\t789019\n";
    let tickers = parse_ticker_txt(text);
    assert_eq!(tickers.len(), 1);
    assert_eq!(tickers[0].symbol.as_str(), "MSFT");
}

#[test]
fn skips_lines_with_non_numeric_cik() {
    let text = "aapl\tNOTANUMBER\nmsft\t789019\n";
    let tickers = parse_ticker_txt(text);
    assert_eq!(tickers.len(), 1);
}

#[test]
fn empty_text_returns_empty_vec() {
    let tickers = parse_ticker_txt("");
    assert!(tickers.is_empty());
}

#[test]
fn handles_whitespace_only_lines() {
    let text = "   \n\t\naapl\t320193\n";
    let tickers = parse_ticker_txt(text);
    assert_eq!(tickers.len(), 1);
}
