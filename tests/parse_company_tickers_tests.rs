/// Unit tests for [`sec_fetcher::parsers::parse_company_tickers_json`] and
/// [`sec_fetcher::parsers::parse_ticker_txt`].
///
/// Two sections:
/// 1. Inline-data unit tests that validate the parser mechanics.
/// 2. Real-data tests against the live `company_tickers.json` fixture (~10 000 entries).
///
/// Run `cargo run --bin refresh-test-fixtures` to recreate the fixture file.
use flate2::read::GzDecoder;
use sec_fetcher::enums::TickerOrigin;
use sec_fetcher::parsers::{parse_company_tickers_json, parse_ticker_txt};
use serde_json::json;
use std::fs::File;
use std::path::PathBuf;

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

// ── Real-data tests (company_tickers.json fixture) ────────────────────────────
//
// Real-world anchor entries verified against SEC EDGAR on 2026-04-05:
//   AAPL → CIK 320193  ("Apple Inc.")
//   MSFT → CIK 789019  ("MICROSOFT CORP")
//   NVDA → CIK 1045810 ("NVIDIA CORP")

fn load_json_fixture(name: &str) -> serde_json::Value {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(format!("{}.gz", name));
    let file = File::open(&path).unwrap_or_else(|_| {
        panic!(
            "missing fixture '{}' — run `cargo run --bin refresh-test-fixtures`",
            name
        )
    });
    serde_json::from_reader(GzDecoder::new(file)).expect("fixture is not valid JSON")
}

fn all_company_tickers() -> Vec<sec_fetcher::models::Ticker> {
    let data = load_json_fixture("company_tickers.json");
    parse_company_tickers_json(&data).unwrap()
}

fn by_symbol<'a>(
    tickers: &'a [sec_fetcher::models::Ticker],
    symbol: &str,
) -> &'a sec_fetcher::models::Ticker {
    tickers
        .iter()
        .find(|t| t.symbol.as_str() == symbol)
        .unwrap_or_else(|| {
            panic!(
                "ticker '{}' not found in company_tickers.json fixture",
                symbol
            )
        })
}

#[test]
fn fixture_has_over_ten_thousand_entries() {
    assert!(
        all_company_tickers().len() > 10_000,
        "company_tickers.json must have 10 000+ operating-company entries"
    );
}

#[test]
fn aapl_entry_is_exact() {
    let tickers = all_company_tickers();
    let aapl = by_symbol(&tickers, "AAPL");
    assert_eq!(aapl.cik.value, 320193);
    assert_eq!(aapl.symbol.as_str(), "AAPL");
    assert!(
        aapl.company_name.to_lowercase().contains("apple"),
        "company_name '{}' should mention Apple",
        aapl.company_name
    );
    assert_eq!(aapl.origin, TickerOrigin::PrimaryListing);
}

#[test]
fn msft_entry_is_exact() {
    let tickers = all_company_tickers();
    let msft = by_symbol(&tickers, "MSFT");
    assert_eq!(msft.cik.value, 789019);
    assert_eq!(msft.symbol.as_str(), "MSFT");
    assert!(msft.company_name.to_lowercase().contains("microsoft"));
    assert_eq!(msft.origin, TickerOrigin::PrimaryListing);
}

#[test]
fn nvda_entry_is_exact() {
    let tickers = all_company_tickers();
    let nvda = by_symbol(&tickers, "NVDA");
    assert_eq!(nvda.cik.value, 1045810);
    assert_eq!(nvda.symbol.as_str(), "NVDA");
    assert!(nvda.company_name.to_lowercase().contains("nvidia"));
    assert_eq!(nvda.origin, TickerOrigin::PrimaryListing);
}

/// AAPL, MSFT, and NVDA each have a distinct CIK.  If the parser blended two
/// entries together, at least one `assert_ne!` below would fail.
#[test]
fn aapl_msft_nvda_have_distinct_ciks() {
    let tickers = all_company_tickers();
    let aapl = by_symbol(&tickers, "AAPL");
    let msft = by_symbol(&tickers, "MSFT");
    let nvda = by_symbol(&tickers, "NVDA");
    assert_ne!(aapl.cik.value, msft.cik.value); // 320193 ≠ 789019
    assert_ne!(aapl.cik.value, nvda.cik.value); // 320193 ≠ 1045810
    assert_ne!(msft.cik.value, nvda.cik.value); // 789019 ≠ 1045810
}

/// Every symbol returned by the parser must be uppercase — TickerSymbol
/// normalizes on construction and the parser must honour that contract.
#[test]
fn all_symbols_are_uppercase() {
    for t in &all_company_tickers() {
        let sym = t.symbol.as_str();
        assert_eq!(
            sym,
            sym.to_uppercase(),
            "symbol '{}' (CIK {}) is not fully uppercase — normalization failed",
            sym,
            t.cik.value
        );
    }
}

/// No two entries in the parsed output should share the same symbol.
#[test]
fn no_duplicate_symbols() {
    use std::collections::HashMap;
    let tickers = all_company_tickers();
    let mut seen: HashMap<&str, u64> = HashMap::new();
    for t in &tickers {
        let sym = t.symbol.as_str();
        if let Some(prev_cik) = seen.insert(sym, t.cik.value) {
            panic!(
                "symbol '{}' appears twice: CIK {} and CIK {}",
                sym, prev_cik, t.cik.value
            );
        }
    }
}
