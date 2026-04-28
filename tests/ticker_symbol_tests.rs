use sec_fetcher::models::TickerSymbol;

#[test]
fn test_ticker_symbol_new_trims_whitespace() {
    let s = TickerSymbol::new("  AAPL  ");
    assert_eq!(s.as_str(), "AAPL");
}

#[test]
fn test_ticker_symbol_new_converts_to_uppercase() {
    let s = TickerSymbol::new("aapl");
    assert_eq!(s.as_str(), "AAPL");
}

#[test]
fn test_ticker_symbol_new_replaces_dot_with_dash() {
    let s = TickerSymbol::new("BRK.B");
    assert_eq!(s.as_str(), "BRK-B");
}

#[test]
fn test_ticker_symbol_new_replaces_slash_with_dash() {
    let s = TickerSymbol::new("BRK/B");
    assert_eq!(s.as_str(), "BRK-B");
}

#[test]
fn test_ticker_symbol_new_handles_mixed_normalization() {
    let s = TickerSymbol::new("  brk.b  ");
    assert_eq!(s.as_str(), "BRK-B");
}

#[test]
fn test_ticker_symbol_display() {
    let s = TickerSymbol::new("AAPL");
    assert_eq!(format!("{}", s), "AAPL");
}

#[test]
fn test_ticker_symbol_deref() {
    let s = TickerSymbol::new("MSFT");
    assert_eq!(*s, *"MSFT");
}

#[test]
fn test_ticker_symbol_as_ref_str() {
    let s = TickerSymbol::new("GOOG");
    let r: &str = s.as_ref();
    assert_eq!(r, "GOOG");
}

#[test]
fn test_ticker_symbol_from_str() {
    let s: TickerSymbol = "NVDA".into();
    assert_eq!(s.as_str(), "NVDA");
}

#[test]
fn test_ticker_symbol_from_string() {
    let s: TickerSymbol = String::from("amd").into();
    assert_eq!(s.as_str(), "AMD");
}

#[test]
fn test_ticker_symbol_eq() {
    let a = TickerSymbol::new("AAPL");
    let b = TickerSymbol::new("aapl");
    let c = TickerSymbol::new("MSFT");
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_ticker_symbol_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(TickerSymbol::new("AAPL"));
    set.insert(TickerSymbol::new("aapl")); // same after normalization
    assert_eq!(set.len(), 1);
}

#[test]
fn test_ticker_symbol_round_trip_serialize_deserialize() {
    let original = TickerSymbol::new("BRK.B");
    let json = serde_json::to_string(&original).unwrap();
    let deserialized: TickerSymbol = serde_json::from_str(&json).unwrap();
    // Deserialization re-normalizes via custom Deserialize impl
    assert_eq!(deserialized.as_str(), "BRK-B");
}
