/// Unit tests for [`sec_fetcher::parsers::parse_13f_xml`].
///
/// These tests use inline XML fixtures that mirror the SEC's informationTable
/// schema.  The fixtures are intentionally small so the expected values can be
/// verified by hand.
///
/// Key invariants that every 13F filing must satisfy:
/// - `value_usd` is in **actual US dollars** after normalization.  Pre-2023
///   filings reported `<value>` in thousands; [`sec_fetcher::normalize::normalize_13f_value_usd`]
///   handles the conversion transparently based on `filing_date`.
/// - `weight_pct` is on the **0–100 percentage scale** (e.g. `75.0000` means 75%).
/// - `weight_pct` values across all holdings must sum to **exactly 100.0000**.
use indoc::indoc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use sec_fetcher::parsers::parse_13f_xml;

/// Minimal two-position 13F informationTable.
///
/// Uses actual USD values as stored in the modern EDGAR 13F-HR XML schema.
/// AAPL: value = $15 000 000 (75% of total $20 000 000)
/// MSFT: value =  $5 000 000 (25% of total $20 000 000)
const TWO_POSITION_13F: &str = indoc! {r#"
    <?xml version="1.0" encoding="UTF-8"?>
    <informationTable>
      <infoTable>
        <nameOfIssuer>APPLE INC</nameOfIssuer>
        <titleOfClass>COM</titleOfClass>
        <cusip>037833100</cusip>
        <value>15000000</value>
        <shrsOrPrnAmt>
          <sshPrnamt>85000</sshPrnamt>
          <sshPrnamtType>SH</sshPrnamtType>
        </shrsOrPrnAmt>
        <investmentDiscretion>SOLE</investmentDiscretion>
      </infoTable>
      <infoTable>
        <nameOfIssuer>MICROSOFT CORP</nameOfIssuer>
        <titleOfClass>COM</titleOfClass>
        <cusip>594918104</cusip>
        <value>5000000</value>
        <shrsOrPrnAmt>
          <sshPrnamt>14000</sshPrnamt>
          <sshPrnamtType>SH</sshPrnamtType>
        </shrsOrPrnAmt>
        <investmentDiscretion>SOLE</investmentDiscretion>
      </infoTable>
    </informationTable>
"#};

/// Three positions with unequal weights; tests rounding and ordering.
///
/// Uses actual USD values as stored in the modern EDGAR 13F-HR XML schema.
/// NVDA: value = $60 000 000 (60.0000%)
/// AAPL: value = $25 000 000 (25.0000%)
/// MSFT: value = $15 000 000 (15.0000%)
/// Total: $100 000 000 → 100.0000%
const THREE_POSITION_13F: &str = indoc! {r#"
    <?xml version="1.0" encoding="UTF-8"?>
    <informationTable>
      <infoTable>
        <nameOfIssuer>NVIDIA CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>67066G104</cusip>
    <value>60000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>500000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>25000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>130000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
  <infoTable>
    <nameOfIssuer>MICROSOFT CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>594918104</cusip>
    <value>15000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>40000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
      </infoTable>
    </informationTable>
"#};

// ── value_usd: stored verbatim from XML (actual USD) ─────────────────────────

#[test]
fn value_stored_verbatim_as_actual_dollars() {
    let holdings = parse_13f_xml(TWO_POSITION_13F, None).unwrap();
    // Parser sorts by value_usd descending, so AAPL ($15 M) is first.
    assert_eq!(holdings[0].cusip, "037833100", "AAPL should be first");
    assert_eq!(
        holdings[0].value_usd,
        dec!(15_000_000),
        "AAPL value_usd must be stored verbatim ($15 000 000 actual dollars)"
    );
    assert_eq!(holdings[1].cusip, "594918104", "MSFT should be second");
    assert_eq!(
        holdings[1].value_usd,
        dec!(5_000_000),
        "MSFT value_usd must be stored verbatim ($5 000 000 actual dollars)"
    );
}

// ── weight_pct scale: must be 0-100, NOT 0-1 ─────────────────────────────────

#[test]
fn weight_pct_is_on_0_to_100_scale() {
    let holdings = parse_13f_xml(TWO_POSITION_13F, None).unwrap();
    // AAPL is 75%, MSFT is 25% — none of these values are valid in a 0-1 world.
    let aapl = holdings.iter().find(|h| h.cusip == "037833100").unwrap();
    let msft = holdings.iter().find(|h| h.cusip == "594918104").unwrap();

    assert_eq!(aapl.weight_pct.value(), dec!(75.0000), "AAPL should be 75%");
    assert_eq!(msft.weight_pct.value(), dec!(25.0000), "MSFT should be 25%");

    // Explicitly assert that no weight is ≤ 1.0 (which would indicate 0-1 scale).
    for h in &holdings {
        assert!(
            h.weight_pct.value() > dec!(1),
            "weight_pct > 1 expected on 0-100 scale; got {} for {}",
            h.weight_pct.value(),
            h.name
        );
    }
}

#[test]
fn weight_pct_is_not_multiplied_by_100_again() {
    // Regression guard: weight_pct is a derived 0–100 percentage.
    // Pct enforces scale, not range; this test checks real computed values stay in [0, 100].
    let holdings = parse_13f_xml(TWO_POSITION_13F, None).unwrap();
    for h in &holdings {
        assert!(
            h.weight_pct.value() <= dec!(100),
            "weight_pct ({}) exceeds 100 for {} — was value divided by total?",
            h.weight_pct.value(),
            h.name
        );
    }
}

// ── weight_pct sum invariant ──────────────────────────────────────────────────

#[test]
fn weights_sum_to_100_two_positions() {
    let holdings = parse_13f_xml(TWO_POSITION_13F, None).unwrap();
    let sum: Decimal = holdings.iter().map(|h| h.weight_pct.value()).sum();
    assert_eq!(sum, dec!(100.0000), "weights must sum to 100%; got {}", sum);
}

#[test]
fn weights_sum_to_100_three_positions() {
    let holdings = parse_13f_xml(THREE_POSITION_13F, None).unwrap();
    let sum: Decimal = holdings.iter().map(|h| h.weight_pct.value()).sum();
    assert_eq!(sum, dec!(100.0000), "weights must sum to 100%; got {}", sum);
}

// ── individual weight correctness ─────────────────────────────────────────────

#[test]
fn three_position_weights_are_correct() {
    let holdings = parse_13f_xml(THREE_POSITION_13F, None).unwrap();
    // Sorted by value_usd descending: NVDA 60%, AAPL 25%, MSFT 15%.
    let nvda = holdings.iter().find(|h| h.cusip == "67066G104").unwrap();
    let aapl = holdings.iter().find(|h| h.cusip == "037833100").unwrap();
    let msft = holdings.iter().find(|h| h.cusip == "594918104").unwrap();

    assert_eq!(nvda.weight_pct.value(), dec!(60.0000), "NVDA should be 60%");
    assert_eq!(aapl.weight_pct.value(), dec!(25.0000), "AAPL should be 25%");
    assert_eq!(msft.weight_pct.value(), dec!(15.0000), "MSFT should be 15%");
}

// ── output sorted by value_usd descending ─────────────────────────────────────

#[test]
fn holdings_sorted_by_value_usd_descending() {
    let holdings = parse_13f_xml(THREE_POSITION_13F, None).unwrap();
    // NVDA (60k) > AAPL (25k) > MSFT (15k)
    assert_eq!(holdings[0].cusip, "67066G104", "NVDA first");
    assert_eq!(holdings[1].cusip, "037833100", "AAPL second");
    assert_eq!(holdings[2].cusip, "594918104", "MSFT third");
}

// ── shares field ──────────────────────────────────────────────────────────────

#[test]
fn shares_and_shares_type_are_parsed() {
    let holdings = parse_13f_xml(TWO_POSITION_13F, None).unwrap();
    let aapl = holdings.iter().find(|h| h.cusip == "037833100").unwrap();
    assert_eq!(aapl.shares, dec!(85000));
    assert_eq!(aapl.shares_type, "SH");
}

// ── edge case: single position → weight must be 100.0000 ──────────────────────

#[test]
fn single_position_has_100_pct_weight() {
    const SINGLE: &str = indoc! {r#"
        <?xml version="1.0" encoding="UTF-8"?>
        <informationTable>
          <infoTable>
            <nameOfIssuer>APPLE INC</nameOfIssuer>
            <titleOfClass>COM</titleOfClass>
            <cusip>037833100</cusip>
            <value>42000000</value>
            <shrsOrPrnAmt>
              <sshPrnamt>200000</sshPrnamt>
              <sshPrnamtType>SH</sshPrnamtType>
            </shrsOrPrnAmt>
            <investmentDiscretion>SOLE</investmentDiscretion>
          </infoTable>
        </informationTable>
    "#};
    let holdings = parse_13f_xml(SINGLE, None).unwrap();
    assert_eq!(holdings.len(), 1);
    assert_eq!(holdings[0].weight_pct.value(), dec!(100.0000));
}

// ── fractional weights round to 4 decimal places ──────────────────────────────

#[test]
fn fractional_weights_rounded_to_4dp() {
    // 1/3 = 33.3333...%; 2/3 = 66.6666...%
    // With round_dp(4): 33.3333 and 66.6667
    const THIRDS: &str = indoc! {r#"
        <?xml version="1.0" encoding="UTF-8"?>
        <informationTable>
          <infoTable>
            <nameOfIssuer>ALPHA INC</nameOfIssuer>
            <titleOfClass>COM</titleOfClass>
            <cusip>000000001</cusip>
            <value>2000000</value>
            <shrsOrPrnAmt><sshPrnamt>1</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
            <investmentDiscretion>SOLE</investmentDiscretion>
          </infoTable>
          <infoTable>
            <nameOfIssuer>BETA INC</nameOfIssuer>
            <titleOfClass>COM</titleOfClass>
            <cusip>000000002</cusip>
            <value>1000000</value>
            <shrsOrPrnAmt><sshPrnamt>1</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
            <investmentDiscretion>SOLE</investmentDiscretion>
          </infoTable>
        </informationTable>
    "#};
    let holdings = parse_13f_xml(THIRDS, None).unwrap();
    // ALPHA: 2/3 * 100 = 66.6666...% → rounds to 66.6667 at 4dp
    // BETA:  1/3 * 100 = 33.3333...% → rounds to 33.3333 at 4dp
    let alpha = holdings.iter().find(|h| h.cusip == "000000001").unwrap();
    let beta = holdings.iter().find(|h| h.cusip == "000000002").unwrap();

    assert_eq!(
        alpha.weight_pct.value(),
        dec!(66.6667),
        "2/3 share must round to 66.6667"
    );
    assert_eq!(
        beta.weight_pct.value(),
        dec!(33.3333),
        "1/3 share must round to 33.3333"
    );
    // Cross-check: distinct inputs must produce distinct outputs.
    assert_ne!(alpha.weight_pct.value(), beta.weight_pct.value());
    // 4 decimal places
    assert_eq!(
        alpha.weight_pct.value().scale(),
        4,
        "weight_pct should have 4 dp; got {}",
        alpha.weight_pct
    );
}
