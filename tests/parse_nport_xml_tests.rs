/// Unit tests for [`sec_fetcher::parsers::parse_nport_xml`].
///
/// These tests use inline XML fixtures that mirror the SEC's NPORT-P schema.
///
/// Key invariants:
/// - `pct_val` is stored exactly as the `<pctVal>` XML element value — on the
///   **0–100 percentage scale** (e.g. `7.7546` means 7.7546%).
/// - `pct_val` must NOT be multiplied by 100 at any level; that would produce
///   values like 775.46 from 7.7546.
/// - `val_usd` is the USD value directly from `<valUSD>` (no scaling applied).
use rust_decimal_macros::dec;
use sec_fetcher::models::Ticker;
use sec_fetcher::parsers::parse_nport_xml;

/// Two-position N-PORT fixture using real-world-derived values.
///
/// NVIDIA:  pctVal = 7.7546   (7.7546% of fund NAV)
/// Prologis: pctVal = 0.0073  (0.0073% of fund NAV — a tiny bond position)
const TWO_POSITION_NPORT: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission>
  <formData>
    <invstOrSec>
      <name>NVIDIA CORP</name>
      <lei>5493003BDMRS0ERX6872</lei>
      <title>NVIDIA CORP</title>
      <cusip>67066G104</cusip>
      <identifiers>
        <isin value="US67066G1040"/>
      </identifiers>
      <balance>100000</balance>
      <units>NS</units>
      <curCd>USD</curCd>
      <valUSD>7754618000.00</valUSD>
      <pctVal>7.7546</pctVal>
      <payoffProfile>Long</payoffProfile>
      <assetCat>EC</assetCat>
      <issuerCat>CORP</issuerCat>
      <invCountry>US</invCountry>
    </invstOrSec>
    <invstOrSec>
      <name>PROLOGIS LP</name>
      <lei>GL16H1DHB0QSHP25F723</lei>
      <title>Prologis LP</title>
      <cusip>74340XCH2</cusip>
      <identifiers>
        <isin value="US74340XCH26"/>
      </identifiers>
      <balance>724000</balance>
      <units>PA</units>
      <curCd>USD</curCd>
      <valUSD>713480.28</valUSD>
      <pctVal>0.007357911161</pctVal>
      <payoffProfile>Long</payoffProfile>
      <assetCat>DBT</assetCat>
      <issuerCat></issuerCat>
      <invCountry>US</invCountry>
    </invstOrSec>
  </formData>
</edgarSubmission>"#;

/// Fixture with a mid-weight position to catch off-by-one-order-of-magnitude errors.
const MID_WEIGHT_NPORT: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission>
  <formData>
    <invstOrSec>
      <name>APPLE INC</name>
      <lei>HWUPKR0MPOU8FGXBT394</lei>
      <title>APPLE INC</title>
      <cusip>037833100</cusip>
      <identifiers>
        <isin value="US0378331005"/>
      </identifiers>
      <balance>500000</balance>
      <units>NS</units>
      <curCd>USD</curCd>
      <valUSD>101500000.00</valUSD>
      <pctVal>5.1234</pctVal>
      <payoffProfile>Long</payoffProfile>
      <assetCat>EC</assetCat>
      <issuerCat>CORP</issuerCat>
      <invCountry>US</invCountry>
    </invstOrSec>
  </formData>
</edgarSubmission>"#;

// ── pct_val scale: must be 0-100, NOT 0-1 ───────────────────────────────────

#[test]
fn pct_val_is_on_0_to_100_scale_for_large_position() {
    let investments = parse_nport_xml(TWO_POSITION_NPORT, &[] as &[Ticker], None).unwrap();
    let nvda = investments.iter().find(|i| i.cusip == "67066G104").unwrap();

    // The XML says pctVal = 7.7546.  That is 7.7546% on the 0-100 scale.
    assert_eq!(nvda.pct_val.value(), dec!(7.7546));
}

#[test]
fn pct_val_not_multiplied_by_100() {
    // Regression guard: verify pct_val is already on the 0–100 scale (not 0–1).
    // Pct enforces scale, not range; this test checks real parsed values stay in [0, 100].
    let investments = parse_nport_xml(TWO_POSITION_NPORT, &[] as &[Ticker], None).unwrap();
    for inv in &investments {
        assert!(
            inv.pct_val.value() <= dec!(100),
            "pct_val ({}) exceeds 100 for {} — value was not multiplied by 100, was it?",
            inv.pct_val.value(),
            inv.name
        );
    }
}

#[test]
fn pct_val_is_on_0_to_100_scale_for_tiny_position() {
    let investments = parse_nport_xml(TWO_POSITION_NPORT, &[] as &[Ticker], None).unwrap();
    let prologis = investments.iter().find(|i| i.cusip == "74340XCH2").unwrap();

    // Tiny bond position: 0.007357911161% on the 0-100 scale.
    // On a 0-1 scale this would be 0.00007357911161 — parser must NOT divide.
    assert_eq!(prologis.pct_val.value(), dec!(0.007357911161));
    assert!(
        prologis.pct_val.value() > dec!(0),
        "pct_val should be a small positive percentage, not zero"
    );
}

#[test]
fn pct_val_mid_weight_exactly_preserved() {
    let investments = parse_nport_xml(MID_WEIGHT_NPORT, &[] as &[Ticker], None).unwrap();
    assert_eq!(investments.len(), 1);
    assert_eq!(
        investments[0].pct_val.value(),
        dec!(5.1234),
        "pct_val must be stored verbatim from <pctVal>; got {}",
        investments[0].pct_val.value()
    );
}

// ── val_usd: stored verbatim (no scaling) ────────────────────────────────────

#[test]
fn val_usd_stored_verbatim_from_xml() {
    let investments = parse_nport_xml(TWO_POSITION_NPORT, &[] as &[Ticker], None).unwrap();
    let nvda = investments.iter().find(|i| i.cusip == "67066G104").unwrap();
    // valUSD in XML is already in actual USD (not thousands).
    assert_eq!(nvda.val_usd, dec!(7754618000.00));
}

// ── sorted by pct_val descending ─────────────────────────────────────────────

#[test]
fn investments_sorted_by_pct_val_descending() {
    let investments = parse_nport_xml(TWO_POSITION_NPORT, &[] as &[Ticker], None).unwrap();
    // NVIDIA (7.7546%) should come before Prologis (0.0073%).
    assert_eq!(
        investments[0].cusip, "67066G104",
        "NVIDIA should be first (highest pct_val)"
    );
    assert_eq!(
        investments[1].cusip, "74340XCH2",
        "Prologis should be second (lower pct_val)"
    );
}

// ── other fields parsed correctly ────────────────────────────────────────────

#[test]
fn name_cusip_and_currency_parsed() {
    let investments = parse_nport_xml(TWO_POSITION_NPORT, &[] as &[Ticker], None).unwrap();
    let nvda = investments.iter().find(|i| i.cusip == "67066G104").unwrap();

    assert_eq!(nvda.name, "NVIDIA CORP");
    assert_eq!(nvda.cur_cd, "USD");
    assert_eq!(nvda.inv_country, "US");
    assert_eq!(nvda.asset_cat, "EC");
    assert_eq!(nvda.payoff_profile, "Long");
}
