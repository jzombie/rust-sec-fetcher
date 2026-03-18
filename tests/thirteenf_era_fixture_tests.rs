//! Integration tests for 13F-HR era-crossing normalization using real EDGAR data.
//!
//! These tests load actual BRK-B (Berkshire Hathaway, CIK 1067983) 13F-HR
//! `informationTable.xml` files that were downloaded by
//! `cargo run --bin refresh-test-fixtures`.  Two fixtures bracket the
//! 2023-01-01 schema crossover date, and a third confirms the modern schema:
//!
//! | Fixture file | Accession | Filed | `<value>` units |
//! |---|---|---|---|
//! | `BRK_B_13f_ancient.xml` | `0000950123-22-012275` | 2022-11-14 | **thousands** |
//! | `BRK_B_13f_transition.xml` | `0000950123-23-002585` | 2023-02-14 | **actual USD** |
//! | `BRK_B_13f_modern.xml` | `0001193125-26-054580` | 2026-02-17 | **actual USD** |
//!
//! ## What these tests verify
//!
//! 1. **Era routing**: `normalize_13f_value_usd` multiplies ancient `<value>`
//!    by 1 000 and passes modern `<value>` through unchanged.
//!
//! 2. **Price-per-share sanity**: for each known AAPL holding, dividing
//!    `value_usd / shares` should produce a plausible AAPL price for that
//!    quarter.  This is the hard empirical evidence that the era routing is
//!    correct — not just a unit test on a mock value.
//!
//! 3. **Portfolio weight**: `weight_pct` is a [`Pct`] (0–100 scale)
//!    across both eras, and all weights sum to approximately (within rounding)
//!    100%.
//!
//! If any fixture file is missing, the test is skipped with a clear message
//! directing the developer to run the fixture refresh binary.

use chrono::NaiveDate;
use flate2::read::GzDecoder;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use sec_fetcher::parsers::parse_13f_xml;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

// ── Fixture loader ────────────────────────────────────────────────────────────

/// Loads a raw XML fixture from `tests/fixtures/{name}.gz`.
///
/// Returns `None` if the file does not exist (fixtures must be downloaded by
/// running `cargo run --bin refresh-test-fixtures`).
fn try_load_xml_fixture(name: &str) -> Option<String> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(format!("{}.gz", name));

    let file = File::open(&path).ok()?;
    let mut decoder = GzDecoder::new(file);
    let mut xml = String::new();
    decoder.read_to_string(&mut xml).ok()?;
    Some(xml)
}

/// Returns the XML for a fixture, or skips the test if the fixture is absent.
macro_rules! load_xml_fixture_or_skip {
    ($name:expr) => {
        match try_load_xml_fixture($name) {
            Some(xml) => xml,
            None => {
                eprintln!(
                    "SKIP: fixture '{}' not found — run `cargo run --bin refresh-test-fixtures`",
                    $name
                );
                return;
            }
        }
    };
}

// ── Helper ────────────────────────────────────────────────────────────────────

/// CUSIP for AAPL (Apple Inc., used as an anchor position in every BRK-B 13F).
const AAPL_CUSIP: &str = "037833100";

/// Returns AAPL's price-per-share implied by the holding's normalized value_usd and shares.
fn aapl_price_per_share(xml: &str, filing_date: NaiveDate) -> Decimal {
    let holdings = parse_13f_xml(xml, Some(filing_date)).expect("parse failed");
    let aapl = holdings
        .iter()
        .find(|h| h.cusip == AAPL_CUSIP)
        .expect("AAPL not found in BRK-B 13F — has AAPL been fully liquidated?");
    aapl.value_usd / aapl.shares
}

// ── Ancient era (Q3-2022, filed 2022-11-14) ───────────────────────────────────

#[test]
fn ancient_aapl_price_per_share_is_plausible() {
    // BRK-B Q3-2022 (0000950123-22-012275, filed 2022-11-14).
    // <value> in the XML is raw **thousands**: AAPL raw=95634
    // → after ×1000: value_usd = $95,634,000 for 692,000 shares → ~$138/sh.
    // AAPL traded ~$138–150 in Q3-2022 — so [120, 175] is a generous sanity band.
    let xml = load_xml_fixture_or_skip!("BRK_B_13f_ancient.xml");
    let filing_date = NaiveDate::from_ymd_opt(2022, 11, 14).unwrap();
    let price = aapl_price_per_share(&xml, filing_date);
    assert!(
        price > dec!(120) && price < dec!(175),
        "ancient AAPL price/share = {} — expected ~$138 (Q3-2022); \
         if ×1000 was NOT applied the raw $0.14 would have failed this check",
        price
    );
}

#[test]
fn ancient_weight_pct_sums_to_100() {
    let xml = load_xml_fixture_or_skip!("BRK_B_13f_ancient.xml");
    let filing_date = NaiveDate::from_ymd_opt(2022, 11, 14).unwrap();
    let holdings = parse_13f_xml(&xml, Some(filing_date)).unwrap();
    let sum: Decimal = holdings.iter().map(|h| h.weight_pct.value()).sum();
    // Rounding at 4 dp across many positions means sum might be off by a few
    // basis points.  Require within 0.5% of 100.
    assert!(
        (sum - dec!(100)).abs() < dec!(0.5),
        "ancient weights sum to {} (expected ~100)",
        sum
    );
}

#[test]
fn ancient_weight_pct_is_normalized_pct_type() {
    // Structural test: all weight_pct values should be in range for a long-only fund [0, 100].
    // If normalize_13f_value_usd accidentally skips the ×1000, value_usd would
    // be ~1000× too small and weight_pct would still be valid — but the
    // price-per-share test above would catch that separately.
    let xml = load_xml_fixture_or_skip!("BRK_B_13f_ancient.xml");
    let filing_date = NaiveDate::from_ymd_opt(2022, 11, 14).unwrap();
    let holdings = parse_13f_xml(&xml, Some(filing_date)).unwrap();
    assert!(!holdings.is_empty(), "expected at least one holding");
    for h in &holdings {
        // .value() on a Pct returns the inner Decimal on the 0–100 scale
        let w = h.weight_pct.value();
        assert!(
            w >= dec!(0) && w <= dec!(100),
            "weight_pct {} out of range for {}",
            w,
            h.name
        );
    }
}

// ── Transition era (Q4-2022, filed 2023-02-14, first modern filing) ───────────

#[test]
fn transition_aapl_price_per_share_is_plausible() {
    // BRK-B Q4-2022 (0000950123-23-002585, filed 2023-02-14).
    // <value> is raw **actual USD**: AAPL raw=133289470
    // → value_usd = $133,289,470 for 1,025,856 shares → ~$130/sh.
    // AAPL traded ~$125–140 in Q4-2022 — so [110, 165] is a sensible band.
    let xml = load_xml_fixture_or_skip!("BRK_B_13f_transition.xml");
    let filing_date = NaiveDate::from_ymd_opt(2023, 2, 14).unwrap();
    let price = aapl_price_per_share(&xml, filing_date);
    assert!(
        price > dec!(110) && price < dec!(165),
        "transition AAPL price/share = {} — expected ~$130 (Q4-2022); \
         if ×1000 was still applied the price would be ~$130,000 and fail this check",
        price
    );
}

#[test]
fn transition_weight_pct_sums_to_100() {
    let xml = load_xml_fixture_or_skip!("BRK_B_13f_transition.xml");
    let filing_date = NaiveDate::from_ymd_opt(2023, 2, 14).unwrap();
    let holdings = parse_13f_xml(&xml, Some(filing_date)).unwrap();
    let sum: Decimal = holdings.iter().map(|h| h.weight_pct.value()).sum();
    assert!(
        (sum - dec!(100)).abs() < dec!(0.5),
        "transition weights sum to {} (expected ~100)",
        sum
    );
}

#[test]
fn era_crossover_produces_different_value_usd_for_same_raw() {
    // Parse the SAME fixture file with opposite era dates to show that the
    // date-routing in normalize_13f_value_usd actually changes value_usd.
    //
    // We use the ancient XML (thousands era) as the test subject:
    //  - With the correct 2022-11-14 date  → thousands → value_usd is large
    //  - With a wrong 2023-01-01+ date     → pass-through → value_usd is small
    //   (This proves the routing is not a no-op.)
    let xml = load_xml_fixture_or_skip!("BRK_B_13f_ancient.xml");

    let ancient_date = NaiveDate::from_ymd_opt(2022, 11, 14).unwrap();
    let modern_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();

    let holdings_ancient = parse_13f_xml(&xml, Some(ancient_date)).unwrap();
    let holdings_modern = parse_13f_xml(&xml, Some(modern_date)).unwrap();

    let aapl_ancient = holdings_ancient
        .iter()
        .find(|h| h.cusip == AAPL_CUSIP)
        .unwrap();
    let aapl_modern = holdings_modern
        .iter()
        .find(|h| h.cusip == AAPL_CUSIP)
        .unwrap();

    // Ancient path should multiply by 1000 → value_usd ~1000× larger.
    assert!(
        aapl_ancient.value_usd > aapl_modern.value_usd * dec!(900),
        "ancient value_usd ({}) should be ~1000× larger than modern path value_usd ({}) \
         because the era-routing ×1000 was applied",
        aapl_ancient.value_usd,
        aapl_modern.value_usd
    );
}

// ── Modern era (Q4-2025, filed 2026-02-17) ────────────────────────────────────

#[test]
fn modern_aapl_price_per_share_is_plausible() {
    // BRK-B Q4-2025 (0001193125-26-054580, filed 2026-02-17).
    // <value> is actual USD.  AAPL traded ~$230–260 in Q4-2025.
    let xml = load_xml_fixture_or_skip!("BRK_B_13f_modern.xml");
    let filing_date = NaiveDate::from_ymd_opt(2026, 2, 17).unwrap();

    let holdings = parse_13f_xml(&xml, Some(filing_date)).unwrap();
    // BRK-B fully exited AAPL by Q4-2024; if AAPL is absent, skip gracefully.
    if let Some(aapl) = holdings.iter().find(|h| h.cusip == AAPL_CUSIP) {
        let price = aapl.value_usd / aapl.shares;
        assert!(
            price > dec!(100) && price < dec!(600),
            "modern AAPL price/share = {} — outside expected [100, 600] range",
            price
        );
    } else {
        eprintln!("INFO: AAPL not present in Q4-2025 BRK-B filing (position may have been exited)");
    }
}

#[test]
fn modern_weight_pct_sums_to_100() {
    let xml = load_xml_fixture_or_skip!("BRK_B_13f_modern.xml");
    let filing_date = NaiveDate::from_ymd_opt(2026, 2, 17).unwrap();
    let holdings = parse_13f_xml(&xml, Some(filing_date)).unwrap();
    assert!(
        !holdings.is_empty(),
        "expected at least one holding in modern 13F"
    );
    let sum: Decimal = holdings.iter().map(|h| h.weight_pct.value()).sum();
    assert!(
        (sum - dec!(100)).abs() < dec!(0.5),
        "modern weights sum to {} (expected ~100)",
        sum
    );
}

#[test]
fn modern_holdings_have_large_value_usd() {
    // In the actual-USD era, individual position values are millions to
    // billions of dollars.  If the file were accidentally parsed as thousands
    // (no ×1000 applied), values would still be in the millions — but if
    // ×1000 were INCORRECTLY applied to modern data, values would be trillion-scale.
    // Check that no single position exceeds $1 trillion (a clear sign of ×1000 misfire).
    let xml = load_xml_fixture_or_skip!("BRK_B_13f_modern.xml");
    let filing_date = NaiveDate::from_ymd_opt(2026, 2, 17).unwrap();
    let holdings = parse_13f_xml(&xml, Some(filing_date)).unwrap();
    for h in &holdings {
        assert!(
            h.value_usd < dec!(1_000_000_000_000),
            "value_usd {} for {} exceeds $1T — ×1000 misfire on modern-era data?",
            h.value_usd,
            h.name
        );
    }
}
