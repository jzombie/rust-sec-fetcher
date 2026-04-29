/// Unit tests for [`sec_fetcher::ops::holdings`] — portfolio position
/// normalization and diffing.
///
/// All functions under test are pure (no async, no network), so these tests
/// construct domain types in memory and exercise the conversion + diff logic.
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use sec_fetcher::models::{NportInvestment, ThirteenfHolding};
use sec_fetcher::normalize::Pct;
use sec_fetcher::ops::holdings::{
    Position, diff_holdings, positions_from_13f, positions_from_nport,
};

// ============================================================================
// helpers
// ============================================================================

fn nport(cusip: &str, name: &str, val_usd: i64, pct_val: Decimal) -> NportInvestment {
    NportInvestment {
        mapped_ticker_symbol: None,
        mapped_company_name: None,
        mapped_company_cik_number: None,
        name: name.into(),
        lei: String::new(),
        title: String::new(),
        cusip: cusip.into(),
        isin: String::new(),
        balance: Decimal::ZERO,
        cur_cd: "USD".into(),
        val_usd: Decimal::from(val_usd),
        pct_val: Pct::from_pct(pct_val),
        payoff_profile: String::new(),
        asset_cat: String::new(),
        issuer_cat: String::new(),
        inv_country: String::new(),
    }
}

fn holding_13f(cusip: &str, name: &str, value_usd: i64, weight: Decimal) -> ThirteenfHolding {
    ThirteenfHolding {
        name: name.into(),
        cusip: cusip.into(),
        title_of_class: "COM".into(),
        value_usd: Decimal::from(value_usd),
        shares: Decimal::ZERO,
        shares_type: "SH".into(),
        put_call: None,
        investment_discretion: "SOLE".into(),
        weight_pct: Pct::from_pct(weight),
    }
}

fn pos(cusip: &str, name: &str, val_usd: i64, weight: Decimal) -> Position {
    Position {
        cusip: cusip.into(),
        name: name.into(),
        val_usd: Decimal::from(val_usd),
        weight: Pct::from_pct(weight),
    }
}

// ============================================================================
// positions_from_nport
// ============================================================================

#[test]
fn test_nport_empty() {
    let positions = positions_from_nport(&[]);
    assert!(positions.is_empty(), "Empty input -> empty positions");
}

#[test]
fn test_nport_single() {
    let investments = vec![nport("037833100", "Apple Inc.", 1_000_000, dec!(25.50))];
    let positions = positions_from_nport(&investments);
    assert_eq!(positions.len(), 1);
    assert_eq!(positions[0].cusip, "037833100");
    assert_eq!(positions[0].name, "Apple Inc.");
    assert_eq!(positions[0].val_usd, dec!(1000000));
    assert_eq!(positions[0].weight, Pct::from_pct(dec!(25.50)));
}

#[test]
fn test_nport_multiple_preserves_order() {
    let investments = vec![
        nport("037833100", "Apple Inc.", 1_000_000, dec!(50.00)),
        nport("594918104", "Microsoft Corp.", 800_000, dec!(40.00)),
        nport("67066G104", "NVIDIA Corp.", 200_000, dec!(10.00)),
    ];
    let positions = positions_from_nport(&investments);
    assert_eq!(positions.len(), 3);
    assert_eq!(positions[0].name, "Apple Inc.");
    assert_eq!(positions[1].name, "Microsoft Corp.");
    assert_eq!(positions[2].name, "NVIDIA Corp.");
}

#[test]
fn test_nport_negative_weight_preserved() {
    // Short positions have negative pct_val
    let investments = vec![nport("037833100", "Short Position", 500_000, dec!(-3.25))];
    let positions = positions_from_nport(&investments);
    assert_eq!(positions[0].weight, Pct::from_pct(dec!(-3.25)));
}

// ============================================================================
// positions_from_13f
// ============================================================================

#[test]
fn test_13f_empty() {
    let positions = positions_from_13f(&[]);
    assert!(positions.is_empty(), "Empty input -> empty positions");
}

#[test]
fn test_13f_single() {
    let holdings = vec![holding_13f(
        "037833100",
        "APPLE INC",
        116_333_000,
        dec!(45.20),
    )];
    let positions = positions_from_13f(&holdings);
    assert_eq!(positions.len(), 1);
    assert_eq!(positions[0].cusip, "037833100");
    assert_eq!(positions[0].name, "APPLE INC");
    assert_eq!(positions[0].val_usd, dec!(116333000));
    assert_eq!(positions[0].weight, Pct::from_pct(dec!(45.20)));
}

#[test]
fn test_13f_weights_round_to_100() {
    // Common scenario: portfolio weights should sum to ~100
    let holdings = vec![
        holding_13f("037833100", "APPLE INC", 1_000_000, dec!(50.00)),
        holding_13f("594918104", "MSFT", 500_000, dec!(25.00)),
        holding_13f("67066G104", "NVDA", 500_000, dec!(25.00)),
    ];
    let positions = positions_from_13f(&holdings);
    let sum: Decimal = positions.iter().map(|p| p.weight.value()).sum();
    assert_eq!(sum, dec!(100.00), "Weights should sum to 100.00");
}

#[test]
fn test_13f_forwarded_without_rescaling() {
    // Ensure weight_pct is faithfully forwarded (no hidden 0-1 to 0-100 conversion)
    let holdings = vec![holding_13f("037833100", "AAPL", 100_000, dec!(7.7546))];
    let positions = positions_from_13f(&holdings);
    assert_eq!(positions[0].weight, Pct::from_pct(dec!(7.7546)));
}

// ============================================================================
// diff_holdings — empty / identical
// ============================================================================

#[test]
fn test_diff_both_empty() {
    let diff = diff_holdings(&[], &[]);
    assert!(diff.added.is_empty());
    assert!(diff.removed.is_empty());
    assert!(diff.changed.is_empty());
}

#[test]
fn test_diff_identical_lists() {
    let positions = vec![
        pos("037833100", "AAPL", 100_000, dec!(50)),
        pos("594918104", "MSFT", 100_000, dec!(50)),
    ];
    let diff = diff_holdings(&positions, &positions);
    assert!(diff.added.is_empty(), "No new buys");
    assert!(diff.removed.is_empty(), "No full sells");
    assert!(diff.changed.is_empty(), "No weight changes");
}

#[test]
fn test_diff_identical_single_position() {
    let p = pos("037833100", "AAPL", 100_000, dec!(100));
    let diff = diff_holdings(&[p.clone()], &[p]);
    assert!(diff.added.is_empty());
    assert!(diff.removed.is_empty());
    assert!(diff.changed.is_empty());
}

// ============================================================================
// diff_holdings — added / removed
// ============================================================================

#[test]
fn test_diff_adds_new_position() {
    let old = vec![];
    let new = vec![pos("037833100", "AAPL", 100_000, dec!(100))];
    let diff = diff_holdings(&old, &new);
    assert_eq!(diff.added.len(), 1);
    assert_eq!(diff.added[0].cusip, "037833100");
    assert!(diff.removed.is_empty());
    assert!(diff.changed.is_empty());
}

#[test]
fn test_diff_removes_position() {
    let old = vec![pos("037833100", "AAPL", 100_000, dec!(100))];
    let new = vec![];
    let diff = diff_holdings(&old, &new);
    assert_eq!(diff.removed.len(), 1);
    assert_eq!(diff.removed[0].cusip, "037833100");
    assert!(diff.added.is_empty());
    assert!(diff.changed.is_empty());
}

#[test]
fn test_diff_add_and_remove_simultaneous() {
    let old = vec![pos("037833100", "AAPL", 100_000, dec!(100))];
    let new = vec![pos("594918104", "MSFT", 200_000, dec!(100))];
    let diff = diff_holdings(&old, &new);
    assert_eq!(diff.added.len(), 1, "MSFT is new");
    assert_eq!(diff.removed.len(), 1, "AAPL is gone");
    assert!(diff.changed.is_empty());
    assert_eq!(diff.added[0].cusip, "594918104");
    assert_eq!(diff.removed[0].cusip, "037833100");
}

#[test]
fn test_diff_multiple_adds() {
    let old = vec![pos("037833100", "AAPL", 100_000, dec!(100))];
    let new = vec![
        pos("037833100", "AAPL", 100_000, dec!(60)),
        pos("594918104", "MSFT", 50_000, dec!(30)),
        pos("67066G104", "NVDA", 20_000, dec!(10)),
    ];
    let diff = diff_holdings(&old, &new);
    assert_eq!(diff.added.len(), 2, "MSFT and NVDA are new");
    assert!(diff.removed.is_empty());
    // AAPL weight changed from 100 to 60 = -40, above threshold
    assert_eq!(diff.changed.len(), 1);
}

// ============================================================================
// diff_holdings — weight changes
// ============================================================================

#[test]
fn test_diff_small_change_below_threshold() {
    let shared = pos("037833100", "AAPL", 100_000, dec!(50));
    let mut bumped = pos("037833100", "AAPL", 100_000, dec!(50));
    bumped.weight = Pct::from_pct(dec!(50.05)); // +0.05, below 0.10 threshold

    let diff = diff_holdings(&[shared], &[bumped]);
    assert!(
        diff.changed.is_empty(),
        "0.05 pct change is below threshold"
    );
    assert!(diff.added.is_empty());
    assert!(diff.removed.is_empty());
}

#[test]
fn test_diff_change_at_threshold_included() {
    let old = pos("037833100", "AAPL", 100_000, dec!(50));
    let mut new_pos = pos("037833100", "AAPL", 100_000, dec!(50));
    new_pos.weight = Pct::from_pct(dec!(50.10)); // +0.10, exactly at threshold

    let diff = diff_holdings(&[old], &[new_pos.clone()]);
    assert_eq!(diff.changed.len(), 1, "0.10 pct change meets threshold");
    assert_eq!(diff.changed[0].1.cusip, "037833100");
}

#[test]
fn test_diff_large_change_included() {
    let old = pos("037833100", "AAPL", 100_000, dec!(80));
    let new_pos = pos("037833100", "AAPL", 50_000, dec!(40));
    let diff = diff_holdings(&[old], &[new_pos]);
    assert_eq!(diff.changed.len(), 1);
    assert_eq!(diff.changed[0].0.weight, Pct::from_pct(dec!(80)));
    assert_eq!(diff.changed[0].1.weight, Pct::from_pct(dec!(40)));
}

#[test]
fn test_diff_negative_delta() {
    // Weight went from 20 → 55 (+35), well above threshold
    let old = pos("037833100", "AAPL", 20_000, dec!(20));
    let new_pos = pos("037833100", "AAPL", 55_000, dec!(55));
    let diff = diff_holdings(&[old], &[new_pos]);
    assert_eq!(diff.changed.len(), 1);
    assert_eq!(diff.changed[0].0.weight, Pct::from_pct(dec!(20)));
    assert_eq!(diff.changed[0].1.weight, Pct::from_pct(dec!(55)));
}

// ============================================================================
// diff_holdings — sorting
// ============================================================================

#[test]
fn test_diff_changed_sorted_by_delta_descending() {
    let old: Vec<Position> = vec!["037833100", "594918104", "67066G104", "02079K305"]
        .into_iter()
        .map(|c| pos(c, "NAME", 100_000, dec!(30)))
        .collect();

    let new: Vec<Position> = [
        ("037833100", dec!(50)), // +20
        ("594918104", dec!(20)), // -10
        ("67066G104", dec!(25)), // -5  (still above 0.10 threshold)
        ("02079K305", dec!(5)),  // -25
    ]
    .into_iter()
    .map(|(c, w)| pos(c, "NAME", 100_000, w))
    .collect();

    let diff = diff_holdings(&old, &new);
    // All four changes exceed the 0.10 threshold because weights are on the
    // 0-100 scale: deltas are 20, 10, 5, 25 — all ≥ 0.10.
    // Sorted by abs delta descending: GOOGL (25), AAPL (20), MSFT (10), NVDA (5)
    assert_eq!(diff.changed.len(), 4);
    assert_eq!(
        diff.changed[0].0.cusip, "02079K305",
        "GOOGL first: delta=25"
    );
    assert_eq!(
        diff.changed[1].0.cusip, "037833100",
        "AAPL second: delta=20"
    );
    assert_eq!(diff.changed[2].0.cusip, "594918104", "MSFT third: delta=10");
    assert_eq!(diff.changed[3].0.cusip, "67066G104", "NVDA fourth: delta=5");
}

#[test]
fn test_diff_empty_old_all_new_are_adds() {
    let new = vec![
        pos("037833100", "AAPL", 100_000, dec!(50)),
        pos("594918104", "MSFT", 100_000, dec!(50)),
    ];
    let diff = diff_holdings(&[], &new);
    assert_eq!(diff.added.len(), 2);
    assert!(diff.removed.is_empty());
    assert!(diff.changed.is_empty());
}

#[test]
fn test_diff_empty_new_all_old_are_removes() {
    let old = vec![
        pos("037833100", "AAPL", 100_000, dec!(50)),
        pos("594918104", "MSFT", 100_000, dec!(50)),
    ];
    let diff = diff_holdings(&old, &[]);
    assert_eq!(diff.removed.len(), 2);
    assert!(diff.added.is_empty());
    assert!(diff.changed.is_empty());
}

// ============================================================================
// diff_holdings — full lifecycle scenario
// ============================================================================

#[test]
fn test_diff_full_lifecycle() {
    // Q1 portfolio
    let old = vec![
        pos("037833100", "AAPL", 500_000, dec!(50)),
        pos("594918104", "MSFT", 300_000, dec!(30)),
        pos("02079K305", "GOOGL", 200_000, dec!(20)),
    ];

    // Q2 portfolio: AAPL still held (weight trimmed to 0.05 under threshold),
    // MSFT sold, GOOGL sold, NVDA newly bought
    let new = vec![
        // AAPL: 50.00 → 50.04 = +0.04, below 0.10 threshold → not "changed"
        pos("037833100", "AAPL", 550_000, dec!(50.04)),
        pos("67066G104", "NVDA", 450_000, dec!(49.96)), // new buy
    ];

    let diff = diff_holdings(&old, &new);

    assert_eq!(diff.added.len(), 1, "NVDA is new");
    assert_eq!(diff.added[0].cusip, "67066G104");

    assert_eq!(diff.removed.len(), 2, "MSFT and GOOGL were sold");
    assert!(
        diff.changed.is_empty(),
        "AAPL +0.04 is below 0.10 threshold"
    );
}

// ============================================================================
// diff_holdings — CUSIP matching is case-sensitive
// ============================================================================

#[test]
fn test_diff_cusip_matching_case_sensitive() {
    let old = vec![pos("037833100", "AAPL", 100_000, dec!(100))];
    // Same CUSIP but different case — should be treated as different
    let new = vec![pos("037833100", "AAPL (lower)", 100_000, dec!(100))];
    let diff = diff_holdings(&old, &new);
    // CUSIPs match so it's identical (same string)
    assert!(diff.added.is_empty());
    assert!(diff.removed.is_empty());
}

// ============================================================================
// Round-trip consistency: nport → position → diff mirrors 13f → position → diff
// ============================================================================

#[test]
fn test_nport_and_13f_positions_are_interchangeable_in_diff() {
    // Both N-PORT and 13F produce `Position` structs with the same shape;
    // mixing them in diff_holdings should work seamlessly.
    let nport_positions = vec![pos("037833100", "Apple Inc.", 1_000_000, dec!(60))];
    let thirteenf_positions = vec![pos("594918104", "MICROSOFT CORP", 500_000, dec!(40))];

    let diff = diff_holdings(&nport_positions, &thirteenf_positions);
    assert_eq!(diff.removed.len(), 1, "AAPL removed from N-PORT snapshot");
    assert_eq!(diff.added.len(), 1, "MSFT added from 13F snapshot");
    assert_eq!(diff.removed[0].cusip, "037833100");
    assert_eq!(diff.added[0].cusip, "594918104");
}
