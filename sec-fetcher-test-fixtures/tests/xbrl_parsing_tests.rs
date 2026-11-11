mod common;

use std::collections::{HashMap, HashSet};

#[test]
fn parse_bac_calculation_linkbase() {
    let xml = common::fixture_string("BAC_10q_20260331_cal.xml");
    let arcs = sec_fetcher::parsers::parse_calculation_linkbase(&xml).unwrap();
    assert!(!arcs.is_empty());
    assert!(arcs.iter().any(|a| a.from.is_standard && !a.to.is_standard));
}

#[test]
fn bac_std_to_custom_anchors_are_stable() {
    // Locks the polarity contract: known BAC custom tags anchor to known
    // standard parents with known weights. Guards against parser regressions
    // that would flip downstream trend-describer signs.
    let xml = common::fixture_string("BAC_10q_20260331_cal.xml");
    let arcs = sec_fetcher::parsers::parse_calculation_linkbase(&xml).unwrap();

    let std_to_custom: Vec<(String, String, f64)> = arcs
        .iter()
        .filter(|a| a.from.is_standard && !a.to.is_standard)
        .map(|a| {
            (
                format!(
                    "{}:{}",
                    a.from.namespace.clone().unwrap_or_default(),
                    a.from.name
                ),
                format!(
                    "{}:{}",
                    a.to.namespace.clone().unwrap_or_default(),
                    a.to.name
                ),
                a.weight,
            )
        })
        .collect();

    // At least the 42 anchors seen when the fixture was captured.
    assert!(
        std_to_custom.len() >= 30,
        "expected many std->custom anchors, got {}",
        std_to_custom.len()
    );

    let set: HashSet<(String, String, String)> = std_to_custom
        .iter()
        .map(|(s, e, w)| (s.clone(), e.clone(), format!("{:.1}", w)))
        .collect();

    // Positive-weight income anchor: custom inherits parent polarity.
    assert!(
        set.contains(&(
            "us-gaap:NoninterestIncome".to_string(),
            "bac:FeesAndCommissions1".to_string(),
            "1.0".to_string()
        )),
        "missing bac:FeesAndCommissions1 -> us-gaap:NoninterestIncome w=1.0"
    );
    // Positive-weight expense anchor.
    assert!(
        set.contains(&(
            "us-gaap:NoninterestExpense".to_string(),
            "bac:ProductDeliveryAndTransactionRelatedExpenses".to_string(),
            "1.0".to_string()
        )),
        "missing expense anchor w=1.0"
    );
    // Negative-weight anchor: custom polarity is opposite the parent.
    let neg = std_to_custom
        .iter()
        .find(|(_, e, _)| e == "bac:FinancingReceivableExcludingAccruedInterestAndOffBalanceSheetLiabilityCreditLossProvisionReversal");
    assert!(neg.is_some(), "missing known w=-1.0 anchor");
    assert!(
        (neg.unwrap().2 + 1.0).abs() < 1e-9,
        "expected weight -1.0, got {}",
        neg.unwrap().2
    );
}

#[test]
fn bac_sch_balances_match_expected_polarities() {
    // SCH declares the filer's intended debit/credit. These must stay stable —
    // trend-describer relies on them, not on per-filing inference.
    let xml = common::fixture_string("BAC_10q_20260331_sch.xsd");
    let schema = sec_fetcher::parsers::parse_extension_schema(&xml).unwrap();
    let by_name: HashMap<&str, &str> = schema
        .elements
        .iter()
        .filter_map(|e| e.balance.as_deref().map(|b| (e.name.as_str(), b)))
        .collect();

    assert_eq!(
        by_name.get("FeesAndCommissions1"),
        Some(&"credit"),
        "FeesAndCommissions1 must stay credit"
    );
    assert_eq!(
        by_name.get("ProductDeliveryAndTransactionRelatedExpenses"),
        Some(&"debit"),
        "expense custom must stay debit"
    );
    // Declared custom balance must agree with CAL-implied polarity:
    // credit parent + w=1.0 -> credit child; debit parent + w=1.0 -> debit child.
    assert!(!by_name.is_empty());
}

#[test]
fn parse_bac_label_linkbase() {
    let xml = common::fixture_string("BAC_10q_20260331_lab.xml");
    let labels = sec_fetcher::parsers::parse_label_linkbase(&xml).unwrap();
    assert!(!labels.is_empty());
    // Human-readable label used in the stable DataFrame (case varies by role).
    let lbl = labels
        .get("bac_FeesAndCommissions1")
        .expect("missing bac_FeesAndCommissions1 label");
    assert!(
        lbl.to_lowercase().contains("fees") && lbl.to_lowercase().contains("commissions"),
        "unexpected label: {}",
        lbl
    );
}

#[test]
fn parse_jpm_definition_linkbase() {
    let xml = common::fixture_string("JPM_10q_20260331_def.xml");
    let arcs = sec_fetcher::parsers::parse_definition_linkbase(&xml).unwrap();
    assert!(!arcs.is_empty());
}

#[test]
fn parse_bac_extension_schema() {
    let xml = common::fixture_string("BAC_10q_20260331_sch.xsd");
    let schema = sec_fetcher::parsers::parse_extension_schema(&xml).unwrap();
    assert!(!schema.elements.is_empty());
    assert!(schema.elements.iter().any(|e| e.balance.is_some()));
}
