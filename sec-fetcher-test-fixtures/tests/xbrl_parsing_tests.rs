mod common;

#[test]
fn parse_bac_calculation_linkbase() {
    let xml = common::fixture_string("BAC_10q_20260331_cal.xml");
    let arcs = sec_fetcher::parsers::parse_calculation_linkbase(&xml).unwrap();
    assert!(!arcs.is_empty());
    assert!(arcs.iter().any(|a| a.from.is_standard && !a.to.is_standard));
}

#[test]
fn parse_bac_label_linkbase() {
    let xml = common::fixture_string("BAC_10q_20260331_lab.xml");
    let labels = sec_fetcher::parsers::parse_label_linkbase(&xml).unwrap();
    assert!(!labels.is_empty());
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
