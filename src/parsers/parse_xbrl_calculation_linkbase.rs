use std::error::Error;

use quick_xml::Reader;
use quick_xml::events::Event;

use crate::parsers::parse_xbrl_definition_linkbase::{
    ConceptRef, is_standard_namespace, local_name,
};

/// A single summation-item arc from the calculation linkbase.
///
/// In XBRL calculation: `from` is the parent/total concept and `to` is the
/// child/detail concept.  When a standard concept is the `from` and a custom
/// concept is the `to`, the custom concept inherits its polarity from the
/// standard parent.
#[derive(Debug, Clone)]
pub struct SummationItemArc {
    /// The parent/total concept (typically the standard GAAP concept).
    pub from: ConceptRef,
    /// The child/detail concept (could be standard or custom).
    pub to: ConceptRef,
    /// The weight of the arc in the calculation (1.0 or -1.0 typically).
    pub weight: f64,
    /// The order of the arc within the presentation.
    pub order: Option<i32>,
}

/// Known summation-item arcrole URIs.
const SUMMATION_ITEM_ARCROLES: &[&str] = &[
    "http://www.xbrl.org/2003/arcrole/summation-item",
    "https://www.xbrl.org/2003/arcrole/summation-item",
    "https://xbrl.org/2023/arcrole/summation-item",
    "http://xbrl.org/2023/arcrole/summation-item",
];

/// Parses an XBRL calculation linkbase XML string and returns the list of
/// summation-item arcs.
///
/// The calculation linkbase (`EX-101.CAL`) defines how concepts roll up into
/// each other via `summation-item` arcs.  These arcs carry a `weight` (1.0 or
/// -1.0) and direction (`from` = total, `to` = detail).
pub fn parse_calculation_linkbase(xml: &str) -> Result<Vec<SummationItemArc>, Box<dyn Error>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut locators: std::collections::HashMap<String, ConceptRef> =
        std::collections::HashMap::new();
    let mut arcs: Vec<(String, String, f64, Option<i32>)> = Vec::new();

    let mut buf = Vec::new();
    let mut in_calculation_link = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let name = local_name(e.name().as_ref());

                match name.as_str() {
                    "calculationLink" => {
                        in_calculation_link = true;
                    }
                    "loc" => {
                        if !in_calculation_link {
                            continue;
                        }
                        let mut href = None;
                        let mut label = None;
                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            let val = std::str::from_utf8(attr.value.as_ref()).unwrap_or("");
                            match attr_local.as_str() {
                                "href" => href = Some(val.to_string()),
                                "label" => label = Some(val.to_string()),
                                _ => {}
                            }
                        }
                        if let (Some(href), Some(label)) = (href, label)
                            && let Some(concept) = parse_href_concept(&href)
                        {
                            locators.insert(label, concept);
                        }
                    }
                    "calculationArc" => {
                        if !in_calculation_link {
                            continue;
                        }
                        let mut from = None;
                        let mut to = None;
                        let mut weight = 1.0f64;
                        let mut order = None;
                        let mut is_summation = false;

                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            let val = std::str::from_utf8(attr.value.as_ref()).unwrap_or("");
                            match attr_local.as_str() {
                                "from" => from = Some(val.to_string()),
                                "to" => to = Some(val.to_string()),
                                "weight" => weight = val.parse::<f64>().unwrap_or(1.0),
                                "order" => order = val.parse::<i32>().ok(),
                                "arcrole" if SUMMATION_ITEM_ARCROLES.contains(&val) => {
                                    is_summation = true;
                                }
                                _ => {}
                            }
                        }

                        if is_summation && let (Some(from), Some(to)) = (from, to) {
                            arcs.push((from, to, weight, order));
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let name = local_name(e.name().as_ref());
                if name.as_str() == "calculationLink" {
                    in_calculation_link = false;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error in calculation linkbase: {}", e).into()),
            _ => {}
        }
        buf.clear();
    }

    let mut results = Vec::new();
    for (from_label, to_label, weight, order) in arcs {
        if let (Some(from), Some(to)) = (locators.get(&from_label), locators.get(&to_label)) {
            results.push(SummationItemArc {
                from: from.clone(),
                to: to.clone(),
                weight,
                order,
            });
        }
    }

    Ok(results)
}

/// Re-export the href parser from the definition linkbase module.
fn parse_href_concept(href: &str) -> Option<ConceptRef> {
    let fragment = href.split('#').nth(1)?;
    let concept_str = fragment.trim();
    if concept_str.is_empty() {
        return None;
    }
    if let Some(underscore_pos) = concept_str.find('_') {
        let prefix = &concept_str[..underscore_pos];
        let name = &concept_str[underscore_pos + 1..];
        if !prefix.is_empty() && !name.is_empty() {
            let is_std = is_standard_namespace(prefix);
            return Some(ConceptRef {
                namespace: Some(prefix.to_string()),
                name: name.to_string(),
                is_standard: is_std,
            });
        }
    }
    Some(ConceptRef {
        namespace: None,
        name: concept_str.to_string(),
        is_standard: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    #[test]
    fn test_parse_simple_calculation_linkbase() {
        let xml = indoc! {r#"
            <?xml version="1.0" encoding="UTF-8"?>
            <link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase"
                           xmlns:xlink="http://www.w3.org/1999/xlink">
              <link:calculationLink xlink:type="extended" xlink:role="http://www.apple.com/role/IncomeStatement">
                <link:loc xlink:type="locator" xlink:href="http://xbrl.sec.gov/stm/2024/us-gaap-2024.xsd#us-gaap_Revenues" xlink:label="rev_lbl"/>
                <link:loc xlink:type="locator" xlink:href="aapl-20240928.xsd#aapl_MyCustomRevenue" xlink:label="myrev_lbl"/>
                <link:calculationArc xlink:type="arc" xlink:arcrole="http://www.xbrl.org/2003/arcrole/summation-item"
                                     xlink:from="rev_lbl" xlink:to="myrev_lbl" weight="1.0" order="1"/>
              </link:calculationLink>
            </link:linkbase>"#};

        let arcs = parse_calculation_linkbase(xml).unwrap();
        assert_eq!(arcs.len(), 1);
        assert!(arcs[0].from.is_standard);
        assert_eq!(arcs[0].from.name, "Revenues");
        assert!(!arcs[0].to.is_standard);
        assert_eq!(arcs[0].to.name, "MyCustomRevenue");
        assert!((arcs[0].weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_negative_weight() {
        let xml = indoc! {r#"
            <?xml version="1.0"?>
            <link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase"
                           xmlns:xlink="http://www.w3.org/1999/xlink">
              <link:calculationLink xlink:type="extended" xlink:role="http://example.com/role/GrossProfit">
                <link:loc xlink:type="locator" xlink:href="us-gaap.xsd#us-gaap_GrossProfit" xlink:label="gp_lbl"/>
                <link:loc xlink:type="locator" xlink:href="ext.xsd#aapl_CustomCostOfRevenue" xlink:label="cost_lbl"/>
                <link:calculationArc xlink:type="arc" xlink:arcrole="http://www.xbrl.org/2003/arcrole/summation-item"
                                     xlink:from="gp_lbl" xlink:to="cost_lbl" weight="-1.0" order="10"/>
              </link:calculationLink>
            </link:linkbase>"#};

        let arcs = parse_calculation_linkbase(xml).unwrap();
        assert_eq!(arcs.len(), 1);
        assert!((arcs[0].weight - (-1.0)).abs() < 1e-10);
        assert_eq!(arcs[0].order, Some(10));
    }

    #[test]
    fn test_parse_empty_calculation_linkbase() {
        let xml = indoc! {r#"
            <?xml version="1.0"?>
            <link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase">
            </link:linkbase>"#};
        let arcs = parse_calculation_linkbase(xml).unwrap();
        assert!(arcs.is_empty());
    }
}
