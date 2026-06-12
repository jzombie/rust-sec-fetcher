use std::collections::HashMap;
use std::error::Error;

use quick_xml::Reader;
use quick_xml::events::Event;

/// A resolved concept reference from a definition linkbase locator.
#[derive(Debug, Clone)]
pub struct ConceptRef {
    /// The namespace prefix (e.g. `"us-gaap"`, `"aapl"`).
    pub namespace: Option<String>,
    /// The local name of the concept (e.g. `"Revenues"`, `"MyCustomTag"`).
    pub name: String,
    /// Whether this is a standard SEC taxonomy concept.
    pub is_standard: bool,
}

/// A single definition arc extracted from the linkbase.
#[derive(Debug, Clone)]
pub struct DefinitionArc {
    /// Resolved source concept (the broader/general concept).
    pub from: ConceptRef,
    /// Resolved target concept (the narrower/special concept).
    pub to: ConceptRef,
    /// The arcrole URI describing the relationship.
    /// Typical: `"http://xbrl.org/arcrole/2008/definition-arcrole/general-special"`
    pub arcrole: String,
}

/// Known standard namespace prefixes used by the SEC taxonomies.
/// Concepts under these namespaces are standard; everything else is custom.
const STANDARD_NAMESPACES: &[&str] = &[
    "us-gaap", "dei", "srt", "invest", "country", "stpr", "naics",
    "sic", "currency", "ecd", "utr", "xref", "rrr", "ffd", "cef",
    "ifrs-full",
];

/// Parses an XBRL definition linkbase XML string and returns the list
/// of definition arcs (custom→standard concept relationships).
///
/// # Arguments
///
/// * `xml` - The raw XML content of a `-def.xml` file (EX-101.DEF).
pub fn parse_definition_linkbase(xml: &str) -> Result<Vec<DefinitionArc>, Box<dyn Error>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    // Map: xlink:label → ConceptRef
    let mut locators: HashMap<String, ConceptRef> = HashMap::new();
    let mut arcs: Vec<(String, String, String)> = Vec::new(); // (from_label, to_label, arcrole)

    let mut buf = Vec::new();
    let mut in_definition_link = false;
    // Track depth within definitionLink to handle nested elements
    let mut depth: usize = 0;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let local = local_name(e.name().as_ref());
                depth += 1;

                match local.as_str() {
                    "definitionLink" => {
                        in_definition_link = true;
                    }
                    "loc" => {
                        if !in_definition_link {
                            continue;
                        }
                        let mut href = None;
                        let mut label = None;
                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            match attr_local.as_str() {
                                "href" => {
                                    href = Some(
                                        std::str::from_utf8(attr.value.as_ref())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                                "label" => {
                                    label = Some(
                                        std::str::from_utf8(attr.value.as_ref())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                                _ => {}
                            }
                        }
                        if let (Some(href), Some(label)) = (href, label)
                            && let Some(concept) = parse_href_concept(&href)
                        {
                            locators.insert(label, concept);
                        }
                    }
                    "definitionArc" => {
                        if !in_definition_link {
                            continue;
                        }
                        let mut from = None;
                        let mut to = None;
                        let mut arcrole = None;
                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            match attr_local.as_str() {
                                "from" => {
                                    from = Some(
                                        std::str::from_utf8(attr.value.as_ref())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                                "to" => {
                                    to = Some(
                                        std::str::from_utf8(attr.value.as_ref())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                                "arcrole" => {
                                    arcrole = Some(
                                        std::str::from_utf8(attr.value.as_ref())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                                _ => {}
                            }
                        }
                        if let (Some(from), Some(to), Some(arcrole)) = (from, to, arcrole) {
                            arcs.push((from, to, arcrole));
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                // Self-closing tags like <link:loc ... />
                let local = local_name(e.name().as_ref());

                match local.as_str() {
                    "loc" => {
                        if !in_definition_link {
                            continue;
                        }
                        let mut href = None;
                        let mut label = None;
                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            match attr_local.as_str() {
                                "href" => {
                                    href = Some(
                                        std::str::from_utf8(attr.value.as_ref())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                                "label" => {
                                    label = Some(
                                        std::str::from_utf8(attr.value.as_ref())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                                _ => {}
                            }
                        }
                        if let (Some(href), Some(label)) = (href, label)
                            && let Some(concept) = parse_href_concept(&href)
                        {
                            locators.insert(label, concept);
                        }
                    }
                    "definitionArc" => {
                        if !in_definition_link {
                            continue;
                        }
                        let mut from = None;
                        let mut to = None;
                        let mut arcrole = None;
                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            match attr_local.as_str() {
                                "from" => {
                                    from = Some(
                                        std::str::from_utf8(attr.value.as_ref())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                                "to" => {
                                    to = Some(
                                        std::str::from_utf8(attr.value.as_ref())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                                "arcrole" => {
                                    arcrole = Some(
                                        std::str::from_utf8(attr.value.as_ref())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                                _ => {}
                            }
                        }
                        if let (Some(from), Some(to), Some(arcrole)) = (from, to, arcrole) {
                            arcs.push((from, to, arcrole));
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let local = local_name(e.name().as_ref());
                if local.as_str() == "definitionLink" {
                    in_definition_link = false;
                }
                depth = depth.saturating_sub(1);
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error in definition linkbase: {}", e).into()),
            _ => {}
        }
        buf.clear();
    }

    // Resolve arcs to ConceptRefs
    let mut results = Vec::new();
    for (from_label, to_label, arcrole) in arcs {
        let from_concept = locators.get(&from_label);
        let to_concept = locators.get(&to_label);
        if let (Some(from), Some(to)) = (from_concept, to_concept) {
            results.push(DefinitionArc {
                from: from.clone(),
                to: to.clone(),
                arcrole,
            });
        }
    }

    Ok(results)
}

/// Parses an `xlink:href` attribute value into a `ConceptRef`.
///
/// Handles formats like:
/// - `http://xbrl.sec.gov/stm/2024/us-gaap-2024.xsd#us-gaap_Revenues`
/// - `aapl-20240928.xsd#aapl_MyCustomTag`
/// - `#us-gaap_Revenues`
///
/// The fragment after `#` is expected to be `{prefix}_{localName}`.
fn parse_href_concept(href: &str) -> Option<ConceptRef> {
    // Split on # to get the fragment
    let fragment = href.split('#').nth(1)?;

    // The fragment is typically `prefix_localName` (e.g. `us-gaap_Revenues`)
    // Some systems use dot notation or other formats; handle the underscore case.
    let concept_str = fragment.trim();

    if concept_str.is_empty() {
        return None;
    }

    // Try splitting on first underscore for (prefix, name)
    if let Some(underscore_pos) = concept_str.find('_') {
        let prefix = &concept_str[..underscore_pos];
        let name = &concept_str[underscore_pos + 1..];
        if !prefix.is_empty() && !name.is_empty() {
            let is_std = STANDARD_NAMESPACES.contains(&prefix);
            return Some(ConceptRef {
                namespace: Some(prefix.to_string()),
                name: name.to_string(),
                is_standard: is_std,
            });
        }
    }

    // Fallback: treat the whole fragment as the concept name
    Some(ConceptRef {
        namespace: None,
        name: concept_str.to_string(),
        is_standard: false,
    })
}

fn local_name(name: &[u8]) -> String {
    let s = std::str::from_utf8(name).unwrap_or("");
    s.rfind(':')
        .map(|i| s[i + 1..].to_string())
        .unwrap_or_else(|| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_definition_linkbase() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase"
               xmlns:xlink="http://www.w3.org/1999/xlink">
  <link:definitionLink xlink:type="extended" xlink:role="http://www.apple.com/role/MyRole">
    <link:loc xlink:type="locator" xlink:href="http://xbrl.sec.gov/stm/2024/us-gaap-2024.xsd#us-gaap_Revenues" xlink:label="us-gaap_Revenues"/>
    <link:loc xlink:type="locator" xlink:href="aapl-20240928.xsd#aapl_MyCustomRevenue" xlink:label="aapl_MyCustomRevenue"/>
    <link:definitionArc xlink:type="arc" xlink:arcrole="http://xbrl.org/arcrole/2008/definition-arcrole/general-special"
                        xlink:from="us-gaap_Revenues" xlink:to="aapl_MyCustomRevenue" order="1"/>
  </link:definitionLink>
</link:linkbase>"#;

        let arcs = parse_definition_linkbase(xml).unwrap();
        assert_eq!(arcs.len(), 1);

        let arc = &arcs[0];
        assert!(arc.from.is_standard);
        assert_eq!(arc.from.namespace.as_deref(), Some("us-gaap"));
        assert_eq!(arc.from.name, "Revenues");

        assert!(!arc.to.is_standard);
        assert_eq!(arc.to.namespace.as_deref(), Some("aapl"));
        assert_eq!(arc.to.name, "MyCustomRevenue");

        assert_eq!(arc.arcrole, "http://xbrl.org/arcrole/2008/definition-arcrole/general-special");
    }

    #[test]
    fn test_parse_with_multiple_arcs() {
        let xml = r#"<?xml version="1.0"?>
<link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase"
               xmlns:xlink="http://www.w3.org/1999/xlink">
  <link:definitionLink xlink:type="extended" xlink:role="http://example.com/role/1">
    <link:loc xlink:type="locator" xlink:href="us-gaap-2024.xsd#us-gaap_Assets" xlink:label="Assets_lbl"/>
    <link:loc xlink:type="locator" xlink:href="ext.xsd#aapl_MyAssets" xlink:label="MyAssets_lbl"/>
    <link:definitionArc xlink:type="arc" xlink:arcrole="http://xbrl.org/arcrole/2008/definition-arcrole/general-special"
                        xlink:from="Assets_lbl" xlink:to="MyAssets_lbl" order="1"/>
  </link:definitionLink>
  <link:definitionLink xlink:type="extended" xlink:role="http://example.com/role/2">
    <link:loc xlink:type="locator" xlink:href="us-gaap-2024.xsd#us-gaap_Revenues" xlink:label="Rev_lbl"/>
    <link:loc xlink:type="locator" xlink:href="ext.xsd#aapl_MyRevenue" xlink:label="MyRev_lbl"/>
    <link:definitionArc xlink:type="arc" xlink:arcrole="http://xbrl.org/arcrole/2008/definition-arcrole/general-special"
                        xlink:from="Rev_lbl" xlink:to="MyRev_lbl" order="2"/>
  </link:definitionLink>
</link:linkbase>"#;

        let arcs = parse_definition_linkbase(xml).unwrap();
        assert_eq!(arcs.len(), 2);

        assert_eq!(arcs[0].from.name, "Assets");
        assert_eq!(arcs[0].to.name, "MyAssets");

        assert_eq!(arcs[1].from.name, "Revenues");
        assert_eq!(arcs[1].to.name, "MyRevenue");
    }

    #[test]
    fn test_parse_empty_linkbase() {
        let xml = r#"<?xml version="1.0"?>
<link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase">
</link:linkbase>"#;

        let arcs = parse_definition_linkbase(xml).unwrap();
        assert!(arcs.is_empty());
    }

    #[test]
    fn test_parse_href_concept_standard() {
        let concept = parse_href_concept("http://xbrl.sec.gov/stm/2024/us-gaap-2024.xsd#us-gaap_NetIncomeLoss")
            .unwrap();
        assert!(concept.is_standard);
        assert_eq!(concept.namespace.as_deref(), Some("us-gaap"));
        assert_eq!(concept.name, "NetIncomeLoss");
    }

    #[test]
    fn test_parse_href_concept_custom() {
        let concept = parse_href_concept("msft-20241231.xsd#msft_MyCustomTag")
            .unwrap();
        assert!(!concept.is_standard);
        assert_eq!(concept.namespace.as_deref(), Some("msft"));
        assert_eq!(concept.name, "MyCustomTag");
    }

    #[test]
    fn test_parse_href_concept_no_namespace() {
        let concept = parse_href_concept("#JustAConcept").unwrap();
        assert!(concept.namespace.is_none());
        assert_eq!(concept.name, "JustAConcept");
        assert!(!concept.is_standard);
    }
}
