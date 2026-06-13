use std::collections::HashMap;
use std::error::Error;

use quick_xml::Reader;
use quick_xml::events::Event;

/// Metadata about a custom element defined in a filing's extension schema.
#[derive(Debug, Clone)]
pub struct CustomElement {
    /// The local name of the element (e.g. `"MyCustomRevenue"`).
    pub name: String,
    /// The element ID used to reference it from linkbases (e.g. `"aapl_MyCustomRevenue"`).
    pub id: Option<String>,
    /// The substitution group (e.g. `"xbrli:item"`, `"xbrli:tuple"`).
    pub substitution_group: Option<String>,
    /// The XBRL type (e.g. `"xbrli:monetaryItemType"`, `"xbrli:stringItemType"`).
    pub type_name: Option<String>,
    /// The balance type for monetary items: `"credit"` or `"debit"`.
    pub balance: Option<String>,
    /// The period type: `"instant"` or `"duration"`.
    pub period_type: Option<String>,
    /// The namespace prefix used for this element in the schema.
    pub namespace_prefix: Option<String>,
}

/// Parsed extension schema metadata.
#[derive(Debug, Clone)]
pub struct ExtensionSchema {
    /// The target namespace URI (e.g. `"http://www.apple.com/20240928"`).
    pub target_namespace: Option<String>,
    /// Map of namespace prefix → namespace URI.
    pub namespace_map: HashMap<String, String>,
    /// Custom elements defined in this schema.
    pub elements: Vec<CustomElement>,
}

/// Parses an XBRL taxonomy extension schema XML string (`EX-101.SCH`).
///
/// Extracts the target namespace, namespace prefix declarations, and
/// element definitions.  This is used alongside the definition linkbase
/// parser to resolve concept references to their namespace prefixes.
pub fn parse_extension_schema(xml: &str) -> Result<ExtensionSchema, Box<dyn Error>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut target_namespace: Option<String> = None;
    let mut namespace_map: HashMap<String, String> = HashMap::new();
    let mut elements: Vec<CustomElement> = Vec::new();

    let mut buf = Vec::new();
    let mut in_schema = false;
    let mut depth: usize = 0;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let local = local_name(e.name().as_ref());
                depth += 1;

                match local.as_str() {
                    "schema" => {
                        in_schema = true;
                        // Extract targetNamespace
                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            let val = std::str::from_utf8(attr.value.as_ref()).unwrap_or("");
                            match attr_local.as_str() {
                                "targetNamespace" => {
                                    target_namespace = Some(val.to_string());
                                }
                                _ => {
                                    // XML namespace declarations like xmlns:aapl="..."
                                    let name_str =
                                        std::str::from_utf8(attr.key.as_ref()).unwrap_or("");
                                    if let Some(prefix) = name_str.strip_prefix("xmlns:") {
                                        namespace_map.insert(prefix.to_string(), val.to_string());
                                    }
                                }
                            }
                        }
                    }
                    "element" => {
                        if !in_schema {
                            continue;
                        }
                        let mut elem_name = String::new();
                        let mut elem_id = None;
                        let mut subst_group = None;
                        let mut type_name = None;
                        let mut balance = None;

                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            let val = std::str::from_utf8(attr.value.as_ref()).unwrap_or("");
                            match attr_local.as_str() {
                                "name" => elem_name = val.to_string(),
                                "id" => elem_id = Some(val.to_string()),
                                "substitutionGroup" => subst_group = Some(val.to_string()),
                                "type" => type_name = Some(val.to_string()),
                                "balance" => balance = Some(val.to_string()),
                                _ => {}
                            }
                        }

                        if !elem_name.is_empty() {
                            elements.push(CustomElement {
                                name: elem_name,
                                id: elem_id,
                                substitution_group: subst_group,
                                type_name,
                                balance,
                                period_type: None, // periodType comes from an annotation or referenced type
                                namespace_prefix: None,
                            });
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                let local = local_name(e.name().as_ref());

                if local.as_str() == "element" && in_schema {
                    let mut elem_name = String::new();
                    let mut elem_id = None;
                    let mut subst_group = None;
                    let mut type_name = None;
                    let mut balance = None;

                    for attr in e.attributes().flatten() {
                        let attr_local = local_name(attr.key.as_ref());
                        let val = std::str::from_utf8(attr.value.as_ref()).unwrap_or("");
                        match attr_local.as_str() {
                            "name" => elem_name = val.to_string(),
                            "id" => elem_id = Some(val.to_string()),
                            "substitutionGroup" => subst_group = Some(val.to_string()),
                            "type" => type_name = Some(val.to_string()),
                            "balance" => balance = Some(val.to_string()),
                            _ => {}
                        }
                    }

                    if !elem_name.is_empty() {
                        elements.push(CustomElement {
                            name: elem_name,
                            id: elem_id,
                            substitution_group: subst_group,
                            type_name,
                            balance,
                            period_type: None,
                            namespace_prefix: None,
                        });
                    }
                }
            }
            Ok(Event::End(ref e)) => {
                let local = local_name(e.name().as_ref());
                if local.as_str() == "schema" {
                    in_schema = false;
                }
                depth = depth.saturating_sub(1);
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error in extension schema: {}", e).into()),
            _ => {}
        }
        buf.clear();
    }

    // Resolve namespace prefix for each element
    // The namespace prefix is typically the prefix used in the target namespace declaration
    let ns_prefix = namespace_map
        .iter()
        .find(|(_, uri)| Some(uri.as_str()) == target_namespace.as_deref())
        .map(|(prefix, _)| prefix.clone());

    for elem in &mut elements {
        elem.namespace_prefix = ns_prefix.clone();
    }

    Ok(ExtensionSchema {
        target_namespace,
        namespace_map,
        elements,
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
    use indoc::indoc;

    #[test]
    fn test_parse_simple_extension_schema() {
        let xml = indoc! {r#"
            <?xml version="1.0" encoding="UTF-8"?>
            <xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                        xmlns:aapl="http://www.apple.com/20240928"
                        xmlns:xbrli="http://www.xbrl.org/2003/instance"
                        targetNamespace="http://www.apple.com/20240928"
                        elementFormDefault="qualified">
              <xsd:element name="MyCustomRevenue" id="aapl_MyCustomRevenue"
                           substitutionGroup="xbrli:item" type="xbrli:monetaryItemType" xbrli:balance="credit"/>
              <xsd:element name="MyCustomAsset" id="aapl_MyCustomAsset"
                           substitutionGroup="xbrli:item" type="xbrli:monetaryItemType" xbrli:balance="debit"/>
            </xsd:schema>"#};

        let schema = parse_extension_schema(xml).unwrap();
        assert_eq!(
            schema.target_namespace.as_deref(),
            Some("http://www.apple.com/20240928")
        );
        assert_eq!(schema.elements.len(), 2);

        assert_eq!(schema.elements[0].name, "MyCustomRevenue");
        assert_eq!(
            schema.elements[0].id.as_deref(),
            Some("aapl_MyCustomRevenue")
        );
        assert_eq!(
            schema.elements[0].substitution_group.as_deref(),
            Some("xbrli:item")
        );
        assert_eq!(
            schema.elements[0].type_name.as_deref(),
            Some("xbrli:monetaryItemType")
        );
        assert_eq!(schema.elements[0].balance.as_deref(), Some("credit"));
        assert_eq!(schema.elements[0].namespace_prefix.as_deref(), Some("aapl"));

        assert_eq!(schema.elements[1].name, "MyCustomAsset");
        assert_eq!(schema.elements[1].balance.as_deref(), Some("debit"));
    }

    #[test]
    fn test_parse_empty_schema() {
        let xml = indoc! {r#"
            <?xml version="1.0"?>
            <xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                        targetNamespace="http://example.com">
            </xsd:schema>"#};

        let schema = parse_extension_schema(xml).unwrap();
        assert_eq!(
            schema.target_namespace.as_deref(),
            Some("http://example.com")
        );
        assert!(schema.elements.is_empty());
    }
}
