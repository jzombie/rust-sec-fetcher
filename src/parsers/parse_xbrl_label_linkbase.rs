use std::collections::HashMap;
use std::error::Error;

use quick_xml::Reader;
use quick_xml::events::Event;

/// Parses an XBRL label linkbase XML string and returns a map of
/// concept ID (as referenced in other linkbases) to its
/// human-readable label.
///
/// The label linkbase (`EX-101.LAB`) uses a three-step indirection:
///
/// ```xml
/// <link:loc xlink:label="loc_bac_X" xlink:href="ext.xsd#bac_X"/>
/// <link:label id="lab_bac_X" xlink:label="lab_bac_X"
///             xlink:role="http://www.xbrl.org/2003/role/label">Friendly Name</link:label>
/// <link:labelArc xlink:from="loc_bac_X" xlink:to="lab_bac_X"/>
/// ```
///
/// This parser follows the `labelArc` chain: loc → labelArc → label → text.
pub fn parse_label_linkbase(xml: &str) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    // Map: xlink:label → concept fragment (from loc href)
    let mut locators: HashMap<String, String> = HashMap::new();
    // Map: xlink:label → label text (from label content)
    let mut label_texts: HashMap<String, String> = HashMap::new();
    // Map: loc_label → lab_label (from labelArc from→to)
    let mut label_arcs: HashMap<String, String> = HashMap::new();

    let mut buf = Vec::new();
    let mut current_label_ref: Option<String> = None;
    let mut capturing_text = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local = local_name(e.name().as_ref());
                match local.as_str() {
                    "loc" => {
                        let mut href = None;
                        let mut label = None;
                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            let val =
                                std::str::from_utf8(attr.value.as_ref()).unwrap_or("");
                            match attr_local.as_str() {
                                "href" => href = Some(val.to_string()),
                                "label" => label = Some(val.to_string()),
                                _ => {}
                            }
                        }
                        if let (Some(href), Some(label)) = (href, label)
                            && let Some(fragment) = href.split('#').nth(1)
                        {
                            let fragment = fragment.to_string();
                            if !fragment.is_empty() {
                                locators.insert(label, fragment);
                            }
                        }
                    }
                    "label" => {
                        let mut label_ref = None;
                        let mut role = String::new();
                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            let val =
                                std::str::from_utf8(attr.value.as_ref()).unwrap_or("");
                            match attr_local.as_str() {
                                "label" => label_ref = Some(val.to_string()),
                                "role" => role = val.to_string(),
                                _ => {}
                            }
                        }
                        if role.ends_with("/role/label") || role.ends_with("/role/terseLabel") {
                            current_label_ref = label_ref;
                            capturing_text = true;
                        }
                    }
                    "labelArc" => {
                        let mut from = None;
                        let mut to = None;
                        for attr in e.attributes().flatten() {
                            let attr_local = local_name(attr.key.as_ref());
                            let val =
                                std::str::from_utf8(attr.value.as_ref()).unwrap_or("");
                            match attr_local.as_str() {
                                "from" => from = Some(val.to_string()),
                                "to" => to = Some(val.to_string()),
                                _ => {}
                            }
                        }
                        if let (Some(from), Some(to)) = (from, to) {
                            label_arcs.insert(from, to);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if capturing_text {
                    if let Some(ref label_key) = current_label_ref {
                        let text = e.decode().unwrap_or_default().trim().to_string();
                        if !text.is_empty() && !label_texts.contains_key(label_key) {
                            label_texts.insert(label_key.clone(), text);
                        }
                    }
                }
            }
            Ok(Event::End(ref e)) => {
                let local = local_name(e.name().as_ref());
                if local.as_str() == "label" {
                    capturing_text = false;
                    current_label_ref = None;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(
                    format!("XML parse error in label linkbase: {}", e).into(),
                )
            }
            _ => {}
        }
        buf.clear();
    }

    // Resolve: loc → labelArc → label_text → concept fragment
    let mut result = HashMap::new();
    for (loc_label, concept_id) in &locators {
        if let Some(lab_label) = label_arcs.get(loc_label) {
            if let Some(label_text) = label_texts.get(lab_label) {
                result.insert(concept_id.clone(), label_text.clone());
            }
        }
    }

    Ok(result)
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
    fn test_parse_label_linkbase_with_label_arc() {
        // Real SEC format: loc → labelArc → label
        let xml = indoc! {r#"
            <?xml version="1.0" encoding="UTF-8"?>
            <link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase"
                           xmlns:xlink="http://www.w3.org/1999/xlink">
              <link:labelLink xlink:type="extended" xlink:role="http://www.apple.com/role/Detail">
                <link:loc xlink:type="locator" xlink:href="aapl.xsd#aapl_MyCustomRevenue" xlink:label="loc_aapl_MyCustomRevenue"/>
                <link:label id="lab_aapl_MyCustomRevenue" xlink:label="lab_aapl_MyCustomRevenue"
                            xlink:role="http://www.xbrl.org/2003/role/label"
                            xlink:type="resource" xml:lang="en-US">My Custom Revenue</link:label>
                <link:labelArc xlink:arcrole="http://www.xbrl.org/2003/arcrole/concept-label"
                               xlink:from="loc_aapl_MyCustomRevenue" xlink:to="lab_aapl_MyCustomRevenue"
                               xlink:type="arc" order="1"/>
              </link:labelLink>
            </link:linkbase>"#};

        let labels = parse_label_linkbase(xml).unwrap();
        assert_eq!(labels.len(), 1);
        assert_eq!(
            labels.get("aapl_MyCustomRevenue").map(|s| s.as_str()),
            Some("My Custom Revenue")
        );
    }

    #[test]
    fn test_parse_multiple_labels_with_arcs() {
        let xml = indoc! {r#"
            <?xml version="1.0"?>
            <link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase"
                           xmlns:xlink="http://www.w3.org/1999/xlink">
              <link:labelLink xlink:type="extended" xlink:role="http://example.com/role/1">
                <link:loc xlink:type="locator" xlink:href="ext.xsd#bac_Fees" xlink:label="loc_bac_Fees"/>
                <link:label xlink:label="lab_bac_Fees"
                            xlink:role="http://www.xbrl.org/2003/role/label">Fees and Commissions</link:label>
                <link:labelArc xlink:from="loc_bac_Fees" xlink:to="lab_bac_Fees"/>
              </link:labelLink>
              <link:labelLink xlink:type="extended" xlink:role="http://example.com/role/2">
                <link:loc xlink:type="locator" xlink:href="ext.xsd#bac_Expense" xlink:label="loc_bac_Expense"/>
                <link:label xlink:label="lab_bac_Expense"
                            xlink:role="http://www.xbrl.org/2003/role/label">Product Delivery Expense</link:label>
                <link:labelArc xlink:from="loc_bac_Expense" xlink:to="lab_bac_Expense"/>
              </link:labelLink>
            </link:linkbase>"#};

        let labels = parse_label_linkbase(xml).unwrap();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels.get("bac_Fees").map(|s| s.as_str()), Some("Fees and Commissions"));
        assert_eq!(labels.get("bac_Expense").map(|s| s.as_str()), Some("Product Delivery Expense"));
    }

    #[test]
    fn test_parse_empty_label_linkbase() {
        let xml = indoc! {r#"
            <?xml version="1.0"?>
            <link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase">
            </link:linkbase>"#};
        let labels = parse_label_linkbase(xml).unwrap();
        assert!(labels.is_empty());
    }
}
