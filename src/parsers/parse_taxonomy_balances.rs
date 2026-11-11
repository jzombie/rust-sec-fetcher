use std::collections::HashMap;
use std::error::Error;

use quick_xml::Reader;

/// Streaming parse of a FASB taxonomy XSD: element local-name -> balance.
///
/// Pure XML parsing only: no fetching, no caching, no disk. Extracts
/// `name` + `xbrli:balance` (`credit`/`debit`) from every `xs:element`.
/// Propagates XML errors — truncated downloads fail instead of returning
/// partial maps.
pub fn parse_all_taxonomy_balances(xml: &str) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut map = HashMap::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf)? {
            quick_xml::events::Event::Start(ref e) | quick_xml::events::Event::Empty(ref e) => {
                let raw = e.name();
                let raw_ref = raw.as_ref();
                let local = raw_ref.rsplit(|&b| b == b':').next().unwrap_or(raw_ref);
                if local == b"element" {
                    let mut el_name: Option<String> = None;
                    let mut balance: Option<String> = None;
                    for attr in e.attributes().flatten() {
                        let key = attr.key.as_ref();
                        let local_key = key.rsplit(|&b| b == b':').next().unwrap_or(key);
                        if local_key == b"name" {
                            el_name =
                                Some(String::from_utf8_lossy(attr.value.as_ref()).into_owned());
                        } else if local_key == b"balance" {
                            let val = attr.value.as_ref();
                            if val.eq_ignore_ascii_case(b"credit") {
                                balance = Some("credit".to_string());
                            } else if val.eq_ignore_ascii_case(b"debit") {
                                balance = Some("debit".to_string());
                            }
                        }
                    }
                    if let (Some(n), Some(b)) = (el_name, balance) {
                        map.insert(n, b);
                    }
                }
            }
            quick_xml::events::Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_balances_and_ignores_non_balance_elements() {
        let xml = r#"<?xml version="1.0"?>
            <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:element name="Revenues" xbrli:balance="credit" />
              <xs:element name="Assets" xbrli:balance="debit" />
              <xs:element name="NoBalance" />
            </xs:schema>"#;
        let map = parse_all_taxonomy_balances(xml).unwrap();
        assert_eq!(map.get("Revenues").map(String::as_str), Some("credit"));
        assert_eq!(map.get("Assets").map(String::as_str), Some("debit"));
        assert!(!map.contains_key("NoBalance"));
    }
}
