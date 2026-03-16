use crate::models::ThirteenfHolding;
use quick_xml::events::Event;
use quick_xml::Reader;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::error::Error;

/// Parses a Form 13F-HR `informationTable.xml` document and returns one
/// [`ThirteenfHolding`] per `<infoTable>` entry, sorted by `value_usd` descending.
///
/// The SEC publishes the informationTable as a separate XML file within the
/// filing index — fetch via [`crate::network::fetch_13f`] which
/// discovers the correct filename from the index before fetching.
///
/// Note: the raw `<value>` element is in **thousands of USD**; this function
/// multiplies by 1 000 so all callers see actual dollar values.
pub fn parse_13f_xml(xml: &str) -> Result<Vec<ThirteenfHolding>, Box<dyn Error>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut holdings: Vec<ThirteenfHolding> = Vec::new();

    // Fields accumulated while inside one <infoTable> block.
    let mut in_info_table = false;
    let mut current_tag = String::new();

    let mut name = String::new();
    let mut cusip = String::new();
    let mut title_of_class = String::new();
    let mut value_raw: Decimal = dec!(0);
    let mut shares: Decimal = dec!(0);
    let mut shares_type = String::new();
    let mut put_call: Option<String> = None;
    let mut investment_discretion = String::new();

    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                // Strip namespace prefix if present (e.g. "ns1:infoTable")
                let name_bytes = e.name();
                let local = local_name(name_bytes.as_ref());
                if local == "infoTable" {
                    in_info_table = true;
                }
                if in_info_table {
                    current_tag = local.to_string();
                }
            }
            Ok(Event::Text(ref e)) if in_info_table => {
                let text = e.unescape()?.trim().to_string();
                if text.is_empty() {
                    buf.clear();
                    continue;
                }
                match current_tag.as_str() {
                    "nameOfIssuer" => name = text,
                    "cusip" => cusip = text,
                    "titleOfClass" => title_of_class = text,
                    "value" => {
                        value_raw = text.parse::<Decimal>().unwrap_or(dec!(0));
                    }
                    "sshPrnamt" => {
                        shares = text.parse::<Decimal>().unwrap_or(dec!(0));
                    }
                    "sshPrnamtType" => shares_type = text,
                    "putCall" => {
                        if !text.is_empty() {
                            put_call = Some(text);
                        }
                    }
                    "investmentDiscretion" => investment_discretion = text,
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let name_bytes = e.name();
                let local = local_name(name_bytes.as_ref());
                if local == "infoTable" {
                    holdings.push(ThirteenfHolding {
                        name: std::mem::take(&mut name),
                        cusip: std::mem::take(&mut cusip),
                        title_of_class: std::mem::take(&mut title_of_class),
                        // 13F value is in thousands of USD → convert to dollars
                        value_usd: value_raw * dec!(1000),
                        shares,
                        shares_type: std::mem::take(&mut shares_type),
                        put_call: put_call.take(),
                        investment_discretion: std::mem::take(&mut investment_discretion),
                    });
                    in_info_table = false;
                    value_raw = dec!(0);
                    shares = dec!(0);
                    current_tag.clear();
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(Box::new(e)),
            _ => {}
        }
        buf.clear();
    }

    holdings.sort_by(|a, b| b.value_usd.cmp(&a.value_usd));
    Ok(holdings)
}

/// Returns the local part of an XML tag name, stripping any namespace prefix.
fn local_name(name: &[u8]) -> &str {
    let s = std::str::from_utf8(name).unwrap_or("");
    s.rfind(':').map(|i| &s[i + 1..]).unwrap_or(s)
}
