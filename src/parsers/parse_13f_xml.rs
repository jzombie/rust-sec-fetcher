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
/// Note: the `<value>` element in the EDGAR 13F-HR `informationTable` XML is
/// in **actual US dollars** — not thousands.  Empirically confirmed by
/// inspecting a real filing (BRK-A Q4 2025, accession 0001193125-26-054580):
/// <https://www.sec.gov/Archives/edgar/data/1067983/000119312526054580/50240.xml>
///
/// The EDGAR Filing Manual (Vol. II, §16) and the 13F XML Technical
/// Specification are the authoritative references:
/// <https://www.sec.gov/info/edgar/edgarfm-vol2.pdf>
/// <https://www.sec.gov/info/edgar/forms/edgarform13f/13fxmltechspec.pdf>
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
            Ok(Event::Text(e)) if in_info_table => {
                let text = e.decode()?.trim().to_string();
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
                        // The modern EDGAR 13F-HR XML stores <value> in actual
                        // US dollars (not thousands), so no conversion is needed.
                        value_usd: value_raw,
                        shares,
                        shares_type: std::mem::take(&mut shares_type),
                        put_call: put_call.take(),
                        investment_discretion: std::mem::take(&mut investment_discretion),
                        // Populated in the second pass below.
                        weight_pct: dec!(0),
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

    // Second pass: compute portfolio weights (0-100 percentage scale).
    let total: Decimal = holdings.iter().map(|h| h.value_usd).sum();
    if !total.is_zero() {
        for h in holdings.iter_mut() {
            h.weight_pct = (h.value_usd / total * dec!(100)).round_dp(4);
        }
    }

    holdings.sort_by(|a, b| b.value_usd.cmp(&a.value_usd));
    Ok(holdings)
}

/// Returns the local part of an XML tag name, stripping any namespace prefix.
fn local_name(name: &[u8]) -> &str {
    let s = std::str::from_utf8(name).unwrap_or("");
    s.rfind(':').map(|i| &s[i + 1..]).unwrap_or(s)
}
