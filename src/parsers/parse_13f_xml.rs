use crate::models::ThirteenfHolding;
use crate::normalize::{Pct, compute_13f_weight_pct, normalize_13f_value_usd};
use chrono::NaiveDate;
use quick_xml::Reader;
use quick_xml::events::Event;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::error::Error;

/// Parses a Form 13F-HR `informationTable.xml` document and returns one
/// [`ThirteenfHolding`] per `<infoTable>` entry, sorted by `value_usd` descending.
///
/// The `filing_date` parameter is required to resolve the `<value>` unit — the
/// EDGAR 13F-HR schema changed from thousands-of-USD to actual-USD around
/// 2023-01-01.  Pass the `filing_date` from the corresponding
/// [`crate::models::CikSubmission`].  All unit conversion is handled by
/// [`crate::normalize::normalize_13f_value_usd`]; do not apply any multiplier
/// at the call site.
///
/// Portfolio weight percentages (0–100 scale) are computed here via
/// [`crate::normalize::compute_13f_weight_pct`] and stored in
/// [`ThirteenfHolding::weight_pct`].
///
/// The filing index and the correct informationTable filename are discovered
/// automatically by [`crate::network::fetch_13f`].
pub fn parse_13f_xml(
    xml: &str,
    filing_date: Option<NaiveDate>,
) -> Result<Vec<ThirteenfHolding>, Box<dyn Error>> {
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
                    "putCall" if !text.is_empty() => {
                        put_call = Some(text);
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
                        // Unit conversion (thousands era vs. actual-USD era) is
                        // handled exclusively by normalize_13f_value_usd.
                        value_usd: normalize_13f_value_usd(value_raw, filing_date),
                        shares,
                        shares_type: std::mem::take(&mut shares_type),
                        put_call: put_call.take(),
                        investment_discretion: std::mem::take(&mut investment_discretion),
                        // Populated in the second pass below.
                        weight_pct: Pct::ZERO,
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

    // Second pass: compute portfolio weights via normalize::percentage.
    // All weight math lives in compute_13f_weight_pct — do not replicate it here.
    let total: Decimal = holdings.iter().map(|h| h.value_usd).sum();
    for h in holdings.iter_mut() {
        h.weight_pct = compute_13f_weight_pct(h.value_usd, total);
    }

    holdings.sort_by_key(|b| std::cmp::Reverse(b.value_usd));
    Ok(holdings)
}

/// Returns the local part of an XML tag name, stripping any namespace prefix.
fn local_name(name: &[u8]) -> &str {
    let s = std::str::from_utf8(name).unwrap_or("");
    s.rfind(':').map(|i| &s[i + 1..]).unwrap_or(s)
}
