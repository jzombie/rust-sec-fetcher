use quick_xml::events::Event;
use quick_xml::Reader;
use std::error::Error;
use crate::models::NportInvestment;

pub fn parse_nport_xml(xml: &str) -> Result<Vec<NportInvestment>, Box<dyn Error>> {
    let mut reader = Reader::from_str(xml);

    let mut investments = Vec::new();
    let mut current_investment = None;
    let mut current_tag = String::new();

    while let Ok(event) = reader.read_event() {
        match event {
            Event::Start(ref e) => {
                let tag = std::str::from_utf8(e.name().as_ref())?.to_string();
                if tag == "invstOrSec" {
                    current_investment = Some(NportInvestment {
                        name: String::new(),
                        lei: String::new(),
                        title: String::new(),
                        cusip: String::new(),
                        isin: String::new(),
                        balance: String::new(),
                        cur_cd: String::new(),
                        val_usd: String::new(),
                        pct_val: String::new(),
                        payoff_profile: String::new(),
                        asset_cat: String::new(),
                        issuer_cat: String::new(),
                        inv_country: String::new(),
                    });
                }
                current_tag = tag;
            }
            Event::Empty(ref e) => {
                // Handle ISIN extraction from attribute inside <isin>
                if current_tag == "identifiers" {
                    if let Some(investment) = &mut current_investment {
                        if let Some(attr) = e
                            .attributes()
                            .find(|a| a.as_ref().map_or(false, |a| a.key.as_ref() == b"value"))
                        {
                            investment.isin = attr?.unescape_value()?.to_string();
                        }
                    }
                }
            }
            Event::Text(ref e) => {
                if let Some(investment) = &mut current_investment {
                    let value = e.unescape()?.trim().to_string(); // Trim whitespace

                    // Only update if value is not empty (ignores whitespace-only nodes)
                    if !value.is_empty() {
                        match current_tag.as_str() {
                            "name" => investment.name = value,
                            "lei" => investment.lei = value,
                            "title" => investment.title = value,
                            "cusip" => investment.cusip = value,
                            "balance" => investment.balance = value,
                            "curCd" => investment.cur_cd = value,
                            "valUSD" => investment.val_usd = value,
                            "pctVal" => investment.pct_val = value,
                            "payoffProfile" => investment.payoff_profile = value,
                            "assetCat" => investment.asset_cat = value,
                            "issuerCat" => investment.issuer_cat = value,
                            "invCountry" => investment.inv_country = value,
                            _ => {}
                        }
                    }
                }
            }
            Event::End(ref e) => {
                if std::str::from_utf8(e.name().as_ref())? == "invstOrSec" {
                    if let Some(investment) = current_investment.take() {
                        investments.push(investment);
                    }
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }

    Ok(investments)
}
