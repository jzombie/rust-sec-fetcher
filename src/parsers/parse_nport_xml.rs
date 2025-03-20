use crate::models::{NportInvestment, Ticker};
use quick_xml::events::Event;
use quick_xml::Reader;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::error::Error;
use std::str::FromStr;

// TODO: Look out for seemingly duplicates such as this. The CUSIP nad ISIN are different, however (as well as the values).
// Investment 4308: NportInvestment { company_ticker: None, name: "PROLOGIS LP", lei: "GL16H1DHB0QSHP25F723", title: "Prologis LP", cusip: "74340XCH2", isin: "US74340XCH26", balance: 724000.00, cur_cd: "USD", val_usd: 713480.28, pct_val: 0.007357911161, payoff_profile: "Long", asset_cat: "DBT", issuer_cat: "", inv_country: "US" }
// Investment 4354: NportInvestment { company_ticker: None, name: "PROLOGIS LP", lei: "GL16H1DHB0QSHP25F723", title: "Prologis LP", cusip: "74340XBK6", isin: "US74340XBK63", balance: 649000.00, cur_cd: "USD", val_usd: 635299.61, pct_val: 0.006551657028, payoff_profile: "Long", asset_cat: "DBT", issuer_cat: "", inv_country: "US" }
pub fn parse_nport_xml(
    xml: &str,
    company_tickers: &[Ticker],
) -> Result<Vec<NportInvestment>, Box<dyn Error>> {
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
                        // company_ticker: None,
                        mapped_company_name: None,
                        mapped_company_cik_number: None,
                        name: String::new(),
                        lei: String::new(),
                        title: String::new(),
                        cusip: String::new(),
                        isin: String::new(),
                        balance: dec!(0.0),
                        cur_cd: String::new(),
                        val_usd: dec!(0.0),
                        pct_val: dec!(0.0),
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
                            "name" => {
                                // TODO: Re-add?
                                // investment.company_ticker =
                                // CompanyTicker::get_by_fuzzy_matched_name(
                                //     company_tickers,
                                //     &value,
                                // );
                                investment.name = value;
                            }
                            "lei" => investment.lei = value,
                            "title" => {
                                // TODO: Re-add?
                                // if investment.company_ticker.is_none() {
                                // investment.company_ticker =
                                //     CompanyTicker::get_by_fuzzy_matched_name(
                                //         company_tickers,
                                //         &value,
                                //     );
                                // }

                                investment.title = value;
                            }
                            "cusip" => investment.cusip = value,
                            "balance" => {
                                investment.balance =
                                    Decimal::from_str(&value).unwrap_or_default().round_dp(2)
                            }
                            "curCd" => investment.cur_cd = value,
                            "valUSD" => {
                                investment.val_usd =
                                    Decimal::from_str(&value).unwrap_or_default().round_dp(2)
                            }
                            "pctVal" => {
                                investment.pct_val = Decimal::from_str(&value).unwrap_or_default()
                            }
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

    NportInvestment::sort_by_pct_val_desc(&mut investments);

    for (i, investment) in investments.iter_mut().enumerate() {
        // TODO: Remove
        // println!("{}, Investment name: {}", i, investment.name);

        let company_ticker =
            match Ticker::get_by_fuzzy_matched_name(&company_tickers, &investment.title, true) {
                Some(company_ticker) => Some(company_ticker),
                None => Ticker::get_by_fuzzy_matched_name(&company_tickers, &investment.name, true),
            };

        // investment.company_ticker = company_ticker;
        if let Some(company_ticker) = company_ticker {
            investment.mapped_company_name = Some(company_ticker.company_name.clone());
            investment.mapped_company_cik_number = Some(company_ticker.cik.to_string());
        }
    }

    Ok(investments)
}
