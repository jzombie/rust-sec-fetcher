use crate::network::SecClient;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::error::Error;

#[derive(Debug)]
pub struct Investment {
    pub name: String,
    pub lei: String,
    pub title: String,
    pub cusip: String,
    pub balance: String,
    pub cur_cd: String,
    pub val_usd: String,
    pub pct_val: String,
    pub payoff_profile: String,
    pub asset_cat: String,
    pub issuer_cat: String,
    pub inv_country: String,
}

pub async fn fetch_n_port_filing(
    sec_client: &SecClient,
    cik: u64,
    accession_number: &str, // TODO: Strip
) -> Result<Vec<Investment>, Box<dyn Error>> {
    let url = format!(
        "https://www.sec.gov/Archives/edgar/data/{}/{}.xml",
        cik, accession_number
    );

    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None)
        .await?;
    let xml_data = response.text().await?;

    parse_nport_xml(&xml_data)
}

fn parse_nport_xml(xml: &str) -> Result<Vec<Investment>, Box<dyn Error>> {
    let mut reader = Reader::from_str(xml);

    let mut investments = Vec::new();
    let mut current_investment = None;
    let mut current_tag = String::new();

    while let Ok(event) = reader.read_event() {
        match event {
            Event::Start(ref e) => {
                let tag = std::str::from_utf8(e.name().as_ref())?.to_string();
                if tag == "invstOrSec" {
                    current_investment = Some(Investment {
                        name: String::new(),
                        lei: String::new(),
                        title: String::new(),
                        cusip: String::new(),
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
