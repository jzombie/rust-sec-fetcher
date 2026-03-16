use crate::enums::Url;
use crate::models::SicCode;
use crate::network::SecClient;
use regex::Regex;
use std::error::Error;

/// Fetches and parses the complete SEC EDGAR SIC code list.
///
/// The list is served as an HTML table at
/// `https://www.sec.gov/info/edgar/siccodes.htm`.  Each row maps a 4-digit SIC
/// code to the EDGAR reviewing office and an industry title.
///
/// The page is relatively stable (SEC rarely adds new SIC codes), so the
/// cached copy is almost always good.
///
/// # Example
///
/// ```no_run
/// # use sec_fetcher::network::{SecClient, fetch_sic_codes};
/// # use sec_fetcher::config::ConfigManager;
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&cfg)?;
/// let codes = fetch_sic_codes(&client).await?;
/// let entry = codes.iter().find(|c| c.code == 3571).unwrap();
/// println!("{} → {}", entry.description, entry.office_short()); // ELECTRONIC COMPUTERS → Technology
/// # Ok(()) }
/// ```
pub async fn fetch_sic_codes(sec_client: &SecClient) -> Result<Vec<SicCode>, Box<dyn Error>> {
    let url = Url::SicCodes.value();
    let response = sec_client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;

    if !response.status().is_success() {
        return Err(format!(
            "SIC codes page returned HTTP {}",
            response.status()
        )
        .into());
    }

    let html = response.text().await?;
    parse_sic_codes_html(&html)
}

fn parse_sic_codes_html(html: &str) -> Result<Vec<SicCode>, Box<dyn Error>> {
    // The page has a straightforward table: <tr><td>code</td><td>office</td><td>title</td></tr>
    let tr_re = Regex::new(r"(?s)<tr[\s>](.*?)</tr>")?;
    let td_re = Regex::new(r"(?s)<td[^>]*>(.*?)</td>")?;
    let tag_re = Regex::new(r"<[^>]+>")?;

    let mut codes = Vec::new();

    for tr_cap in tr_re.captures_iter(html) {
        let cells: Vec<String> = td_re
            .captures_iter(&tr_cap[1])
            .map(|c| decode_html_entities(tag_re.replace_all(&c[1], "").trim()))
            .collect();

        if cells.len() == 3 {
            if let Ok(code) = cells[0].parse::<u16>() {
                codes.push(SicCode {
                    code,
                    office: cells[1].clone(),
                    description: cells[2].clone(),
                });
            }
        }
    }

    Ok(codes)
}

fn decode_html_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}
