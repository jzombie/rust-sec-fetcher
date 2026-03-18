use crate::enums::Url;
use crate::models::SicCode;
use crate::network::SecClient;
use regex::Regex;
use std::error::Error;
use std::sync::LazyLock as Lazy;

static TR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<tr[\s>](.*?)</tr>").unwrap());
static TD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<td[^>]*>(.*?)</td>").unwrap());

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
        return Err(format!("SIC codes page returned HTTP {}", response.status()).into());
    }

    let html = response.text().await?;
    parse_sic_codes_html(&html)
}

fn parse_sic_codes_html(html: &str) -> Result<Vec<SicCode>, Box<dyn Error>> {
    let mut codes = Vec::new();

    for tr_cap in TR_RE.captures_iter(html) {
        let cells: Vec<String> = TD_RE
            .captures_iter(&tr_cap[1])
            .filter_map(|c| {
                // html2text handles entity decoding and tag stripping.
                // plain_no_decorate() suppresses link-reference footnotes.
                html2text::config::plain_no_decorate()
                    .string_from_read(c[1].as_bytes(), 1_000_000)
                    .ok()
                    .map(|s| s.trim().to_string())
            })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_sic_table() {
        let html = r#"
            <table>
                <tr><th>SIC Code</th><th>Office</th><th>Industry Title</th></tr>
                <tr><td>3571</td><td>Office of Technology</td><td>ELECTRONIC COMPUTERS</td></tr>
                <tr><td>6020</td><td>Office of Finance</td><td>SAVINGS INSTITUTION, FEDERALLY CHARTERED</td></tr>
            </table>
        "#;
        let codes = parse_sic_codes_html(html).unwrap();
        assert_eq!(codes.len(), 2);
        assert_eq!(codes[0].code, 3571);
        assert_eq!(codes[0].description, "ELECTRONIC COMPUTERS");
        assert_eq!(codes[0].office, "Office of Technology");
        assert_eq!(codes[1].code, 6020);
    }

    #[test]
    fn decodes_html_entities_in_cells() {
        let html = r#"
            <table>
                <tr><td>0100</td><td>Office of Finance</td><td>CROPS &amp; FARMING</td></tr>
            </table>
        "#;
        let codes = parse_sic_codes_html(html).unwrap();
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0].description, "CROPS & FARMING");
    }

    #[test]
    fn strips_nested_tags_from_cells() {
        let html = r#"
            <table>
                <tr><td>7372</td><td><b>Office of Technology</b></td><td>PREPACKAGED SOFTWARE</td></tr>
            </table>
        "#;
        let codes = parse_sic_codes_html(html).unwrap();
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0].office, "Office of Technology");
    }

    #[test]
    fn skips_header_and_non_numeric_rows() {
        let html = r#"
            <table>
                <tr><th>SIC</th><th>Office</th><th>Title</th></tr>
                <tr><td>not-a-number</td><td>x</td><td>y</td></tr>
                <tr><td>0111</td><td>Office of Finance</td><td>WHEAT</td></tr>
            </table>
        "#;
        let codes = parse_sic_codes_html(html).unwrap();
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0].code, 111);
    }
}
