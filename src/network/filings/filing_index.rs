use crate::enums::Url;
use crate::models::{CikSubmission, FilingDocument, FilingIndex};
use crate::network::SecClient;
use regex::Regex;
use std::error::Error;
use std::sync::LazyLock as Lazy;

static TR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<tr[^>]*>(.*?)</tr>").unwrap());
static TD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<td[^>]*>(.*?)</td>").unwrap());
static HREF_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"(?i)href="([^"]+)""#).unwrap());
static TAGS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]*>").unwrap());

/// Parses the EDGAR HTML filing index page into a [`FilingIndex`].
///
/// EDGAR's `-index.htm` page contains an HTML table listing every document in
/// the filing with columns: Seq, Description, Document (linked filename), Type,
/// Size.  This function extracts the filename and SEC document type from each
/// row.
///
/// Note: some hrefs use EDGAR's iXBRL viewer prefix (`/ix?doc=/Archives/...`);
/// the actual filename is extracted from those transparently.
fn parse_filing_index_html(html: &str) -> Result<FilingIndex, Box<dyn Error>> {
    let mut documents = Vec::new();

    for tr_cap in TR_RE.captures_iter(html) {
        let row_html = &tr_cap[1];

        let cells: Vec<String> = TD_RE
            .captures_iter(row_html)
            .map(|c| c[1].to_string())
            .collect();

        // We need at least 4 cells: seq, description, document, type.
        if cells.len() < 4 {
            continue;
        }

        // Cell index 2: the document link.  Extract just the filename from the href.
        let doc_cell = &cells[2];
        let name = match HREF_RE.captures(doc_cell) {
            Some(cap) => {
                let href = &cap[1];
                // iXBRL viewer URLs look like: /ix?doc=/Archives/edgar/data/.../file.htm
                let path = if let Some(pos) = href.find("?doc=") {
                    &href[pos + 5..]
                } else {
                    href
                };
                // Take only the final path segment (the filename itself).
                path.rsplit('/').next().unwrap_or("").to_string()
            }
            None => continue,
        };

        if name.is_empty() {
            continue;
        }

        // Cell index 3: the SEC document type (e.g. "EX-99.1", "8-K", "GRAPHIC").
        // Strip any HTML tags, then collapse whitespace and HTML entities.
        let type_stripped = TAGS_RE.replace_all(&cells[3], "");
        let document_type = type_stripped
            .replace("&nbsp;", "")
            .replace('\u{00a0}', "")
            .trim()
            .to_string();

        documents.push(FilingDocument {
            name,
            document_type,
        });
    }

    Ok(FilingIndex { documents })
}

/// Fetches the EDGAR filing index for a given [`CikSubmission`], returning the
/// list of all documents (primary form, exhibits, stylesheets, etc.) stored in
/// that filing's archive directory.
///
/// The index is sourced from EDGAR's HTML filing index page:
/// `https://www.sec.gov/Archives/edgar/data/{CIK}/{ACCESSION}/{ACCESSION}-index.htm`
///
/// Use [`FilingIndex::exhibits`] on the result to filter down to exhibit
/// documents only (those whose SEC type starts with `"EX-"`).
///
/// # Example
/// ```rust,no_run
/// # use sec_fetcher::network::{fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_filing_index, SecClient};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::{CikSubmission, TickerSymbol};
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
/// let submissions = fetch_cik_submissions(&client, cik).await?;
/// let latest = CikSubmission::by_form(&submissions, "8-K").into_iter().next().unwrap();
///
/// let index = fetch_filing_index(&client, latest).await?;
/// for exhibit in index.exhibits() {
///     println!("{} — {}", exhibit.document_type, exhibit.name);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn fetch_filing_index(
    client: &SecClient,
    filing: &CikSubmission,
) -> Result<FilingIndex, Box<dyn Error>> {
    let url = Url::CikAccessionIndex(filing.cik.clone(), filing.accession_number.clone()).value();

    let response = client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;

    let html = response.text().await?;
    parse_filing_index_html(&html)
}
