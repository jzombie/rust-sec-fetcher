/// Fetches the latest 8-K filing for a given ticker symbol and prints it as
/// embedding-friendly text: prose is preserved as-is, and HTML tables are
/// flattened into "Header: value, Header: value." sentences.
///
/// Usage:
///   cargo run --example latest_8k_as_markdown -- <TICKER_SYMBOL>
///
/// Example:
///   cargo run --example latest_8k_as_markdown -- LLY

use html_to_markdown_rs::{convert, ConversionOptions, HeadingStyle, PreprocessingPreset};
use regex::Regex;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_8k_filings_by_ticker_symbol, fetch_company_tickers, SecClient};
use std::env;
use std::error::Error;
use tokio;

// ---------------------------------------------------------------------------
// XBRL noise removal
// ---------------------------------------------------------------------------

/// Strips SEC inline XBRL (iXBRL) noise from HTML before markdown conversion.
///
/// Modern SEC filings embed two kinds of machine-readable garbage:
///  1. `<ix:header>` — the XBRL filing manifest (CIKs, member references, dates).
///     Pure metadata; zero human-readable value.
///  2. `<div style="display:none">` — hidden divs that often contain duplicate
///     XBRL-tagged data the browser never renders.
fn strip_xbrl_noise(html: &str) -> String {
    // ix:header block — (?is): case-insensitive + dot matches newline
    let ix_header = Regex::new(r"(?is)<ix:header\b[^>]*>.*?</ix:header>").unwrap();
    // display:none divs (single-level; covers the vast majority of cases)
    let hidden_div = Regex::new(
        r#"(?is)<div\b[^>]*style\s*=\s*["'][^"']*display\s*:\s*none[^"']*["'][^>]*>.*?</div>"#,
    )
    .unwrap();

    let s = ix_header.replace_all(html, "");
    let s = hidden_div.replace_all(&s, "");
    s.into_owned()
}

// ---------------------------------------------------------------------------
// Table flattening
// ---------------------------------------------------------------------------

fn parse_table_row(line: &str) -> Vec<String> {
    line.trim()
        .trim_matches('|')
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn is_separator_row(line: &str) -> bool {
    line.trim()
        .trim_matches('|')
        .split('|')
        .all(|cell| cell.trim().chars().all(|c| c == '-' || c == ':' || c == ' '))
}

/// Converts a collected block of markdown table lines into sentences.
fn table_to_sentences(lines: &[&str]) -> String {
    let data_rows: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|l| !is_separator_row(l))
        .collect();

    if data_rows.is_empty() {
        return String::new();
    }

    let headers = parse_table_row(data_rows[0]);
    let has_headers = headers.iter().any(|h| !h.is_empty());
    let mut out = String::new();

    for row_line in &data_rows[1..] {
        let cells = parse_table_row(row_line);
        if cells.iter().all(|c| c.is_empty()) {
            continue;
        }

        let sentence: String = if has_headers {
            headers
                .iter()
                .zip(cells.iter().chain(std::iter::repeat(&String::new())))
                .filter(|(h, v)| !h.is_empty() && !v.is_empty())
                .map(|(h, v)| format!("{}: {}", h, v))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            cells
                .iter()
                .filter(|c| !c.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        };

        if !sentence.is_empty() {
            out.push_str(&sentence);
            out.push_str(".\n");
        }
    }

    out
}

/// Walks the markdown output line-by-line; when a markdown table is detected
/// (lines starting with `|`), it is replaced with flattened sentences.
fn flatten_tables(markdown: &str) -> String {
    let mut result = String::new();
    let mut table_lines: Vec<&str> = Vec::new();

    for line in markdown.lines() {
        if line.trim().starts_with('|') {
            table_lines.push(line.trim());
        } else {
            if !table_lines.is_empty() {
                result.push_str(&table_to_sentences(&table_lines));
                result.push('\n');
                table_lines.clear();
            }
            result.push_str(line);
            result.push('\n');
        }
    }

    if !table_lines.is_empty() {
        result.push_str(&table_to_sentences(&table_lines));
    }

    result
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <TICKER_SYMBOL>", args[0]);
        std::process::exit(1);
    }

    let ticker_symbol = args[1].to_uppercase();

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client).await?;

    let filings =
        fetch_8k_filings_by_ticker_symbol(&client, &company_tickers, &ticker_symbol).await?;

    let latest = filings
        .first()
        .ok_or_else(|| format!("No 8-K filings found for '{}'", ticker_symbol))?;

    let url = latest.as_primary_document_url();
    let date = latest
        .filing_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "unknown date".to_string());

    eprintln!("Fetching 8-K from {} ({})", date, url);

    let response = client
        .raw_request(reqwest::Method::GET, &url, None, None)
        .await?;

    let html = response.text().await?;
    let html = strip_xbrl_noise(&html);

    let mut options = ConversionOptions::default();

    // ATX headings (###) are clean for embedding context
    options.heading_style = HeadingStyle::Atx;

    // Strip SEC boilerplate/navigation junk; let the library convert tables
    // to markdown table format so we can flatten them ourselves below.
    options.preprocessing.enabled = true;
    options.preprocessing.preset = PreprocessingPreset::Aggressive;

    let markdown = convert(&html, Some(options))
        .map_err(|e| format!("HTML to Markdown conversion failed: {}", e))?;

    // Flatten markdown tables into "Header: value, Header: value." sentences
    let embedding_text = flatten_tables(&markdown);

    println!("{}", embedding_text);

    Ok(())
}

