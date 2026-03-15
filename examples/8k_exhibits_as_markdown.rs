/// Fetches all exhibit documents attached to the latest 8-K filing for a given
/// ticker symbol and prints each one as embedding-friendly Markdown text.
///
/// Exhibits are documents filed alongside the 8-K whose SEC type starts with
/// "EX-" (e.g. EX-99.1 press releases, EX-10.1 contracts, EX-23.1 consents).
/// HTML exhibits are converted to Markdown with tables flattened into sentences.
/// Plain-text exhibits are printed as-is. Binary formats (PDF, XLSX, images)
/// are skipped with a notice.
///
/// Usage:
///   cargo run --example 8k_exhibits_as_markdown -- <TICKER_SYMBOL>
///
/// Example:
///   cargo run --example 8k_exhibits_as_markdown -- LLY
use html_to_markdown_rs::{convert, ConversionOptions, HeadingStyle, PreprocessingPreset};
use regex::Regex;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{
    fetch_8k_filings_by_ticker_symbol, fetch_company_tickers, fetch_filing_index, SecClient,
};
use std::env;
use std::error::Error;
use tokio;

// ---------------------------------------------------------------------------
// XBRL noise removal  (same helper used by latest_8k_as_markdown)
// ---------------------------------------------------------------------------

fn strip_xbrl_noise(html: &str) -> String {
    let ix_header = Regex::new(r"(?is)<ix:header\b[^>]*>.*?</ix:header>").unwrap();
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
    line.trim().trim_matches('|').split('|').all(|cell| {
        cell.trim()
            .chars()
            .all(|c| c == '-' || c == ':' || c == ' ')
    })
}

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
// HTML → Markdown conversion
// ---------------------------------------------------------------------------

fn html_to_markdown(html: &str) -> Result<String, Box<dyn Error>> {
    let html = strip_xbrl_noise(html);

    let mut options = ConversionOptions::default();
    options.heading_style = HeadingStyle::Atx;
    options.preprocessing.enabled = true;
    options.preprocessing.preset = PreprocessingPreset::Aggressive;

    let markdown = convert(&html, Some(options))
        .map_err(|e| format!("HTML to Markdown conversion failed: {}", e))?;

    Ok(flatten_tables(&markdown))
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

    let date = latest
        .filing_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "unknown date".to_string());

    eprintln!(
        "Fetching filing index for {} 8-K ({})  items={}",
        ticker_symbol,
        date,
        latest.items.join(",")
    );

    let path = if latest.is_earnings_release() {
        "Path 1 (Earnings Release — Item 2.02)"
    } else {
        "Path 2 (Mid-Quarter Event)"
    };
    eprintln!("Routing: {}", path);

    let index = fetch_filing_index(&client, latest).await?;
    let exhibits = index.exhibits();

    if exhibits.is_empty() {
        eprintln!("No exhibits found for this filing.");
        return Ok(());
    }

    eprintln!("Found {} exhibit(s):", exhibits.len());
    for ex in &exhibits {
        eprintln!("  {} — {}", ex.document_type, ex.name);
    }
    eprintln!();

    let base_url = latest.as_edgar_archive_url();

    for exhibit in exhibits {
        let exhibit_url = format!("{}/{}", base_url, exhibit.name);

        println!("---");
        println!("## {} ({})", exhibit.document_type, exhibit.name);
        println!();

        if exhibit.is_html() {
            eprintln!("Fetching HTML exhibit: {}", exhibit_url);

            let response = client
                .raw_request(reqwest::Method::GET, &exhibit_url, None, None)
                .await?;
            let html = response.text().await?;
            let markdown = html_to_markdown(&html)?;
            println!("{}", markdown);
        } else if exhibit.is_text() {
            eprintln!("Fetching text exhibit: {}", exhibit_url);

            let response = client
                .raw_request(reqwest::Method::GET, &exhibit_url, None, None)
                .await?;
            let text = response.text().await?;
            println!("{}", text);
        } else {
            eprintln!(
                "Skipping binary exhibit: {} ({})",
                exhibit.name, exhibit_url
            );
            println!("*Binary exhibit — not rendered ({}).*", exhibit_url);
        }

        println!();
    }

    Ok(())
}
