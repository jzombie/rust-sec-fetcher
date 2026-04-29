//! Extracts the full text of any section(s) from a 10-K filing for a given ticker.
//!
//! By default the most recent 10-K is used.  Pass `--year YYYY` to select a
//! specific fiscal year, or `--list` to print every available 10-K filing date.
//!
//! # Usage
//!
//! ```text
//! cargo run --example tenk_sections -- <TICKER> [OPTIONS]
//!
//! cargo run --example tenk_sections -- AAPL
//! cargo run --example tenk_sections -- AAPL --list
//! cargo run --example tenk_sections -- AAPL --year 2022
//! cargo run --example tenk_sections -- PG --section item_7
//! cargo run --example tenk_sections -- KO --section item_1 --year 2020
//! cargo run --example tenk_sections -- WMT --section item_1a
//! cargo run --example tenk_sections -- COST --section all > cost_10k.txt
//! ```

use chrono::Datelike;
use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{
    SecClient, fetch_10k_filings, fetch_10k_sections, fetch_10k_sections_for_filing,
    fetch_cik_by_ticker_symbol,
};
use std::error::Error;

#[derive(Parser)]
#[command(
    about = "Extract sections from a 10-K filing",
    long_about = "Fetches the primary 10-K document for the given ticker and extracts the full \
                 untruncated text of any numbered section.\n\n\
                 --section accepts any normalized key: item_1, item_1a, item_2, item_3, \
                 item_7, item_7a, item_8, etc.  Short forms without the underscore also work: \
                 item1, item7a.  Use 'all' to print every section found (default).\n\n\
                 Progress messages go to stderr; section text goes to stdout."
)]
struct Args {
    /// Ticker symbol (e.g. AAPL, PG, KO, WMT, COST)
    ticker: String,

    /// Which section(s) to output.  Use a key like "item_7", "item_1a", "item_7a", or "all".
    /// Short forms without underscore also work: "item7", "item1a".  Default: all.
    #[arg(long, default_value = "all")]
    section: String,

    /// Select the 10-K whose filing date falls in this calendar year (e.g. --year 2022).
    /// If omitted, the most recent 10-K is used.
    #[arg(long)]
    year: Option<i32>,

    /// List all available 10-K filings for this ticker (date, accession number) and exit.
    #[arg(long)]
    list: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let cfg = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&cfg)?;
    let ticker = TickerSymbol::new(&args.ticker);

    eprintln!("Resolving CIK for {}…", ticker);
    let cik = fetch_cik_by_ticker_symbol(&client, &ticker).await?;

    // --list: print all available 10-K filings and exit.
    if args.list {
        eprintln!("Fetching 10-K filing list for CIK {}…", cik);
        let filings = fetch_10k_filings(&client, cik).await?;
        if filings.is_empty() {
            eprintln!("No 10-K filings found.");
            return Ok(());
        }
        println!("{:<12}  {:<20}  Form", "Date", "Accession Number");
        for f in &filings {
            let date = f
                .filing_date
                .map(|d| d.to_string())
                .unwrap_or_else(|| "unknown".into());
            println!("{:<12}  {:<20}  {}", date, f.accession_number, f.form);
        }
        return Ok(());
    }

    let sections = if let Some(year) = args.year {
        // --year: pick the first filing whose date falls in that calendar year.
        eprintln!("Fetching 10-K list for CIK {} to find year {}…", cik, year);
        let filings = fetch_10k_filings(&client, cik).await?;
        let filing = filings
            .iter()
            .find(|f| f.filing_date.map(|d| d.year() == year).unwrap_or(false))
            .ok_or_else(|| format!("No 10-K filing found for {} in year {}", ticker, year))?;
        let date = filing
            .filing_date
            .map(|d| d.to_string())
            .unwrap_or_default();
        eprintln!("Using {} filing from {}…", filing.form, date);
        fetch_10k_sections_for_filing(&client, filing).await?
    } else {
        eprintln!("Fetching most recent 10-K for CIK {}…", cik);
        fetch_10k_sections(&client, cik).await?
    };

    let section_key = normalize_section_key(&args.section);

    if section_key == "all" {
        let mut keys: Vec<&str> = sections.keys().collect();
        keys.sort_by_key(|a| section_sort_key(a));
        if keys.is_empty() {
            eprintln!("Warning: no sections found in this filing.");
        }
        for key in keys {
            if let Some(text) = sections.get(key) {
                eprintln!("{}: {} chars", key, text.len());
                println!("=== {} ===", key.to_ascii_uppercase());
                println!();
                println!("{text}");
                println!();
            }
        }
    } else {
        match sections.get(&section_key) {
            Some(text) => {
                eprintln!("{}: {} chars", section_key, text.len());
                println!("=== {} ===", section_key.to_ascii_uppercase());
                println!();
                println!("{text}");
                println!();
            }
            None => {
                eprintln!("Warning: {} not found in this filing.", section_key);
                eprintln!("Available sections: {}", {
                    let mut keys: Vec<&str> = sections.keys().collect();
                    keys.sort_by_key(|a| section_sort_key(a));
                    keys.join(", ")
                });
            }
        }
    }

    Ok(())
}

/// Normalize a CLI section argument to a canonical map key.
///
/// Accepts `item7`, `item_7`, `item7a`, `item_7a`, `all`, `both`, etc.
fn normalize_section_key(s: &str) -> String {
    let lower = s.to_ascii_lowercase();
    if lower == "all" || lower == "both" {
        return "all".to_string();
    }
    let rest = lower
        .strip_prefix("item_")
        .or_else(|| lower.strip_prefix("item"))
        .unwrap_or(&lower);
    format!("item_{rest}")
}

/// Sort key for section map keys (`"item_7a"` → `(7, "a")`).
fn section_sort_key(key: &str) -> (u32, String) {
    let rest = key.strip_prefix("item_").unwrap_or(key);
    let num_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let num: u32 = rest[..num_end].parse().unwrap_or(0);
    let suffix = rest[num_end..].to_string();
    (num, suffix)
}
