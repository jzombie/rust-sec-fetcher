//! Extracts the **full text** of Item 1 (Business) and/or Item 7 (MD&A) from
//! a 10-K filing for any given ticker.
//!
//! By default the most recent 10-K is used.  Pass `--year YYYY` to select a
//! specific fiscal year, or `--list` to print every available 10-K with its
//! filing date so you can pick the one you want.
//!
//! These two sections are the raw textual corpus needed to connect a company's
//! products and supply chain narrative back to consumer-goods signal data:
//!
//! | Section | What companies write here |
//! |---------|--------------------------|
//! | **Item 1 — Business** | Every product category, brand, distribution channel, and geographic market the company operates in.  Legally mandated to detail *exactly* what they make and sell. |
//! | **Item 7 — MD&A** | Management's year-in-review: revenue drivers, supply chain pressures, input cost trends, margin commentary, and consumer demand outlook. |
//!
//! Output is **untruncated** plain text — typically 5,000–50,000+ words per
//! section — suitable for vector embedding, RAG retrieval, LLM context, or any
//! NLP workload that needs the full corpus.
//!
//! # Usage
//!
//! ```text
//! cargo run --example tenk_sections -- <TICKER> [OPTIONS]
//!
//! cargo run --example tenk_sections -- AAPL
//! cargo run --example tenk_sections -- AAPL --list
//! cargo run --example tenk_sections -- AAPL --year 2022
//! cargo run --example tenk_sections -- PG --section item7
//! cargo run --example tenk_sections -- KO --section item1 --year 2020
//! cargo run --example tenk_sections -- WMT --section both > wmt_10k_corpus.txt
//! cargo run --example tenk_sections -- COST --section both 2>/dev/null | wc -c
//! ```

use clap::{Parser, ValueEnum};
use chrono::Datelike;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{
    SecClient, fetch_10k_filings, fetch_10k_sections, fetch_10k_sections_for_filing,
    fetch_cik_by_ticker_symbol,
};
use std::error::Error;

#[derive(Parser)]
#[command(
    about = "Extract full Item 1 (Business) and/or Item 7 (MD&A) from a 10-K",
    long_about = "Fetches the primary 10-K document for the given ticker and extracts the full \
                 untruncated text of Item 1 and/or Item 7 as a plain-text corpus.\n\n\
                 Item 1 — Business: legally-detailed description of every product category, \
                 brand, and market the company operates in.\n\n\
                 Item 7 — MD&A: supply chain costs, input price trends, consumer demand \
                 commentary, margin analysis, and management's business outlook.\n\n\
                 Progress messages are written to stderr; section text goes to stdout, \
                 so you can redirect stdout to a file while still seeing progress."
)]
struct Args {
    /// Ticker symbol (e.g. AAPL, PG, KO, WMT, COST)
    ticker: String,

    /// Which section(s) to output
    #[arg(long, value_enum, default_value = "both")]
    section: SectionArg,

    /// Select the 10-K whose filing date falls in this calendar year (e.g. --year 2022).
    /// If omitted, the most recent 10-K is used.
    #[arg(long)]
    year: Option<i32>,

    /// List all available 10-K filings for this ticker (date, accession number) and exit.
    #[arg(long)]
    list: bool,
}

#[derive(Clone, ValueEnum)]
enum SectionArg {
    /// Item 1 — Business description only
    Item1,
    /// Item 7 — MD&A only
    Item7,
    /// Both Item 1 and Item 7 (default)
    Both,
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
        println!("{:<12}  {:<20}  {}", "Date", "Accession Number", "Form");
        for f in &filings {
            let date = f.filing_date.map(|d| d.to_string()).unwrap_or_else(|| "unknown".into());
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
        let date = filing.filing_date.map(|d| d.to_string()).unwrap_or_default();
        eprintln!("Using {} filing from {}…", filing.form, date);
        fetch_10k_sections_for_filing(&client, filing).await?
    } else {
        eprintln!("Fetching most recent 10-K for CIK {}…", cik);
        fetch_10k_sections(&client, cik).await?
    };

    let want_item1 = matches!(args.section, SectionArg::Item1 | SectionArg::Both);
    let want_item7 = matches!(args.section, SectionArg::Item7 | SectionArg::Both);

    if want_item1 {
        match sections.item1 {
            Some(ref text) => {
                eprintln!("Item 1: {} chars", text.len());
                println!("=== ITEM 1. BUSINESS ===");
                println!();
                println!("{text}");
                println!();
            }
            None => eprintln!("Warning: Item 1 not found in 10-K primary document."),
        }
    }

    if want_item7 {
        match sections.item7 {
            Some(ref text) => {
                eprintln!("Item 7: {} chars", text.len());
                println!("=== ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS ===");
                println!();
                println!("{text}");
                println!();
            }
            None => eprintln!("Warning: Item 7 not found in 10-K primary document."),
        }
    }

    Ok(())
}

