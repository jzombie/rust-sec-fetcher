//! Fetches and displays custom→standard tag polarity anchoring for a single ticker.
//!
//! Scope: custom extension tags ONLY (e.g. `bac:FeesAndCommissions1`).
//! Standard `us-gaap:*` tags have fixed FASB polarity and need no lookup.
//! Use BAC (156 customs) to demo; AAPL returns empty (0 customs).
//!
//! Scans periodic filings (10-K, 10-Q, 20-F) for XBRL calculation
//! linkbases (`EX-101.CAL`), extension schemas (`EX-101.SCH`) and labels
//! (`EX-101.LAB`). Returns one stable row per custom tag: parent, weight,
//! declared vs effective balance, conflict flag.
//!
//! Downstream (e.g. trend-describer) must use `effective_balance` + `weight`
//! and skip or pin rows where `conflict=true` — never flip signs per filing.
//!
//! # Usage
//!
//! ```sh
//! cargo run --example xbrl_anchoring -- --ticker BAC
//! cargo run --example xbrl_anchoring -- --ticker BAC --print-csv
//! ```

use clap::Parser;
use polars::prelude::*;
use sec_fetcher::{
    config::ConfigManager,
    models::TickerSymbol,
    network::{SecClient, fetch_company_tickers, fetch_custom_tag_anchoring},
};

#[derive(Parser)]
#[command(about = "Fetch and display XBRL custom→standard tag anchoring for a ticker")]
struct Args {
    /// Ticker symbol (e.g. AAPL, MSFT, BAC)
    #[arg(long, short = 't')]
    ticker: String,

    /// Print as CSV instead of formatted table
    #[arg(long, default_value_t = false)]
    print_csv: bool,

    /// Only scan the N most-recent periodic filings. Omit for full history.
    #[arg(long)]
    max_filings: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy(),
        )
        .init();

    let args = Args::parse();
    let ticker = TickerSymbol::new(&args.ticker);

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client, false).await?;

    println!("Fetching custom tag anchoring for {}...", ticker);
    let df =
        fetch_custom_tag_anchoring(&client, &company_tickers, &ticker, args.max_filings).await?;

    if df.height() == 0 || df.column("ext_concept").is_err() {
        println!("No custom tag anchoring found for {}.", ticker);
        println!("This means either:");
        println!("  - The company has no periodic filings with XBRL calculation linkbases");
        println!("  - The filings exist but contain only standard taxonomy concepts");
        println!("  - The filings use no filer-specific extension tags");
        return Ok(());
    }

    println!(
        "Found {} stable custom-tag polarities (one row per tag):",
        df.height()
    );
    if let (Ok(conflict), Ok(consistent)) = (df.column("conflict"), df.column("consistent")) {
        let count_true = |c: &polars::prelude::Column| -> usize {
            c.bool()
                .map(|b| b.into_iter().filter(|v| *v == Some(true)).count())
                .unwrap_or(0)
        };
        println!(
            "  consistent={}/{}  conflict={} (pin or drop these downstream)",
            count_true(consistent),
            df.height(),
            count_true(conflict)
        );
    }

    if args.print_csv {
        let mut buf = Vec::new();
        CsvWriter::new(&mut buf)
            .include_header(true)
            .finish(&mut df.clone())?;
        println!("{}", String::from_utf8_lossy(&buf));
    } else {
        println!("{}", df);
    }

    Ok(())
}
