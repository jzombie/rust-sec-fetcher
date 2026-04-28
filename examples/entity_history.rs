//! Shows the complete 10-K filing history for one or more tickers, including
//! filings from predecessor CIKs that resulted from holding-company reorganizations.
//!
//! When a company restructures as a holding company (e.g. Google Inc. →
//! Alphabet Inc. in 2015), the SEC assigns a new CIK to the successor entity.
//! A plain ticker-to-CIK lookup returns only post-reorganization filings.
//! This example uses [`fetch_all_entity_submissions`] which automatically
//! discovers and merges predecessor CIKs from EDGAR's own co-registrant
//! records — no configuration required.
//!
//! # Usage
//!
//! ```text
//! cargo run --example entity_history -- GOOG
//! cargo run --example entity_history -- GOOG META
//! cargo run --example entity_history -- GOOG META AAPL
//! ```

use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{
    SecClient, fetch_all_entity_submissions, fetch_cik_by_ticker_symbol, fetch_related_ciks,
};
use std::error::Error;

#[derive(Parser)]
#[command(
    about = "Show the complete 10-K filing history including predecessor-CIK filings",
    long_about = None
)]
struct Args {
    /// One or more ticker symbols (e.g. GOOG META AAPL)
    #[arg(required = true)]
    tickers: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    for ticker_str in &args.tickers {
        let ticker = TickerSymbol::new(ticker_str);
        let cik = fetch_cik_by_ticker_symbol(&client, &ticker).await?;

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  {ticker}  →  primary CIK {}", cik.value);

        // Show what predecessor CIKs were found (if any).
        let related = fetch_related_ciks(&client, &cik).await?;
        if related.is_empty() {
            println!("  Predecessor CIKs: none (no holding-company reorganization on record)");
        } else {
            let ids: Vec<String> = related.iter().map(|c| c.value.to_string()).collect();
            println!(
                "  Predecessor CIKs found via EFTS co-registrant records: {}",
                ids.join(", ")
            );
        }
        println!();

        // Fetch the full merged submission history across all CIKs.
        let all_submissions = fetch_all_entity_submissions(&client, cik.clone()).await?;

        // Show only 10-K filings, newest first.
        let tenk: Vec<_> = all_submissions
            .iter()
            .filter(|s| s.form == "10-K")
            .collect();

        if tenk.is_empty() {
            println!("  No 10-K filings found.");
        } else {
            println!("  {:<12} {:<14} {}", "Filed", "CIK", "Accession");
            println!("  {}", "─".repeat(60));
            for s in &tenk {
                let date = s
                    .filing_date
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                println!("  {:<12} {:<14} {}", date, s.cik.value, s.accession_number);
            }
            println!();
            println!(
                "  Total 10-K filings: {}  (spanning CIKs: {})",
                tenk.len(),
                {
                    let mut seen_ciks: Vec<u64> = tenk.iter().map(|s| s.cik.value).collect();
                    seen_ciks.dedup();
                    seen_ciks.sort_unstable();
                    seen_ciks.dedup();
                    seen_ciks
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            );
        }
        println!();
    }

    Ok(())
}
