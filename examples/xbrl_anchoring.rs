//! Fetches and displays custom→standard tag anchoring for a single ticker.
//!
//! Scans recent periodic filings (10-K, 10-Q, 20-F) for XBRL definition
//! linkbases and reports any relationships that link a filer's custom
//! extension tag to a standard taxonomy concept.
//!
//! # Usage
//!
//! ```sh
//! cargo run --example xbrl_anchoring -- --ticker AAPL
//! cargo run --example xbrl_anchoring -- --ticker MSFT --print-csv
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
    let df = fetch_custom_tag_anchoring(&client, &company_tickers, &ticker).await?;

    if df.height() == 0 {
        println!("No custom tag anchoring found for {}.", ticker);
        println!("This means either:");
        println!("  - The company has no periodic filings with XBRL definition linkbases");
        println!("  - The filings exist but contain only standard taxonomy concepts");
        println!("  - The filings use no filer-specific extension tags");
        return Ok(());
    }

    println!("Found {} anchoring relationships:", df.height());

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
