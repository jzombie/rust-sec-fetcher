//! Fetches all exhibit documents attached to the latest 8-K filing for a given
//! ticker symbol and prints each one as rendered text.
//!
//! HTML exhibits are converted using the chosen view.  Plain-text exhibits are
//! printed as-is (with blank-line collapsing for `embedding` view).  Binary
//! formats (PDF, XLSX, images) are represented by a notice string.
//!
//! # Usage
//!
//! ```text
//! cargo run --example 8k_exhibits_as_markdown -- <TICKER_SYMBOL> [--view markdown|embedding]
//! cargo run --example 8k_exhibits_as_markdown -- LLY
//! cargo run --example 8k_exhibits_as_markdown -- AAPL --view markdown
//! ```
use clap::{Parser, ValueEnum};
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{SecClient, fetch_8k_filings, fetch_cik_by_ticker_symbol};
use sec_fetcher::ops::render_all_exhibits;
use sec_fetcher::views::{EmbeddingTextView, FilingView, MarkdownView};
use std::error::Error;

#[derive(Parser)]
#[command(
    about = "Render EX-* exhibit attachments only (not the primary document) for a ticker",
    long_about = "Fetches the latest 8-K and renders every EX-* exhibit — press releases, \n\
                  earnings tables, executive statements — skipping the primary cover page. \n\
                  Use `8k_full_render` to render the primary document and all exhibits \n\
                  together in one output, or `latest_8k_as_markdown` for the primary \n\
                  document alone."
)]
struct Args {
    /// Ticker symbol (e.g. LLY)
    ticker: String,

    /// Rendering view
    #[arg(long, value_enum, default_value = "embedding")]
    view: ViewArg,
}

#[derive(Clone, ValueEnum)]
enum ViewArg {
    Markdown,
    Embedding,
}

async fn run<V: FilingView>(
    client: &SecClient,
    args: &Args,
    view: &V,
) -> Result<(), Box<dyn Error>> {
    let ticker_symbol = TickerSymbol::new(&args.ticker);

    let cik = fetch_cik_by_ticker_symbol(client, &ticker_symbol).await?;
    let filings = fetch_8k_filings(client, cik).await?;

    let latest = filings
        .first()
        .ok_or_else(|| format!("No 8-K filings found for '{}'", ticker_symbol))?;

    let date = latest
        .filing_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    eprintln!(
        "Fetching exhibit index for {} 8-K ({})  items={}",
        ticker_symbol,
        date,
        latest.items.join(",")
    );

    let exhibits = render_all_exhibits(client, latest, view).await?;

    if exhibits.is_empty() {
        eprintln!("No exhibits found for this filing.");
        return Ok(());
    }

    eprintln!("Found {} exhibit(s):", exhibits.len());
    for ex in &exhibits {
        eprintln!("  {} — {}", ex.document_type, ex.name);
    }
    eprintln!();

    for ex in exhibits {
        eprintln!("Rendering: {}", ex.url);
        println!("---");
        println!("## {} ({})", ex.document_type, ex.name);
        println!();
        println!("{}", ex.content);
        println!();
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    match args.view {
        ViewArg::Markdown => run(&client, &args, &MarkdownView).await?,
        ViewArg::Embedding => run(&client, &args, &EmbeddingTextView).await?,
    }

    Ok(())
}
