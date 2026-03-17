/// Fetches all exhibit documents attached to the latest 8-K filing for a given
/// ticker symbol and prints each one as rendered text.
///
/// HTML exhibits are converted using the chosen view.  Plain-text exhibits are
/// printed as-is (with blank-line collapsing for `embedding` view).  Binary
/// formats (PDF, XLSX, images) are represented by a notice string.
///
/// Usage:
///   cargo run --example 8k_exhibits_as_markdown -- <TICKER_SYMBOL> [--view markdown|embedding]
///
/// Example:
///   cargo run --example 8k_exhibits_as_markdown -- LLY
use clap::{Parser, ValueEnum};
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{
    fetch_8k_filings, fetch_and_render, fetch_cik_by_ticker_symbol, fetch_filing_index, SecClient,
};
use sec_fetcher::views::{EmbeddingTextView, MarkdownView};
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let ticker_symbol = TickerSymbol::new(&args.ticker);

    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let cik = fetch_cik_by_ticker_symbol(&client, &ticker_symbol).await?;
    let filings = fetch_8k_filings(&client, cik).await?;

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
        let url = format!("{}/{}", base_url, exhibit.name);

        println!("---");
        println!("## {} ({})", exhibit.document_type, exhibit.name);
        println!();

        eprintln!("Rendering: {}", url);

        let text = match args.view {
            ViewArg::Markdown => fetch_and_render(&client, &url, &MarkdownView).await?,
            ViewArg::Embedding => fetch_and_render(&client, &url, &EmbeddingTextView).await?,
        };

        println!("{}", text);
        println!();
    }

    Ok(())
}
