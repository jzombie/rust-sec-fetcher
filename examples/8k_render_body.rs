/// Fetches the latest 8-K filing for a given ticker and renders it as text.
///
/// Usage:
///   cargo run --example latest_8k_as_markdown -- <TICKER> [--view markdown|embedding]
///
/// Example:
///   cargo run --example latest_8k_as_markdown -- LLY
///   cargo run --example latest_8k_as_markdown -- LLY --view markdown
use clap::{Parser, ValueEnum};
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_8k_filings, fetch_and_render, fetch_cik_by_ticker_symbol, SecClient};
use sec_fetcher::views::{EmbeddingTextView, MarkdownView};
use std::error::Error;
use tokio;

#[derive(Parser)]
#[command(
    about = "Render the primary 8-K document only (not exhibits) for a ticker",
    long_about = "Fetches the latest 8-K and renders the primary document — typically a brief \n\
                 HTML cover page that lists which items are being filed.  The substantive \n\
                 content (earnings releases, financial tables) usually lives in the EX-99.x \n\
                 exhibits; use `8k_full_render` to capture both the primary document \n\
                 and all exhibits together."
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
    /// Lossless Markdown with tables as pipe tables
    Markdown,
    /// Tables flattened to labeled prose sentences
    Embedding,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let ticker_symbol = args.ticker.to_uppercase();

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let cik = fetch_cik_by_ticker_symbol(&client, &ticker_symbol).await?;
    let filings = fetch_8k_filings(&client, cik).await?;

    let latest = filings
        .first()
        .ok_or_else(|| format!("No 8-K filings found for '{}'", ticker_symbol))?;

    let url = latest.as_primary_document_url();
    let date = latest
        .filing_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    eprintln!("Fetching 8-K from {} ({})", date, url);

    let text = match args.view {
        ViewArg::Markdown => fetch_and_render(&client, &url, &MarkdownView).await?,
        ViewArg::Embedding => fetch_and_render(&client, &url, &EmbeddingTextView).await?,
    };

    println!("{}", text);

    Ok(())
}
