/// Renders the complete latest 8-K filing for a ticker: the primary document
/// followed by every attached exhibit, in a single combined output.
///
/// # When to use this vs. the other 8-K examples
///
/// | Example                  | What it fetches                                      |
/// |--------------------------|------------------------------------------------------|
/// | `8k_full_render`         | Primary document **and** all EX-* exhibits together  |
/// | `latest_8k_as_markdown`  | Primary document only (the 8-K cover page / body)    |
/// | `8k_exhibits_as_markdown`| EX-* exhibit attachments only (press releases, etc.) |
///
/// For most analytical purposes `8k_full_render` is the right default because
/// 8-K filings often have minimal text in the primary document (it just lists
/// the items filed) with the substantive content — earnings releases, financial
/// tables, executive statements — living in the EX-99.1 or EX-99.2 exhibits.
///
/// # Usage
///
///   cargo run --example 8k_full_render -- <TICKER> [--view markdown|embedding]
///
/// # Examples
///
///   cargo run --example 8k_full_render -- AAPL
///   cargo run --example 8k_full_render -- LLY --view markdown
///   cargo run --example 8k_full_render -- MSFT --view embedding > msft_8k.txt
use clap::{Parser, ValueEnum};
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{
    fetch_8k_filings, fetch_and_render, fetch_cik_by_ticker_symbol, fetch_filing_index, SecClient,
};
use sec_fetcher::rendering::{EmbeddingTextView, FilingView, MarkdownView};
use std::error::Error;
use tokio;

#[derive(Parser)]
#[command(
    about = "Render the complete 8-K filing (primary document + all exhibits) for a ticker",
    long_about = "Fetches the latest 8-K for a ticker and renders both the primary filing \
                 document and every EX-* attachment into one combined output stream.\n\n\
                 The primary 8-K document is usually a brief cover page that identifies \
                 which items are being filed; the substantive content (earnings releases, \
                 financial tables, executive commentary) lives in the EX-99.x exhibits. \
                 This example renders both together so nothing is missed."
)]
struct Args {
    /// Ticker symbol (e.g. AAPL)
    ticker: String,

    /// Rendering view
    #[arg(long, value_enum, default_value = "embedding")]
    view: ViewArg,
}

#[derive(Clone, ValueEnum)]
enum ViewArg {
    /// Lossless Markdown — tables preserved as pipe tables; best for citation / RAG retrieval
    Markdown,
    /// Embedding prose — tables flattened to labeled sentences; best for vector embedding
    Embedding,
}

async fn run<V: FilingView>(client: &SecClient, ticker: &str, view: &V) -> Result<(), Box<dyn Error>> {
    let cik = fetch_cik_by_ticker_symbol(client, ticker).await?;
    let filings = fetch_8k_filings(client, cik).await?;

    let latest = filings
        .first()
        .ok_or_else(|| format!("No 8-K filings found for '{}'", ticker))?;

    let date = latest
        .filing_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let items = latest.items.join(", ");
    eprintln!("Latest 8-K — {} — items: {}", date, items);

    println!("# {} 8-K — {}", ticker, date);
    println!();

    // ------------------------------------------------------------------
    // Primary document
    // ------------------------------------------------------------------
    let primary_url = latest.as_primary_document_url();
    eprintln!("Rendering primary document: {}", primary_url);

    println!("## Primary Document");
    println!();
    let primary_text = fetch_and_render(client, &primary_url, view).await?;
    println!("{}", primary_text);

    // ------------------------------------------------------------------
    // Exhibits
    // ------------------------------------------------------------------
    let index = fetch_filing_index(client, latest).await?;
    let exhibits = index.exhibits();

    if exhibits.is_empty() {
        eprintln!("No exhibits found for this filing.");
        return Ok(());
    }

    eprintln!("Found {} exhibit(s):", exhibits.len());
    for ex in &exhibits {
        eprintln!("  {} — {}", ex.document_type, ex.name);
    }

    let base_url = latest.as_edgar_archive_url();

    for exhibit in exhibits {
        let url = format!("{}/{}", base_url, exhibit.name);
        eprintln!("Rendering: {}", url);

        println!("---");
        println!("## Exhibit: {} ({})", exhibit.document_type, exhibit.name);
        println!();

        let text = fetch_and_render(client, &url, view).await?;
        println!("{}", text);
        println!();
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let ticker = args.ticker.to_uppercase();

    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    match args.view {
        ViewArg::Markdown => run(&client, &ticker, &MarkdownView).await?,
        ViewArg::Embedding => run(&client, &ticker, &EmbeddingTextView).await?,
    }

    Ok(())
}
