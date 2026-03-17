/// Finds the most recent earnings press release for a ticker and renders it.
///
/// SEC press releases are `EX-99.x` exhibits attached to 8-K filings.
/// Companies use them to announce quarterly and annual earnings, Regulation FD
/// voluntary disclosures, and other material events.
///
/// This tool searches the company's 8-K history for the most recent filing
/// that contains at least one `EX-99.x` exhibit, then renders every press
/// release exhibit found in that filing.
///
/// # Narrowing to earnings releases
///
/// Use `--earnings-only` to restrict the search to 8-Ks that reported
/// operating results (Item 2.02 — "Results of Operations and Financial
/// Condition", or the legacy Item 12 designation used before 2004).
/// This filters out voluntary Reg FD disclosures (Item 7.01) and other
/// announcements that happen to attach an EX-99.x, leaving only true
/// quarterly/annual earnings releases.
///
/// # Usage
///
///   cargo run --example press_release_show -- <TICKER> [OPTIONS]
///
///   Options:
///     --view markdown|embedding   Rendering style [default: embedding]
///     --earnings-only             Restrict to earnings-results 8-Ks (Item 2.02)
///     --include-body              Also render the 8-K primary document
///
/// # Examples
///
///   cargo run --example press_release_show -- LLY
///   cargo run --example press_release_show -- AAPL --view markdown
///   cargo run --example press_release_show -- MSFT --earnings-only
///   cargo run --example press_release_show -- GOOGL --earnings-only --include-body
use clap::{Parser, ValueEnum};
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{
    fetch_8k_filings, fetch_and_render, fetch_cik_by_ticker_symbol, fetch_filing_index, SecClient,
};
use sec_fetcher::views::{EmbeddingTextView, FilingView, MarkdownView};
use std::error::Error;

#[derive(Parser)]
#[command(
    about = "Render the most recent press release (EX-99.x) for a ticker",
    long_about = "Searches the company's 8-K history for the most recent filing with an \
                 EX-99.x press release exhibit and renders it.\n\n\
                 Use --earnings-only to restrict to earnings-results 8-Ks (Item 2.02).\n\
                 Use --include-body to also render the 8-K cover page alongside the press release."
)]
struct Args {
    /// Ticker symbol (e.g. LLY, AAPL, MSFT)
    ticker: String,

    /// Rendering style
    #[arg(long, value_enum, default_value = "embedding")]
    view: ViewArg,

    /// Only search 8-Ks that reported operating results (Item 2.02 earnings release)
    #[arg(long)]
    earnings_only: bool,

    /// Also render the 8-K primary document before the press release exhibits
    #[arg(long)]
    include_body: bool,
}

#[derive(Clone, ValueEnum)]
enum ViewArg {
    /// Lossless Markdown — tables preserved as pipe tables; best for citation and RAG retrieval
    Markdown,
    /// Embedding prose — tables flattened to labeled sentences; best for vector embedding
    Embedding,
}

async fn run<V: FilingView>(
    client: &SecClient,
    args: &Args,
    view: &V,
) -> Result<(), Box<dyn Error>> {
    let ticker = args.ticker.to_uppercase();

    let cik = fetch_cik_by_ticker_symbol(client, &ticker).await?;
    let all_8ks = fetch_8k_filings(client, cik).await?;

    // Filter to the subset matching the requested criteria.
    let candidates: Vec<_> = all_8ks
        .iter()
        .filter(|f| !args.earnings_only || f.is_earnings_release())
        .collect();

    if candidates.is_empty() {
        if args.earnings_only {
            eprintln!(
                "No earnings-release 8-Ks found for '{}'. \
                 Try without --earnings-only to broaden the search.",
                ticker
            );
        } else {
            eprintln!("No 8-K filings found for '{}'.", ticker);
        }
        return Ok(());
    }

    // Walk filings newest-first until we find one with a press release exhibit.
    let mut found = false;
    for filing in &candidates {
        let index = fetch_filing_index(client, filing).await?;
        let press_releases = index.press_releases();

        if press_releases.is_empty() {
            continue;
        }

        let date = filing
            .filing_date
            .map(|d| d.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let items = filing.items.join(", ");

        eprintln!(
            "Found {} press release(s) in {} 8-K — {} (items: {})",
            press_releases.len(),
            ticker,
            date,
            items
        );

        println!("# {} Press Release — {}", ticker, date);
        if !items.is_empty() {
            println!("_8-K items: {}_", items);
        }
        println!();

        // Optionally render the primary 8-K body first.
        if args.include_body {
            let primary_url = filing.as_primary_document_url();
            eprintln!("Rendering primary document: {}", primary_url);

            println!("## 8-K Body");
            println!();
            let text = fetch_and_render(client, &primary_url, view).await?;
            println!("{}", text);
            println!();
        }

        // Render every press release exhibit.
        let base_url = filing.as_edgar_archive_url();
        for pr in &press_releases {
            let url = format!("{}/{}", base_url, pr.name);
            eprintln!("Rendering press release: {} ({})", pr.document_type, url);

            println!("---");
            println!("## {} — {}", pr.document_type, pr.name);
            println!();
            let text = fetch_and_render(client, &url, view).await?;
            println!("{}", text);
            println!();
        }

        found = true;
        break;
    }

    if !found {
        let qualifier = if args.earnings_only {
            "earnings-release"
        } else {
            ""
        };
        eprintln!(
            "No {} 8-K with EX-99.x press release exhibits found for '{}'.",
            qualifier, ticker
        );
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
