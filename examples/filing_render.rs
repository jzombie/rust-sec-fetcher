//! Generic SEC filing document renderer.
//!
//! Fetches any SEC EDGAR URL and renders it using the chosen view.  Useful
//! for inspecting how a specific filing artifact looks before building a
//! larger pipeline, and for comparing the output of different views
//! side-by-side.
//!
//! # Usage
//!
//! ```text
//! cargo run --example filing_render -- <URL> [--view markdown|embedding]
//! ```
//!
//! # Examples
//!
//! Render a 10-K as lossless Markdown (tables kept as tables):
//!
//! ```text
//! cargo run --example filing_render -- \
//!   "https://www.sec.gov/Archives/edgar/data/320193/000032019325000008/aapl-20241228.htm" \
//!   --view markdown
//! ```
//!
//! Render an 8-K press release row-by-row for dense vector embeddings:
//!
//! ```text
//! cargo run --example filing_render -- \
//!   "https://www.sec.gov/Archives/edgar/data/320193/000114036126006577/aapl-20260224.htm" \
//!   --view embedding
//! ```
use clap::{Parser, ValueEnum};
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_and_render, SecClient};
use sec_fetcher::views::{EmbeddingTextView, MarkdownView};
use std::error::Error;

#[derive(Parser)]
#[command(
    about = "Fetch any SEC EDGAR document URL and render it as clean text",
    long_about = None
)]
struct Args {
    /// URL of the document to render (HTML, plain-text, or binary)
    url: String,

    /// Rendering view to apply
    #[arg(long, value_enum, default_value = "markdown")]
    view: ViewArg,
}

/// Selects the rendering strategy applied to the fetched document.
#[derive(Clone, ValueEnum)]
enum ViewArg {
    /// Lossless Markdown — HTML tables are preserved as pipe tables.
    /// Best for RAG retrieval, citation-accurate QA, and structured extraction.
    Markdown,
    /// Embedding-optimized prose — HTML tables are flattened to labeled
    /// sentences (`Metric — Period: value, …`).  Row labels are always
    /// preserved.  Best for dense vector embeddings and semantic search.
    Embedding,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    eprintln!(
        "Fetching {} (view: {:?})",
        args.url,
        args.view.to_possible_value().unwrap().get_name()
    );

    let text = match args.view {
        ViewArg::Markdown => fetch_and_render(&client, &args.url, &MarkdownView).await?,
        ViewArg::Embedding => fetch_and_render(&client, &args.url, &EmbeddingTextView).await?,
    };

    println!("{}", text);

    Ok(())
}
