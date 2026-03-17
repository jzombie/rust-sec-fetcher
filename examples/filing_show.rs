/// Renders any SEC filing — body, exhibits, or both — for a given ticker and form type.
///
/// This is the general-purpose filing renderer.  It accepts any EDGAR form type
/// string and renders the most recent matching filing.  Pass `--form 10-K` for
/// annual reports, `--form DEF 14A` for proxy statements, `--form 8-K` (the
/// default) for current reports, and so on.
///
/// # What gets rendered
///
/// | `--part`      | Output                                                         |
/// |---------------|----------------------------------------------------------------|
/// | `all`         | Primary document **and** all substantive exhibits (default)    |
/// | `body`        | Primary document only                                         |
/// | `exhibits`    | Substantive exhibits only (no primary document)               |
///
/// "Substantive" means exhibits that contain human-readable prose or financial
/// tables.  The following exhibit types are automatically excluded because they
/// are short legal boilerplate or machine-readable data with no analytical value:
///
/// | Excluded exhibit type | Description |
/// |---|---|
/// | `EX-31.x` | SOX § 302 CEO/CFO certifications |
/// | `EX-32.x` | SOX § 906 CEO/CFO certifications |
/// | `EX-23.x` | Auditor / accountant consent |
/// | `EX-101.*` | XBRL instance, schema, label, and calculation files |
/// | `GRAPHIC`  | Image files (logos, signature scans, charts) |
///
/// The excluded types represent the vast majority of exhibit count in a typical
/// 10-K or 10-Q while containing essentially zero prose content.
///
/// # Usage
///
///   cargo run --example filing_show -- <TICKER> [OPTIONS]
///
///   Options:
///     --form <FORM>        EDGAR form type [default: 8-K]
///     --view markdown|embedding  Rendering style [default: embedding]
///     --part all|body|exhibits   What to render [default: all]
///
/// # Examples
///
///   cargo run --example filing_show -- AAPL
///   cargo run --example filing_show -- AAPL --form 10-K
///   cargo run --example filing_show -- LLY --form 10-Q --view markdown
///   cargo run --example filing_show -- MSFT --form "DEF 14A" --part body
///   cargo run --example filing_show -- AMZN --part exhibits > amzn_exhibits.txt
use clap::{Parser, ValueEnum};
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{
    fetch_and_render, fetch_cik_by_ticker_symbol, fetch_filing_index, fetch_filings, SecClient,
};
use sec_fetcher::views::{EmbeddingTextView, FilingView, MarkdownView};
use std::error::Error;
use tokio;

#[derive(Parser)]
#[command(
    about = "Render any SEC filing (body, exhibits, or both) for a ticker and form type",
    long_about = "Fetches the most recent filing matching the given form type and renders \
                 all substantive content.  Boilerplate exhibits (SOX certs, auditor consents, \
                 XBRL schemas, graphics) are excluded from rendering by default.\n\n\
                 Any EDGAR form type string is accepted: 8-K, 10-K, 10-Q, DEF 14A, \
                 S-1, SC 13D, SC 13G, 424B4, etc."
)]
struct Args {
    /// Ticker symbol (e.g. AAPL, LLY, MSFT)
    ticker: String,

    /// EDGAR form type to fetch
    #[arg(long, default_value = "8-K")]
    form: String,

    /// Rendering style
    #[arg(long, value_enum, default_value = "embedding")]
    view: ViewArg,

    /// Which part of the filing to render
    #[arg(long, value_enum, default_value = "all")]
    part: FilingPart,
}

#[derive(Clone, ValueEnum)]
enum ViewArg {
    /// Lossless Markdown — tables preserved as pipe tables; best for citation and RAG retrieval
    Markdown,
    /// Embedding prose — tables flattened to labeled sentences; best for vector embedding
    Embedding,
}

#[derive(Clone, ValueEnum)]
enum FilingPart {
    /// Primary document and all substantive exhibits (default)
    All,
    /// Primary document only
    Body,
    /// Substantive exhibits only (no primary document)
    Exhibits,
}

async fn run<V: FilingView>(
    client: &SecClient,
    args: &Args,
    view: &V,
) -> Result<(), Box<dyn Error>> {
    let ticker = args.ticker.to_uppercase();

    let cik = fetch_cik_by_ticker_symbol(client, &ticker).await?;
    let filings = fetch_filings(client, cik, &args.form).await?;

    let latest = filings
        .first()
        .ok_or_else(|| format!("No {} filings found for '{}'", args.form, ticker))?;

    let date = latest
        .filing_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    eprintln!("{} {} — {}", ticker, args.form, date);

    println!("# {} {} — {}", ticker, args.form, date);
    println!();

    // ------------------------------------------------------------------
    // Primary document (body or all)
    // ------------------------------------------------------------------
    if matches!(args.part, FilingPart::All | FilingPart::Body) {
        let primary_url = latest.as_primary_document_url();
        eprintln!("Rendering primary document: {}", primary_url);

        println!("## Primary Document");
        println!();
        let text = fetch_and_render(client, &primary_url, view).await?;
        println!("{}", text);
    }

    // ------------------------------------------------------------------
    // Substantive exhibits (exhibits or all)
    // ------------------------------------------------------------------
    if matches!(args.part, FilingPart::All | FilingPart::Exhibits) {
        let index = fetch_filing_index(client, latest).await?;
        let exhibits = index.substantive_exhibits();

        if exhibits.is_empty() {
            eprintln!("No substantive exhibits found for this filing.");
            return Ok(());
        }

        eprintln!(
            "Found {} substantive exhibit(s) ({} total in filing):",
            exhibits.len(),
            index.documents.len()
        );
        for ex in &exhibits {
            eprintln!("  {} — {}", ex.document_type, ex.name);
        }

        let base_url = latest.as_edgar_archive_url();

        for ex in exhibits {
            let url = format!("{}/{}", base_url, ex.name);
            eprintln!("Rendering: {}", url);

            println!("---");
            println!("## Exhibit: {} ({})", ex.document_type, ex.name);
            println!();

            let text = fetch_and_render(client, &url, view).await?;
            println!("{}", text);
            println!();
        }
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
