//! Shows IPO registration statement filings for a given company.
//!
//! Fetches all S-1 and S-1/A (or F-1 / F-1/A for foreign issuers) filings
//! for the company and renders the selected filing's content.
//!
//! # How to identify a company
//!
//! Pre-IPO companies **do not yet have a ticker symbol**. Use `ipo_list` to
//! discover the CIK (Central Index Key) assigned by the SEC, then pass it
//! with `--cik`. Once a company has completed its IPO and been assigned a
//! ticker you can use `--ticker` instead.
//!
//! ```text
//! # Discovered by ipo_list:  "1713445  Reddit, Inc.  …"
//! cargo run --example ipo_show -- --cik 1713445        # pre-IPO or post-IPO
//! cargo run --example ipo_show -- --ticker RDDT        # only after IPO
//! ```
//!
//! # What this shows
//!
//! This example is IPO-aware: it first prints the **full IPO filing timeline**
//! (initial S-1, each amendment, and the final pricing prospectus if present),
//! giving a clear picture of how far through the process a company is before
//! optionally rendering the document text.
//!
//! | Form    | Stage in the IPO process                                      |
//! |---------|---------------------------------------------------------------|
//! | S-1     | Initial registration filed with the SEC                       |
//! | S-1/A   | Amendment responding to SEC comments or updated financials    |
//! | 424B4   | Pricing prospectus — final offer price, size, underwriter     |
//! | F-1     | Foreign private issuer equivalent of S-1                      |
//! | F-1/A   | Amendment to F-1                                              |
//!
//! # Usage
//!
//! ```text
//! cargo run --example ipo_show -- --cik <CIK_NUMBER> [OPTIONS]
//! cargo run --example ipo_show -- --ticker <TICKER>   [OPTIONS]
//! cargo run --example ipo_show -- --cik 1713445 --part summary
//! cargo run --example ipo_show -- --ticker RDDT --index -1 --part body
//! cargo run --example ipo_show -- --ticker RDDT --part all
//! ```
use clap::{Parser, ValueEnum};
use sec_fetcher::config::ConfigManager;
use sec_fetcher::enums::Url;
use sec_fetcher::models::{Cik, TickerSymbol};
use sec_fetcher::network::{SecClient, fetch_cik_by_ticker_symbol, fetch_company_profile};
use sec_fetcher::ops::{get_ipo_registration_filings, render_filing};
use sec_fetcher::views::{EmbeddingTextView, FilingView, MarkdownView};
use std::error::Error;

#[derive(Parser)]
#[command(
    about = "Show IPO registration statement details for a company",
    long_about = "Fetches all S-1 / S-1/A (or F-1 / F-1/A) filings for a company and prints \
                  the full IPO filing timeline before optionally rendering the prospectus text. \
                  Pre-IPO companies do not have a ticker — use --cik (from ipo_list output). \
                  --ticker works only for companies that have already completed their IPO."
)]
#[command(group(
    clap::ArgGroup::new("identity")
        .required(true)
        .args(["cik", "ticker"])
))]
struct Args {
    /// SEC CIK number — works for pre-IPO and post-IPO companies.
    /// Find it in the output of `ipo_list` (the "CIK" column).
    /// Accepts both plain (2039972) and zero-padded (0002039972) forms.
    #[arg(long, group = "identity")]
    cik: Option<String>,

    /// Ticker symbol — only valid after the IPO has completed and a ticker
    /// has been assigned. Use --cik for companies still in registration.
    #[arg(long, group = "identity")]
    ticker: Option<String>,

    /// Rendering style applied to the document text
    #[arg(long, value_enum, default_value = "markdown")]
    view: ViewArg,

    /// Which part of the filing to show
    #[arg(long, value_enum, default_value = "summary")]
    part: FilingPart,

    /// Index of the filing to render (0 = newest, -1 = oldest).
    /// Negative values count from the end: -1 is the initial S-1, -2 is the
    /// first amendment, etc.
    #[arg(long, default_value_t = 0, allow_negative_numbers = true)]
    index: i64,
}

#[derive(Clone, ValueEnum)]
enum ViewArg {
    /// Lossless Markdown — tables preserved as pipe tables
    Markdown,
    /// Embedding-optimised prose — tables flattened to labeled sentences
    Embedding,
}

#[derive(Clone, ValueEnum)]
enum FilingPart {
    /// Print the IPO filing timeline only (no document text)
    Summary,
    /// Print the timeline and render the primary document
    Body,
    /// Print the timeline and render the primary document and all exhibits
    All,
}

async fn run<V: FilingView>(
    client: &SecClient,
    args: &Args,
    view: &V,
) -> Result<(), Box<dyn Error>> {
    // Resolve identity: CIK takes precedence; ticker requires a live lookup.
    let cik: Cik = if let Some(ref raw_cik) = args.cik {
        Cik::from_str(raw_cik).map_err(|e| format!("Invalid CIK '{}': {}", raw_cik, e))?
    } else {
        let symbol = TickerSymbol::new(args.ticker.as_deref().unwrap());
        fetch_cik_by_ticker_symbol(client, &symbol).await?
    };

    // Fetch company name — reuses the same cached HTTP response as fetch_filings.
    let company_name = fetch_company_profile(client, cik.clone())
        .await
        .map(|p| p.name)
        .unwrap_or_else(|_| format!("CIK {}", cik));

    // Human-readable label for output headers.
    let company_label = args
        .ticker
        .as_deref()
        .map(|t| format!("{} ({})", company_name, t.to_uppercase()))
        .unwrap_or(company_name.clone());

    // Fetch all four IPO registration form types — S-1 and F-1 families.
    let all_filings = get_ipo_registration_filings(client, cik.clone()).await?;

    if all_filings.is_empty() {
        eprintln!(
            "No S-1, S-1/A, F-1, or F-1/A filings found for {}.",
            company_label
        );
        return Ok(());
    }

    // ── IPO filing timeline ─────────────────────────────────────────────────
    println!("# {} — IPO Filing Timeline", company_label);
    println!("  CIK: {}", cik);
    println!();

    for (i, filing) in all_filings.iter().enumerate() {
        let date = filing
            .filing_date
            .map(|d| d.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let doc_url = filing.as_primary_document_url();
        // Index page lists every document in the filing (exhibits, etc.)
        let index_url =
            Url::CikAccessionIndex(filing.cik.clone(), filing.accession_number.clone()).value();

        println!("[{}] {}  Filed: {}", i, filing.form, date);
        println!("    Document:  {}", doc_url);
        println!("    Index:     {}", index_url);
        println!();
    }

    // Tally by form type.
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for f in &all_filings {
        *counts.entry(f.form.as_str()).or_insert(0) += 1;
    }
    let summary: Vec<String> = counts
        .iter()
        .map(|(k, v)| format!("{}: {}", k, v))
        .collect();
    println!(
        "{} total filing(s)  ({})",
        all_filings.len(),
        summary.join(", ")
    );
    println!();
    println!("Tip: use --index <N> --part body to render any filing above.");

    // ── Resolve the target filing ─────────────────────────────────────────
    if matches!(args.part, FilingPart::Summary) {
        return Ok(());
    }

    let n = all_filings.len() as i64;
    let target_idx = if args.index >= 0 {
        args.index
    } else {
        // Negative index counts from the end (-1 = oldest).
        n + args.index
    };

    if target_idx < 0 || target_idx >= n {
        eprintln!(
            "Index {} is out of range (0..{} for {} filings).",
            args.index,
            n - 1,
            n
        );
        std::process::exit(1);
    }

    let target = &all_filings[target_idx as usize];
    let date = target
        .filing_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let label = format!("{} {} — {}", company_label, target.form, date);

    eprintln!("Rendering: {}", label);
    println!("---");
    println!("# {}", label);
    println!();

    let render_exhibits = matches!(args.part, FilingPart::All);
    let rendered = render_filing(client, target, true, render_exhibits, view).await?;

    // ── Primary document ──────────────────────────────────────────────────
    eprintln!("Primary document: {}", target.as_primary_document_url());
    println!("## Primary Document");
    println!();
    if let Some(ref body) = rendered.body {
        println!("{}", body);
    }

    // ── Exhibits (only when --part=all) ───────────────────────────────────
    if !rendered.exhibits.is_empty() {
        eprintln!("Found {} substantive exhibit(s):", rendered.exhibits.len());
        for ex in &rendered.exhibits {
            eprintln!("  {} — {}", ex.document_type, ex.name);
        }
        for ex in &rendered.exhibits {
            eprintln!("Rendering exhibit: {}", ex.url);
            println!();
            println!("---");
            println!("## Exhibit: {} ({})", ex.document_type, ex.name);
            println!();
            println!("{}", ex.content);
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
        ViewArg::Markdown => run(&client, &args, &MarkdownView).await,
        ViewArg::Embedding => run(&client, &args, &EmbeddingTextView).await,
    }
}
