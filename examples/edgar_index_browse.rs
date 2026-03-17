/// Browse EDGAR's quarterly full-index (`master.idx`) files.
///
/// # What is the full-index?
///
/// The SEC publishes a `master.idx` file for every calendar quarter going back
/// to **Q4 1993**. Each file is a pipe-delimited text file that lists *every
/// filing* accepted by EDGAR during that quarter — typically 200,000–500,000
/// entries per quarter in recent years.
///
/// This is fundamentally different from the live Atom feed:
///
/// | Source          | Depth          | Update frequency | Use case                     |
/// |-----------------|---------------|------------------|------------------------------|
/// | Atom feed       | ~days/weeks    | near-real-time   | alerting, delta polling      |
/// | `master.idx`    | **30+ years**  | nightly          | historical research, backfill|
///
/// Each row contains:
/// ```text
/// CIK | Company Name | Form Type | Date Filed | Filename (→ URL)
/// ```
///
/// # Usage
///
///   cargo run --example full_index
///       → 50 filings from the current quarter (2026 Q1 as of today)
///
///   cargo run --example full_index -- --form 10-K
///       → first 50 annual reports from this quarter
///
///   cargo run --example full_index -- --year 2024 --quarter 4 --form 8-K
///       → first 50 8-K filings from Q4 2024
///
///   cargo run --example full_index -- --year 1994 --quarter 1
///       → oldest available data (Q1 1994; Q4 1993 is earliest)
///
///   cargo run --example full_index -- --form 10-K --company apple
///       → Apple's 10-K filings this quarter
///
///   cargo run --example full_index -- --form 10-K --top 200
///       → 200 annual reports from this quarter
///
///   cargo run --example full_index -- --form 10-K --top 0
///       → ALL 10-K filings from this quarter (no limit)
///
/// # Options
///
///   --year YYYY       Calendar year (default: current year)
///   --quarter Q       Quarter 1–4 (default: current quarter)
///   --form TYPE       Filter by form type substring, case-insensitive
///                       e.g. "10-K", "8-K", "10-K/A", "NPORT-P", "4"
///   --company NAME    Filter by company name substring, case-insensitive
///   --top N           Maximum rows to display (default: 50; 0 = no limit)
use chrono::{Datelike, Local};
use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::MasterIndexEntry;
use sec_fetcher::network::{fetch_edgar_master_index, SecClient};
use std::error::Error;
use std::fmt;

#[derive(Parser)]
#[command(about = "Browse EDGAR quarterly full-index (master.idx) files")]
struct Args {
    /// Calendar year (default: current year)
    #[arg(long)]
    year: Option<u16>,

    /// Quarter 1–4 (default: current quarter)
    #[arg(long)]
    quarter: Option<u8>,

    /// Filter by form type substring, case-insensitive (e.g. \"10-K\", \"8-K\")
    #[arg(long)]
    form: Option<String>,

    /// Filter by company name substring, case-insensitive
    #[arg(long)]
    company: Option<String>,

    /// Maximum rows to display (default: 50; 0 = no limit)
    #[arg(long)]
    top: Option<usize>,
}

/// A single full-index entry formatted as a table row.
struct IndexEntryRow<'a>(&'a MasterIndexEntry);

impl<'a> fmt::Display for IndexEntryRow<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let e = self.0;
        write!(
            f,
            "{:<12}  {:<12}  {:<45}  {:<18}  {}",
            e.date_filed.to_string(),
            e.cik,
            e.company_name.chars().take(45).collect::<String>(),
            e.form_type,
            e.as_url(),
        )
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let today = Local::now().date_naive();
    let year = args.year.unwrap_or(today.year() as u16);
    let quarter = args.quarter.unwrap_or(((today.month0() / 3) + 1) as u8);

    if quarter < 1 || quarter > 4 {
        eprintln!("Error: --quarter must be 1–4");
        std::process::exit(1);
    }

    let form_filter = args.form.map(|s| s.to_lowercase());
    let company_filter = args.company.map(|s| s.to_lowercase());

    // ── Fetch ──────────────────────────────────────────────────────────────
    eprintln!("Fetching EDGAR full-index for {year} Q{quarter}…");
    let cfg = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&cfg)?;
    let all_entries = fetch_edgar_master_index(&client, year, quarter).await?;
    eprintln!("  Total filings in index: {}", fmt_count(all_entries.len()));

    // ── Filter ─────────────────────────────────────────────────────────────
    let filtered: Vec<&MasterIndexEntry> = all_entries
        .iter()
        .filter(|e| {
            if let Some(ref f) = form_filter {
                if !e.form_type.to_lowercase().contains(f.as_str()) {
                    return false;
                }
            }
            if let Some(ref c) = company_filter {
                if !e.company_name.to_lowercase().contains(c.as_str()) {
                    return false;
                }
            }
            true
        })
        .collect();

    let filter_desc = match (&form_filter, &company_filter) {
        (Some(f), Some(c)) => format!("form: {f}  company: {c}"),
        (Some(f), None) => format!("form: {f}"),
        (None, Some(c)) => format!("company: {c}"),
        (None, None) => String::new(),
    };
    if !filter_desc.is_empty() {
        eprintln!(
            "  After filter ({filter_desc}): {}",
            fmt_count(filtered.len())
        );
    }

    // Default to top 50 when the user hasn't asked for a specific limit.
    let limit = args.top.unwrap_or(50);
    let display: Vec<&MasterIndexEntry> = if limit == 0 {
        filtered.iter().copied().collect()
    } else {
        filtered.iter().copied().take(limit).collect()
    };

    if limit > 0 && display.len() < filtered.len() {
        eprintln!(
            "  Showing first {} of {} (use --top 0 for all)",
            display.len(),
            fmt_count(filtered.len())
        );
    }

    // ── Display ────────────────────────────────────────────────────────────
    println!();
    println!(
        "EDGAR full-index — {year} Q{quarter}{}",
        if filter_desc.is_empty() {
            String::new()
        } else {
            format!("   [{filter_desc}]")
        }
    );
    println!();
    println!(
        "{:<12}  {:<12}  {:<45}  {:<18}  {}",
        "Filed", "CIK", "Company", "Form", "URL"
    );
    println!("{}", "-".repeat(150));

    for entry in &display {
        println!("{}", IndexEntryRow(entry));
    }

    println!();
    Ok(())
}

fn fmt_count(n: usize) -> String {
    // Format with thousands separators for readability.
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}
