// TODO: Convert into a stream producer that can be consumed by other tools...

/// Demonstrates delta polling of the SEC EDGAR filing Atom feed.
///
/// # How it works
///
/// EDGAR exposes a live Atom feed. Each "page" contains the 40 most-recently
/// filed documents. To go further back in time, you pass a `before=` cursor
/// to get the next oldest page of 40, and so on.
///
/// **Without `--since`** the tool just fetches the first page (or `--pages` N
/// pages) and prints what it finds. There is no filtering.
///
/// **With `--since <timestamp>`** the tool works differently:
///
///   1. It fetches pages backwards in time, one at a time.
///   2. It stops as soon as the oldest entry on a page is *at or before* the
///      timestamp you gave — meaning all newer filings are now in memory.
///   3. It discards everything at or before the timestamp, then prints only
///      entries that are strictly newer.
///   4. It prints the timestamp of the *newest* entry it found. That is the
///      "high-water mark" — copy it as your `--since` value next time to pick
///      up exactly where you left off, with no gaps and no duplicates.
///
/// So the loop looks like:
///
///   First run (no --since):
///     → prints newest 40 filings
///     → shows: --since "2026-03-13T21:00:00-04:00"   ← your new mark
///
///   Next run (--since "2026-03-13T21:00:00-04:00"):
///     → fetches pages until it reaches that timestamp
///     → prints only what is strictly newer  (0 entries on a quiet weekend)
///     → if anything new was found, shows an updated --since mark
///
/// Without `--pages`, `--since` mode auto-paginates up to 25 pages (1 000
/// entries). That covers several days of normal SEC filing volume. If you
/// need to backfill further than that, combine `--since` with a large
/// `--pages N` value, or use the SEC bulk submission dataset instead.
///
/// # Usage
///
///   cargo run --example edgar_firehose
///       → newest 40 filings across all form types
///
///   cargo run --example edgar_firehose -- --filter "8-K,NPORT-P,SC 13G"
///       → one parallel request per type, newest 40 of each
///
///   cargo run --example edgar_firehose -- --filter "8-K" --since "2026-03-13T17:30:01-04:00"
///       → only 8-K filings strictly newer than that timestamp
///       → auto-paginates until the mark is covered (up to 25 pages)
///
///   cargo run --example edgar_firehose -- --filter "8-K" --pages 5
///       → newest 200 8-K filings, no timestamp filter
///
///   cargo run --example edgar_firehose -- --filter "8-K" --since "<mark>" --pages 50
///       → catch up after a long gap (up to 2 000 entries back)
///
/// # Recommended form-type filters
///
///   "8-K"            — corporate events and earnings releases
///   "NPORT-P"        — monthly fund portfolio holdings
///   "13F-HR"         — quarterly institutional holdings
///   "SC 13G"         — passive ownership crossing 5%
///   "SC 13D"         — active ownership crossing 5%
use chrono::DateTime;
use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::FeedEntry;
use sec_fetcher::network::{fetch_edgar_feeds_since, SecClient, EDGAR_PAGE_SIZE};
use std::error::Error;
use std::fmt;

/// When --since is given without --pages, paginate up to this many pages.
const SINCE_DEFAULT_MAX_PAGES: usize = 25;

#[derive(Parser)]
#[command(
    about = "Delta-poll the SEC EDGAR filing Atom feed",
    long_about = "Fetches recent EDGAR filings. Without --since, prints the newest page(s). \
                  With --since, prints only entries strictly newer than the given ISO-8601 \
                  timestamp and outputs the new high-water mark for the next run."
)]
struct Args {
    /// Comma-separated form-type filter (e.g. \"8-K,NPORT-P\"). No filter = all types.
    #[arg(long, alias = "type", default_value = "")]
    filter: String,

    /// Only show filings strictly after this ISO-8601 timestamp (e.g. \"2026-03-13T17:30:01-04:00\")
    #[arg(long, alias = "after")]
    since: Option<String>,

    /// Number of 40-entry pages to fetch. Default: 1 without --since, 25 with --since.
    #[arg(long)]
    pages: Option<usize>,
}

/// A single feed entry formatted as a table row.
struct FeedEntryRow<'a>(&'a FeedEntry);

impl<'a> fmt::Display for FeedEntryRow<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let e = self.0;
        let route = if matches!(e.form_type.to_uppercase().as_str(), "8-K" | "8-K/A") {
            if e.is_earnings_release() {
                "→ Path 1 (Earnings / 2.02)"
            } else if e.is_mid_quarter_event() {
                "→ Path 2 (Mid-quarter event)"
            } else {
                "→ 8-K (no items of interest)"
            }
        } else {
            ""
        };

        let items_str = if e.items.is_empty() {
            String::new()
        } else {
            format!("[{}]  ", e.items.join(","))
        };

        write!(
            f,
            "{:<12}  {:<8}  {:<12}  {:<45}  {:<28}  {}{}",
            e.filing_date.map(|d| d.to_string()).unwrap_or_default(),
            e.form_type,
            e.cik.as_ref().map(|c| c.to_string()).unwrap_or_default(),
            e.company_name.chars().take(45).collect::<String>(),
            e.accession_number,
            items_str,
            route,
        )
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let since: Option<DateTime<chrono::FixedOffset>> = match &args.since {
        None => None,
        Some(s) => match DateTime::parse_from_rfc3339(s) {
            Ok(dt) => Some(dt),
            Err(_) => {
                eprintln!("Invalid --since timestamp: '{}'", s);
                eprintln!("Expected ISO 8601, e.g. \"2026-03-13T17:30:01-04:00\"");
                std::process::exit(1);
            }
        },
    };

    let type_filter: Vec<&str> = args
        .filter
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let max_pages = args.pages.unwrap_or(if since.is_some() {
        SINCE_DEFAULT_MAX_PAGES
    } else {
        1
    });

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    if let Some(s) = since {
        eprintln!("Delta mode — entries filed after: {}", s.to_rfc3339());
    }

    let types: Vec<&str> = if type_filter.is_empty() {
        vec![""]
    } else {
        type_filter.clone()
    };

    let delta = fetch_edgar_feeds_since(&client, &types, since, max_pages).await?;
    let all_entries = &delta.entries;
    let new_high_water = delta.high_water;

    for ft in &types {
        let label = if ft.is_empty() { "all types" } else { ft };
        let n = all_entries
            .iter()
            .filter(|e| ft.is_empty() || e.form_type == *ft)
            .count();
        eprintln!(
            "  {} → {} new entries (fetched up to {} pages × {} each)",
            label, n, max_pages, EDGAR_PAGE_SIZE
        );
    }

    println!(
        "{:<12}  {:<8}  {:<12}  {:<45}  {:<28}  Items / Route",
        "Filed", "Form", "CIK", "Company", "Accession"
    );
    println!("{}", "-".repeat(145));

    let mut type_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for entry in all_entries {
        *type_counts.entry(entry.form_type.as_str()).or_insert(0) += 1;
        println!("{}", FeedEntryRow(entry));
    }

    eprintln!();

    if !type_counts.is_empty() {
        let mut summary: Vec<_> = type_counts.iter().collect();
        summary.sort_by_key(|(ft, _)| *ft);
        eprintln!(
            "Entries shown — {}",
            summary
                .iter()
                .map(|(ft, n)| format!("{}: {}", ft, n))
                .collect::<Vec<_>>()
                .join("  |  ")
        );
    }

    if let Some(since_val) = since {
        if all_entries.is_empty() {
            eprintln!(
                "No new filing(s) since {}. Mark unchanged.",
                since_val.to_rfc3339()
            );
        } else {
            eprintln!(
                "Delta complete: {} new filing(s) since {}.",
                all_entries.len(),
                since_val.to_rfc3339()
            );
        }
    }

    if new_high_water != since {
        if let Some(mark) = new_high_water {
            let filter_part = if args.filter.is_empty() {
                String::new()
            } else {
                format!("--filter \"{}\" ", args.filter)
            };
            let mark_str = mark.to_rfc3339();
            eprintln!();
            eprintln!("┌─ Next delta poll ──────────────────────────────────────────────────");
            eprintln!("│  {}", mark_str);
            eprintln!("└────────────────────────────────────────────────────────────────────");
            eprintln!(
                "  cargo run --example edgar_firehose -- {}--since \"{}\"",
                filter_part, mark_str
            );
        }
    }

    Ok(())
}
