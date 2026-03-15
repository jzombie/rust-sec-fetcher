/// Demonstrates event-driven delta polling of the SEC EDGAR filing Atom feed.
///
/// The EDGAR "current filings" feed is sorted newest-first and returns up to 40
/// entries per request. By storing the `updated` timestamp of the most recent
/// entry (the "high-water mark") after each poll, subsequent polls process only
/// the delta — filings that arrived since the last check.
///
/// This is the foundation for an on-demand pipeline that never pulls more than
/// it needs: instead of re-fetching all of a company's historical submissions,
/// you poll the feed at an interval and process only what is new.
///
/// # Usage
///
///   cargo run --example edgar_firehose
///       → latest 40 8-K filings from the global feed
///
///   cargo run --example edgar_firehose -- 10-K
///       → latest 40 10-K filings
///
///   cargo run --example edgar_firehose -- ""
///       → ALL form types (the full firehose, 40 entries)
///
///   cargo run --example edgar_firehose -- 8-K "2026-03-13T17:30:01-04:00"
///       → delta mode: only entries newer than the given high-water mark
///
/// # How delta polling works
///
///  1. First run:  print all entries, output the high-water mark at the end.
///  2. Next run:   paste the high-water mark as the second argument.
///                 Only entries filed after that timestamp are printed.
///  3. In production: store the high-water mark in a file or database and
///                    pass it automatically on each scheduled invocation.

use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_edgar_feed, SecClient};
use std::env;
use std::error::Error;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    // First argument: form type filter. Empty string = all form types.
    let form_type = args.get(1).map(|s| s.as_str()).unwrap_or("8-K");

    // Second argument (optional): ISO 8601 high-water mark for delta mode.
    let last_seen: Option<&str> = args.get(2).map(|s| s.as_str());

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let form_label = if form_type.is_empty() {
        "all form types"
    } else {
        form_type
    };

    eprintln!(
        "Fetching latest 40 {} filings from the EDGAR feed...",
        form_label
    );

    if let Some(mark) = last_seen {
        eprintln!(
            "Delta mode — high-water mark: {} (skipping entries at or before this timestamp)",
            mark
        );
    }

    let entries = fetch_edgar_feed(&client, form_type, 40).await?;

    let mut new_count = 0usize;
    let mut high_water: Option<&str> = None;

    println!(
        "{:<12}  {:<8}  {:<45}  {:<28}  {}",
        "Filed", "Form", "Company", "Accession", "Items / Route"
    );
    println!("{}", "-".repeat(130));

    for entry in &entries {
        // On the first (newest) entry, capture the high-water mark for next run.
        if high_water.is_none() {
            high_water = Some(&entry.updated);
        }

        // Delta cutoff: stop once we reach entries we've already seen.
        if let Some(mark) = last_seen {
            if entry.updated.as_str() <= mark {
                break;
            }
        }

        new_count += 1;

        // Build routing label for 8-K entries.
        let route = if entry.form_type.to_uppercase() == "8-K"
            || entry.form_type.to_uppercase() == "8-K/A"
        {
            if entry.is_earnings_release() {
                "→ Path 1 (Earnings / 2.02)"
            } else if entry.is_mid_quarter_event() {
                "→ Path 2 (Mid-quarter event)"
            } else {
                "→ 8-K (no meaningful items)"
            }
        } else {
            ""
        };

        let items_str = if entry.items.is_empty() {
            String::new()
        } else {
            format!("[{}]  ", entry.items.join(","))
        };

        println!(
            "{:<12}  {:<8}  {:<45}  {:<28}  {}{}",
            entry
                .filing_date
                .map(|d| d.to_string())
                .unwrap_or_default(),
            entry.form_type,
            entry.company_name.chars().take(45).collect::<String>(),
            entry.accession_number,
            items_str,
            route,
        );
    }

    eprintln!();

    if let Some(mark) = last_seen {
        eprintln!(
            "Delta complete: {} new filing(s) since {}.",
            new_count, mark
        );
    }

    if let Some(mark) = high_water {
        eprintln!();
        eprintln!("┌─ High-water mark for next delta poll ─────────────────────────────");
        eprintln!("│  {}", mark);
        eprintln!("└────────────────────────────────────────────────────────────────────");
        eprintln!(
            "Pass this as the second argument on the next run to process only new filings:"
        );
        eprintln!(
            "  cargo run --example edgar_firehose -- {} \"{}\"",
            form_type, mark
        );
    }

    Ok(())
}
