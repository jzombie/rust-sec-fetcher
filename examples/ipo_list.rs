/// Lists recent IPO registration statement filings from the EDGAR live feed.
///
/// IPO-related filings are identified by their SEC form type:
///
/// | Form    | Meaning                                                       |
/// |---------|---------------------------------------------------------------|
/// | S-1     | Initial registration statement — the primary IPO prospectus   |
/// | S-1/A   | Amendment filed during SEC review or before pricing           |
/// | F-1     | Foreign private issuer equivalent of S-1                      |
/// | F-1/A   | Amendment to an F-1                                           |
/// | 424B4   | Final prospectus filed after pricing (contains deal terms)    |
///
/// This tool polls the EDGAR Atom feed — the fastest publicly available source
/// for freshly accepted filings — and filters for the above form types.
/// Because each page of the feed covers the most recent 40 entries *across all
/// form types*, you typically need multiple pages to surface a meaningful number
/// of IPO filings.
///
/// # Usage
///
///   cargo run --example ipo_list
///   cargo run --example ipo_list -- --pages 10
///   cargo run --example ipo_list -- --pages 5 --include-finals
///
/// # Options
///
///   --pages <N>        Number of EDGAR feed pages to scan (default: 5; each page = 40 entries)
///   --include-finals   Also show 424B4 "pricing" prospectuses (final deal-term filings)
///   --since <ts>       Only show filings strictly newer than this ISO-8601 timestamp
///
/// # Interpreting the output
///
///   Filings are printed newest-first. An S-1 is the *start* of the IPO
///   process; S-1/A amendments follow during SEC review; the 424B4 is filed
///   at the moment of pricing, after which shares begin trading.
///
/// # Notes on coverage
///
///   The EDGAR feed covers all registrants. However, not every S-1 leads to
///   an IPO — some are withdrawn, others are for secondary offerings or SPAC
///   mergers. Use `ipo_show` to inspect the full prospectus of any filing
///   that looks interesting.
use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::enums::FormType;
use sec_fetcher::network::SecClient;
use sec_fetcher::ops::get_ipo_feed_entries;
use std::error::Error;

#[derive(Parser)]
#[command(
    about = "List recent IPO registration filings from the EDGAR live feed",
    long_about = "Polls the EDGAR Atom feed and filters for S-1, S-1/A, F-1, F-1/A (and \
                  optionally 424B4) filings. Each entry represents an IPO registration \
                  statement or an amendment. Use --pages to look further back in time."
)]
struct Args {
    /// Number of 40-entry EDGAR feed pages to scan
    #[arg(long, default_value_t = 5)]
    pages: usize,

    /// Also include 424B4 final pricing prospectuses
    #[arg(long, default_value_t = false)]
    include_finals: bool,

    /// Only show filings strictly after this ISO-8601 timestamp
    /// (e.g. \"2026-03-13T17:30:01-04:00\")
    #[arg(long)]
    since: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let since = match &args.since {
        None => None,
        Some(s) => match chrono::DateTime::parse_from_rfc3339(s) {
            Ok(dt) => Some(dt),
            Err(_) => {
                eprintln!("Invalid --since timestamp: '{}'", s);
                eprintln!("Expected ISO 8601, e.g. \"2026-03-13T17:30:01-04:00\"");
                std::process::exit(1);
            }
        },
    };

    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    // Determine which form types to fetch.
    let mut form_types: Vec<FormType> = FormType::IPO_REGISTRATION_FORM_TYPES.to_vec();
    if args.include_finals {
        form_types.extend_from_slice(FormType::IPO_PRICING_FORM_TYPES);
    }

    eprintln!(
        "Scanning {} page(s) × 40 entries for: {}",
        args.pages,
        form_types
            .iter()
            .map(|ft| ft.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if let Some(s) = &since {
        eprintln!("Delta mode — filings after: {}", s.to_rfc3339());
    }

    // Fetch from the EDGAR feed, exact-match filter, and deduplicate in ops.
    let (entries, high_water) =
        get_ipo_feed_entries(&client, &form_types, since, args.pages).await?;

    if entries.is_empty() {
        println!("No IPO registration filings found in the scanned range.");
        if let Some(hw) = high_water {
            eprintln!("High-water mark: {}", hw.to_rfc3339());
        }
        return Ok(());
    }

    // ── Entries (newest-first) ──────────────────────────────────────────────
    for entry in &entries {
        // filing_date is parsed from the feed summary and is occasionally
        // absent; fall back to the EDGAR acceptance timestamp's date.
        let filed = entry
            .filing_date
            .map(|d| d.to_string())
            .unwrap_or_else(|| entry.updated.date_naive().to_string());

        let cik = entry
            .cik
            .as_ref()
            .map(|c| c.to_string())
            .unwrap_or_default();

        println!(
            "Filed: {} — {} — {}  (CIK {})",
            filed, entry.form_type, entry.company_name, cik
        );
        println!("    {}", entry.filing_href);
        println!();
    }

    println!(
        "Total: {} filing(s) across {} form type(s)",
        entries.len(),
        form_types.len()
    );

    // ── High-water mark for delta polling ───────────────────────────────────
    if let Some(hw) = high_water {
        eprintln!();
        eprintln!(
            "Next run — use --since \"{}\" to see only newer filings.",
            hw.to_rfc3339()
        );
    }

    Ok(())
}
