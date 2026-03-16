/// Shows how the holdings of a fund (N-PORT) or institutional manager (13F)
/// have changed across consecutive filings, and insider transactions (Form 4)
/// for operating companies.
///
/// Supports any ticker:
///   - **NPORT-P** — ETFs and mutual funds (monthly, ~60-day lag)
///                   e.g. SPY, QQQ, IVV, ARKK
///   - **13F-HR**  — Institutional managers ≥ $100 M AUM (quarterly, 45-day lag)
///                   e.g. BRK-A
///   - **Form 4**  — Insider transactions for any public company (≤ 2 business days lag)
///                   e.g. AAPL, TSLA, NVDA
///
/// For N-PORT and 13F, consecutive filings are diffed to show adds, removes,
/// and weight changes. For Form 4, recent insider buy/sell activity is listed.
///
/// # Usage
///
///   cargo run --example holdings_history -- SPY
///       → last 3 NPORT-P filings for SPY, changes only
///
///   cargo run --example holdings_history -- BRK-A
///       → last 4 13F-HR filings for Berkshire Hathaway, changes only
///
///   cargo run --example holdings_history -- AAPL
///       → last 20 Form 4 insider transactions for Apple
///
///   cargo run --example holdings_history -- AAPL --filings 50
///       → last 50 Form 4 filings (may cover many transactions each)
///
///   cargo run --example holdings_history -- SPY --all
///       → show full holdings for each filing, not just the diff
///
/// # What you can and cannot see via Form 4
///
/// Form 4 covers corporate insiders: directors, officers, and anyone owning
/// more than 10% of a class of equity. It does NOT cover Members of Congress
/// or other government officials — those are reported under the STOCK Act to
/// the House Clerk and Senate Secretary (disclosures.house.gov / efts.senate.gov),
/// which is a completely separate system that is NOT accessible through the SEC.
/// A Congress member would only appear in Form 4 if they were also a corporate
/// director or officer of a public company, which is rare.
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::{Cik, CikSubmission, NportInvestment, ThirteenfHolding};
use sec_fetcher::network::{
    fetch_13f, fetch_13f_filings, fetch_cik_by_ticker_symbol, fetch_form4, fetch_form4_filings,
    fetch_nport, fetch_nport_filings, SecClient,
};
use std::collections::HashMap;
use std::env;
use std::error::Error;

// ── config ───────────────────────────────────────────────────────────────────

/// Default number of filings to fetch when --filings is not specified.
const DEFAULT_FILINGS_NPORT: usize = 3;
const DEFAULT_FILINGS_13F: usize = 4;
const DEFAULT_FILINGS_FORM4: usize = 20;

/// Minimum absolute weight change (percentage points for N-PORT, approximate
/// percentage of total value for 13F) to appear in the "Weight changes" list.
const WEIGHT_CHANGE_THRESHOLD: Decimal = dec!(0.10);

// ── common snapshot type ──────────────────────────────────────────────────────

/// A normalised single-position snapshot used for diffing.
/// Produced from both NportInvestment and ThirteenfHolding.
#[derive(Clone)]
struct Position {
    cusip: String,
    name: String,
    val_usd: Decimal,
    /// Weight as a percentage of total portfolio (0–100 scale).
    /// For N-PORT this comes directly from the filing; for 13F it is derived
    /// from the position's share of total reported value.
    weight: Decimal,
}

fn positions_from_nport(investments: &[NportInvestment]) -> Vec<Position> {
    investments
        .iter()
        .map(|h| Position {
            cusip: h.cusip.clone(),
            name: h.name.clone(),
            val_usd: h.val_usd,
            weight: h.pct_val * dec!(100), // pct_val is 0–1, convert to 0–100
        })
        .collect()
}

fn positions_from_13f(holdings: &[ThirteenfHolding]) -> Vec<Position> {
    let total: Decimal = holdings.iter().map(|h| h.value_usd).sum();
    holdings
        .iter()
        .map(|h| Position {
            cusip: h.cusip.clone(),
            name: h.name.clone(),
            val_usd: h.value_usd,
            weight: if total.is_zero() {
                dec!(0)
            } else {
                (h.value_usd / total * dec!(100)).round_dp(4)
            },
        })
        .collect()
}

// ── diff ──────────────────────────────────────────────────────────────────────

struct Diff {
    added: Vec<Position>,               // present in new, absent in old
    removed: Vec<Position>,             // present in old, absent in new
    changed: Vec<(Position, Position)>, // (old, new) where |weight delta| ≥ threshold
}

fn diff_snapshots(old: &[Position], new: &[Position]) -> Diff {
    let old_map: HashMap<&str, &Position> = old.iter().map(|p| (p.cusip.as_str(), p)).collect();
    let new_map: HashMap<&str, &Position> = new.iter().map(|p| (p.cusip.as_str(), p)).collect();

    let added: Vec<Position> = new
        .iter()
        .filter(|p| !old_map.contains_key(p.cusip.as_str()))
        .cloned()
        .collect();

    let removed: Vec<Position> = old
        .iter()
        .filter(|p| !new_map.contains_key(p.cusip.as_str()))
        .cloned()
        .collect();

    let mut changed: Vec<(Position, Position)> = new
        .iter()
        .filter_map(|n| {
            old_map.get(n.cusip.as_str()).map(|o| {
                let delta = (n.weight - o.weight).abs();
                if delta >= WEIGHT_CHANGE_THRESHOLD {
                    Some(((*o).clone(), n.clone()))
                } else {
                    None
                }
            })
        })
        .flatten()
        .collect();

    // Sort by absolute weight change, largest first.
    changed.sort_by(|(o1, n1), (o2, n2)| {
        let d1 = (n1.weight - o1.weight).abs();
        let d2 = (n2.weight - o2.weight).abs();
        d2.cmp(&d1)
    });

    Diff {
        added,
        removed,
        changed,
    }
}

// ── formatting ────────────────────────────────────────────────────────────────

fn fmt_usd(v: Decimal) -> String {
    if v >= dec!(1_000_000_000) {
        format!("${:.2}B", v / dec!(1_000_000_000))
    } else if v >= dec!(1_000_000) {
        format!("${:.2}M", v / dec!(1_000_000))
    } else {
        format!("${:.0}", v)
    }
}

fn print_diff(label_old: &str, label_new: &str, diff: &Diff) {
    println!("\n── {} → {} {}", label_old, label_new, "─".repeat(50));

    if diff.added.is_empty() && diff.removed.is_empty() && diff.changed.is_empty() {
        println!("  (no significant changes)");
        return;
    }

    if !diff.added.is_empty() {
        println!("  ADDED ({}):", diff.added.len());
        for p in &diff.added {
            println!(
                "    {:9}  {:<45}  {:>10}  {:>6.2}%",
                p.cusip,
                p.name.chars().take(45).collect::<String>(),
                fmt_usd(p.val_usd),
                p.weight
            );
        }
    }

    if !diff.removed.is_empty() {
        println!("  REMOVED ({}):", diff.removed.len());
        for p in &diff.removed {
            println!(
                "    {:9}  {:<45}  {:>10}  {:>6.2}%",
                p.cusip,
                p.name.chars().take(45).collect::<String>(),
                fmt_usd(p.val_usd),
                p.weight
            );
        }
    }

    if !diff.changed.is_empty() {
        println!(
            "  WEIGHT CHANGES ≥ {:.2}% ({}):",
            WEIGHT_CHANGE_THRESHOLD,
            diff.changed.len()
        );
        for (old, new) in &diff.changed {
            let delta = new.weight - old.weight;
            let sign = if delta >= dec!(0) { "+" } else { "" };
            println!(
                "    {:9}  {:<45}  {:>6.2}% → {:>6.2}%  ({}{:.2}%)",
                new.cusip,
                new.name.chars().take(45).collect::<String>(),
                old.weight,
                new.weight,
                sign,
                delta
            );
        }
    }
}

fn print_full(label: &str, positions: &[Position]) {
    println!("\n── {} {}", label, "─".repeat(60));
    println!(
        "  {:9}  {:<45}  {:>12}  {:>7}",
        "CUSIP", "Name", "Value", "Weight"
    );
    println!("  {}", "-".repeat(80));
    for p in positions {
        println!(
            "  {:9}  {:<45}  {:>12}  {:>6.2}%",
            p.cusip,
            p.name.chars().take(45).collect::<String>(),
            fmt_usd(p.val_usd),
            p.weight
        );
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    let mut ticker = String::new();
    let mut num_filings: Option<usize> = None;
    let mut show_all = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--filings" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    num_filings = Some(v.parse().unwrap_or(3).max(2));
                }
            }
            "--all" => show_all = true,
            other if !other.starts_with('-') => ticker = other.to_string(),
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!("Usage: holdings_history <TICKER> [--filings N] [--all]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if ticker.is_empty() {
        eprintln!("Usage: holdings_history <TICKER> [--filings N] [--all]");
        eprintln!("  <TICKER>     fund, institutional manager, or company, e.g. SPY, BRK-A, AAPL");
        eprintln!(
            "  --filings N  how many recent filings to fetch (default: 3 NPORT / 4 13F / 20 Form4)"
        );
        eprintln!("  --all        print full holdings for every filing (N-PORT/13F only), not just the diff");
        std::process::exit(1);
    }

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    // ── CIK lookup ────────────────────────────────────────────────────────────
    eprintln!("Looking up CIK for {}…", ticker);
    let cik: Cik = fetch_cik_by_ticker_symbol(&client, &ticker).await?;
    eprintln!("  CIK: {}", cik.to_string());

    // ── Submission list ───────────────────────────────────────────────────────
    eprintln!("Fetching submission history…");

    // Detect which form type this issuer uses.
    let nport_subs = fetch_nport_filings(&client, cik.clone()).await?;
    let thirteenf_subs = fetch_13f_filings(&client, cik.clone()).await?;
    let form4_subs = fetch_form4_filings(&client, cik.clone()).await?;

    enum FilingKind {
        NPort,
        ThirteenF,
        Form4,
    }

    let (kind, relevant) = if !nport_subs.is_empty() {
        eprintln!(
            "  Found {} NPORT-P filings → using N-PORT path",
            nport_subs.len()
        );
        (FilingKind::NPort, nport_subs)
    } else if !thirteenf_subs.is_empty() {
        eprintln!(
            "  Found {} 13F-HR filings → using 13F path",
            thirteenf_subs.len()
        );
        (FilingKind::ThirteenF, thirteenf_subs)
    } else if !form4_subs.is_empty() {
        eprintln!(
            "  Found {} Form 4/4A filings → showing insider transactions",
            form4_subs.len()
        );
        (FilingKind::Form4, form4_subs)
    } else {
        eprintln!(
            "No NPORT-P, 13F-HR, or Form 4 filings found for {}.",
            ticker
        );
        std::process::exit(1);
    };

    let default_n = match kind {
        FilingKind::NPort => DEFAULT_FILINGS_NPORT,
        FilingKind::ThirteenF => DEFAULT_FILINGS_13F,
        FilingKind::Form4 => DEFAULT_FILINGS_FORM4,
    };
    let n = num_filings.unwrap_or(default_n).min(relevant.len());

    // Take the N most recent filings. Submissions are returned newest-first.
    let selected: Vec<&CikSubmission> = relevant.iter().take(n).collect();

    // ── Form 4 path — fetch and print, then return early ─────────────────────
    if let FilingKind::Form4 = kind {
        eprintln!("Fetching {} Form 4 filings…", n);
        println!();
        println!(
            "{} (CIK: {})  —  insider transactions (most recent {} filings)",
            ticker.to_uppercase(),
            cik.to_string(),
            n
        );
        println!();
        println!(
            "{:<12}  {:<12}  {:<30}  {:<20}  {:<12}  {:>12}  {:>10}  {:>14}",
            "Filed", "Txn Date", "Insider", "Role", "Type", "Shares", "Price", "Owned After"
        );
        println!("{}", "-".repeat(130));

        for sub in &selected {
            let txns = fetch_form4(&client, sub).await?;
            for t in &txns {
                let role = if t.is_officer {
                    t.officer_title.as_deref().unwrap_or("Officer")
                } else if t.is_director {
                    "Director"
                } else if t.is_ten_pct_owner {
                    "10%+ Owner"
                } else {
                    "Insider"
                };
                let action = format!("{} ({})", t.code_description(), t.transaction_code);
                let price_str = t
                    .price_per_share
                    .map(|p| format!("${:.2}", p))
                    .unwrap_or_else(|| "-".to_string());
                let after_str = t
                    .shares_owned_after
                    .map(|s| format!("{:.0}", s))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "{:<12}  {:<12}  {:<30}  {:<20}  {:<12}  {:>12}  {:>10}  {:>14}",
                    sub.filing_date.map(|d| d.to_string()).unwrap_or_default(),
                    t.transaction_date
                        .map(|d| d.to_string())
                        .unwrap_or_default(),
                    t.filer_name.chars().take(30).collect::<String>(),
                    role.chars().take(20).collect::<String>(),
                    action,
                    format!("{:.0}", t.shares),
                    price_str,
                    after_str,
                );
            }
        }
        println!();
        return Ok(());
    }

    eprintln!("Fetching {} filings in sequence…", n);

    // ── Fetch each filing ─────────────────────────────────────────────────────
    let mut snapshots: Vec<(String, Vec<Position>)> = Vec::with_capacity(n);

    for sub in &selected {
        let date_label = sub
            .filing_date
            .map(|d| d.to_string())
            .unwrap_or_else(|| sub.accession_number.to_string());
        eprintln!("  {} ({})", date_label, sub.accession_number.to_string());

        let positions: Vec<Position> = match kind {
            FilingKind::NPort => {
                let investments: Vec<NportInvestment> = fetch_nport(&client, sub).await?;
                positions_from_nport(&investments)
            }
            FilingKind::ThirteenF => {
                let holdings: Vec<ThirteenfHolding> = fetch_13f(&client, sub).await?;
                positions_from_13f(&holdings)
            }
            FilingKind::Form4 => unreachable!("Form4 path returns early above"),
        };

        snapshots.push((date_label, positions));
    }

    // Snapshots are newest-first; reverse to oldest-first for display.
    snapshots.reverse();

    // ── Print results ─────────────────────────────────────────────────────────
    println!();
    println!(
        "{} (CIK: {})  —  {} filings",
        ticker.to_uppercase(),
        cik.to_string(),
        snapshots.len()
    );

    for i in 0..snapshots.len() {
        let (label, positions) = &snapshots[i];

        if show_all {
            print_full(label, positions);
        }

        if i + 1 < snapshots.len() {
            let (next_label, next_positions) = &snapshots[i + 1];
            let d = diff_snapshots(positions, next_positions);
            print_diff(label, next_label, &d);
        }
    }

    println!();
    Ok(())
}
