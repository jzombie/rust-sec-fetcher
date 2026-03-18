//! Shows how the holdings of a fund (N-PORT) or institutional manager (13F)
//! have changed across consecutive filings, and insider transactions (Form 4)
//! for operating companies.
//!
//! Supports any ticker:
//!   - **NPORT-P** — ETFs and mutual funds (monthly, ~60-day lag)
//!     e.g. SPY, QQQ, IVV, ARKK
//!   - **13F-HR**  — Institutional managers ≥ $100 M AUM (quarterly, 45-day lag)
//!     e.g. BRK-A
//!   - **Form 4**  — Insider transactions for any public company (≤ 2 business days lag)
//!     e.g. AAPL, TSLA, NVDA
//!
//! For N-PORT and 13F, consecutive filings are diffed to show adds, removes,
//! and weight changes. For Form 4, recent insider buy/sell activity is listed.
//!
//! # Usage
//!
//! ```text
//! cargo run --example holdings_show -- SPY
//! cargo run --example holdings_show -- BRK-A
//! cargo run --example holdings_show -- AAPL
//! cargo run --example holdings_show -- AAPL --filings 50
//! cargo run --example holdings_show -- SPY --all
//! ```
//!
//! # What you can and cannot see via Form 4
//!
//! Form 4 covers corporate insiders: directors, officers, and anyone owning
//! more than 10% of a class of equity. It does NOT cover Members of Congress
//! or other government officials — those are reported under the STOCK Act to
//! the House Clerk and Senate Secretary, which is a completely separate system.
use clap::Parser;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::{Cik, CikSubmission, NportInvestment, ThirteenfHolding, TickerSymbol};
use sec_fetcher::network::{
    SecClient, fetch_13f, fetch_13f_filings, fetch_cik_by_ticker_symbol, fetch_form4,
    fetch_form4_filings, fetch_nport, fetch_nport_filings,
};
use sec_fetcher::ops::{
    Diff, Position, WEIGHT_CHANGE_THRESHOLD, diff_holdings, positions_from_13f,
    positions_from_nport,
};
use std::error::Error;
use std::fmt;

#[derive(Parser)]
#[command(
    about = "Show holdings changes or insider transactions across recent filings for a ticker"
)]
struct Args {
    /// Ticker symbol: fund (SPY), institutional manager (BRK-A), or company (AAPL)
    ticker: String,

    /// Number of recent filings to fetch (default: 3 NPORT / 4 13F / 20 Form4)
    #[arg(long)]
    filings: Option<usize>,

    /// Print full holdings for every filing (N-PORT/13F only), not just the diff
    #[arg(long)]
    all: bool,
}

// ── config ───────────────────────────────────────────────────────────────────

/// Default number of filings to fetch when --filings is not specified.
const DEFAULT_FILINGS_NPORT: usize = 3;
const DEFAULT_FILINGS_13F: usize = 4;
const DEFAULT_FILINGS_FORM4: usize = 20;

/// A single Form 4 insider transaction row ready for display.
struct Form4Row {
    filed: String,
    txn_date: String,
    filer_name: String,
    role: String,
    action: String,
    shares: String,
    price: String,
    owned_after: String,
}

impl fmt::Display for Form4Row {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<12}  {:<12}  {:<30}  {:<20}  {:<12}  {:>12}  {:>10}  {:>14}",
            self.filed,
            self.txn_date,
            self.filer_name,
            self.role,
            self.action,
            self.shares,
            self.price,
            self.owned_after,
        )
    }
}

// ── formatting ────────────────────────────────────────────────────────────────

fn format_position(p: &Position) -> String {
    format!(
        "  {:9}  {:<45}  {:>12}  {:>6.2}%",
        p.cusip,
        p.name.chars().take(45).collect::<String>(),
        fmt_usd(p.val_usd),
        p.weight,
    )
}

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
            println!("  {}", format_position(p));
        }
    }

    if !diff.removed.is_empty() {
        println!("  REMOVED ({}):", diff.removed.len());
        for p in &diff.removed {
            println!("  {}", format_position(p));
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
        println!("{}", format_position(p));
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let ticker = TickerSymbol::new(&args.ticker);
    let show_all = args.all;

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    // ── CIK lookup ────────────────────────────────────────────────────────────
    eprintln!("Looking up CIK for {}…", ticker);
    let cik: Cik = fetch_cik_by_ticker_symbol(&client, &ticker).await?;
    eprintln!("  CIK: {}", cik);

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
    let n = args.filings.unwrap_or(default_n).min(relevant.len());

    // Take the N most recent filings. Submissions are returned newest-first.
    let selected: Vec<&CikSubmission> = relevant.iter().take(n).collect();

    // ── Form 4 path — fetch and print, then return early ─────────────────────
    if let FilingKind::Form4 = kind {
        eprintln!("Fetching {} Form 4 filings…", n);
        println!();
        println!(
            "{} (CIK: {})  —  insider transactions (most recent {} filings)",
            ticker, cik, n
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
                let row = Form4Row {
                    filed: sub.filing_date.map(|d| d.to_string()).unwrap_or_default(),
                    txn_date: t
                        .transaction_date
                        .map(|d| d.to_string())
                        .unwrap_or_default(),
                    filer_name: t.filer_name.chars().take(30).collect(),
                    role: role.chars().take(20).collect(),
                    action: format!("{} ({})", t.code_description(), t.transaction_code),
                    shares: format!("{:.0}", t.shares),
                    price: t
                        .price_per_share
                        .map(|p| format!("${:.2}", p))
                        .unwrap_or_else(|| "-".to_string()),
                    owned_after: t
                        .shares_owned_after
                        .map(|s| format!("{:.0}", s))
                        .unwrap_or_else(|| "-".to_string()),
                };
                println!("{}", row);
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
        eprintln!("  {} ({})", date_label, sub.accession_number);

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
    println!("{} (CIK: {})  —  {} filings", ticker, cik, snapshots.len());

    for i in 0..snapshots.len() {
        let (label, positions) = &snapshots[i];

        if show_all {
            print_full(label, positions);
        }

        if i + 1 < snapshots.len() {
            let (next_label, next_positions) = &snapshots[i + 1];
            let d = diff_holdings(positions, next_positions);
            print_diff(label, next_label, &d);
        }
    }

    println!();
    Ok(())
}
