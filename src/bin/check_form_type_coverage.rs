//! # check_form_type_coverage
//!
//! Downloads EDGAR master indexes starting from the most recently completed
//! calendar quarter and scans backwards until every named [`FormType`] variant
//! has been observed in real EDGAR data, or until [`MAX_LOOKBACK_QUARTERS`] is
//! exhausted.
//!
//! ```sh
//! cargo run --bin check_form_type_coverage
//! ```
//!
//! Exits 0 on success, 1 if coverage gaps are found.
//!
//! ## What is checked
//!
//! **Forward (our enum → EDGAR):** every named `FormType` variant must appear
//! at least once across the scanned quarters.  A variant never seen suggests a
//! typo or a fully retired form type.
//!
//! **Reverse (EDGAR → our enum):** any form type appearing at least
//! [`MINIMUM_FILINGS_THRESHOLD`] times in the *most recent* quarter must be a
//! named variant.  A miss means a high-frequency type is unaccounted for.

use chrono::Datelike;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::enums::FormType;
use sec_fetcher::network::{fetch_edgar_master_index, SecClient};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use strum::IntoEnumIterator;

/// Maximum number of quarters to scan backwards when looking for rare variants.
/// 8 quarters = 2 years of history.
const MAX_LOOKBACK_QUARTERS: u8 = 8;

/// A form type must appear at least this many times in the sampled quarter to
/// require a named `FormType` variant.  Lower this threshold as coverage improves.
const MINIMUM_FILINGS_THRESHOLD: usize = 2_000;

/// Returns the most recently fully-completed EDGAR calendar quarter as
/// `(year, quarter)`.
///
/// EDGAR finalises a quarter's master index within a few days of quarter-end,
/// so the previous calendar quarter is always safe to query.
///
/// Examples (evaluated at the given wall-clock date):
/// - March 17, 2026  →  (2025, 4)
/// - April 15, 2026  →  (2026, 1)
/// - July  20, 2026  →  (2026, 2)
/// - Oct    5, 2026  →  (2026, 3)
fn last_completed_quarter() -> (u16, u8) {
    let today = chrono::Local::now().date_naive();
    let month = today.month();
    let year = today.year() as u16;
    let current_q = ((month - 1) / 3 + 1) as u8;
    if current_q == 1 {
        (year - 1, 4)
    } else {
        (year, current_q - 1)
    }
}

fn prev_quarter(year: u16, quarter: u8) -> (u16, u8) {
    if quarter == 1 {
        (year - 1, 4)
    } else {
        (year, quarter - 1)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (start_year, start_quarter) = last_completed_quarter();

    println!("sec-fetcher FormType coverage check");
    println!(
        "Starting from {} Q{}; scanning back up to {} quarters",
        start_year, start_quarter, MAX_LOOKBACK_QUARTERS
    );
    println!(
        "Reverse-check threshold: {} filings in most recent quarter",
        MINIMUM_FILINGS_THRESHOLD
    );
    println!();

    let cfg = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&cfg)?;

    // ── Scan quarters until all named variants are found (or budget exhausted) ──

    // For the forward check: record which quarter each variant was first seen in.
    let mut found_in: HashMap<FormType, (u16, u8)> = HashMap::new();
    // For the reverse check: counts from only the most recent quarter.
    let mut recent_counts: HashMap<String, usize> = HashMap::new();
    // Named variants not yet observed in any scanned quarter.
    // Retired variants are excluded — they are not expected in recent EDGAR data.
    let mut not_yet_found: HashSet<FormType> = FormType::iter()
        .filter(|ft| !ft.is_retired())
        .collect();
    let active_count = not_yet_found.len();

    let (mut year, mut quarter) = (start_year, start_quarter);
    let mut quarters_scanned = 0u8;
    #[allow(unused_assignments)] // initial value is always overwritten in loop
    let mut oldest_quarter: (u16, u8) = (start_year, start_quarter);

    loop {
        print!("  Downloading {} Q{} ... ", year, quarter);
        std::io::stdout().flush()?;

        let entries = fetch_edgar_master_index(&client, year, quarter).await?;

        let mut q_counts: HashMap<String, usize> = HashMap::new();
        for entry in &entries {
            *q_counts.entry(entry.form_type.clone()).or_insert(0) += 1;
        }

        if quarters_scanned == 0 {
            recent_counts = q_counts.clone();
        }
        oldest_quarter = (year, quarter);

        let newly_found: Vec<FormType> = not_yet_found
            .iter()
            .filter(|ft| q_counts.contains_key(ft.as_edgar_str()))
            .cloned()
            .collect();
        let newly_count = newly_found.len();
        for ft in newly_found {
            found_in.insert(ft.clone(), (year, quarter));
            not_yet_found.remove(&ft);
        }

        quarters_scanned += 1;
        println!(
            "{} filings  ({} new, {}/{} found total)",
            entries.len(),
            newly_count,
            found_in.len(),
            active_count
        );

        if not_yet_found.is_empty() || quarters_scanned >= MAX_LOOKBACK_QUARTERS {
            break;
        }
        (year, quarter) = prev_quarter(year, quarter);
    }

    println!();
    if quarters_scanned == 1 {
        println!("Scanned 1 quarter: {} Q{}", start_year, start_quarter);
    } else {
        println!(
            "Scanned {} quarters: {} Q{} → {} Q{}",
            quarters_scanned,
            start_year,
            start_quarter,
            oldest_quarter.0,
            oldest_quarter.1
        );
    }
    println!();

    let mut any_failure = false;

    // ── Forward: named variants → EDGAR ──────────────────────────────────────

    // Variants found but only in an older quarter (rare, not necessarily wrong).
    let mut found_in_history: Vec<(String, u16, u8)> = found_in
        .iter()
        .filter(|(_, &(y, q))| (y, q) != (start_year, start_quarter))
        .map(|(ft, &(y, q))| (ft.to_string(), y, q))
        .collect();
    found_in_history.sort();

    // Variants never seen in any scanned quarter → likely typo or retired form.
    let mut never_found: Vec<String> = not_yet_found
        .iter()
        .map(|ft| format!("  \"{}\"\t(FormType::{:?})", ft.as_edgar_str(), ft))
        .collect();
    never_found.sort();

    if never_found.is_empty() {
        println!(
            "OK   forward: all {} active variants found across {} quarter(s)",
            active_count,
            quarters_scanned
        );
    } else {
        any_failure = true;
        println!(
            "FAIL forward: {} variant(s) not found in any of the last {} quarters:",
            never_found.len(),
            quarters_scanned
        );
        println!("     (typo? check src/enums/form_type_enum.rs; or add retired = \"true\" prop)");
        for line in &never_found {
            println!("{}", line);
        }
    }

    // Retired variants — known to not appear in recent data, shown as INFO.
    let mut retired_list: Vec<String> = FormType::iter()
        .filter(|ft| ft.is_retired())
        .map(|ft| format!("  \"{}\"\t(FormType::{:?})", ft.as_edgar_str(), ft))
        .collect();
    retired_list.sort();
    if !retired_list.is_empty() {
        println!();
        println!(
            "INFO forward: {} variant(s) marked retired (skipped in forward check):",
            retired_list.len()
        );
        for line in &retired_list {
            println!("{}", line);
        }
    }

    if !found_in_history.is_empty() {
        println!();
        println!(
            "INFO forward: {} variant(s) absent from {} Q{} but present in earlier quarters:",
            found_in_history.len(),
            start_year,
            start_quarter
        );
        for (edgar_str, y, q) in &found_in_history {
            println!("  \"{}\"  (last seen {} Q{})", edgar_str, y, q);
        }
    }

    // ── Reverse: EDGAR → named variants (most recent quarter only) ───────────

    let mut unaccounted: Vec<(String, usize)> = recent_counts
        .iter()
        .filter(|(_, &count)| count >= MINIMUM_FILINGS_THRESHOLD)
        .filter(|(form, _)| matches!(form.parse::<FormType>().unwrap(), FormType::Other(_)))
        .map(|(form, &count)| (form.clone(), count))
        .collect();
    unaccounted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    println!();
    if unaccounted.is_empty() {
        println!(
            "OK   reverse: all form types >={} filings in {} Q{} have named variants",
            MINIMUM_FILINGS_THRESHOLD, start_year, start_quarter
        );
    } else {
        any_failure = true;
        println!(
            "FAIL reverse: {} form type(s) >={} filings in {} Q{} without a named variant:",
            unaccounted.len(),
            MINIMUM_FILINGS_THRESHOLD,
            start_year,
            start_quarter
        );
        println!("     (add to src/enums/form_type_enum.rs then lower the threshold)");
        for (form, count) in &unaccounted {
            println!("  {:>8} filings   \"{}\"", count, form);
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────

    let unique_recent = recent_counts.len();
    let named_in_recent = recent_counts
        .keys()
        .filter(|f| !matches!(f.parse::<FormType>().unwrap(), FormType::Other(_)))
        .count();
    let found_in_current = found_in
        .iter()
        .filter(|(_, &(y, q))| (y, q) == (start_year, start_quarter))
        .count();

    println!();
    println!("Summary:");
    println!(
        "  {} named variants in crate ({} active, {} retired)  |  {} found in {} Q{}  |  {} required looking back",
        FormType::iter().count(),
        active_count,
        FormType::iter().filter(|ft| ft.is_retired()).count(),
        found_in_current,
        start_year,
        start_quarter,
        found_in_history.len()
    );
    println!(
        "  {} unique form types in {} Q{}  ({} named  /  {} Other)",
        unique_recent,
        start_year,
        start_quarter,
        named_in_recent,
        unique_recent - named_in_recent
    );
    println!();

    if any_failure {
        println!("COVERAGE GAPS FOUND.  See above.");
        std::process::exit(1);
    } else {
        println!("All checks passed.");
    }

    Ok(())
}
