//! Join 10-K audit rows to fiscal periods via fundamentals accessions.
//!
//! Reads the audit CSV (default `data/tenk_audit.csv`) and the per-ticker
//! fundamentals CSVs (default
//! `data/latest-us-gaap/us-gaap-fundamentals/`), and writes an enriched CSV
//! (default `data/tenk_audit_periods.csv`) with eight appended columns:
//! `fy`, `fp`, `period_key`, `is_canonical`, `canonical_accession`,
//! `join_status`, `superseded_by`, and `pairing_rule`.
//!
//! Grouping rule per `(cik_or_ticker, fy)` period: latest `filed_date` among
//! adequate rows is canonical (restated figures supersede the original);
//! when no row is adequate, the latest row overall is canonical so every
//! group still has a representative. Rows with no fundamentals match
//! (pre-XBRL era, legacy tickers) keep blank `fy`/`fp`/`period_key` and
//! stand as their own canonical row. No heuristic pairing is performed:
//! a superseded original without its own fundamentals match is never
//! assigned to another filing's period, so wrong-period association is
//! impossible by construction. `pairing_rule` is `fundamentals-join` for
//! joined rows and `none` otherwise; `superseded_by` is always empty
//! (columns kept for schema stability).
//!
//! Read-only over fundamentals — nothing outside the output CSV is written.
//! There are **no network calls**.
//!
//! ```sh
//! cargo run -p sec-fetcher-tenk-parser-audit --bin tenk-period-join --release
//! ```
//!
//! Defaults resolve from the shared data root; run with `--help` for paths.

use clap::{CommandFactory, FromArgMatches, Parser};
use sec_fetcher::parsers::{
    resolve_fundamentals_dir, resolve_tenk_audit_csv, resolve_tenk_periods_csv,
};
use sec_fetcher_tenk_parser_audit::{
    load_period_map, normalize_accession, period_group_key, pick_canonical,
};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::info;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "tenk-period-join",
    about = "Join 10-K audit rows to fiscal periods for amendment grouping"
)]
struct Args {
    /// Audit CSV to enrich. Defaults to `data/tenk_audit.csv`.
    #[arg(long, short = 'a')]
    audit: Option<PathBuf>,

    /// Per-ticker fundamentals directory. Defaults to
    /// `data/latest-us-gaap/us-gaap-fundamentals`.
    #[arg(long, short = 'f')]
    fundamentals_dir: Option<PathBuf>,

    /// Path to the output CSV. Defaults to `data/tenk_audit_periods.csv`.
    #[arg(long, short = 'o')]
    output: Option<PathBuf>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy(),
        )
        .init();

    let args = Args::from_arg_matches(
        &Args::command()
            .mut_arg("audit", |a| {
                a.help(format!(
                    "Audit CSV to enrich. Defaults to `{}`.",
                    resolve_tenk_audit_csv(None)
                ))
            })
            .mut_arg("fundamentals_dir", |a| {
                a.help(format!(
                    "Per-ticker fundamentals directory. Defaults to `{}`.",
                    resolve_fundamentals_dir(None)
                ))
            })
            .mut_arg("output", |a| {
                a.help(format!(
                    "Path to the output CSV. Defaults to `{}`.",
                    resolve_tenk_periods_csv(None)
                ))
            })
            .get_matches(),
    )?;
    let audit_path = PathBuf::from(resolve_tenk_audit_csv(
        args.audit.as_ref().and_then(|p| p.to_str()),
    ));
    let fundamentals_dir = PathBuf::from(resolve_fundamentals_dir(
        args.fundamentals_dir.as_ref().and_then(|p| p.to_str()),
    ));
    let output = PathBuf::from(resolve_tenk_periods_csv(
        args.output.as_ref().and_then(|p| p.to_str()),
    ));

    // ── Load the accession→(fy, fp) map ─────────────────────────────────────
    let period_map = load_period_map(&fundamentals_dir);
    info!(
        "Period map: {} accession(s) loaded from {:?}",
        period_map.len(),
        fundamentals_dir
    );

    // ── Read the audit CSV ──────────────────────────────────────────────────
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(&audit_path)?;
    let headers = rdr.headers()?.clone();
    let find = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("audit CSV is missing column `{name}`"))
    };
    let (ticker_idx, cik_idx, acc_idx, filed_idx, adequate_idx) = (
        find("ticker")?,
        find("cik")?,
        find("accession_number")?,
        find("filed_date")?,
        find("is_adequate")?,
    );
    let records: Vec<csv::StringRecord> = rdr.records().collect::<Result<_, _>>()?;
    let total = records.len();
    info!("Audit rows: {total} loaded from {audit_path:?}");

    // ── Join ──────────────────────────────────────────────────────────────────
    let mut fys: Vec<String> = Vec::with_capacity(total);
    let mut fps: Vec<String> = Vec::with_capacity(total);
    let mut period_keys: Vec<String> = Vec::with_capacity(total);
    let mut join_status: Vec<&str> = Vec::with_capacity(total);
    let mut pairing_rule: Vec<&str> = Vec::with_capacity(total);
    let superseded_by: Vec<&str> = vec![""; total];
    for rec in records.iter() {
        let get = |idx: usize| rec.get(idx).unwrap_or_default();
        let (fy, fp) = period_map
            .get(&normalize_accession(get(acc_idx)))
            .cloned()
            .unwrap_or_default();
        if fy.is_empty() {
            fys.push(String::new());
            fps.push(String::new());
            period_keys.push(String::new());
            join_status.push("missing-no-fundamentals");
            pairing_rule.push("none");
        } else {
            let key = period_group_key(get(cik_idx), get(ticker_idx), &fy);
            fys.push(fy);
            fps.push(fp);
            period_keys.push(key);
            join_status.push("joined");
            pairing_rule.push("fundamentals-join");
        }
    }

    // ── No heuristic pairing ──────────────────────────────────────────────
    //
    // A superseded original without its own fundamentals match keeps blank
    // `fy`/`fp`/`period_key` and stands alone. Assigning it to a joined
    // amendment's period by date proximity risks wrong-period association,
    // which is not tolerated: grouping must come from fundamentals only.

    // ── Group + pick canonical row per group ────────────────────────────────
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, key) in period_keys.iter().enumerate() {
        if !key.is_empty() {
            groups.entry(key.clone()).or_default().push(i);
        }
    }
    let mut is_canonical = vec![false; total];
    let mut canonical_accession = vec![""; total];
    for idxs in groups.values() {
        let dates: Vec<&str> = idxs
            .iter()
            .map(|&i| records[i].get(filed_idx).unwrap_or_default())
            .collect();
        let adequate: Vec<bool> = idxs
            .iter()
            .map(|&i| {
                records[i]
                    .get(adequate_idx)
                    .unwrap_or_default()
                    .eq_ignore_ascii_case("true")
            })
            .collect();
        if let Some(local) = pick_canonical(&dates, &adequate) {
            let winner = idxs[local];
            is_canonical[winner] = true;
            canonical_accession[winner] = records[winner].get(acc_idx).unwrap_or_default();
            // Non-winners point at the winner.
            for &i in idxs {
                if i != winner {
                    canonical_accession[i] = records[winner].get(acc_idx).unwrap_or_default();
                }
            }
        }
    }
    // Ungrouped rows stand alone as their own canonical row.
    for i in 0..total {
        if period_keys[i].is_empty() {
            is_canonical[i] = true;
            canonical_accession[i] = records[i].get(acc_idx).unwrap_or_default();
        }
    }

    let joined = join_status.iter().filter(|&&s| s == "joined").count();
    let canonical_count = is_canonical.iter().filter(|&&b| b).count();
    info!(
        "Groups: {} period(s); joined {joined}/{total}; paired 0 (disabled); canonical rows: {canonical_count}",
        groups.len()
    );

    // ── Write enriched CSV atomically ───────────────────────────────────────
    let mut header: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
    header.extend(
        [
            "fy",
            "fp",
            "period_key",
            "is_canonical",
            "canonical_accession",
            "join_status",
            "superseded_by",
            "pairing_rule",
        ]
        .iter()
        .map(|h| h.to_string()),
    );
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_path = output.with_extension("csv.tmp");
    {
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_path(&tmp_path)?;
        wtr.write_record(&header)?;
        for i in 0..total {
            let mut row: Vec<&str> = records[i].iter().collect();
            row.push(&fys[i]);
            row.push(&fps[i]);
            row.push(&period_keys[i]);
            row.push(if is_canonical[i] { "true" } else { "false" });
            row.push(canonical_accession[i]);
            row.push(join_status[i]);
            row.push(superseded_by[i]);
            row.push(pairing_rule[i]);
            wtr.write_record(&row)?;
        }
        wtr.flush()?;
    }
    std::fs::rename(&tmp_path, &output)?;
    info!("Wrote {total} enriched row(s) to {output:?}");

    Ok(())
}
