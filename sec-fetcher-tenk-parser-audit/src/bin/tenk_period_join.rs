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

// ── Pure join helpers (unit-tested) ───────────────────────────────────────────
//
// Extracted from `main` so grouping and canonical selection are covered by
// unit tests. Behavior is identical to the inline loop: fundamentals-only
// grouping, no heuristic pairing.

/// Classify one audit row against the fundamentals period map.
///
/// Returns `(fy, fp, period_key, join_status, pairing_rule)`. Unmatched rows
/// keep blank fiscal identity and stand alone downstream.
fn classify_join(
    accession: &str,
    cik: &str,
    ticker: &str,
    period_map: &HashMap<String, (String, String)>,
) -> (String, String, String, &'static str, &'static str) {
    match period_map.get(&normalize_accession(accession)).cloned() {
        None => (
            String::new(),
            String::new(),
            String::new(),
            "missing-no-fundamentals",
            "none",
        ),
        Some((fy, fp)) => {
            let key = period_group_key(cik, ticker, &fy);
            (fy, fp, key, "joined", "fundamentals-join")
        }
    }
}

/// Assign canonical flags for all rows.
///
/// Groups row indices by non-empty `period_key`, picks the latest adequate
/// row per group via [`pick_canonical`], and marks ungrouped rows canonical
/// on their own. Returns `(is_canonical, canonical_accession)`.
fn assign_canonical(
    period_keys: &[String],
    filed_dates: &[&str],
    adequate: &[bool],
    accessions: &[&str],
) -> (Vec<bool>, Vec<String>) {
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, key) in period_keys.iter().enumerate() {
        if !key.is_empty() {
            groups.entry(key.clone()).or_default().push(i);
        }
    }
    let mut is_canonical = vec![false; period_keys.len()];
    let mut canonical_accession = vec![String::new(); period_keys.len()];
    for idxs in groups.values() {
        let dates: Vec<&str> = idxs.iter().map(|&i| filed_dates[i]).collect();
        let flags: Vec<bool> = idxs.iter().map(|&i| adequate[i]).collect();
        if let Some(local) = pick_canonical(&dates, &flags) {
            let winner = idxs[local];
            is_canonical[winner] = true;
            for &i in idxs {
                canonical_accession[i] = accessions[winner].to_string();
            }
        }
    }
    for (i, key) in period_keys.iter().enumerate() {
        if key.is_empty() {
            is_canonical[i] = true;
            canonical_accession[i] = accessions[i].to_string();
        }
    }
    (is_canonical, canonical_accession)
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
    // Fundamentals only. A superseded original without its own fundamentals
    // match keeps blank fiscal identity and stands alone; assigning it to a
    // joined amendment's period by date proximity risks wrong-period
    // association, which is not tolerated.
    let mut fys: Vec<String> = Vec::with_capacity(total);
    let mut fps: Vec<String> = Vec::with_capacity(total);
    let mut period_keys: Vec<String> = Vec::with_capacity(total);
    let mut join_status: Vec<&str> = Vec::with_capacity(total);
    let mut pairing_rule: Vec<&str> = Vec::with_capacity(total);
    let superseded_by: Vec<&str> = vec![""; total];
    for rec in records.iter() {
        let get = |idx: usize| rec.get(idx).unwrap_or_default();
        let (fy, fp, key, status, rule) =
            classify_join(get(acc_idx), get(cik_idx), get(ticker_idx), &period_map);
        fys.push(fy);
        fps.push(fp);
        period_keys.push(key);
        join_status.push(status);
        pairing_rule.push(rule);
    }

    // ── Group + pick canonical row per group ────────────────────────────────
    let filed_dates: Vec<&str> = records
        .iter()
        .map(|r| r.get(filed_idx).unwrap_or_default())
        .collect();
    let adequate: Vec<bool> = records
        .iter()
        .map(|r| {
            r.get(adequate_idx)
                .unwrap_or_default()
                .eq_ignore_ascii_case("true")
        })
        .collect();
    let accessions: Vec<&str> = records
        .iter()
        .map(|r| r.get(acc_idx).unwrap_or_default())
        .collect();
    let (is_canonical, canonical_accession) =
        assign_canonical(&period_keys, &filed_dates, &adequate, &accessions);

    let joined = join_status.iter().filter(|&&s| s == "joined").count();
    let canonical_count = is_canonical.iter().filter(|&&b| b).count();
    let group_count = {
        let mut seen = std::collections::HashSet::new();
        for key in &period_keys {
            if !key.is_empty() {
                seen.insert(key);
            }
        }
        seen.len()
    };
    info!(
        "Groups: {group_count} period(s); joined {joined}/{total}; paired 0 (disabled); canonical rows: {canonical_count}"
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
            row.push(canonical_accession[i].as_str());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn period_map(pairs: &[(&str, &str, &str)]) -> HashMap<String, (String, String)> {
        pairs
            .iter()
            .map(|(acc, fy, fp)| (normalize_accession(acc), (fy.to_string(), fp.to_string())))
            .collect()
    }

    #[test]
    fn classify_join_matches_fundamentals_only() {
        let map = period_map(&[("0001-01-000001", "2024", "FY")]);
        let (fy, fp, key, status, rule) = classify_join("0001-01-000001", "123", "AAPL", &map);
        assert_eq!(fy, "2024");
        assert_eq!(fp, "FY");
        assert_eq!(key, "123|2024");
        assert_eq!(status, "joined");
        assert_eq!(rule, "fundamentals-join");

        let (fy, fp, key, status, rule) = classify_join("9999-99-999999", "123", "AAPL", &map);
        assert!(fy.is_empty() && fp.is_empty() && key.is_empty());
        assert_eq!(status, "missing-no-fundamentals");
        assert_eq!(rule, "none");
    }

    #[test]
    fn classify_join_accepts_undashed_accessions() {
        let map = period_map(&[("0001-01-000001", "2024", "FY")]);
        let (fy, _, _, status, _) = classify_join("000101000001", "123", "AAPL", &map);
        assert_eq!(fy, "2024");
        assert_eq!(status, "joined");
    }

    #[test]
    fn assign_canonical_prefers_latest_adequate_and_isolates_unmatched() {
        let keys = vec![
            "123|2024".to_string(),
            "123|2024".to_string(),
            String::new(),
        ];
        let dates = vec!["2024-02-01", "2024-08-01", "2024-03-01"];
        let adequate = vec![true, true, false];
        let accs = vec!["acc-1", "acc-2", "acc-3"];
        let (is_canon, canon_acc) = assign_canonical(&keys, &dates, &adequate, &accs);
        assert_eq!(is_canon, vec![false, true, true]);
        assert_eq!(canon_acc, vec!["acc-2", "acc-2", "acc-3"]);
    }

    #[test]
    fn assign_canonical_falls_back_to_latest_when_none_adequate() {
        let keys = vec!["123|2024".to_string(), "123|2024".to_string()];
        let dates = vec!["2024-02-01", "2024-05-01"];
        let adequate = vec![false, false];
        let accs = vec!["acc-1", "acc-2"];
        let (is_canon, canon_acc) = assign_canonical(&keys, &dates, &adequate, &accs);
        assert_eq!(is_canon, vec![false, true]);
        assert_eq!(canon_acc, vec!["acc-2", "acc-2"]);
    }

    #[test]
    fn assign_canonical_empty_input_yields_empty_output() {
        let (is_canon, canon_acc) = assign_canonical(&[], &[], &[], &[]);
        assert!(is_canon.is_empty() && canon_acc.is_empty());
    }
}
