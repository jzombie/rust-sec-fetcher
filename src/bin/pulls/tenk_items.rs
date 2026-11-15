//! Download 10-K filings from EDGAR to a local directory.
//!
//! For every ticker in the full company-tickers dataset, fetches all 10-K and
//! 10-K405 filings and downloads the best available document for each one.
//! Amendment filings (`10-K/A`, `10-K405/A`) are included by default because
//! they carry corrections and restatements. Pass `--skip-amendments` to leave
//! them out, and add `--prune-amendments` to also delete previously-downloaded
//! amendment files. A plain pull never deletes anything on disk.
//! Documents are saved to `<output_dir>/<TICKER>/<ACCESSION>.<ext>`.
//!
//! A filing manifest (`.tenk_manifest.csv`) is maintained inside the output
//! directory: one `ticker,cik,accession,filed_date,form,primary_document` row
//! per filing in every fetched filing list — including skipped and
//! already-on-disk files. Reruns merge into it without duplicating, so a
//! resume pass over an existing corpus backfills the manifest with no
//! re-downloads. `tenk-parser-audit` reads it to tag `cik`, `filed_date`,
//! and `form_type` on each audit row.
//!
//! This binary does **no parsing or audit-CSV generation** — it only pulls
//! raw documents.  Run `tenk-parser-audit` afterwards to evaluate the parser
//! against the downloaded corpus.
//!
//! # Usage
//!
//! ```sh
//! cargo run --bin pull-tenk-items --release
//! ```
//!
//! # Resuming
//!
//! If a file already exists on disk for a given `(ticker, accession_number)`,
//! it is skipped.  This makes the run safe to interrupt and resume.
//!
//! To re-fetch all documents from scratch, delete the output directory or
//! use a fresh `--output-dir`.

use clap::{CommandFactory, FromArgMatches, Parser};
use futures::StreamExt;
use sec_fetcher::{
    config::ConfigManager,
    network::{SecClient, fetch_10k_filings, fetch_best_10k_document, fetch_company_tickers},
};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{error, info};

// ── Global counters ───────────────────────────────────────────────────────────

static GLOBAL_TICKERS_OK: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_TICKERS_ERR: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_DOWNLOADED: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_SKIPPED: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_FAILED: AtomicUsize = AtomicUsize::new(0);

// ── Accession→form manifest ───────────────────────────────────────────────────
//
// One row per filing in every fetched filing list — including skipped and
// already-on-disk files, since the rows come from the filing-list metadata,
// not from downloads. Merged key-first into the existing manifest (when
// present) so reruns and partial runs never duplicate or lose rows.

/// A single accession→form manifest record.
#[derive(Clone, Debug)]
struct ManifestRow {
    ticker: String,
    cik: String,
    accession: String,
    filed_date: String,
    form: String,
    primary_document: String,
}

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "pull-tenk-items",
    about = "Download 10-K filings from EDGAR to a local directory"
)]
struct Args {
    /// Output directory (default resolved from the shared data root).
    #[arg(long, short = 'o')]
    output_dir: Option<PathBuf>,

    /// Skip 10-K/A and 10-K405/A amendment filings (never download them).
    /// Amendments are included by default because they carry corrections and
    /// restatements — the unamended original alone can be stale. They often
    /// contain only the amended excerpt, so expect them to score inadequate
    /// as standalone audit rows.
    #[arg(long, default_value_t = false)]
    skip_amendments: bool,

    /// Delete previously-downloaded amendment files (`10-K/A`, `10-K405/A`)
    /// from the output directory. Requires `--skip-amendments`. Off by
    /// default — a plain pull never deletes anything.
    #[arg(long, default_value_t = false, requires = "skip_amendments")]
    prune_amendments: bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy()
                .add_directive("html5ever=off".parse().unwrap())
                .add_directive("selectors=off".parse().unwrap()),
        )
        .init();

    let args = Args::from_arg_matches(
        &Args::command()
            .mut_arg("output_dir", |a| {
                a.help(format!(
                    "Output directory. Defaults to `{}`. Pass explicitly for an alternate location.",
                    sec_fetcher_shared::resolve_tenk_items_dir(None)
                ))
            })
            .get_matches(),
    )?;
    let output_dir = PathBuf::from(sec_fetcher_shared::resolve_tenk_items_dir(
        args.output_dir.as_ref().and_then(|p| p.to_str()),
    ));
    let include_amendments = !args.skip_amendments;
    // `requires = "skip_amendments"` on the flag guarantees prune implies
    // skip, so no runtime combination check is needed.
    let prune_amendments = args.prune_amendments;

    let config_manager = ConfigManager::load()?;
    let max_concurrent = config_manager
        .get_config()
        .max_concurrent
        .unwrap_or(1)
        .max(1);
    let client = Arc::new(SecClient::from_config_manager(&config_manager)?);
    info!(
        "Pulling 10-K documents into {:?} (max_concurrent={}, include_amendments={}, prune_amendments={})",
        output_dir, max_concurrent, include_amendments, prune_amendments,
    );

    // ── Ensure output directory exists ────────────────────────────────────────

    std::fs::create_dir_all(&output_dir)?;

    // ── Fetch full ticker list (deduplicated by CIK) ──────────────────────────

    let all_tickers = fetch_company_tickers(&client, false).await?;
    let mut seen_ciks = std::collections::HashSet::new();
    let company_tickers: Vec<_> = all_tickers
        .into_iter()
        .filter(|t| seen_ciks.insert(t.cik.to_string()))
        .collect();
    let total_tickers = company_tickers.len();
    info!("Total unique CIKs to process: {}", total_tickers);

    // ── Track current ticker index globally for progress ──────────────────────

    let ticker_index = Arc::new(AtomicUsize::new(0));

    // ── Pipelined producer / consumer ─────────────────────────────────────────
    //
    // Each per-ticker future:
    //   1. Fetches the filing list for that ticker.
    //   2. For each filing not yet on disk, downloads and saves the document.
    //   3. Logs a summary line with per-ticker status.

    let log_dir = output_dir.clone();

    let producer = async move {
        let mut stream = futures::stream::iter(company_tickers.into_iter())
            .map(|entry| {
                let output_dir = output_dir.clone();
                let client = Arc::clone(&client);
                let idx = ticker_index.fetch_add(1, Ordering::Relaxed);
                async move {
                    let ticker = entry.symbol.to_string();
                    let cik_str = entry.cik.to_string();
                    let ticker_idx = idx + 1; // 1-based for display

                    // ── Per-ticker counters ───────────────────────────────────
                    let mut downloaded: usize = 0;
                    let mut skipped: usize = 0;
                    let mut failed: usize = 0;
                    // Amendment accession stems seen this run — pruned in one
                    // pass after the loop instead of scanning the directory
                    // per filing.
                    let mut prune_stems: std::collections::HashSet<String> =
                        std::collections::HashSet::new();

                    let filings = match fetch_10k_filings(&client, entry.cik).await {
                        Ok(f) => f,
                        Err(e) => {
                            error!(
                                "[{ticker}] [{ticker_idx}/{total_tickers}] filing list failed: {e}"
                            );
                            GLOBAL_TICKERS_ERR.fetch_add(1, Ordering::Relaxed);
                            return Vec::new();
                        }
                    };

                    // Manifest rows come from the filing-list metadata, so
                    // skipped and already-on-disk filings are recorded too —
                    // that is what lets the offline audit tag amendments.
                    // Keys are sanitized exactly like the on-disk paths so
                    // the auditor's path-derived lookups match.
                    let manifest_rows: Vec<ManifestRow> = filings
                        .iter()
                        .map(|filing| ManifestRow {
                            ticker: sanitize_dir_name(&ticker),
                            cik: cik_str.clone(),
                            accession: sanitize_dir_name(
                                &filing.accession_number.to_string(),
                            ),
                            filed_date: filing
                                .filing_date
                                .map(|d| d.to_string())
                                .unwrap_or_default(),
                            form: filing.form.clone(),
                            primary_document: filing.primary_document.display().to_string(),
                        })
                        .collect();

                    let total_filings = filings.len();
                    info!(
                        "[{ticker}] [{ticker_idx}/{total_tickers}] found {total_filings} 10-K filing(s)"
                    );

                    for filing in &filings {
                        let acc = filing.accession_number.to_string();
                        let date_str = filing
                            .filing_date
                            .map(|d| d.to_string())
                            .unwrap_or_else(|| "unknown-date".to_string());

                        // Amendments are included by default (corrections and
                        // restatements); with `--skip-amendments` they are
                        // skipped here and, with `--prune-amendments`,
                        // previously-downloaded copies are removed in one
                        // pass after the loop. A plain pull never deletes.
                        if !include_amendments
                            && filing.form.trim().to_ascii_uppercase().ends_with("/A")
                        {
                            if prune_amendments {
                                prune_stems.insert(sanitize_dir_name(&acc));
                            }
                            skipped += 1;
                            info!(
                                "[{ticker}]   {date_str}  {acc}  ({}) — SKIP (amendment)",
                                filing.form,
                            );
                            continue;
                        }

                        // Determine filename on disk.
                        let ticker_dir = output_dir.join(sanitize_dir_name(&ticker));
                        let ext = std::path::Path::new(&filing.primary_document)
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("txt");
                        let dest = ticker_dir.join(format!("{}.{}", sanitize_dir_name(&acc), ext));

                        // Skip if already downloaded.
                        if dest.exists() {
                            skipped += 1;
                            info!(
                                "[{ticker}]   {date_str}  {acc}  — SKIP (exists on disk)"
                            );
                            continue;
                        }

                        info!(
                            "[{ticker}]   {date_str}  {acc}  — DOWNLOADING ..."
                        );

                        let bytes = match fetch_best_10k_document(&client, filing).await {
                            Ok(b) => b,
                            Err(e) => {
                                failed += 1;
                                error!(
                                    "[{ticker}]   {date_str}  {acc}  — FAILED: {e}"
                                );
                                continue;
                            }
                        };

                        // Save to disk.
                        if let Err(e) = std::fs::create_dir_all(&ticker_dir) {
                            failed += 1;
                            error!("[{ticker}]   {date_str}  {acc}  — FAILED: cannot create dir {ticker_dir:?}: {e}");
                            continue;
                        }
                        if let Err(e) = std::fs::write(&dest, bytes.as_ref()) {
                            failed += 1;
                            error!("[{ticker}]   {date_str}  {acc}  — FAILED: cannot write {dest:?}: {e}");
                            continue;
                        }

                        downloaded += 1;
                        info!(
                            "[{ticker}]   {date_str}  {acc}  — saved ({size}B)",
                            size = bytes.len()
                        );
                    }

                    // ── Prune skipped amendments in one pass ────────────────────
                    //
                    // Single `read_dir` scan per ticker (not per filing), run
                    // after all downloads so a mid-loop `remove_dir` can never
                    // force later filings to recreate the directory.
                    if !prune_stems.is_empty() {
                        let ticker_dir = output_dir.join(sanitize_dir_name(&ticker));
                        if ticker_dir.is_dir()
                            && let Ok(entries) = std::fs::read_dir(&ticker_dir)
                        {
                            for e in entries.flatten() {
                                let p = e.path();
                                if p.is_file()
                                    && let Some(stem) = p.file_stem().and_then(|s| s.to_str())
                                    && prune_stems.contains(stem)
                                {
                                    let _ = std::fs::remove_file(&p);
                                }
                            }
                            let _ = std::fs::remove_dir(&ticker_dir);
                        }
                    }

                    // ── Per-ticker summary ────────────────────────────────────
                    GLOBAL_DOWNLOADED.fetch_add(downloaded, Ordering::Relaxed);
                    GLOBAL_SKIPPED.fetch_add(skipped, Ordering::Relaxed);
                    GLOBAL_FAILED.fetch_add(failed, Ordering::Relaxed);
                    GLOBAL_TICKERS_OK.fetch_add(1, Ordering::Relaxed);

                    info!(
                        "[{ticker}] [{ticker_idx}/{total_tickers}] done — \
                         {total_filings} filing(s): \
                         {downloaded} downloaded, \
                         {skipped} skipped, \
                         {failed} failed"
                    );

                    manifest_rows
                }
            })
            .buffer_unordered(max_concurrent);

        let mut fresh_rows: Vec<ManifestRow> = Vec::new();
        let mut tickers_since_flush = 0usize;
        let dir_str = output_dir.to_string_lossy().into_owned();
        let manifest_path = std::path::PathBuf::from(
            sec_fetcher_shared::resolve_tenk_manifest_csv(Some(&dir_str)),
        );
        while let Some(rows) = stream.next().await {
            fresh_rows.extend(rows);
            tickers_since_flush += 1;
            // Checkpoint so an interrupted run keeps rows for all completed
            // tickers instead of losing the whole run's manifest progress.
            if tickers_since_flush >= 250 && !fresh_rows.is_empty() {
                let batch = std::mem::take(&mut fresh_rows);
                if let Err(e) = merge_manifest_rows(&manifest_path, batch) {
                    error!("Manifest {:?} checkpoint write failed: {e}", manifest_path);
                }
                tickers_since_flush = 0;
            }
        }

        // ── Merge + write the accession→form manifest ───────────────────────
        //
        // Keyed merge into any existing manifest: reruns overwrite identical
        // rows, partial runs keep rows for tickers that failed this time.
        // The final merge also persists whatever the checkpoints left behind.
        match merge_manifest_rows(&manifest_path, std::mem::take(&mut fresh_rows)) {
            Ok((total, fresh)) => info!(
                "Manifest {:?}: {} filing(s) recorded ({} fresh this run)",
                manifest_path, total, fresh
            ),
            Err(e) => error!(
                "Manifest {:?} could not be written: {e} (corpus is unaffected)",
                manifest_path
            ),
        }
    };

    producer.await;

    // ── Final report ──────────────────────────────────────────────────────────

    let final_downloaded = GLOBAL_DOWNLOADED.load(Ordering::Relaxed);
    let final_skipped = GLOBAL_SKIPPED.load(Ordering::Relaxed);
    let final_failed = GLOBAL_FAILED.load(Ordering::Relaxed);
    let final_ok = GLOBAL_TICKERS_OK.load(Ordering::Relaxed);
    let final_err = GLOBAL_TICKERS_ERR.load(Ordering::Relaxed);

    info!(
        "═══ Done — documents saved under {:?} ═══\n\
         Tickers processed: {final_ok} OK, {final_err} failed\n\
         Documents: {final_downloaded} downloaded, {final_skipped} skipped, {final_failed} failed",
        log_dir,
    );
    Ok(())
}

// TODO: Move into a shared utility?
/// Replace characters unsafe for filesystem paths with `_`.
fn sanitize_dir_name(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect()
}

// ── Manifest read / merge / write ─────────────────────────────────────────────
//
// CSV is handled with the `csv` crate, not hand-rolled parsing: fields are
// quoted on write when they contain commas, quotes, or newlines, and the
// reader decodes them back. A line-splitting parser would corrupt records
// with embedded newlines.

const MANIFEST_HEADER: [&str; 6] = [
    "ticker",
    "cik",
    "accession",
    "filed_date",
    "form",
    "primary_document",
];

/// Load an existing manifest into a map keyed by `(ticker, accession)`.
/// Missing file → empty map. Malformed lines are skipped.
fn load_manifest_map(
    path: &std::path::Path,
) -> std::collections::HashMap<(String, String), ManifestRow> {
    let mut map = std::collections::HashMap::new();
    let Ok(mut rdr) = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
    else {
        return map;
    };
    for record in rdr.records().flatten() {
        if record.len() < 6 {
            continue;
        }
        let key = (record[0].to_string(), record[2].to_string());
        map.insert(
            key,
            ManifestRow {
                ticker: record[0].to_string(),
                cik: record[1].to_string(),
                accession: record[2].to_string(),
                filed_date: record[3].to_string(),
                form: record[4].to_string(),
                primary_document: record[5].to_string(),
            },
        );
    }
    map
}

/// Merge `fresh` rows into the manifest at `path` (keyed by
/// `(ticker, accession)`, so reruns never duplicate) and write it back
/// sorted for determinism. The write goes to a sibling temp file followed
/// by an atomic rename, so a crash mid-write never truncates the existing
/// manifest. Returns `(total_rows, fresh_rows)`.
fn merge_manifest_rows(
    path: &std::path::Path,
    fresh: Vec<ManifestRow>,
) -> std::io::Result<(usize, usize)> {
    let mut map = load_manifest_map(path);
    let fresh_count = fresh.len();
    for row in fresh {
        map.insert((row.ticker.clone(), row.accession.clone()), row);
    }
    let mut rows: Vec<&ManifestRow> = map.values().collect();
    rows.sort_by(|a, b| (&a.ticker, &a.accession).cmp(&(&b.ticker, &b.accession)));
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension("csv.tmp");
    {
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_path(&tmp_path)?;
        wtr.write_record(MANIFEST_HEADER)?;
        for r in &rows {
            wtr.write_record([
                &r.ticker,
                &r.cik,
                &r.accession,
                &r.filed_date,
                &r.form,
                &r.primary_document,
            ])?;
        }
        wtr.flush()?;
    }
    std::fs::rename(&tmp_path, path)?;
    Ok((rows.len(), fresh_count))
}

#[cfg(test)]
mod manifest_tests {
    use super::*;

    fn row(ticker: &str, acc: &str, form: &str) -> ManifestRow {
        ManifestRow {
            ticker: ticker.into(),
            cik: "1".into(),
            accession: acc.into(),
            filed_date: "2024-01-01".into(),
            form: form.into(),
            primary_document: "doc.htm".into(),
        }
    }

    #[test]
    fn rerun_does_not_duplicate() {
        let dir = std::env::temp_dir().join("tenk_manifest_test_dedupe");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("manifest.csv");
        let rows = vec![row("ADM", "0000007084-05-000350", "10-K")];
        let (total, _) = merge_manifest_rows(&path, rows.clone()).unwrap();
        assert_eq!(total, 1);
        let (total2, _) = merge_manifest_rows(&path, rows).unwrap();
        assert_eq!(total2, 1, "rerun must overwrite, not append");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn partial_run_keeps_old_rows() {
        let dir = std::env::temp_dir().join("tenk_manifest_test_partial");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("manifest.csv");
        merge_manifest_rows(&path, vec![row("A", "acc-1", "10-K")]).unwrap();
        let (total, _) = merge_manifest_rows(&path, vec![row("B", "acc-2", "10-K/A")]).unwrap();
        assert_eq!(
            total, 2,
            "rows for tickers that failed this run must survive"
        );
        let map = load_manifest_map(&path);
        assert_eq!(map[&("B".to_string(), "acc-2".to_string())].form, "10-K/A");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_file_loads_empty() {
        let map = load_manifest_map(std::path::Path::new("/nonexistent-xyz/manifest.csv"));
        assert!(map.is_empty());
    }

    #[test]
    fn csv_round_trip_with_comma_field() {
        let dir = std::env::temp_dir().join("tenk_manifest_test_quote");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("manifest.csv");
        let mut r = row("A,B", "acc-1", "10-K");
        r.primary_document = "doc,name.htm".into();
        merge_manifest_rows(&path, vec![r]).unwrap();
        let map = load_manifest_map(&path);
        assert_eq!(map.len(), 1);
        assert_eq!(
            map[&("A,B".to_string(), "acc-1".to_string())].primary_document,
            "doc,name.htm"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn csv_round_trip_with_multiline_field() {
        // A line-splitting parser would fracture this record across two
        // iterations and drop it; the `csv` reader must survive it.
        let dir = std::env::temp_dir().join("tenk_manifest_test_multiline");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("manifest.csv");
        let mut r = row("A", "acc-1", "10-K");
        r.primary_document = "line1\nline2".into();
        merge_manifest_rows(&path, vec![r]).unwrap();
        let map = load_manifest_map(&path);
        assert_eq!(map.len(), 1);
        assert_eq!(
            map[&("A".to_string(), "acc-1".to_string())].primary_document,
            "line1\nline2"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn keys_match_sanitized_disk_paths() {
        // Tickers with path-unsafe chars (e.g. "BF/A") live on disk under
        // sanitized names; manifest keys must use the same form or the
        // auditor's path-derived lookup silently misses.
        assert_eq!(sanitize_dir_name("BF/A"), "BF_A");
        let dir = std::env::temp_dir().join("tenk_manifest_test_sanitize");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("manifest.csv");
        let r = ManifestRow {
            ticker: sanitize_dir_name("BF/A"),
            cik: "1".into(),
            accession: sanitize_dir_name("0001-23-000001"),
            filed_date: "2024-01-01".into(),
            form: "10-K".into(),
            primary_document: "doc.htm".into(),
        };
        merge_manifest_rows(&path, vec![r]).unwrap();
        let map = load_manifest_map(&path);
        assert!(map.contains_key(&("BF_A".to_string(), "0001-23-000001".to_string())));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn atomic_write_leaves_no_temp_file() {
        let dir = std::env::temp_dir().join("tenk_manifest_test_atomic");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("manifest.csv");
        merge_manifest_rows(&path, vec![row("A", "acc-1", "10-K")]).unwrap();
        assert!(path.is_file());
        assert!(
            !dir.join("manifest.csv.tmp").exists(),
            "temp file must be renamed away"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
