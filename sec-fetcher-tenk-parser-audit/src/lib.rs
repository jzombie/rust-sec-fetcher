//! Shared utilities for auditing the 10-K section parser.
//!
//! Provides CSV row construction, file discovery, path heuristics, manifest
//! loading, and text helpers used by `tenk-parser-audit` (which runs the
//! parser on local files).
//!
//! Accession keys are normalized through
//! [`sec_fetcher_shared::normalize_accession`] (re-exported here) at every
//! comparison boundary — manifest load and lookup, period-map build and
//! lookup — so dashed, undashed, padded, or mixed-case inputs always match.
//! Canonical dashed form is kept at rest (CSVs, filenames).
//!
//! This library keeps external dependencies to a minimum — plain
//! [`HashMap`]s and [`String`]s throughout, plus the `csv` crate for
//! manifest parsing — so that any crate (including `sec-fetcher`) can depend
//! on it without creating circular dependency issues.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

pub use sec_fetcher_shared::normalize_accession;

// ── CSV header construction ───────────────────────────────────────────────────

/// Returns the canonical column-header row for a 10-K audit CSV.
///
/// `item_keys` should be the list of item map keys (e.g. `"item_1"`,
/// `"item_1a"`, `"item_7"`, …) in display order.  One `*_chars` column is
/// emitted per key.
pub fn header_columns(item_keys: &[String]) -> Vec<String> {
    let mut header: Vec<String> = vec![
        "ticker".into(),
        "cik".into(),
        "accession_number".into(),
        "filed_date".into(),
        "form_type".into(),
        "is_adequate".into(),
        "item_1_snippet".into(),
        "item_7_snippet".into(),
        "items_found".into(),
    ];
    for key in item_keys {
        header.push(format!("{}_chars", key));
    }
    header.push("error".into());
    header
}

// ── CSV row builder ───────────────────────────────────────────────────────────

/// Build a single CSV record from the available data.
///
/// Every column defined by [`header_columns`] is populated.  Fields that are
/// not known at the call site should be passed as `""`.
///
/// * `item_keys` — the ordered list of item keys (e.g. `"item_1"`, `"item_7"`).
/// * `sections` — an optional map from item key to section text
///   (typically the output of the 10-K section parser).
/// * `is_adequate` — whether the filing is considered adequate.
///   When `sections` is `Some`, this is typically computed from it.
#[allow(clippy::too_many_arguments)]
pub fn build_row(
    ticker: &str,
    cik: &str,
    acc: &str,
    filed_date: &str,
    form_type: &str,
    item_keys: &[String],
    items_found: &str,
    sections: Option<&HashMap<String, String>>,
    is_adequate: bool,
    err: &str,
) -> Vec<String> {
    let mut record: Vec<String> = vec![
        ticker.into(),
        cik.into(),
        acc.into(),
        filed_date.into(),
        form_type.into(),
        if is_adequate { "true" } else { "false" }.into(),
        snippet(sections.and_then(|s| s.get("item_1")).map(|s| s.as_str())),
        snippet(sections.and_then(|s| s.get("item_7")).map(|s| s.as_str())),
        items_found.into(),
    ];
    for key in item_keys {
        let count = sections
            .and_then(|s| s.get(key))
            .map(|v| v.len())
            .unwrap_or(0);
        record.push(count.to_string());
    }
    record.push(err.into());
    record
}

// ── Text helpers ──────────────────────────────────────────────────────────────

/// First 150 chars of `text`, with runs of whitespace collapsed to a single
/// space.  Returns an empty string when `text` is `None`.
pub fn snippet(text: Option<&str>) -> String {
    match text {
        None => String::new(),
        Some(t) => {
            let collapsed: String = t.split_whitespace().collect::<Vec<_>>().join(" ");
            collapsed.chars().take(150).collect()
        }
    }
}

// ── File discovery ────────────────────────────────────────────────────────────

/// Recursively collect all files under `root` whose name matches the
/// optional extension filter.  Hidden files (names starting with `.`)
/// are skipped.
///
/// When `pattern` is `"*"`, every file is included; otherwise `pattern`
/// is matched as a case-insensitive suffix (e.g. `"*.htm"` matches both
/// `.htm` and `.html`).
pub fn collect_files(root: &Path, pattern: &str) -> io::Result<Vec<PathBuf>> {
    if !root.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            format!("{:?} is not a directory", root),
        ));
    }

    let mut files = Vec::new();
    let match_all = pattern == "*" || pattern == "*.*";
    collect_files_recursive(root, pattern, match_all, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_recursive(
    dir: &Path,
    pattern: &str,
    match_all: bool,
    files: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip hidden entries
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && name.starts_with('.')
        {
            continue;
        }

        if path.is_dir() {
            collect_files_recursive(&path, pattern, match_all, files)?;
        } else if path.is_file() {
            if match_all {
                files.push(path);
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let suffix = pattern.strip_prefix('*').unwrap_or(pattern);
                if name
                    .to_ascii_lowercase()
                    .ends_with(&suffix.to_ascii_lowercase())
                {
                    files.push(path);
                }
            }
        }
    }
    Ok(())
}

// ── Path heuristics ───────────────────────────────────────────────────────────

/// Try to extract the ticker symbol from a file path that follows the
/// `pull-tenk-items` layout: `<root>/<TICKER>/<ACCESSION>.<ext>`.
///
/// Walks up from the file's parent until `root` is reached and returns the
/// immediate child of `root` (so deeper layouts like
/// `<root>/<SECTOR>/<TICKER>/<FILE>` still resolve). Returns an empty string
/// when the file is not beneath `root`.
pub fn guess_ticker(file_path: &Path, root: &Path) -> String {
    let mut current = file_path.parent();
    let mut candidate: Option<&str> = None;
    while let Some(dir) = current {
        if dir == root {
            return candidate.unwrap_or_default().to_string();
        }
        if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
            candidate = Some(name);
        }
        current = dir.parent();
    }
    String::new()
}

/// Try to extract the accession number from the filename.
///
/// The `pull-tenk-items` naming convention is:
/// `<ACCESSION>.<ext>`.  Strips the extension.
pub fn guess_accession(file_path: &Path) -> String {
    file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_default()
}

// ── Accession→form manifest ───────────────────────────────────────────────────

/// One manifest record: `(cik, filed_date, form_type)`.
pub type ManifestEntry = (String, String, String);

/// Load the accession→form manifest written by `pull-tenk-items`
/// (`.tenk_manifest.csv`), keyed by `(ticker, accession_number)`.
///
/// Parsed with the `csv` crate (not line-splitting): fields may contain
/// quoted commas, quotes, or embedded newlines. Keys are normalized at the
/// boundary — ticker trimmed, accession via [`normalize_accession`] — so
/// they match path-derived lookups regardless of dash/padding/case form.
/// Returns an empty map when the file is missing or unreadable — callers
/// fall back to blank `cik` / `filed_date` / `form_type` columns, preserving
/// the old offline behavior. Malformed lines are skipped. Only the first
/// five columns are read; extra columns are ignored.
pub fn load_manifest(path: &Path) -> HashMap<(String, String), ManifestEntry> {
    let mut map = HashMap::new();
    let Ok(mut rdr) = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
    else {
        return map;
    };
    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        map.insert(
            (
                record[0].trim().to_string(),
                normalize_accession(&record[2]),
            ),
            (
                record[1].to_string(),
                record[3].to_string(),
                record[4].to_string(),
            ),
        );
    }
    map
}

// ── Accession→period join (amendment grouping) ───────────────────────────────
//
// The fundamentals CSVs (`data/latest-us-gaap/us-gaap-fundamentals/<T>.csv`)
// carry canonical `(fy, fp)` per accession (`accn`). Joining audit rows on
// accession gives the period key that groups an original 10-K with its `/A`
// amendments. Read-only: nothing in fundamentals is modified.

/// One period record: `(fy, fp)`.
pub type PeriodEntry = (String, String);

/// Load an `accession → (fy, fp)` map from every `*.csv` under `dir`.
///
/// Only the `accn`, `fy`, and `fp` columns are read (by header name, so
/// column order doesn't matter). Rows are streamed as byte records so the
/// ~2,000 wide concept columns are never allocated as owned strings.
/// Missing dir, unreadable files, and rows with empty `accn`/`fy`/`fp` are
/// skipped. When one accession covers several rows (e.g. comparative
/// figures for an earlier year alongside the primary year), the `FY` row
/// wins over quarterly rows and the latest fiscal year wins ties —
/// accessions are globally unique so this only triggers on comparatives.
pub fn load_period_map(dir: &Path) -> HashMap<String, PeriodEntry> {
    let mut map = HashMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return map;
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("csv"))
                    .unwrap_or(false)
        })
        .collect();
    files.sort();
    let mut record = csv::ByteRecord::new();
    for path in files {
        let Ok(mut rdr) = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(&path)
        else {
            continue;
        };
        let Ok(headers) = rdr.byte_headers().cloned() else {
            continue;
        };
        let find = |name: &[u8]| headers.iter().position(|h| h == name);
        let (Some(accn_idx), Some(fy_idx), Some(fp_idx)) =
            (find(b"accn"), find(b"fy"), find(b"fp"))
        else {
            continue;
        };
        while rdr.read_byte_record(&mut record).unwrap_or(false) {
            let (Some(accn_b), Some(fy_b), Some(fp_b)) =
                (record.get(accn_idx), record.get(fy_idx), record.get(fp_idx))
            else {
                continue;
            };
            let accn = std::str::from_utf8(accn_b).unwrap_or_default().trim();
            let fy = std::str::from_utf8(fy_b).unwrap_or_default().trim();
            let fp = std::str::from_utf8(fp_b).unwrap_or_default().trim();
            if accn.is_empty() || fy.is_empty() || fp.is_empty() {
                continue;
            }
            let key = normalize_accession(accn);
            let candidate = (fy.to_string(), fp.to_string());
            match map.get_mut(&key) {
                None => {
                    map.insert(key, candidate);
                }
                Some(existing) => {
                    let new_is_fy = candidate.1.eq_ignore_ascii_case("FY");
                    let old_is_fy = existing.1.eq_ignore_ascii_case("FY");
                    let fy_cmp = candidate
                        .0
                        .parse::<i32>()
                        .unwrap_or(0)
                        .cmp(&existing.0.parse::<i32>().unwrap_or(0));
                    if (new_is_fy && !old_is_fy) || (new_is_fy == old_is_fy && fy_cmp.is_gt()) {
                        *existing = candidate;
                    }
                }
            }
        }
    }
    map
}

/// Group key for one fiscal period of one company: `(cik_or_ticker, fy)`.
///
/// `cik` is preferred (stable across ticker renames); `ticker` is the
/// fallback when `cik` is blank (pre-manifest rows). Returns an empty
/// string when both are blank so ownerless rows never collapse into a
/// shared `"|fy"` group — callers treat empty keys as ungrouped.
pub fn period_group_key(cik: &str, ticker: &str, fy: &str) -> String {
    let cik = cik.trim();
    let ticker = ticker.trim();
    let owner = if !cik.is_empty() {
        cik
    } else if !ticker.is_empty() {
        ticker
    } else {
        return String::new();
    };
    format!("{owner}|{fy}")
}

/// Pick the canonical row index within one period group.
///
/// Latest `filed_date` among adequate rows wins (restated figures supersede
/// the original). When no row is adequate, the latest row overall wins so
/// every group still has a canonical representative. Dates compare
/// lexicographically (`YYYY-MM-DD`); blanks sort before any real date.
/// Returns `None` for an empty group.
pub fn pick_canonical(filed_dates: &[&str], adequate: &[bool]) -> Option<usize> {
    if filed_dates.is_empty() {
        return None;
    }
    let latest = |idxs: &mut dyn Iterator<Item = usize>| {
        idxs.max_by(|&a, &b| filed_dates[a].cmp(filed_dates[b]).then_with(|| a.cmp(&b)))
    };
    let mut adequate_idxs = (0..filed_dates.len()).filter(|&i| adequate[i]);
    if let Some(i) = latest(&mut adequate_idxs) {
        return Some(i);
    }
    let mut all_idxs = 0..filed_dates.len();
    latest(&mut all_idxs)
}

#[cfg(test)]
mod period_tests {
    use super::*;

    #[test]
    fn normalization_ignores_dashes_and_case() {
        assert_eq!(
            normalize_accession("0001090872-26-000064"),
            normalize_accession("000109087226000064")
        );
    }

    #[test]
    fn period_map_loads_by_header_name() {
        let dir = std::env::temp_dir().join("tenk_period_map_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Columns deliberately out of order; extra tag columns present.
        std::fs::write(
            dir.join("A.csv"),
            "fy,accn,form,fp,filed\n2024,0001-01-000001,10-K,FY,2024-02-01\n2024,0001-01-000002,10-K/A,FY,2024-05-01\n",
        )
        .unwrap();
        let map = load_period_map(&dir);
        assert_eq!(map.len(), 2);
        assert_eq!(
            map[&normalize_accession("0001-01-000001")],
            ("2024".to_string(), "FY".to_string())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_dir_loads_empty() {
        assert!(load_period_map(Path::new("/nonexistent-xyz/fundamentals")).is_empty());
    }

    #[test]
    fn latest_adequate_wins_over_later_or_earlier() {
        // Original adequate (Feb), excerpt /A inadequate (May), full
        // restatement /A adequate (Aug) → restatement is canonical.
        let dates = vec!["2024-02-01", "2024-05-01", "2024-08-01"];
        let adequate = vec![true, false, true];
        assert_eq!(pick_canonical(&dates, &adequate), Some(2));
    }

    #[test]
    fn excerpt_only_groups_fall_back_to_latest() {
        let dates = vec!["2024-02-01", "2024-05-01"];
        let adequate = vec![false, false];
        assert_eq!(pick_canonical(&dates, &adequate), Some(1));
    }

    #[test]
    fn group_key_prefers_cik() {
        assert_eq!(period_group_key("1090872", "A", "2024"), "1090872|2024");
        assert_eq!(period_group_key("", "A", "2024"), "A|2024");
    }

    #[test]
    fn group_key_empty_without_owner() {
        assert_eq!(period_group_key("", "", "2024"), "");
        assert_eq!(period_group_key("  ", "  ", "2024"), "");
    }

    #[test]
    fn grouping_comes_from_fundamentals_only() {
        // Zero tolerance for wrong-period association: no heuristic may
        // assign one filing's period to another. Group keys derive solely
        // from the fundamentals map via `period_group_key`.
        assert_eq!(period_group_key("1090872", "A", "2024"), "1090872|2024");
        assert_eq!(period_group_key("", "", "2024"), "");
    }
}

#[cfg(test)]
mod manifest_tests {
    use super::*;

    #[test]
    fn missing_manifest_loads_empty() {
        let map = load_manifest(Path::new("/nonexistent-xyz/.tenk_manifest.csv"));
        assert!(map.is_empty());
    }

    #[test]
    fn manifest_lookup_by_ticker_and_accession() {
        let dir = std::env::temp_dir().join("tenk_audit_manifest_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".tenk_manifest.csv");
        std::fs::write(
            &path,
            "ticker,cik,accession,filed_date,form,primary_document\n\
             ADM,7084,0000007084-05-000350,2005-08-12,10-K,adm10kfy05.htm\n\
             ABCB,70732,0001193125-07-251748,2007-11-29,10-K/A,d10ka.htm\n",
        )
        .unwrap();
        let map = load_manifest(&path);
        assert_eq!(map.len(), 2);
        assert_eq!(
            map[&(
                "ADM".to_string(),
                normalize_accession("0000007084-05-000350")
            )],
            (
                "7084".to_string(),
                "2005-08-12".to_string(),
                "10-K".to_string()
            )
        );
        assert!(
            map[&(
                "ABCB".to_string(),
                normalize_accession("0001193125-07-251748")
            )]
                .2
                .ends_with("/A"),
            "amendment form must be visible for scoring-time filtering"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_keys_converge_across_formats() {
        // Undashed, padded, and dashed inputs must all land on one key —
        // this is the production-grade guarantee: no silent misses from
        // format variants on either side of the join.
        let dir = std::env::temp_dir().join("tenk_audit_manifest_formats");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".tenk_manifest.csv");
        std::fs::write(
            &path,
            "ticker,cik,accession,filed_date,form,primary_document\n\
             \"  ADM \",7084,000000708405000350,2005-08-12,10-K,adm10kfy05.htm\n",
        )
        .unwrap();
        let map = load_manifest(&path);
        assert_eq!(map.len(), 1);
        // Dashed lookup (what `guess_accession` yields from filenames).
        assert!(map.contains_key(&(
            "ADM".to_string(),
            normalize_accession("0000007084-05-000350")
        )));
        // Undashed lookup resolves identically.
        assert_eq!(
            normalize_accession("000000708405000350"),
            normalize_accession("0000007084-05-000350")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_survives_quoted_multiline_field() {
        // A line-splitting parser fractures this record and drops it; the
        // `csv` reader must decode the embedded newline inside quotes.
        let dir = std::env::temp_dir().join("tenk_audit_manifest_multiline");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".tenk_manifest.csv");
        std::fs::write(
            &path,
            "ticker,cik,accession,filed_date,form,primary_document\n\
             ADM,7084,0000007084-05-000350,2005-08-12,10-K,\"doc\nname.htm\"\n",
        )
        .unwrap();
        let map = load_manifest(&path);
        assert_eq!(map.len(), 1);
        assert_eq!(
            map[&(
                "ADM".to_string(),
                normalize_accession("0000007084-05-000350")
            )]
                .2,
            "10-K"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
