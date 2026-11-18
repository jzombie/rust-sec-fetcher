# 10-K Parser Audit and Period Join

Two-stage offline pipeline. No network calls. Single on-disk tree under `data/`.

## Stage 1: `tenk-parser-audit`

Measures the 10-K section parser against local filings.

- Input: filings under `data/tenk_items/`, manifest at `data/tenk_items/.tenk_manifest.csv`
- Output: `data/tenk_audit.csv` (one row per filing)
- Columns: `ticker`, `cik`, `accession_number`, `filed_date`, `form_type`, `is_adequate`, `item_1_snippet`, `item_7_snippet`, `items_found`, per-item `*_chars`, `error`
- Adequacy: `item_1_chars >= 400 AND item_7_chars >= 2000`
- Amendments (`/A`) are included and tagged via `form_type` for scoring-time filtering, never deleted at collection
- Never reads the periods file

Run:

```sh
cargo run -p sec-fetcher-tenk-parser-audit --bin tenk-parser-audit --release
```

## Stage 2: `tenk-period-join`

Adds fiscal identity for scoring. Reads the audit file, never modifies it.

- Inputs: `data/tenk_audit.csv`, fundamentals at `data/latest-us-gaap/us-gaap-fundamentals/`
- Output: `data/tenk_audit_periods.csv` (same rows, same order, 8 columns appended)
- Appended: `fy`, `fp`, `period_key`, `is_canonical`, `canonical_accession`, `join_status`, `superseded_by`, `pairing_rule`

Rules:

- Join key is the EDGAR accession, normalized (trim, strip dashes, uppercase) at every comparison. Canonical dashed form is kept at rest.
- Group key is `(cik_or_ticker, fy)`. CIK preferred; ticker fallback; blank owner stays ungrouped.
- Canonical per group: latest `filed_date` among adequate rows wins; if none adequate, latest row overall wins.
- No heuristic pairing. Unmatched rows keep blank `fy`/`fp`/`period_key` and stand alone as their own canonical row. Wrong-period association is impossible by construction. `pairing_rule` is `fundamentals-join` or `none`; `superseded_by` is always empty (schema stability).

Run:

```sh
cargo run -p sec-fetcher-tenk-parser-audit --bin tenk-period-join --release
```

## Stage 3: `tenk-itemized-to-parquet`

Exports full Item 1 / Item 7 text plus frozen audit annotations to Parquet for upload.

- Inputs: filings under `data/tenk_items/` **and** `data/tenk_audit_periods.csv` from Stage 2 (join key: normalized accession)
- Output: `batch_NNNN.parquet` shards under `hf_upload_folder/tenk_itemized/`
- Columns: `ticker_symbol`, `accession_number`, `filed_date`, `form_type`, `fy`, `fp`, `is_adequate`, `is_canonical`, `canonical_accession`, `item_1`, `item_7`
- Item columns are null when the section is absent or the document fails to parse (counted as parse errors, never fatal)
- Filings missing from the periods CSV get blank annotations and `false` flags, counted as `unannotated` in the summary

**Ordering: run Stage 2 before Stage 3.** The export embeds a frozen copy of the join; re-running the audit or join afterwards orphans the embedded flags. Full pipeline after any parser change:

```sh
cargo run -p sec-fetcher-tenk-parser-audit --bin tenk-parser-audit --release
cargo run -p sec-fetcher-tenk-parser-audit --bin tenk-period-join --release
cargo run --release --bin tenk-itemized-to-parquet
```

The exporter enforces this: it refuses to run when `data/tenk_audit_periods.csv` is missing or older than `data/tenk_audit.csv`, naming the join command to re-run. Column names on both sides of the join are shared constants in `sec-fetcher-shared` (`TENK_AUDIT_COL_*`, `TENK_PERIOD_COL_*`, `TENK_PERIOD_APPENDED_COLUMNS`), so a rename breaks at compile time instead of silently misjoining.

Run:

```sh
cargo run --release --bin tenk-itemized-to-parquet
```

## Scoring guidance

- Score canonical rows only (`is_canonical=true AND is_adequate=true`). The flags are embedded per row, so this is a plain `WHERE` clause — no sidecar CSV needed.
- Filter amendments at scoring time via `form_type`, not at collection.
- Pre-2010 and legacy rows are ungrouped by design (no fundamentals) and score standalone or out of scope.
- Current corpus: 97,526 audit rows, 53,359 joined, 53,015 groups, 92.6% grouped canonical adequacy.
