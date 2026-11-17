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

## Scoring guidance

- Score canonical rows only (`is_canonical=true`).
- Filter amendments at scoring time via `form_type`, not at collection.
- Pre-2010 and legacy rows are ungrouped by design (no fundamentals) and score standalone or out of scope.
- Current corpus: 97,526 audit rows, 53,359 joined, 53,015 groups, 92.6% grouped canonical adequacy.
