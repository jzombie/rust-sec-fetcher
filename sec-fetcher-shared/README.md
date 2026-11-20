Shared definitions for [sec-fetcher](https://crates.io/crates/sec-fetcher). Single source of truth for names both sides of a join build from, so a rename breaks at compile time instead of silently misjoining. Not published to crates.io.

- Fiscal-period parsing and sort ranks: `parse_period`, `parse_period_slot`, `parse_period_slot_token`, `normalize_fp_label` (including the `Q4` → `FY` normalization), `Period` / `PeriodSlot`.
- Key normalization: `normalize_accession` (trim, strip dashes, uppercase; dashed form kept at rest), `normalize_symbol` (dot/dash class-share variants).
- Canonical US-GAAP bulk CSV columns: `US_GAAP_CSV_META_COLUMNS` (`canonical_order`, `fy`, `fp`, `period_end`, `filed`, `form`, `is_amendment`, `accn`, `filing_url`), newest-first row order.
- 10-K pipeline columns: `TENK_AUDIT_COL_*` (audit CSV), `TENK_PERIOD_COL_*` + `TENK_PERIOD_APPENDED_COLUMNS` (period join), `TENK_MANIFEST_COL_*` + `TENK_MANIFEST_HEADER` (pull manifest; note `accession`/`form` differ from audit `accession_number`/`form_type` by design).
- Parquet schemas: `TENK_RAW_PARQUET_COLUMNS`, `TENK_ITEMIZED_PARQUET_COLUMNS`, and the Arrow field-metadata key `PARQUET_FIELD_METADATA_DESCRIPTION`.
- Path resolvers: `resolve_data_root`, `resolve_data_dir`, `resolve_tenk_items_dir`, `resolve_tenk_audit_csv`, `resolve_tenk_manifest_csv`, `resolve_fundamentals_dir`, `resolve_tenk_periods_csv`, `resolve_hf_upload_dir`, `resolve_tenk_raw_parquet_dir`, `resolve_tenk_itemized_parquet_dir`.
