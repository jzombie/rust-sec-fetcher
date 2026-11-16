Done:

1. Amendments included by default — `pull-tenk-items` fetches `10-K/A`/`10-K405/A`
   (corrections/restatements; original alone can be stale). `--skip-amendments`
   opts out of new downloads, `--prune-amendments` (with `--skip-amendments`)
   deletes previously-downloaded copies; a plain pull never deletes. Expect
   `/A` rows to score inadequate as standalone docs (~82% of empty
   `items_found` rows) — filter them at scoring time, not collection time.
   (Next step if wanted: pull writes accession→form manifest so the offline
   audit can tag `form_type`/`is_amendment` instead of guessing.)
2. Large-file guard raised 20MB → 100MB, plus large-file html2text bypass —
   files over 20MB use tag-strip (html2text stack-overflows on the 89MB BPOP
   2013 10-K: `thread 'main' has overflowed its stack`). Tag-strip path now
   decodes heading entities (`&nbsp;`/`&#160;` → space, `&#146;`/`&#8217;` →
   `'`), splits block tags to newlines (table-glued `...INC. ITEM 1. BUSINESS`
   headings), and the Item-7 fallback matches multi-line titles. Verified on
   BPOP: 9 items, `is_adequate=true` (i1 ~117k, i7 ~362k), ~14s, no crash.
   (Review pass: Cow header strip, single-pass entity decode, td/th as spaces
   with whole-tag collision-safe boundaries, case-normalized amendment check
   with empty-dir cleanup. Review application had one real bug — prefix-only
   tag replacement left attribute junk, BPOP dropped to 0 items; fixed to
   consume whole tags, mirror shows 41 heading hits. Full workspace suite
   green after all changes.)
3. html2text panic fallback — `html_to_plain` falls back to tag-strip when
   html2text panics instead of erroring. Verified on the 10 panic files, now
   also via the improved tag-strip path: 0 errors, 9/10 adequate (CXDO 15
   items, SIGI 14 items recovered; GJR item_14 only — likely amendment).
   Tag-strip path also strips `<!-- -->`, `<style>`, `<script>` blocks first
   (commented-out draft headings leaked phantom ITEMs); new regression test
   `tag_strip_removes_comments_style_and_script`. Spot-checks stable after:
   BPOP 9 items adequate, panic subset 0 errors 9/10 adequate.
4. Old-era empties — sampled empties (AAL 2006, AAON 1997, AAPI 2010, AAOI
   2015) are all `10-K/A`; covered by fix 1. Remaining ~18% non-amendment
   empties (≈800 files) still need pattern work — slice
   `is_adequate=false AND error=''` after the next pull+audit to build the list.

Review round 2 (all applied + verified, full workspace suite green):
- `prune_amendments` requires `skip_amendments` at clap parse time (was
  silent no-op downloading amendments); simplified the runtime combination.
- Amendment pruning batched to one `read_dir`+`remove_dir` per ticker after
  the filing loop (was O(K·N) scans with mid-loop rmdir/mkdir churn).
- Review's block-tag rewrite had a real application bug (prefix-only match
  left attribute junk, BPOP → 0 items); fixed to consume whole tags.

Baseline scores — 2026-11-14 audit run (`data/tenk_audit.csv`, 48M, Nov 14 15:36;
corpus `data/tenk_items` 223G / 97,504 files / 5,653 tickers; parser includes
inline-tag unwrap fix, ABUS-class recovery confirmed in-run):
- Rows: 97,503 (unchanged corpus — pure parser-effect read). `cik`/`filed_date`/
  `form_type` all blank (offline audit limitation — back up CSV before re-audit).
- Adequate: 83,381 (85.5%; +53 vs pre-fix 83,328). Inadequate: 14,122.
- Errors: 3 total, all `skip: binary` (zero `read:` errors — no truncation signal).
- Inadequate-no-error (`is_adequate=false AND error=''`): 14,119 (−53).
  Empty `items_found`: 4,652 (−5).
- `item_1a`-without-`item_1` signature: 569 (−7 recovered, all still inadequate).
  Near-full (≥16 items) inadequate-no-error: 538 (−20).
- Slice detail (15:36 run): of 538 near-full, 452 have 0<`i7`<2000 (short-Item-7
  threshold failures — stub vs truncation TBD), 54 missing `item_1`, 42 missing
  `item_7`; most-missed keys are `item_15`/`item_16` (adequacy-irrelevant).
  `item_7`-only singletons (367): 361 have adequate-sized `i7` — prime suspect
  for missed-Item-1 recovery. `item_8`-only (371) / `item_9a`-only (345) have no
  adequate body — likely genuine stubs/amendments.
- Top inadequate-no-error `items_found` patterns: `(empty)` 4,657;
  `item_10;item_11;item_12;item_13;item_14` 1,973 (Part-III-only, amendment
  excerpts); `item_8` 371; `item_7` 367; `item_9a` 345;
  `item_10..item_16` 258; `item_10;item_11;item_12` 231;
  near-full (≥16 items, e.g. `item_1..item_14`) 558 — mostly threshold
  failures (`i7` ~500-800 chars, e.g. ADM 05/06; or missing `item_1`,
  e.g. ABUS 15-001398).
- Next parser work: slice the 558 near-full threshold failures first
  (headings found, body short), then `item_7`-only with huge `i7`
  (missing `item_1`, e.g. ABCB/ACFN/ACI), then empties.

5. Inline-tag split headings — html2text wraps table cells at inline-tag
   boundaries, so `<font>Item </font>1.` (several EDGAR agents) rendered as
   `Item  │Business` / `1.    │` on two rows and the section was lost: ABUS
   2014 10-K had Item 1A–14 but no Item 1 (inadequate). Fix: unwrap purely
   presentational inline tags (`font/b/strong/i/em/u/span/small`, never `a`
   so TOC links keep `[...]` brackets) pre-html2text; new regression test
   `inline_tag_split_heading_recovered`. Verified: ABUS flipped to adequate
   (item_1 114k chars, 17 items); ABCB still inadequate (genuine 10-K/A,
   Item 7 only — correct); ADM still inadequate (incorporate-by-reference
   stub, `i7=798`, no standalone MD&A heading — correct). BPOP stable
   (9 items adequate; char counts drift slightly from re-wrapped table
   borders). Corpus-wide opportunity: 576 rows have `item_1a` but no
   `item_1` (all inadequate) — re-audit to measure recovery. Full workspace
   suite green (158 lib). (Review: size-gate runs on the pre-unwrap length —
   unwrapping first allocated a full copy and could shrink tag-dense filings
   below the 20MB guard back into html2text; tag-strip needs no unwrap.
   Verified BPOP byte-identical after reorder: i1 117841, i7 362482.)

6. Accession→form manifest — `pull-tenk-items` now maintains
   `data/tenk_items/.tenk_manifest.csv`
   (`ticker,cik,accession,filed_date,form,primary_document`), one row per
   filing in every fetched filing list including skipped/existing files.
   Keyed merge (no dupes on rerun; partial runs keep old rows); hidden
   dotfile so the audit walk never parses it as a filing. `tenk-parser-audit`
   loads it and fills the previously-blank `cik`/`filed_date`/`form_type`
   columns (blanks preserved when no manifest). Full suite green; smoke test
   on 1 real file tags correctly and skips the manifest. NOT YET DEPLOYED TO
   CORPUS: run `pull-tenk-items --release` once (all SKIP, no re-download) to
   write the manifest, then `tenk-parser-audit --release` to tag all 97,503
   rows. Back up `data/tenk_audit.csv` first. (Review: CSV via `csv` crate both
   sides — no hand-rolled parsers, multiline fields safe; atomic tmp+rename
   writes; 250-ticker checkpoints so interrupts keep progress; manifest keys
   sanitized exactly like disk paths so slash-tickers e.g. BF/A match.
   Verified BF_A end-to-end.)

7. Accession→period join — `tenk-period-join` reads `data/tenk_audit.csv` +
   `data/latest-us-gaap/us-gaap-fundamentals/` and writes
   `data/tenk_audit_periods.csv` (+`fy`,`fp`,`period_key`,`is_canonical`,
   `canonical_accession`,`join_status`,`superseded_by`,`pairing_rule`).
   Read-only over fundamentals; per-ticker CSVs scanned once (~280k rows,
   224,776 unique accn; accn→(fy,fp) is 1:1, first-wins safe). Measured:
   53,359/97,523 joined (misses are era: 100% pre-2010, 27% 2010s, 13%
   2020s); 3,462 stranded originals paired to joined `/A` via
   nearest-earlier-unmapped-base (median gap 46d, p90 253d, 26 >730d to
   review, 0 negative); 3,407 mixed-form groups; grouped canonical adequacy
   96.8% (51,326/53,015). Canonical double-count eliminated: 93,772
   canonical rows, one per period + one per ungrouped filing. (Review:
   claimed-set blocks double-claims and cross-period stealing; form checks
   case-insensitive/trimmed; FY+latest-fy preference on accession
   collisions; byte-record streaming skips ~2,000 wide columns; blank-owner
   rows stay ungrouped. Re-ran: 3,461 paired, 3,460 mixed groups, canonical
   adequacy 96.9% (51,350/53,015), zero ownerless groups.)
