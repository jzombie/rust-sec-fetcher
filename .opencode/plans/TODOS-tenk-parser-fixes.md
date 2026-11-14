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

Baseline scores — 2026-11-14 audit run (`data/tenk_audit.csv`, 48M, Nov 14 03:13;
corpus `data/tenk_items` 223G / 97,504 files / 5,653 tickers):
- Rows: 97,503 (prior 09:57 run: 97,496; +7). `cik`/`filed_date`/`form_type`
  all blank (offline audit limitation — back up CSV before re-audit).
- Adequate: 83,328 (85.5%). Prior: 82,826. Inadequate: 14,175.
- Errors: 3 total, all `skip: binary` (prior: 464 = 451 `too_large` + 10
  `parse` + 3 `binary`). Zero `read:` errors — no truncation signal.
  Large-file guard (100MB) + tag-strip bypass + panic fallback cleared the
  backlog.
- Inadequate-no-error (`is_adequate=false AND error=''`): 14,172. Empty
  `items_found`: 4,660 total (4,657 inadequate-no-error + 3 binary).
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
