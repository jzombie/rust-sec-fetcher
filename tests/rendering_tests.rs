/// Integration tests for the `sec_fetcher::rendering` module.
///
/// # Test strategy
///
/// ## Unit tests (always run — no network, no fixtures required)
///
/// Use realistic inline HTML strings that mirror common SEC filing patterns:
///
/// 1. Financial tables with blank first header (row-label pattern) — the most
///    common information-loss vector in SEC filings.
/// 2. Standard tables (all headers non-blank).
/// 3. Tables with empty cells — values should render as "Col: —", never silently
///    disappear.
/// 4. XBRL noise (`<ix:header>`, hidden `display:none` divs) — must be stripped.
/// 5. Plain-text content — blank lines collapsed, visible text preserved.
/// 6. View invariants — `EmbeddingTextView` must not emit raw pipe-table lines;
///    `MarkdownView` must preserve them.
///
/// ## Fixture-based tests (run when fixture files are present)
///
/// The fixtures are real SEC filing HTML files downloaded by
/// `cargo run --example refresh_test_fixtures -- <TICKER>`.  If the fixture
/// files do not exist, the tests are skipped (not failed) so CI stays green on
/// a fresh clone.
///
/// Fixture files expected at:
///   tests/fixtures/rendering/<TICKER>_8k_primary.html
///   tests/fixtures/rendering/<TICKER>_8k_exhibit.html
use flate2::read::GzDecoder;
use sec_fetcher::rendering::{EmbeddingTextView, FilingView, MarkdownView};
use std::io::Read;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn embedding(html: &str) -> String {
    EmbeddingTextView
        .render_html(html)
        .expect("EmbeddingTextView::render_html failed")
}

fn markdown(html: &str) -> String {
    MarkdownView
        .render_html(html)
        .expect("MarkdownView::render_html failed")
}

#[track_caller]
fn assert_contains(output: &str, expected: &str, label: &str) {
    assert!(
        output.contains(expected),
        "{label}: expected text {:?} not found in rendered output:\n---\n{output}\n---",
        expected
    );
}

#[track_caller]
fn assert_not_contains(output: &str, unexpected: &str, label: &str) {
    assert!(
        !output.contains(unexpected),
        "{label}: text {:?} should NOT appear in rendered output:\n---\n{output}\n---",
        unexpected
    );
}

// ---------------------------------------------------------------------------
// Inline HTML fixtures
// ---------------------------------------------------------------------------

/// Financial table with a blank first header — the row-label pattern found in
/// virtually every SEC earnings release.  `headers[0]` is empty; `cells[0]` is
/// the metric name (Revenue, Net Income, …).
const FINANCIAL_TABLE_HTML: &str = r#"<!DOCTYPE html>
<html><body>
<h1>Quarterly Financial Results</h1>
<table>
  <tr><th></th><th>Q1 2025</th><th>Q1 2024</th></tr>
  <tr><td>Revenue</td><td>$12,345</td><td>$10,987</td></tr>
  <tr><td>Net Income</td><td>$2,100</td><td>$1,870</td></tr>
  <tr><td>Earnings per Share</td><td>$1.23</td><td>$1.08</td></tr>
</table>
</body></html>"#;

/// Standard table where all header cells are non-blank; each row value is
/// paired with its column header.
const STANDARD_TABLE_HTML: &str = r#"<!DOCTYPE html>
<html><body>
<h2>Segment Results</h2>
<table>
  <tr><th>Segment</th><th>Revenue</th><th>Operating Income</th></tr>
  <tr><td>Cloud Services</td><td>$7,200</td><td>$2,160</td></tr>
  <tr><td>Enterprise Software</td><td>$3,500</td><td>$875</td></tr>
  <tr><td>Professional Services</td><td>$1,645</td><td>$329</td></tr>
</table>
</body></html>"#;

/// Table with an empty cell in the last column — the empty value must not be
/// silently dropped.  `EmbeddingTextView` should emit "Period B: —".
const SPARSE_TABLE_HTML: &str = r#"<!DOCTYPE html>
<html><body>
<table>
  <tr><th></th><th>Period A</th><th>Period B</th></tr>
  <tr><td>Revenue</td><td>$1,000</td><td></td></tr>
  <tr><td>EBITDA</td><td>$400</td><td>$380</td></tr>
</table>
</body></html>"#;

/// iXBRL filing: contains an `<ix:header>` block (pure machine metadata) and a
/// hidden `<div style="display:none">` (duplicate XBRL data).  Both must be
/// stripped; only the visible prose must appear in the output.
const XBRL_NOISE_HTML: &str = r#"<!DOCTYPE html>
<html><body>
<ix:header>
  <ix:resources>
    <ix:context id="ctx-1" MACHINE_ONLY_TOKEN_XBRL_MANIFEST>
      <ix:entity><ix:identifier>0000320193</ix:identifier></ix:entity>
    </ix:context>
  </ix:resources>
</ix:header>
<div style="display:none">
  <ix:nonNumeric name="dei:EntityName">HIDDEN_DUPLICATE_CONTENT</ix:nonNumeric>
</div>
<h1>Annual Report</h1>
<p>The company achieved record results in the fiscal year.</p>
<p>Revenue grew 12% year over year to $391 billion.</p>
</body></html>"#;

/// Tests `render_text` (plain-text path): excessive blank lines must collapse.
const MULTI_BLANK_TEXT: &str =
    "First paragraph.\n\n\n\nSecond paragraph.\n\n\n\n\n\nThird paragraph.";

// ---------------------------------------------------------------------------
// 1. Row-label pattern — financial tables (blank first header)
// ---------------------------------------------------------------------------

#[test]
fn financial_table_row_labels_preserved_in_embedding() {
    let out = embedding(FINANCIAL_TABLE_HTML);
    // All row labels must survive
    assert_contains(&out, "Revenue", "row label: Revenue");
    assert_contains(&out, "Net Income", "row label: Net Income");
    assert_contains(&out, "Earnings per Share", "row label: Earnings per Share");
    // All column headers must survive
    assert_contains(&out, "Q1 2025", "column header: Q1 2025");
    assert_contains(&out, "Q1 2024", "column header: Q1 2024");
    // Representative cell values must survive
    assert_contains(&out, "12,345", "cell value: 12,345");
    assert_contains(&out, "1,870", "cell value: 1,870");
    assert_contains(&out, "1.08", "cell value: 1.08");
}

#[test]
fn financial_table_row_labels_preserved_in_markdown() {
    let out = markdown(FINANCIAL_TABLE_HTML);
    assert_contains(&out, "Revenue", "row label: Revenue");
    assert_contains(&out, "Net Income", "row label: Net Income");
    assert_contains(&out, "Q1 2025", "column header: Q1 2025");
    assert_contains(&out, "12,345", "cell value: 12,345");
}

// ---------------------------------------------------------------------------
// 2. Standard table — all headers non-blank
// ---------------------------------------------------------------------------

#[test]
fn standard_table_all_data_preserved_in_embedding() {
    let out = embedding(STANDARD_TABLE_HTML);
    // Column headers
    assert_contains(&out, "Segment", "column header: Segment");
    assert_contains(&out, "Revenue", "column header: Revenue");
    assert_contains(&out, "Operating Income", "column header: Operating Income");
    // Row values
    assert_contains(&out, "Cloud Services", "row value: Cloud Services");
    assert_contains(&out, "Enterprise Software", "row value: Enterprise Software");
    assert_contains(&out, "Professional Services", "row value: Professional Services");
    // Numeric values
    assert_contains(&out, "7,200", "cell value: 7,200");
    assert_contains(&out, "875", "cell value: 875");
    assert_contains(&out, "329", "cell value: 329");
}

#[test]
fn standard_table_all_data_preserved_in_markdown() {
    let out = markdown(STANDARD_TABLE_HTML);
    assert_contains(&out, "Segment", "column header: Segment");
    assert_contains(&out, "Cloud Services", "row value: Cloud Services");
    assert_contains(&out, "7,200", "cell value: 7,200");
}

// ---------------------------------------------------------------------------
// 3. Empty cells — rendered as dashes, not silently dropped
// ---------------------------------------------------------------------------

#[test]
fn empty_cell_gets_dash_not_silently_dropped() {
    let out = embedding(SPARSE_TABLE_HTML);
    // Row label must appear
    assert_contains(&out, "Revenue", "row label: Revenue");
    // The non-empty column must be present
    assert_contains(&out, "Period A", "column header: Period A");
    assert_contains(&out, "1,000", "cell value: 1,000");
    // The empty column header must STILL appear (not silently skipped)
    assert_contains(&out, "Period B", "empty column header: Period B");
    // The empty cell should be rendered as an em-dash sentinel
    assert_contains(&out, "Period B: \u{2014}", "empty cell dash: Period B: —");
    // EBITDA row with values in both columns
    assert_contains(&out, "EBITDA", "row label: EBITDA");
    assert_contains(&out, "380", "cell value: 380");
}

// ---------------------------------------------------------------------------
// 4. XBRL noise — ix:header and hidden divs stripped
// ---------------------------------------------------------------------------

#[test]
fn ix_header_block_is_stripped() {
    let emb = embedding(XBRL_NOISE_HTML);
    let md = markdown(XBRL_NOISE_HTML);
    for out in [&emb, &md] {
        assert_not_contains(out, "MACHINE_ONLY_TOKEN_XBRL_MANIFEST", "ix:header content");
        assert_not_contains(out, "ix:context", "ix:header element");
    }
}

#[test]
fn hidden_div_content_is_stripped() {
    let emb = embedding(XBRL_NOISE_HTML);
    let md = markdown(XBRL_NOISE_HTML);
    for out in [&emb, &md] {
        assert_not_contains(out, "HIDDEN_DUPLICATE_CONTENT", "display:none content");
    }
}

#[test]
fn visible_content_survives_xbrl_stripping() {
    let emb = embedding(XBRL_NOISE_HTML);
    let md = markdown(XBRL_NOISE_HTML);
    for out in [&emb, &md] {
        assert_contains(out, "Annual Report", "visible heading");
        assert_contains(out, "record results", "visible paragraph");
        assert_contains(out, "391 billion", "visible numeric");
    }
}

// ---------------------------------------------------------------------------
// 5. View invariants — pipe tables vs. prose
// ---------------------------------------------------------------------------

#[test]
fn embedding_view_produces_no_pipe_table_lines() {
    // For HTML that actually has a <table>, EmbeddingTextView must flatten it —
    // no lines that begin with '|' should appear.
    for html in [FINANCIAL_TABLE_HTML, STANDARD_TABLE_HTML, SPARSE_TABLE_HTML] {
        let out = embedding(html);
        let pipe_line = out.lines().find(|l| l.trim().starts_with('|'));
        assert!(
            pipe_line.is_none(),
            "EmbeddingTextView must not produce pipe-table lines — found: {:?}\nfull output:\n{}",
            pipe_line,
            out
        );
    }
}

#[test]
fn markdown_view_preserves_pipe_tables() {
    for html in [FINANCIAL_TABLE_HTML, STANDARD_TABLE_HTML] {
        let out = markdown(html);
        let has_pipe = out.lines().any(|l| l.trim().starts_with('|'));
        assert!(
            has_pipe,
            "MarkdownView must preserve pipe-table lines:\n{}",
            out
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Plain-text path — consecutive blank lines are collapsed
// ---------------------------------------------------------------------------

#[test]
fn embedding_text_view_collapses_consecutive_blank_lines() {
    let out = EmbeddingTextView.render_text(MULTI_BLANK_TEXT);

    // No two consecutive blank lines should remain
    let has_double_blank = out
        .lines()
        .collect::<Vec<_>>()
        .windows(2)
        .any(|w| w[0].trim().is_empty() && w[1].trim().is_empty());

    assert!(
        !has_double_blank,
        "EmbeddingTextView::render_text left consecutive blank lines:\n{}",
        out
    );

    // The three paragraphs must still be present
    assert_contains(&out, "First paragraph", "paragraph 1");
    assert_contains(&out, "Second paragraph", "paragraph 2");
    assert_contains(&out, "Third paragraph", "paragraph 3");
}

// ---------------------------------------------------------------------------
// 7. Fixture-based tests (skipped when fixtures not present)
//    Run: cargo run --example refresh_test_fixtures -- AAPL
// ---------------------------------------------------------------------------

/// Loads a compressed `.gz` fixture, returning `None` when the file is absent
/// so the calling test skips rather than fails.
///
/// Fixture files are written by `cargo run --bin refresh_test_fixtures`.
fn load_gz_fixture(name: &str) -> Option<String> {
    let path = std::path::Path::new("tests/fixtures").join(format!("{}.gz", name));
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => panic!("Error opening fixture {}: {}", path.display(), e),
    };
    let mut decoder = GzDecoder::new(file);
    let mut buf = String::new();
    decoder
        .read_to_string(&mut buf)
        .unwrap_or_else(|e| panic!("Error decompressing {}: {}", path.display(), e));
    Some(buf)
}

/// Property: every visible table header word (≥4 chars, no markup) that appears
/// in the raw HTML must also appear in the rendered embedding output.
///
/// This provides a structural guarantee that no column header is silently dropped
/// during the HTML→text pipeline.  It is not a precise diff — we look for
/// occurrence of each extracted token, not exact table reconstruction.
fn assert_table_headers_preserved(html: &str, rendered: &str, source: &str) {
    // Extract text between <th> and </th> (case-insensitive, minimal match)
    let re = regex::Regex::new(r"(?i)<th[^>]*>(.*?)</th>").unwrap();
    for cap in re.captures_iter(html) {
        let raw_header = &cap[1];
        // Strip any sub-tags and trim
        let clean = regex::Regex::new(r"<[^>]+>")
            .unwrap()
            .replace_all(raw_header, "")
            .trim()
            .to_string();
        if clean.is_empty() {
            continue; // blank first header is the row-label pattern — skip
        }
        assert_contains(
            rendered,
            &clean,
            &format!("{source}: table header {:?}", clean),
        );
    }
}

#[test]
fn fixture_aapl_8k_primary_visible_content_preserved() {
    let Some(html) = load_gz_fixture("AAPL_8k_primary.html") else {
        eprintln!("SKIP: AAPL_8k_primary.html.gz not found — run: cargo run --bin refresh_test_fixtures");
        return;
    };

    let emb = embedding(&html);
    let md = markdown(&html);

    // Neither output should be empty
    assert!(!emb.trim().is_empty(), "EmbeddingTextView produced empty output for AAPL primary");
    assert!(!md.trim().is_empty(), "MarkdownView produced empty output for AAPL primary");

    // No raw XBRL machine tokens in output
    for out in [&emb, &md] {
        assert_not_contains(out, "<ix:", "raw iXBRL tag in output");
        assert_not_contains(out, "display:none", "hidden CSS in output");
    }

    // All table headers in the HTML should appear in the embedding output
    assert_table_headers_preserved(&html, &emb, "AAPL primary (embedding)");
}

#[test]
fn fixture_aapl_8k_exhibit_table_headers_preserved() {
    let Some(html) = load_gz_fixture("AAPL_8k_exhibit.html") else {
        eprintln!("SKIP: AAPL_8k_exhibit.html.gz not found — run: cargo run --bin refresh_test_fixtures");
        return;
    };

    let emb = embedding(&html);

    assert!(!emb.trim().is_empty(), "EmbeddingTextView produced empty output for AAPL exhibit");
    assert_not_contains(&emb, "<ix:", "raw iXBRL tag in output");

    // Property: all non-blank table headers from the raw HTML appear in the embedding
    assert_table_headers_preserved(&html, &emb, "AAPL exhibit (embedding)");

    // EmbeddingTextView must not contain pipe-table lines
    let pipe_line = emb.lines().find(|l| l.trim().starts_with('|'));
    assert!(
        pipe_line.is_none(),
        "EmbeddingTextView must not produce pipe-table lines — found: {:?}",
        pipe_line
    );
}

#[test]
fn fixture_aapl_8k_exhibit_markdown_preserves_tables() {
    let Some(html) = load_gz_fixture("AAPL_8k_exhibit.html") else {
        eprintln!("SKIP: AAPL_8k_exhibit.html.gz not found — run: cargo run --bin refresh_test_fixtures");
        return;
    };

    let md = markdown(&html);
    assert_table_headers_preserved(&html, &md, "AAPL exhibit (markdown)");
}
