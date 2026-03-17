use html_to_markdown_rs::{convert, ConversionOptions, HeadingStyle, PreprocessingPreset};
use once_cell::sync::Lazy;
use regex::Regex;
use std::error::Error;

// ---------------------------------------------------------------------------
// Statics
// ---------------------------------------------------------------------------

static IX_HEADER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<ix:header\b[^>]*>.*?</ix:header>").unwrap());

static HIDDEN_DIV_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<div\b[^>]*style\s*=\s*["'][^"']*display\s*:\s*none[^"']*["'][^>]*>.*?</div>"#,
    )
    .unwrap()
});

// ---------------------------------------------------------------------------
// Public-to-sibling helpers
// ---------------------------------------------------------------------------

/// Strips SEC inline XBRL (iXBRL) noise from HTML before rendering.
///
/// Removes:
/// - `<ix:header>` blocks — filing manifests that are pure machine-readable
///   metadata with no human-readable content.
/// - `<div style="display:none">` — hidden divs that typically contain
///   duplicate XBRL-tagged data the browser never shows.
pub(super) fn strip_xbrl_noise(html: &str) -> String {
    let s = IX_HEADER_RE.replace_all(html, "");
    HIDDEN_DIV_RE.replace_all(&s, "").into_owned()
}

/// Full HTML → clean Markdown pipeline shared by all views.
///
/// 1. `strip_xbrl_noise` — drops XBRL scaffolding
/// 2. `html_to_markdown_rs::convert` — ATX headings, aggressive preprocessing
/// 3. `collapse_blank_lines` — caps consecutive blank lines at one
pub(super) fn render_html_to_clean_markdown(html: &str) -> Result<String, Box<dyn Error>> {
    let clean = strip_xbrl_noise(html);

    let mut options = ConversionOptions::default();
    options.heading_style = HeadingStyle::Atx;
    options.preprocessing.enabled = true;
    options.preprocessing.preset = PreprocessingPreset::Aggressive;

    let md = convert(&clean, Some(options))
        .map_err(|e| format!("HTML→Markdown conversion failed: {}", e))?;

    Ok(collapse_blank_lines(&md))
}

/// Collapses runs of blank lines down to a single blank line and trims
/// trailing whitespace from each line.  Keeps the output compact without
/// losing structural paragraph breaks.
pub(super) fn collapse_blank_lines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut consecutive_blank: usize = 0;

    for line in s.lines() {
        if line.trim().is_empty() {
            consecutive_blank += 1;
            if consecutive_blank == 1 {
                out.push('\n');
            }
            // additional consecutive blanks are dropped
        } else {
            consecutive_blank = 0;
            out.push_str(line.trim_end());
            out.push('\n');
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Markdown table → labeled-prose conversion (EmbeddingTextView)
// ---------------------------------------------------------------------------

fn parse_table_row(line: &str) -> Vec<String> {
    line.trim()
        .trim_matches('|')
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn is_separator_row(line: &str) -> bool {
    line.trim().trim_matches('|').split('|').all(|cell| {
        cell.trim()
            .chars()
            .all(|c| c == '-' || c == ':' || c == ' ')
    })
}

/// Converts a collected block of markdown table lines to labeled prose sentences.
///
/// # Row-label pattern (financial tables)
///
/// SEC financial tables almost universally place the metric name in the first
/// column and leave its header blank.  When `headers[0]` is empty, `cells[0]`
/// is treated as the row label:
///
/// ```text
/// |             | Q1 2025   | Q4 2024   |
/// | Revenue     | 95,359    | 90,753    |
/// | Net Income  |  25,000   | 22,000    |
/// ```
/// → `Revenue — Q1 2025: 95,359, Q4 2024: 90,753.`
/// → `Net Income — Q1 2025: 25,000, Q4 2024: 22,000.`
///
/// # Standard labeled tables
///
/// When all header cells are non-empty, cells are paired with their header:
///
/// ```text
/// | Name | Value | Date |
/// | AAPL | $175  | 2025 |
/// ```
/// → `Name: AAPL, Value: $175, Date: 2025.`
///
/// # Empty cells
///
/// Columns whose value is absent are rendered as `Header: —` to preserve the
/// structural fact that the period / column exists but has no value.
fn table_to_sentences(lines: &[&str]) -> String {
    let data_rows: Vec<Vec<String>> = lines
        .iter()
        .filter(|l| !is_separator_row(l))
        .map(|l| parse_table_row(l))
        .collect();

    if data_rows.is_empty() {
        return String::new();
    }

    let headers = &data_rows[0];
    let has_any_header = headers.iter().any(|h| !h.trim().is_empty());

    if !has_any_header {
        // No header row — just join each row's non-empty cells
        return data_rows
            .iter()
            .map(|row| {
                let parts: Vec<&str> = row
                    .iter()
                    .map(|c| c.trim())
                    .filter(|c| !c.is_empty())
                    .collect();
                if parts.is_empty() {
                    String::new()
                } else {
                    format!("{}.\n", parts.join(", "))
                }
            })
            .collect();
    }

    let first_header_empty = headers[0].trim().is_empty();
    let mut out = String::new();

    for row in &data_rows[1..] {
        if row.iter().all(|c| c.trim().is_empty()) {
            continue;
        }

        let sentence = if first_header_empty {
            // Row-label pattern: cells[0] is the metric name
            let row_label = row.get(0).map(|s| s.trim()).unwrap_or("");
            let col_pairs: Vec<String> = headers[1..]
                .iter()
                .enumerate()
                .filter(|(_, h)| !h.trim().is_empty())
                .map(|(i, h)| {
                    let v = row.get(i + 1).map(|s| s.trim()).unwrap_or("");
                    if v.is_empty() {
                        format!("{}: —", h.trim())
                    } else {
                        format!("{}: {}", h.trim(), v)
                    }
                })
                .collect();

            match (row_label.is_empty(), col_pairs.is_empty()) {
                (true, true) => String::new(),
                (true, false) => col_pairs.join(", "),
                (false, true) => row_label.to_string(),
                (false, false) => format!("{} — {}", row_label, col_pairs.join(", ")),
            }
        } else {
            // Standard: pair each header with its corresponding cell
            headers
                .iter()
                .enumerate()
                .filter(|(_, h)| !h.trim().is_empty())
                .map(|(i, h)| {
                    let v = row.get(i).map(|s| s.trim()).unwrap_or("");
                    if v.is_empty() {
                        format!("{}: —", h.trim())
                    } else {
                        format!("{}: {}", h.trim(), v)
                    }
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        if !sentence.is_empty() {
            out.push_str(&sentence);
            out.push_str(".\n");
        }
    }

    out
}

/// Walks a Markdown string line-by-line; pipe-table blocks are replaced with
/// the labeled prose produced by [`table_to_sentences`].
pub(super) fn flatten_tables(markdown: &str) -> String {
    let mut result = String::new();
    let mut table_lines: Vec<&str> = Vec::new();

    for line in markdown.lines() {
        if line.trim().starts_with('|') {
            table_lines.push(line.trim());
        } else {
            if !table_lines.is_empty() {
                result.push_str(&table_to_sentences(&table_lines));
                result.push('\n');
                table_lines.clear();
            }
            result.push_str(line);
            result.push('\n');
        }
    }

    if !table_lines.is_empty() {
        result.push_str(&table_to_sentences(&table_lines));
    }

    result
}
