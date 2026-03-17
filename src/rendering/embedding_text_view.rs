use super::html_helpers::{collapse_blank_lines, flatten_tables, render_html_to_clean_markdown};
use super::FilingView;
use std::error::Error;

/// Renders an SEC filing document as embedding-optimized prose text.
///
/// Applies the full [`MarkdownView`] pipeline, then converts every pipe-table
/// block into labeled prose sentences designed for dense semantic search:
///
/// ```text
/// |             | Q1 2025   | Q4 2024   |
/// | Revenue     | 95,359    | 90,753    |
/// | Net Income  | 25,000    | 22,000    |
/// ```
/// becomes:
/// ```text
/// Revenue — Q1 2025: 95,359, Q4 2024: 90,753.
/// Net Income — Q1 2025: 25,000, Q4 2024: 22,000.
/// ```
///
/// The row-label pattern (blank first header = metric-name-in-first-column)
/// is handled automatically, so metric names are always present in the output.
/// Columns whose value is absent in a given row are rendered as `Header: —`
/// to preserve the structural information that the column exists.
///
/// Plain-text filings have excess blank lines collapsed (same as [`MarkdownView`]).
///
/// [`MarkdownView`]: crate::rendering::MarkdownView
pub struct EmbeddingTextView;

impl FilingView for EmbeddingTextView {
    fn render_html(&self, html: &str) -> Result<String, Box<dyn Error>> {
        let md = render_html_to_clean_markdown(html)?;
        Ok(flatten_tables(&md))
    }

    fn render_text(&self, text: &str) -> String {
        collapse_blank_lines(text)
    }
}
