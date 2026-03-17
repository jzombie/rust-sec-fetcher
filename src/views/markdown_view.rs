use super::html_helpers::render_html_to_clean_markdown;
use super::FilingView;
use std::error::Error;

/// Renders an SEC filing document as clean ATX Markdown.
///
/// XBRL metadata noise is stripped; HTML tables are preserved as standard
/// pipe-table Markdown.  Consecutive blank lines are collapsed to one.
///
/// This view is **lossless** — table structure including row labels, column
/// headers, and cell values is fully preserved.  Recommended when you need
/// high-fidelity retrieval, citation accuracy, or downstream structured
/// extraction.
pub struct MarkdownView;

impl FilingView for MarkdownView {
    fn render_html(&self, html: &str) -> Result<String, Box<dyn Error>> {
        render_html_to_clean_markdown(html)
    }
}
