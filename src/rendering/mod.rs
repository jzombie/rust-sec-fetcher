mod html_helpers;

mod embedding_text_view;
mod markdown_view;

pub use embedding_text_view::EmbeddingTextView;
pub use markdown_view::MarkdownView;

use std::error::Error;

/// The content type of a fetched filing document, used by [`fetch_and_render`]
/// to dispatch to the correct rendering path.
///
/// [`fetch_and_render`]: crate::network::fetch_and_render
#[derive(Debug, Clone)]
pub enum FilingContentType {
    /// HTML or XHTML — rendered via [`FilingView::render_html`].
    Html,
    /// Plain text — passed to [`FilingView::render_text`].
    Text,
    /// Binary format (PDF, XLSX, image, XBRL schema, …) — not decoded as text.
    Binary { mime_type: String },
}

/// A rendering strategy for fetched SEC filing documents.
///
/// Implementations control how HTML, plain-text, and binary documents are
/// converted to the output string returned by [`fetch_and_render`].
///
/// # Provided implementations
///
/// | Type | Tables | Best suited for |
/// |---|---|---|
/// | [`MarkdownView`] | Preserved as pipe tables | Lossless archiving, RAG retrieval |
/// | [`EmbeddingTextView`] | Flattened to labeled prose | Embedding generation, semantic search |
///
/// # Implementing a custom view
///
/// Only `render_html` is required.  The default `render_text` passes content
/// through unchanged; the default `render_binary` emits a bracketed notice.
///
/// ```rust
/// use sec_fetcher::rendering::FilingView;
/// use std::error::Error;
///
/// pub struct PlainTextView;
///
/// impl FilingView for PlainTextView {
///     fn render_html(&self, html: &str) -> Result<String, Box<dyn Error>> {
///         Ok(html2text::from_read(html.as_bytes(), 120))
///     }
/// }
/// ```
///
/// [`fetch_and_render`]: crate::network::fetch_and_render
pub trait FilingView: Send + Sync {
    /// Convert an HTML document body into this view's text format.
    fn render_html(&self, html: &str) -> Result<String, Box<dyn Error>>;

    /// Convert a plain-text document body into this view's text format.
    ///
    /// The default implementation passes the content through unchanged.
    fn render_text(&self, text: &str) -> String {
        text.to_string()
    }

    /// Produce a placeholder string for documents that cannot be decoded as
    /// text (PDF, XLSX, images, XBRL schemas, etc.).
    ///
    /// The default returns a bracketed notice embedding the MIME type and URL,
    /// suitable for inclusion inline with other rendered content.
    fn render_binary(&self, url: &str, mime: &str) -> String {
        format!("*[Binary document ({mime}) — not rendered: {url}]*")
    }
}
