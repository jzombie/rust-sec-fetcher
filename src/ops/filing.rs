use crate::enums::Url;
use crate::models::{CikSubmission, FilingDocument};
use crate::network::{fetch_and_render, fetch_filing_index, SecClient};
use crate::views::FilingView;
use std::error::Error;

/// A single rendered exhibit from a filing.
pub struct RenderedExhibit {
    /// SEC document type (e.g. `"EX-99.1"`, `"EX-10.1"`).
    pub document_type: String,
    /// Filename within the filing archive (e.g. `"ex991.htm"`).
    pub name: String,
    /// Full EDGAR URL to the exhibit document.
    pub url: String,
    /// Rendered text content (Markdown or embedding prose).
    pub content: String,
}

/// The rendered content of a filing — body and/or exhibits.
pub struct RenderedFiling {
    /// Rendered text of the primary document, `None` if not requested.
    pub body: Option<String>,
    /// Rendered text of each exhibit included per the request.
    pub exhibits: Vec<RenderedExhibit>,
}

/// Renders a filing's primary document and/or its **substantive** exhibits.
///
/// "Substantive" means exhibits containing human-readable prose or financial
/// tables — press releases, material contracts, subsidiary lists, etc.
/// Boilerplate is excluded: SOX certifications (`EX-31.x`, `EX-32.x`),
/// auditor consents (`EX-23.x`), XBRL data files (`EX-101.*`), and graphics.
///
/// Use [`render_all_exhibits`] if you need every exhibit without filtering.
///
/// # Example
///
/// ```rust,no_run
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::network::{fetch_cik_by_ticker_symbol, fetch_filings, SecClient};
/// # use sec_fetcher::models::TickerSymbol;
/// # use sec_fetcher::ops::filing::render_filing;
/// # use sec_fetcher::views::EmbeddingTextView;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("LLY")).await?;
/// let filings = fetch_filings(&client, cik, "8-K").await?;
/// let latest = filings.first().unwrap();
/// let rendered = render_filing(&client, latest, true, true, &EmbeddingTextView).await?;
/// if let Some(body) = rendered.body {
///     println!("{}", body);
/// }
/// for ex in rendered.exhibits {
///     println!("--- {} ---\n{}", ex.document_type, ex.content);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn render_filing<V: FilingView>(
    client: &SecClient,
    filing: &CikSubmission,
    render_body: bool,
    render_exhibits: bool,
    view: &V,
) -> Result<RenderedFiling, Box<dyn Error>> {
    let body = if render_body {
        let url = filing.as_primary_document_url();
        Some(fetch_and_render(client, &url, view).await?)
    } else {
        None
    };

    let exhibits = if render_exhibits {
        let index = fetch_filing_index(client, filing).await?;
        let docs: Vec<FilingDocument> = index.substantive_exhibits().into_iter().cloned().collect();
        render_exhibit_docs(client, filing, &docs, view).await?
    } else {
        Vec::new()
    };

    Ok(RenderedFiling { body, exhibits })
}

/// Renders **all** exhibits from a filing, including non-substantive ones
/// (SOX certifications, auditor consents, XBRL files, and graphics).
///
/// Most callers should prefer [`render_filing`] with `render_exhibits: true`,
/// which filters to only human-readable content.  Use this variant when
/// you need the complete set.
pub async fn render_all_exhibits<V: FilingView>(
    client: &SecClient,
    filing: &CikSubmission,
    view: &V,
) -> Result<Vec<RenderedExhibit>, Box<dyn Error>> {
    let index = fetch_filing_index(client, filing).await?;
    let docs: Vec<FilingDocument> = index.exhibits().into_iter().cloned().collect();
    render_exhibit_docs(client, filing, &docs, view).await
}

/// Renders a single [`FilingDocument`] from a filing, returning the text
/// content together with its URL.
///
/// Use this when you already have the exhibit list and want to render one
/// document at a time (e.g. after filtering press releases from a 13F walk).
pub async fn render_exhibit_doc<V: FilingView>(
    client: &SecClient,
    filing: &CikSubmission,
    doc: &FilingDocument,
    view: &V,
) -> Result<RenderedExhibit, Box<dyn Error>> {
    let url = Url::CikAccessionDocument(
        filing.cik.clone(),
        filing.accession_number.clone(),
        doc.name.clone(),
    )
    .value();
    let content = fetch_and_render(client, &url, view).await?;
    Ok(RenderedExhibit {
        document_type: doc.document_type.clone(),
        name: doc.name.clone(),
        url,
        content,
    })
}

// ── internal helper ──────────────────────────────────────────────────────────

async fn render_exhibit_docs<V: FilingView>(
    client: &SecClient,
    filing: &CikSubmission,
    docs: &[FilingDocument],
    view: &V,
) -> Result<Vec<RenderedExhibit>, Box<dyn Error>> {
    let mut rendered = Vec::with_capacity(docs.len());
    for doc in docs {
        rendered.push(render_exhibit_doc(client, filing, doc, view).await?);
    }
    Ok(rendered)
}
