use crate::network::SecClient;
use crate::rendering::{FilingContentType, FilingView};
use std::error::Error;

/// Fetches an SEC EDGAR document URL and renders it as text using `view`.
///
/// # Content-type detection
///
/// Detection follows this priority order:
///
/// 1. **URL extension fast-path** — known binary extensions (`.pdf`, `.xlsx`,
///    `.zip`, `.xsd`, `.png`, `.jpg`, `.gif`) skip the HTTP request entirely
///    and return the view's `render_binary` placeholder immediately.
/// 2. **HTTP `Content-Type` header** — `text/html` / `application/xhtml+xml`
///    → HTML path; any other `text/*` → plain-text path; everything else →
///    binary (response body is not read).
/// 3. **URL extension fallback** when no `Content-Type` header is present —
///    `.htm`/`.html` → HTML; default → plain text.
///
/// # Example
///
/// ```rust,no_run
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::network::{fetch_and_render, SecClient};
/// # use sec_fetcher::rendering::MarkdownView;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&config)?;
/// let url = "https://www.sec.gov/Archives/edgar/data/320193/000114036126006577/aapl-20260224.htm";
/// let text = fetch_and_render(&client, url, &MarkdownView).await?;
/// println!("{}", text);
/// # Ok(())
/// # }
/// ```
pub async fn fetch_and_render<V: FilingView>(
    client: &SecClient,
    url: &str,
    view: &V,
) -> Result<String, Box<dyn Error>> {
    // Fast-path: known binary extensions — skip the network round-trip
    if let Some(mime) = known_binary_extension(url) {
        return Ok(view.render_binary(url, mime));
    }

    let response = client
        .raw_request(reqwest::Method::GET, url, None, None)
        .await?;

    let content_type = sniff_content_type(&response, url);

    match content_type {
        FilingContentType::Binary { mime_type } => Ok(view.render_binary(url, &mime_type)),
        FilingContentType::Html => {
            let html = response.text().await?;
            view.render_html(&html)
        }
        FilingContentType::Text => {
            let text = response.text().await?;
            Ok(view.render_text(&text))
        }
    }
}

/// Returns a static MIME-type string if the URL's extension identifies a
/// known binary format that should never be decoded as text.  Returns `None`
/// if the content should be fetched and sniffed via the `Content-Type` header.
fn known_binary_extension(url: &str) -> Option<&'static str> {
    let without_query = url.split('?').next().unwrap_or(url);
    let path = without_query.split('#').next().unwrap_or(without_query);

    // rsplit gives the extension as the first element; guard against no-dot paths
    let ext = path.rsplit('.').next().unwrap_or("");
    if ext.len() > 5 || ext == path {
        return None;
    }

    match ext.to_ascii_lowercase().as_str() {
        "pdf" => Some("application/pdf"),
        "xlsx" | "xls" => Some("application/vnd.ms-excel"),
        "zip" => Some("application/zip"),
        "xsd" => Some("application/xml"),
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        _ => None,
    }
}

/// Determines [`FilingContentType`] for a response that has already been
/// received (headers available, body not yet read).
fn sniff_content_type(response: &reqwest::Response, url: &str) -> FilingContentType {
    if let Some(ct_header) = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
    {
        let ct_lower = ct_header.to_ascii_lowercase();
        let mime: &str = ct_lower.split(';').next().unwrap_or(ct_lower.as_str()).trim();

        if mime == "text/html" || mime == "application/xhtml+xml" {
            return FilingContentType::Html;
        }
        // EDGAR serves inline XBRL (iXBRL) HTML filings as application/xml.
        // Treat these as HTML when the URL extension confirms they are .htm/.html.
        if mime == "application/xml" {
            let path_lower = url.split('?').next().unwrap_or(url).to_ascii_lowercase();
            if path_lower.ends_with(".htm") || path_lower.ends_with(".html") {
                return FilingContentType::Html;
            }
        }
        if mime.starts_with("text/") {
            return FilingContentType::Text;
        }
        // Non-text MIME type → binary (do not read body)
        return FilingContentType::Binary {
            mime_type: mime.to_string(),
        };
    }

    // No Content-Type header — guess from URL extension
    let without_query = url.split('?').next().unwrap_or(url);
    let path = without_query.split('#').next().unwrap_or(without_query);
    let path_lower = path.to_ascii_lowercase();

    if path_lower.ends_with(".htm") || path_lower.ends_with(".html") {
        FilingContentType::Html
    } else {
        // Optimistic default: treat unknown extensions as plain text
        FilingContentType::Text
    }
}
