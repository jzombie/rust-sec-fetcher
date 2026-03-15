/// Represents a single document entry within an SEC EDGAR filing index.
///
/// Parsed from the EDGAR HTML filing index page (`{accession}-index.htm`).
///
/// Common `document_type` values:
/// - `"8-K"` — the primary form body
/// - `"EX-99.1"` — a press release or earnings release
/// - `"EX-10.1"`, `"EX-10.2"`, … — material contracts
/// - `"EX-23.1"` — auditor consent
/// - `"GRAPHIC"`, `"XML"`, `""` — ancillary files
#[derive(Debug, Clone)]
pub struct FilingDocument {
    /// File name within the filing archive (e.g. `"ex991.htm"`).
    pub name: String,

    /// SEC document type string (e.g. `"EX-99.1"`, `"8-K"`).
    pub document_type: String,
}

impl FilingDocument {
    /// Returns `true` if this document is an exhibit (type starts with `"EX-"`).
    pub fn is_exhibit(&self) -> bool {
        self.document_type.to_uppercase().starts_with("EX-")
    }

    /// Returns `true` if the file is an HTML document.
    pub fn is_html(&self) -> bool {
        let n = self.name.to_lowercase();
        n.ends_with(".htm") || n.ends_with(".html")
    }

    /// Returns `true` if the file is a plain-text document.
    pub fn is_text(&self) -> bool {
        self.name.to_lowercase().ends_with(".txt")
    }
}

/// The full document listing for a single SEC EDGAR filing.
///
/// Fetch via [`crate::network::fetch_filing_index`].
#[derive(Debug)]
pub struct FilingIndex {
    /// All documents listed in this filing archive.
    pub documents: Vec<FilingDocument>,
}

impl FilingIndex {
    /// Returns only the exhibit documents from this filing.
    pub fn exhibits(&self) -> Vec<&FilingDocument> {
        self.documents
            .iter()
            .filter(|doc| doc.is_exhibit())
            .collect()
    }
}
