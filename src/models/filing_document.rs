/// Represents a single document entry within an SEC EDGAR filing index.
///
/// Parsed from the EDGAR HTML filing index page (`{accession}-index.htm`).
///
/// # Common `document_type` values
///
/// | Type pattern | Description |
/// |---|---|
/// | `"8-K"`, `"10-K"`, `"10-Q"` | Primary form body |
/// | `"EX-99.1"`, `"EX-99.2"` | Press release / earnings release |
/// | `"EX-10.1"` … `"EX-10.x"` | Material contracts |
/// | `"EX-21.1"` | Subsidiaries of the registrant |
/// | `"EX-23.1"` | Auditor / accountant consent |
/// | `"EX-31.1"`, `"EX-31.2"` | SOX § 302 certifications (CEO/CFO) |
/// | `"EX-32.1"`, `"EX-32.2"` | SOX § 906 certifications (CEO/CFO) |
/// | `"EX-101.INS"` … `"EX-101.PRE"` | XBRL instance / schema / label files |
/// | `"GRAPHIC"` | Image files (logos, charts) |
/// | `"XML"`, `""` | Ancillary machine-readable files |
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

    /// Returns `true` if this is a press release or earnings release exhibit.
    ///
    /// Matches `EX-99.x` — the standard type used for press releases,
    /// earnings tables, and Regulation FD voluntary disclosures.
    pub fn is_press_release(&self) -> bool {
        let t = self.document_type.to_uppercase();
        t.starts_with("EX-99")
    }

    /// Returns `true` if this is a Sarbanes-Oxley certification exhibit.
    ///
    /// Matches `EX-31.x` (§ 302 CEO/CFO certifications) and `EX-32.x`
    /// (§ 906 certifications).  These are short, formulaic legal documents
    /// with no analytical content.
    pub fn is_sarbanes_oxley_cert(&self) -> bool {
        let t = self.document_type.to_uppercase();
        t.starts_with("EX-31") || t.starts_with("EX-32")
    }

    /// Returns `true` if this is an auditor or accountant consent exhibit (`EX-23.x`).
    ///
    /// Consent forms are short single-page documents required when financial
    /// statements are incorporated by reference; they contain no analytical
    /// content.
    pub fn is_auditor_consent(&self) -> bool {
        self.document_type.to_uppercase().starts_with("EX-23")
    }

    /// Returns `true` if this is an XBRL structured data exhibit (`EX-101.*`).
    ///
    /// These are machine-readable schema, instance, label, and calculation
    /// documents that duplicate the financial data already present in the
    /// primary HTML document.  They contain no human-readable prose.
    pub fn is_xbrl_data(&self) -> bool {
        self.document_type.to_uppercase().starts_with("EX-101")
    }

    /// Returns `true` if this exhibit contains substantive, human-readable content.
    ///
    /// An exhibit is substantive if it:
    /// - Is an `EX-*` document, **and**
    /// - Is **not** a SOX certification (`EX-31.x`, `EX-32.x`)
    /// - Is **not** an auditor consent (`EX-23.x`)
    /// - Is **not** an XBRL data file (`EX-101.*`)
    /// - Is **not** a raw graphic (`GRAPHIC`)
    ///
    /// Substantive exhibits include press releases (`EX-99.x`), material
    /// contracts (`EX-10.x`), subsidiary lists (`EX-21.x`), and any other
    /// exhibit that contains actual prose or financial tables.
    ///
    /// Use [`FilingIndex::substantive_exhibits`] to filter a filing's documents
    /// to only those worth rendering or embedding.
    pub fn is_substantive_exhibit(&self) -> bool {
        if !self.is_exhibit() {
            return false;
        }
        if self.is_sarbanes_oxley_cert() || self.is_auditor_consent() || self.is_xbrl_data() {
            return false;
        }
        if self.document_type.to_uppercase() == "GRAPHIC" {
            return false;
        }
        true
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
    /// Returns all exhibit documents from this filing (any `EX-*` type).
    ///
    /// Includes everything: press releases, SOX certs, auditor consents,
    /// XBRL data files, and graphics.  Use [`Self::substantive_exhibits`]
    /// to filter to only human-readable content.
    pub fn exhibits(&self) -> Vec<&FilingDocument> {
        self.documents
            .iter()
            .filter(|doc| doc.is_exhibit())
            .collect()
    }

    /// Returns only the substantive exhibits — those containing human-readable
    /// prose or financial tables.
    ///
    /// Excludes SOX certifications, auditor consents, XBRL data files, and
    /// graphics.  This is the right default filter when rendering or embedding
    /// a filing's attachments.
    ///
    /// See [`FilingDocument::is_substantive_exhibit`] for the exact criteria.
    pub fn substantive_exhibits(&self) -> Vec<&FilingDocument> {
        self.documents
            .iter()
            .filter(|doc| doc.is_substantive_exhibit())
            .collect()
    }

    /// Returns only press release exhibits (`EX-99.x`).
    ///
    /// These are typically earnings announcements, Regulation FD voluntary
    /// disclosures, and other company-authored statements attached to 8-K
    /// filings.  To find earnings releases specifically, check the parent
    /// [`CikSubmission::is_earnings_release`] before fetching the index.
    ///
    /// [`CikSubmission::is_earnings_release`]: crate::models::CikSubmission::is_earnings_release
    pub fn press_releases(&self) -> Vec<&FilingDocument> {
        self.documents
            .iter()
            .filter(|doc| doc.is_press_release())
            .collect()
    }
}
