use chrono::NaiveDate;

use crate::models::{AccessionNumber, Cik, TickerSymbol};

/// Represents a single custom→standard tag anchoring relationship extracted
/// from a filing's XBRL definition linkbase (`EX-101.DEF`).
///
/// When a filer defines a custom extension tag (e.g. `aapl:MyCustomRevenue`),
/// the definition linkbase links it to a standard US-GAAP concept via a
/// `definitionArc` with a `general-special` arcrole.  This struct captures
/// that relationship.
///
/// # Note
///
/// The SEC's companyfacts API does **not** surface filer-specific extension
/// tags — it only contains facts from recognised taxonomies (`us-gaap`, `dei`,
/// `ffd`, `cef`, `ifrs-full`, etc.).  Filer-specific extension tags live only
/// in per-filing XBRL packages on EDGAR.  This struct is useful when processing
/// those packages, but cannot be populated from companyfacts alone.
#[derive(Debug, Clone)]
pub struct CustomTagAnchoring {
    /// Ticker symbol of the filer.
    pub ticker: TickerSymbol,
    /// Central Index Key of the filer.
    pub cik: Cik,
    /// Accession number of the filing containing this anchoring.
    pub accn: AccessionNumber,
    /// Filing date (if available from the submission metadata).
    pub filing_date: Option<NaiveDate>,
    /// SEC form type string (e.g. `"10-K"`, `"10-Q"`).
    // TODO: unify with CikSubmission.form — both should use `FormType`
    //       instead of String for stronger typing.
    pub form: String,
    /// Namespace prefix of the custom tag (e.g. `"aapl"`).
    pub custom_namespace: String,
    /// Local name of the custom tag (e.g. `"MyCustomRevenue"`).
    pub custom_tag: String,
    /// Namespace prefix of the standard tag (e.g. `"us-gaap"`).
    pub standard_namespace: String,
    /// Local name of the standard tag (e.g. `"Revenues"`).
    pub standard_tag: String,
    /// The definition arcrole describing the relationship.
    /// Typical values:
    /// - `"general-special"` — the custom tag is a specialization of the standard
    /// - `"domain-member"` — dimensional relationship
    pub arcrole: String,
}
