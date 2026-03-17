use crate::enums::Url;
use chrono::NaiveDate;

/// One row from an EDGAR full-index `master.idx` file.
///
/// The SEC publishes a `master.idx` for every calendar quarter going back to
/// Q4 1993. Each row records a single filing submission with the filer's CIK,
/// company name, form type, the date it was accepted, and the path to the
/// filing document within the EDGAR archive.
///
/// Fetch the file via [`crate::network::fetch_edgar_master_index`].
#[derive(Debug, Clone)]
pub struct MasterIndexEntry {
    /// Filer's CIK (stored as a string; some historic entries omit leading zeros).
    pub cik: String,
    pub company_name: String,
    pub form_type: String,
    pub date_filed: NaiveDate,
    /// Relative EDGAR archive path, e.g.
    /// `edgar/data/1000032/0001000032-24-000006-index.htm`.
    ///
    /// Use [`MasterIndexEntry::as_url`] to obtain the fully qualified URL.
    pub filename: String,
}

impl MasterIndexEntry {
    /// Returns the full SEC URL to the filing document.
    pub fn as_url(&self) -> String {
        Url::EdgarArchive(self.filename.clone()).value()
    }
}
