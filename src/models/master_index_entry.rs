use crate::enums::{FormType, Url};
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

    /// Returns the [`FormType`] for this entry, parsed from the raw `form_type` string.
    ///
    /// Unknown form types are returned as [`FormType::Other`].
    pub fn form_type(&self) -> FormType {
        self.form_type.parse().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn make_entry() -> MasterIndexEntry {
        MasterIndexEntry {
            cik: "320193".to_string(),
            company_name: "APPLE INC".to_string(),
            form_type: "10-K".to_string(),
            date_filed: NaiveDate::from_ymd_opt(2024, 11, 1).unwrap(),
            filename: "edgar/data/320193/0000320193-24-000123-index.htm".to_string(),
        }
    }

    #[test]
    fn as_url_returns_edgar_archive_url() {
        let entry = make_entry();
        let url = entry.as_url();
        assert!(url.contains("edgar/data/320193/0000320193-24-000123-index.htm"));
        assert!(url.starts_with("https://"));
    }

    #[test]
    fn form_type_parses_known_type() {
        let entry = make_entry();
        let ft = entry.form_type();
        assert_eq!(ft.as_edgar_str(), "10-K");
    }

    #[test]
    fn form_type_parses_unknown_as_other() {
        let entry = MasterIndexEntry {
            form_type: "SOMETHING_WEIRD".to_string(),
            ..make_entry()
        };
        let ft = entry.form_type();
        assert_eq!(ft.as_edgar_str(), "Other");
    }

    #[test]
    fn debug_format_includes_fields() {
        let entry = make_entry();
        let debug = format!("{:?}", entry);
        assert!(debug.contains("320193"));
        assert!(debug.contains("APPLE INC"));
        assert!(debug.contains("10-K"));
    }
}
