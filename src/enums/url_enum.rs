use crate::models::{AccessionNumber, Cik};
pub type Year = usize;

/// A convenience enum for accessing structured SEC URLs.
pub enum Url {
    /// Points to the SEC's Investment Company Series and Class dataset CSV
    /// for a given year.
    InvestmentCompanySeriesAndClassDataset(Year),

    /// Points to the JSON submission metadata for a specific CIK.
    CikSubmission(Cik),

    /// Points to a paginated submissions file (e.g. CIK0000320193-submissions-001.json).
    CikSubmissionPage(String),

    CikAccession(Cik, AccessionNumber),

    /// Points to the human-readable EDGAR filing index page for a specific filing.
    /// Format: https://www.sec.gov/Archives/edgar/data/{CIK}/{accn_unformatted}/{accn_formatted}-index.htm
    CikAccessionIndex(Cik, AccessionNumber),

    /// Points to the `primary_doc.xml` of a specific filing, using
    /// CIK and Accession Number.
    CikAccessionPrimaryDocument(Cik, AccessionNumber),

    /// Points to a named document within a specific filing archive.
    /// Format: https://www.sec.gov/Archives/edgar/data/{CIK}/{accn_unformatted}/{filename}
    CikAccessionDocument(Cik, AccessionNumber, String),

    /// Points to the full company ticker list JSON from the SEC.
    CompanyTickers,

    /// Points to the `companyfacts` XBRL JSON API for a specific CIK.
    CompanyFacts(Cik),

    /// The EDGAR "current filings" Atom feed, optionally filtered by form type.
    ///
    /// Set `form_type` to an empty string to receive all form types (the full
    /// firehose). EDGAR caps `count` at 40 entries per request. Entries are
    /// ordered newest-first by their acceptance timestamp — use the `<updated>`
    /// value of the most-recent entry as a high-water mark for delta polling.
    ///
    /// `before` is the optional `dateb` parameter (format: `"YYYYMMDDHHmmss"`).
    /// Leave empty for the latest entries. Set to the acceptance timestamp of
    /// the *oldest* entry in a batch to walk backwards (pagination).
    ///
    /// See [`crate::network::fetch_edgar_feed`] and
    /// [`crate::network::fetch_edgar_feed_page`].
    EdgarCurrentFeed {
        form_type: String,
        count: usize,
        before: String,
    },

    /// Per-company Atom feed for a specific CIK, optionally filtered by form type.
    ///
    /// Useful for tracking a watchlist of companies. Returns richer structured
    /// data per entry than the global feed (company metadata, SIC code, etc.).
    /// Same `count` cap and `updated`-based delta semantics as `EdgarCurrentFeed`.
    /// Supports the same `before` / `dateb` pagination parameter.
    EdgarCompanyFeed {
        cik: Cik,
        form_type: String,
        count: usize,
        before: String,
    },
}

impl Url {
    /// Returns the fully qualified URL string corresponding to the
    /// specified variant.
    pub fn value(&self) -> String {
        match &self {
            Url::InvestmentCompanySeriesAndClassDataset(year) => format!(
                "https://www.sec.gov/files/investment/data/other/investment-company-series-and-class-information/investment-company-series-class-{}.csv",
                year
            ),
            Url::CikSubmission(cik) => format!(
                "https://data.sec.gov/submissions/CIK{}.json",
                cik.to_string()
            ),
            Url::CikSubmissionPage(filename) => format!(
                "https://data.sec.gov/submissions/{}",
                filename
            ),
            Url::CikAccession(cik, accession_number) => format!(
                "https://www.sec.gov/Archives/edgar/data/{}/{}",
                cik.to_string(),
                accession_number.to_unformatted_string()
            ),
            Url::CikAccessionIndex(cik, accession_number) => format!(
                "https://www.sec.gov/Archives/edgar/data/{}/{}/{}-index.htm",
                cik.to_string(),
                accession_number.to_unformatted_string(),
                accession_number.to_string(),
            ),
            Url::CikAccessionPrimaryDocument(cik, accession_number ) => format!(
                "{}/primary_doc.xml",
                Url::CikAccession(cik.clone(), accession_number.clone()).value(),
            ),
            Url::CikAccessionDocument(cik, accession_number, filename) => format!(
                "{}/{}",
                Url::CikAccession(cik.clone(), accession_number.clone()).value(),
                filename,
            ),
            Url::CompanyTickers => "https://www.sec.gov/files/company_tickers.json".to_string(),
            Url::CompanyFacts(cik) => format!(
                "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json",
                cik.to_string()
            ),
            Url::EdgarCurrentFeed { form_type, count, before } => format!(
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={}&dateb={}&owner=include&count={}&search_text=&output=atom",
                form_type, before, count
            ),
            Url::EdgarCompanyFeed { cik, form_type, count, before } => format!(
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type={}&dateb={}&owner=include&count={}&search_text=&output=atom",
                cik.to_string(),
                form_type,
                before,
                count
            ),
        }
    }
}
