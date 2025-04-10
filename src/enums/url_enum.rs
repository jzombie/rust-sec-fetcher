use crate::models::{AccessionNumber, Cik};
pub type Year = usize;

/// A convenience enum for accessing structured SEC URLs.
pub enum Url {
    /// Points to the SEC's Investment Company Series and Class dataset CSV
    /// for a given year.
    InvestmentCompanySeriesAndClassDataset(Year),

    /// Points to the JSON submission metadata for a specific CIK.
    CikSubmission(Cik),

    CikAccession(Cik, AccessionNumber),

    /// Points to the `primary_doc.xml` of a specific filing, using
    /// CIK and Accession Number.
    CikAccessionPrimaryDocument(Cik, AccessionNumber),

    /// Points to the full company ticker list JSON from the SEC.
    CompanyTickers,

    /// Points to the `companyfacts` XBRL JSON API for a specific CIK.
    CompanyFacts(Cik),
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
            Url::CikAccession(cik, accession_number ) => format!(
                "https://www.sec.gov/Archives/edgar/data/{}/{}",
                cik.to_string(),
                accession_number.to_unformatted_string()
            ),
            Url::CikAccessionPrimaryDocument(cik, accession_number ) => format!(
                "{}/primary_doc.xml",
                Url::CikAccession(cik.clone(), accession_number.clone()).value(),
            ),
            Url::CompanyTickers => "https://www.sec.gov/files/company_tickers.json".to_string(),
            Url::CompanyFacts(cik) => format!(
                "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json",
                cik.to_string()
            ),
        }
    }
}
