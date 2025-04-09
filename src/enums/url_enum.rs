use crate::models::{AccessionNumber, Cik};
pub type Year = usize;

pub enum Url {
    InvestmentCompanySeriesAndClassDataset(Year),
    CikSubmission(Cik),
    CikAccessionPrimaryDocument(Cik, AccessionNumber),
    CompanyTickers,
    CompanyFacts(Cik),
}

impl Url {
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
            Url::CikAccessionPrimaryDocument(cik, accession_number ) => format!(
                "https://www.sec.gov/Archives/edgar/data/{}/{}/primary_doc.xml",
                cik.to_string(),
                accession_number.to_unformatted_string()
            ),
            Url::CompanyTickers => "https://www.sec.gov/files/company_tickers.json".to_string(),
            Url::CompanyFacts(cik) => format!(
                "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json",
                cik.to_string()
            ),
        }
    }
}
