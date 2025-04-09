pub type Year = usize;

pub enum Url {
    InvestmentCompanySeriesAndClassDataset(Year),
}

impl Url {
    pub fn value(&self) -> String {
        match &self {
            Url::InvestmentCompanySeriesAndClassDataset(year) => format!(
                "https://www.sec.gov/files/investment/data/other/investment-company-series-and-class-information/investment-company-series-class-{}.csv",
                year
            )
        }
    }
}
