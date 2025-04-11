pub enum CacheNamespacePrefix {
    LatestFundsYear,
    CompanyTickerFuzzyMatch,
}

impl CacheNamespacePrefix {
    pub fn value(&self) -> &[u8] {
        match self {
            CacheNamespacePrefix::LatestFundsYear => {
                b"network::fetch_investment_company_series_and_class_dataset::latest_funds_year"
            }
            CacheNamespacePrefix::CompanyTickerFuzzyMatch => {
                b"CompanyTicker::get_by_fuzzy_matched_name"
            }
        }
    }
}
