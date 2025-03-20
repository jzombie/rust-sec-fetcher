use serde::{Deserialize, Serialize};
use std::cmp::{Eq, PartialEq};
use strum_macros::{Display, EnumString};

#[derive(Eq, PartialEq, Hash, Clone, EnumString, Display, Debug, Serialize, Deserialize)]
pub enum TickerOrigin {
    // TODO: Rename to `OperatingCompanies`?
    /// If derived directly from: fetch_company_tickers
    CompanyTickers,

    // TODO: Rename to `InvestmentCompanies`?
    /// If derived from: fetch_investment_company_series_and_class_dataset
    Funds,
}
