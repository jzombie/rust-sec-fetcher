use serde::{Deserialize, Serialize};
use std::cmp::{Eq, PartialEq};
use strum_macros::{Display, EnumString};

#[derive(Eq, PartialEq, Hash, Clone, EnumString, Display, Debug, Serialize, Deserialize)]
pub enum TickerOrigin {
    /// If derived directly from: fetch_company_tickers
    CompanyTickers,

    /// If derived from: fetch_investment_company_series_and_class_dataset
    Funds,
}
