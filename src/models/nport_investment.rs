// use crate::models::Ticker;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};

// TODO: Document NPORT, etc.

#[serde_as]
#[derive(Debug, Serialize, Deserialize)]
pub struct NportInvestment {
    // #[serde(default)] // Ensures empty string instead of omitting the column
    // pub company_ticker: Option<CompanyTicker>,
    pub mapped_ticker_symbol: Option<String>,
    pub mapped_company_name: Option<String>,
    pub mapped_company_cik_number: Option<String>,

    pub name: String,
    pub lei: String, // Legal Entity Identifier
    pub title: String,
    pub cusip: String, // Committee on Uniform Securities Identification Procedures
    pub isin: String,  // International Securities Identification Number

    #[serde_as(as = "DisplayFromStr")]
    pub balance: Decimal,

    pub cur_cd: String,

    #[serde_as(as = "DisplayFromStr")]
    pub val_usd: Decimal,

    #[serde_as(as = "DisplayFromStr")]
    pub pct_val: Decimal,

    pub payoff_profile: String,
    pub asset_cat: String,
    pub issuer_cat: String,
    pub inv_country: String,
}

impl NportInvestment {
    /// Sorts a list of investments by `pct_val` in descending order.
    pub fn sort_by_pct_val_desc(investments: &mut Vec<NportInvestment>) {
        investments.sort_by(|a, b| b.pct_val.cmp(&a.pct_val));
    }
}
