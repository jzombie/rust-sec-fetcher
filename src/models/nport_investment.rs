use rust_decimal::Decimal;

// TODO: Document NPORT, etc.

#[derive(Debug)]
pub struct NportInvestment {
    pub name: String,
    pub lei: String, // Legal Entity Identifier
    pub title: String,
    pub cusip: String, // Committee on Uniform Securities Identification Procedures
    pub isin: String,  // International Securities Identification Number
    pub balance: Decimal,
    pub cur_cd: String,
    pub val_usd: Decimal,
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
