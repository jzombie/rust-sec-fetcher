// TODO: Use numeric values where possible
// TODO: Document NPORT, etc.

#[derive(Debug)]
pub struct NportInvestment {
    pub name: String,
    pub lei: String, // Legal Entity Identifier
    pub title: String,
    pub cusip: String, // Committee on Uniform Securities Identification Procedures
    pub isin: String,  // International Securities Identification Number
    pub balance: String,
    pub cur_cd: String,
    pub val_usd: String,
    pub pct_val: String,
    pub payoff_profile: String,
    pub asset_cat: String,
    pub issuer_cat: String,
    pub inv_country: String,
}
