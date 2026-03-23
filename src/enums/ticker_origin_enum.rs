use bitcode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::cmp::{Eq, PartialEq};
use strum_macros::{Display, EnumString};

#[derive(
    Eq, PartialEq, Hash, Clone, EnumString, Display, Debug, Serialize, Deserialize, Encode, Decode,
)]
pub enum TickerOrigin {
    /// Primary listing from `company_tickers.json`.
    ///
    /// One entry per exchange-listed **operating company** common-stock ticker.
    /// Always has a company name.  These are the only tickers expected to have
    /// US-GAAP XBRL data at the `companyfacts` endpoint.
    ///
    /// Not to be confused with [`TickerOrigin::InvestmentCompany`], which
    /// covers mutual funds and ETFs registered under the Investment Company Act.
    PrimaryListing,

    /// Derived instrument from `ticker.txt`.
    ///
    /// Includes warrants (`-WT`), units (`-UN`), preferred share classes
    /// (`-PA`, `-PB`, …), ADRs, and defunct/delisted tickers that no longer
    /// appear in `company_tickers.json`.  These entries typically share a
    /// CIK with a [`PrimaryListing`] entry and almost never carry independent
    /// US-GAAP XBRL data.
    ///
    /// [`PrimaryListing`]: TickerOrigin::PrimaryListing
    DerivedInstrument,

    /// Fund share class from `fetch_investment_company_series_and_class_dataset`.
    ///
    /// Mutual fund and ETF share classes registered under the Investment
    /// Company Act.  Not expected to have operating-company US-GAAP facts.
    InvestmentCompany,
}
