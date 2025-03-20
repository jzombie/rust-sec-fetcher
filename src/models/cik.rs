use serde::Deserialize;
use serde::Serialize;

use crate::models::AccessionNumber;
use crate::models::Ticker;

use std::error::Error;

/// Represents an SEC **CIK (Central Index Key)**, a unique identifier for entities.
///
/// A **CIK** is a **permanent 10-digit identifier** assigned by the SEC to:
/// - Public companies
/// - Mutual funds
/// - Insiders (e.g., executives, large shareholders)
/// - Broker-dealers
///
/// # Format:
/// - Always a **10-digit zero-padded number**.
/// - Example: `"0000320193"` (Apple Inc.)
///
/// # Example:
/// ```rust
/// use sec_fetcher::models::Cik;
///
/// let cik = Cik { value: 320193 };
/// assert_eq!(cik.to_string(), "0000320193");
/// ```
///
/// # Notes:
/// - CIKs are **not reassigned or reused**.
/// - Used in SEC filings to track registrants, insiders, and related entities.
///
/// # Reference:
/// - [SEC CIK Lookup](https://www.sec.gov/edgar/searchedgar/cik)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cik {
    pub value: u64,
}

#[derive(Debug)]
pub enum CikError {
    InvalidLength,
    ParseError(std::num::ParseIntError),
}

impl std::fmt::Display for CikError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CikError::InvalidLength => write!(f, "CIK length exceeds 10 digits"),
            CikError::ParseError(e) => write!(f, "Failed to parse CIK: {}", e),
        }
    }
}

impl std::error::Error for CikError {}

impl From<std::num::ParseIntError> for CikError {
    fn from(err: std::num::ParseIntError) -> Self {
        CikError::ParseError(err)
    }
}

impl Cik {
    /// Creates a `Cik` instance from a `AccessionNumber` instance.
    ///
    /// # Example:
    /// ```rust
    /// use sec_fetcher::models::{AccessionNumber, Cik};
    ///
    /// let accession = AccessionNumber::from_str("0001234567-23-000045").unwrap();
    /// let cik = Cik::from_accession_number(&accession);
    /// assert_eq!(cik.to_u64(), accession.cik.to_u64());
    /// ```
    pub fn from_accession_number(accession_number: &AccessionNumber) -> Self {
        accession_number.cik.clone()
    }

    /// Creates a `Cik` instance from a `u64` value.
    ///
    /// # Errors
    /// - Returns `CikError::InvalidLength` if the numeric value exceeds 10 digits.
    pub fn from_u64(cik_u64: u64) -> Result<Self, CikError> {
        if cik_u64 > 9_999_999_999 {
            return Err(CikError::InvalidLength);
        }

        Ok(Self { value: cik_u64 })
    }

    /// Parses a zero-padded CIK string into a `Cik` struct.
    ///
    /// # Errors
    /// - Returns `CikError::InvalidLength` if the string exceeds 10 characters.
    /// - Returns `CikError::ParseError` if the string is not a valid number.
    pub fn from_str(cik_str: &str) -> Result<Self, CikError> {
        if cik_str.len() > 10 {
            return Err(CikError::InvalidLength);
        }

        let value = cik_str.parse::<u64>()?;
        Ok(Self { value })
    }

    /// Converts the CIK to a zero-padded 10-digit string.
    pub fn to_string(&self) -> String {
        format!("{:010}", self.value)
    }

    /// Converts the CIK to a u64.
    pub fn to_u64(&self) -> u64 {
        self.value
    }
}

impl Cik {
    /// Retrieves the **CIK (Central Index Key)** for a given **ticker symbol**
    /// from the **SEC tickers DataFrame**.
    ///
    ///
    /// # Arguments
    /// - `company_tickers` - A slice of `CompanyTicker` structs.
    /// - `ticker_symbol` - A **stock ticker symbol** (case-sensitive) as a `&str`.
    ///
    /// # Returns
    /// - `Ok(Cik)` - A `Cik` model instance.
    /// - `Err(Box<dyn Error>)` - If:
    ///   - The ticker column is not a string type.
    ///   - The CIK column is not a string type.
    ///   - The ticker symbol is not found in the DataFrame.
    ///   - The corresponding CIK value is missing.
    ///
    /// # Response Format
    /// The returned **CIK** will be a **zero-padded 10-digit string** (e.g., `"0000320193"`),
    /// ensuring consistency with SEC filings.
    ///
    /// # Errors
    /// - Returns an **error** if the dataset does not contain the given ticker,
    ///   or if the data is incorrectly formatted.
    pub fn get_company_cik_by_ticker_symbol(
        company_tickers: &[Ticker],
        ticker_symbol: &str,
    ) -> Result<Cik, Box<dyn Error>> {
        company_tickers
            .iter()
            .find(|pred| pred.symbol == ticker_symbol)
            .map(|company_ticker| company_ticker.cik.clone())
            .ok_or_else(|| format!("Ticker symbol '{}' not found", ticker_symbol).into())
    }
}
