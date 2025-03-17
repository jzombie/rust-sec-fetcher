use crate::models::AccessionNumber;
use crate::models::CompanyTicker;
use std::collections::HashMap;
use std::error::Error;
use strsim::jaro_winkler;

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
#[derive(Debug, Clone)]
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
    /// - `company_tickers` - A slice of `CompanyTicker` instances.
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
        company_tickers: &[CompanyTicker],
        ticker_symbol: &str,
    ) -> Result<Cik, Box<dyn Error>> {
        company_tickers
            .iter()
            .find(|pred| pred.ticker_symbol == ticker_symbol)
            .map(|company_ticker| company_ticker.cik.clone())
            .ok_or_else(|| format!("Ticker symbol '{}' not found", ticker_symbol).into())
    }

    /// Finds the best-matching CIK based on fuzzy name similarity.
    ///
    /// - Uses **Jaro-Winkler similarity** for fuzzy matching.
    /// - Exact matches score highest.
    /// - Shorter tickers (common stock) get a boost.
    /// - Preferred stock symbols (containing '-') are penalized.
    /// - CIK frequency boosts score.
    ///
    /// # Arguments
    /// - `companies`: Slice of `CompanyTicker` structs.
    /// - `query`: The company name to search for.
    ///
    /// # Returns
    /// - `Some(Cik)`: Best matching CIK.
    /// - `None`: If no suitable match is found.
    pub fn get_cik_by_company_name(companies: &[CompanyTicker], query: &str) -> Option<Cik> {
        let mut cik_counts: HashMap<u64, usize> = HashMap::with_capacity(companies.len());
        let mut candidates: Vec<(u64, &str, Box<str>, f64)> = Vec::new(); // (CIK, ticker, title, score)

        let query_lower = query.to_lowercase();

        for company in companies {
            let cik = company.cik.value; // Extract raw u64
            let ticker = company.ticker_symbol.as_str();
            let title = company.company_name.to_lowercase();

            // Compute Jaro-Winkler similarity (range: 0.0 - 1.0)
            let similarity = jaro_winkler(&query_lower, &title);

            // TODO: Move scoring to constants
            // Only consider if similarity > 0.7 (adjust threshold as needed)
            if similarity > 0.7 {
                let mut score = similarity * 10.0; // Convert to a weighted score

                if title == query_lower {
                    score += 5.0; // Exact match boost
                }
                if ticker.len() <= 4 {
                    score += 3.0; // Common stock
                } else if ticker.contains('-') {
                    score -= 2.0; // Preferred stock penalty
                }

                // Count occurrences of CIK
                *cik_counts.entry(cik).or_insert(0) += 1;

                let boxed_title = title.into_boxed_str();
                candidates.push((cik, ticker, boxed_title, score));
            }
        }

        // Boost score based on CIK frequency
        for candidate in &mut candidates {
            candidate.3 += *cik_counts.get(&candidate.0).unwrap_or(&0) as f64;
        }

        // Return the best-matching CIK wrapped in `Cik`
        candidates
            .into_iter()
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
            .map(|(cik_value, _, _, _)| Cik { value: cik_value })
    }
}
