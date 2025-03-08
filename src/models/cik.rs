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
    // TODO: From `AccessionNumber`

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
