use crate::models::{Cik, CikError};

/// Represents an SEC **Accession Number**, which is a unique identifier for a filing.
///
/// An **Accession Number** is always **18 digits long** and consists of:
/// - **CIK (Central Index Key)** (10 digits): The SEC identifier of the filer.
/// - **Year (YY)** (2 digits): The last two digits of the filing year.
/// - **Sequence Number (NNNNNN)** (6 digits): A unique sequence number for filings within the same year.
///
/// # Format:
/// - `XXXXXXXXXX-YY-NNNNNN` (with dashes)
/// - `XXXXXXXXXXXXXXXXXX` (without dashes)
///
/// # Example:
/// ```rust
/// use sec_fetcher::models::{AccessionNumber, Cik};
///
/// let accession_number = AccessionNumber {
///     cik: Cik { value: 1234567890 },
///     year: 24,
///     sequence: 123456,
/// };
/// assert_eq!(accession_number.to_string(), "1234567890-24-123456");
/// ```
///
/// # Notes:
/// - The **CIK in an Accession Number** represents the **filer**, not necessarily the referenced entity.
/// - To retrieve filings, use the **Accession Number** in SEC's EDGAR system.
///
/// # Reference:
/// - [Search SEC Filings by Accession Number](https://www.sec.gov/edgar/searchedgar/companysearch.html)
/// - [SEC CIK Lookup](https://www.sec.gov/edgar/searchedgar/cik)
/// - [SEC CIK Data API (JSON)](https://www.sec.gov/files/company_tickers.json)

#[derive(Debug, Clone)]
pub struct AccessionNumber {
    pub cik: Cik,      // First 10 digits (zero-padded)
    pub year: u16,     // Next 2 digits (filing year)
    pub sequence: u32, // Last 6 digits (filing sequence number)
}

#[derive(Debug)]
pub enum AccessionNumberError {
    InvalidLength,
    ParseError(std::num::ParseIntError),
    CikError(CikError),
}

impl std::fmt::Display for AccessionNumberError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessionNumberError::InvalidLength => write!(
                f,
                "Accession number must be 18 digits (XXXXXXXXXX-YY-NNNNNN)"
            ),
            AccessionNumberError::ParseError(e) => {
                write!(f, "Failed to parse Accession Number: {}", e)
            }
            AccessionNumberError::CikError(e) => write!(f, "CIK Error: {}", e), // Display `CikError`
        }
    }
}

impl std::error::Error for AccessionNumberError {}

impl From<std::num::ParseIntError> for AccessionNumberError {
    fn from(err: std::num::ParseIntError) -> Self {
        AccessionNumberError::ParseError(err)
    }
}

impl From<CikError> for AccessionNumberError {
    fn from(err: CikError) -> Self {
        AccessionNumberError::CikError(err)
    }
}

impl AccessionNumber {
    /// Parses an **Accession Number** string (with or without dashes).
    ///
    /// - Expected format: `"XXXXXXXXXX-YY-NNNNNN"`
    /// - Example: `"0001234567-23-000045"`
    ///
    /// # Errors
    /// - Returns `AccessionNumberError::InvalidLength` if the cleaned string is not **18 digits**.
    /// - Returns `AccessionNumberError::ParseError` if parsing fails.
    pub fn from_str(accession_str: &str) -> Result<Self, AccessionNumberError> {
        // Remove dashes so we can handle both formatted and unformatted cases
        // let clean_str: String = accession_str.chars().filter(|c| c.is_ascii_digit()).collect();
        let clean_str: String = accession_str.chars().filter(|&c| c != '-').collect();

        // Ensure the total length is exactly 18 digits
        if clean_str.len() != 18 {
            return Err(AccessionNumberError::InvalidLength);
        }

        // Extract fields based on fixed positions
        let cik_part = &clean_str[0..10];
        let year_part = &clean_str[10..12];
        let sequence_part = &clean_str[12..18];

        // Ensure fields contain only digits before parsing
        if !cik_part.chars().all(|c| c.is_ascii_digit())
            || !year_part.chars().all(|c| c.is_ascii_digit())
            || !sequence_part.chars().all(|c| c.is_ascii_digit())
        {
            return Err(AccessionNumberError::ParseError(
                "Non-numeric character found".parse::<u64>().unwrap_err(),
            ));
        }

        // Parse fields into numbers
        let cik_u64 = cik_part.parse::<u64>()?;
        let cik = Cik::from_u64(cik_u64).map_err(AccessionNumberError::from)?;

        let year = year_part.parse::<u16>()?;
        let sequence = sequence_part.parse::<u32>()?;

        Ok(Self {
            cik,
            year,
            sequence,
        })
    }

    /// Creates an **Accession Number** from numeric components.
    ///
    /// - `cik` → 10-digit **CIK** number
    /// - `year` → 2-digit **filing year**
    /// - `sequence` → 6-digit **filing sequence**
    ///
    /// # Errors
    /// - Returns `AccessionNumberError::InvalidLength` if any field exceeds its expected length.
    pub fn from_parts(
        cik_u64: u64,
        year: u16,
        sequence: u32,
    ) -> Result<Self, AccessionNumberError> {
        if cik_u64 > 9_999_999_999 {
            return Err(AccessionNumberError::InvalidLength);
        }
        if year > 99 {
            return Err(AccessionNumberError::InvalidLength);
        }
        if sequence > 999_999 {
            return Err(AccessionNumberError::InvalidLength);
        }

        let cik = Cik::from_u64(cik_u64).map_err(AccessionNumberError::from)?;

        Ok(Self {
            cik,
            year,
            sequence,
        })
    }

    /// Converts the Accession Number to its **standard dash-separated format**.
    ///
    /// - Example Output: `"0001234567-23-000045"`
    pub fn to_string(&self) -> String {
        format!(
            "{:010}-{:02}-{:06}",
            self.cik.to_string(),
            self.year,
            self.sequence
        )
    }

    /// Returns the Accession Number as a **plain numeric string** (without dashes).
    ///
    /// # Example:
    /// ```rust
    /// use sec_fetcher::models::AccessionNumber;
    ///
    /// let accession = AccessionNumber::from_str("0001234567-23-000045").unwrap();
    /// assert_eq!(accession.to_unformatted_string(), "000123456723000045");
    /// ```
    pub fn to_unformatted_string(&self) -> String {
        format!(
            "{:010}{:02}{:06}",
            self.cik.to_string(),
            self.year,
            self.sequence
        )
    }
}
