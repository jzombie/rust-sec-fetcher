// TODO: Document terminology
#[derive(Debug, Clone)]
pub struct AccessionNumber {
    // TODO: Use Cik model type here
    pub cik: u64,       // First 10 digits (zero-padded)
    pub year: u16,      // Next 2 digits (filing year)
    pub sequence: u32,  // Last 6 digits (filing sequence number)
}

#[derive(Debug)]
pub enum AccessionNumberError {
    InvalidLength,
    ParseError(std::num::ParseIntError),
}

impl std::fmt::Display for AccessionNumberError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessionNumberError::InvalidLength => write!(f, "Accession number must be 18 digits (XXXXXXXXXX-YY-NNNNNN)"),
            AccessionNumberError::ParseError(e) => write!(f, "Failed to parse Accession Number: {}", e),
        }
    }
}

impl std::error::Error for AccessionNumberError {}

impl From<std::num::ParseIntError> for AccessionNumberError {
    fn from(err: std::num::ParseIntError) -> Self {
        AccessionNumberError::ParseError(err)
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
        let clean_str: String = accession_str.chars().filter(|c| c.is_ascii_digit()).collect();
    
        // Ensure the total length is exactly 18 digits
        if clean_str.len() != 18 {
            return Err(AccessionNumberError::InvalidLength);
        }
    
        // Extract fields based on fixed positions
        let cik_part = &clean_str[0..10];
        let year_part = &clean_str[10..12];
        let sequence_part = &clean_str[12..18];
    
        // Ensure fields contain only numbers before parsing
        if !clean_str.chars().all(|c| c.is_ascii_digit()) {
            return Err(AccessionNumberError::ParseError(
                "Non-numeric character found".parse::<u64>().unwrap_err(),
            ));
        }

        // Parse fields into numbers
        let cik = cik_part.parse::<u64>()?;
        let year = year_part.parse::<u16>()?;
        let sequence = sequence_part.parse::<u32>()?;
    
        Ok(Self { cik, year, sequence })
    }    
     

    /// Creates an **Accession Number** from numeric components.
    ///
    /// - `cik` → 10-digit **CIK** number
    /// - `year` → 2-digit **filing year**
    /// - `sequence` → 6-digit **filing sequence**
    ///
    /// # Errors
    /// - Returns `AccessionNumberError::InvalidLength` if any field exceeds its expected length.
    pub fn from_parts(cik: u64, year: u16, sequence: u32) -> Result<Self, AccessionNumberError> {
        if cik > 9_999_999_999 {
            return Err(AccessionNumberError::InvalidLength);
        }
        if year > 99 {
            return Err(AccessionNumberError::InvalidLength);
        }
        if sequence > 999_999 {
            return Err(AccessionNumberError::InvalidLength);
        }

        Ok(Self { cik, year, sequence })
    }

    /// Converts the Accession Number to its **standard dash-separated format**.
    ///
    /// - Example Output: `"0001234567-23-000045"`
    pub fn to_string(&self) -> String {
        format!("{:010}-{:02}-{:06}", self.cik, self.year, self.sequence)
    }
}
