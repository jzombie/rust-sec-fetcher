/// Converts a numeric CIK to a zero-padded 10-digit string.
pub fn cik_to_string(cik: u64) -> String {
    format!("{:010}", cik)
}

/// Parses a zero-padded CIK string into a `u64`.
///
/// # Errors
/// Returns an error if the string is not a valid numeric CIK
/// or if it exceeds 10 characters.
pub fn cik_to_u64(cik_str: &str) -> Result<u64, std::num::ParseIntError> {
    if cik_str.len() > 10 {
        return "CIK length exceeds 10 digits".parse::<u64>(); // Forces an error
    }

    cik_str.parse::<u64>()
}
