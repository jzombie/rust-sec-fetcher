/// Formats a CIK by padding it with leading zeros to ensure 10 digits
pub fn format_cik(cik: i64) -> String {
    format!("{:010}", cik)
}
