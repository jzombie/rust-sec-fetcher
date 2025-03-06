use std::fs::File;
use std::io::{self, Write};

/// Writes a file from a byte array.
///
/// # Arguments
/// * `file_path` - The path where the CSV file should be written
/// * `data` - A byte array containing CSV data
///
/// # Returns
/// * `Ok(())` if successful, or an `io::Error` if writing fails
pub fn write_file(file_path: &str, data: &[u8]) -> io::Result<()> {
    let mut file = File::create(file_path)?;
    file.write_all(data)?;
    file.flush()?;
    Ok(())
}
