use csv::Writer;
use serde::Serialize;
use std::error::Error;
use std::fs::File;

pub trait VecExtensions<T> {
    fn head(&self, count: usize) -> &[T];
    fn tail(&self, count: usize) -> &[T];

    /// Writes the vector's contents to a CSV file.
    ///
    /// # Arguments
    /// - `file_path`: The path to the output CSV file.
    ///
    /// # Errors
    /// Returns an error if writing fails.
    fn write_to_csv(&self, file_path: &str) -> Result<(), Box<dyn Error>>
    where
        T: Serialize;
}

impl<T> VecExtensions<T> for Vec<T> {
    fn head(&self, count: usize) -> &[T] {
        &self[..count.min(self.len())]
    }

    fn tail(&self, count: usize) -> &[T] {
        &self[self.len().saturating_sub(count)..]
    }

    fn write_to_csv(&self, file_path: &str) -> Result<(), Box<dyn Error>>
    where
        T: Serialize,
    {
        if self.is_empty() {
            println!("Warning: No data to write.");
            return Ok(());
        }

        println!("Writing {} records to CSV...", self.len());

        let file = File::create(file_path)?;
        let mut writer = Writer::from_writer(file);

        // Write headers explicitly from the first record
        let first_record = serde_json::to_value(&self[0])?;
        if let Some(obj) = first_record.as_object() {
            writer.write_record(obj.keys())?;
        } else {
            return Err("Failed to extract headers".into());
        }

        // Write all records as CSV rows
        for record in self.iter() {
            let record_value = serde_json::to_value(record)?;
            if let Some(obj) = record_value.as_object() {
                let row: Vec<String> = obj
                    .values()
                    .map(|v| v.as_str().unwrap_or("").to_string()) // Ensure consistent column count
                    .collect();
                writer.write_record(&row)?;
            }
        }

        writer.flush()?;
        println!("CSV writing completed successfully!");
        Ok(())
    }
}
