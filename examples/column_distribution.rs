use csv::ReaderBuilder;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Reads all CSV files in the given directory and counts occurrences of column headers.
fn count_column_occurrences(dir: &str) -> HashMap<String, usize> {
    let mut column_counts = HashMap::new();

    let files: Vec<_> = fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "csv"))
        .collect();

    let results: Vec<HashMap<String, usize>> = files
        .par_iter()
        .map(|entry| {
            let path = entry.path();
            if let Some(counts) = process_csv(&path) {
                counts
            } else {
                HashMap::new()
            }
        })
        .collect();

    // Aggregate results
    for map in results {
        for (col, count) in map {
            *column_counts.entry(col).or_insert(0) += count;
        }
    }

    column_counts
}

/// Processes a single CSV file and counts its column headers.
fn process_csv(path: &Path) -> Option<HashMap<String, usize>> {
    let mut counts = HashMap::new();
    let file = fs::File::open(path).ok()?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    if let Some(headers) = rdr.headers().ok() {
        for col in headers.iter() {
            *counts.entry(col.to_string()).or_insert(0) += 1;
        }
    }

    Some(counts)
}

fn main() {
    let dir = "data/us-gaap"; // Change this to your directory path
    let column_distribution = count_column_occurrences(dir);

    let mut sorted_columns: Vec<_> = column_distribution.iter().collect();
    sorted_columns.sort_by(|a, b| b.1.cmp(a.1)); // Sort by frequency (descending)

    println!("Column Distribution:");
    for (col, count) in sorted_columns {
        println!("{}: {}", col, count);
    }
}
