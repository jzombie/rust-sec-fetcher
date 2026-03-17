use clap::Parser;
use csv::ReaderBuilder;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(about = "Count column-header frequencies across all CSV files in a directory")]
struct Args {
    /// Directory containing the CSV files to analyze
    #[arg(default_value = "data/us-gaap")]
    dir: String,
}

/// A column name together with the number of CSV files it appears in.
struct ColumnCount {
    name: String,
    count: usize,
}

impl fmt::Display for ColumnCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.count)
    }
}

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
    let args = Args::parse();
    let column_distribution = count_column_occurrences(&args.dir);

    let mut sorted_columns: Vec<ColumnCount> = column_distribution
        .into_iter()
        .map(|(name, count)| ColumnCount { name, count })
        .collect();
    sorted_columns.sort_by(|a, b| b.count.cmp(&a.count));

    println!("Column Distribution:");
    for col in &sorted_columns {
        println!("{}", col);
    }
}
