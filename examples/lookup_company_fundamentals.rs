use csv::Reader;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::enums::FundamentalConcept;
use sec_fetcher::network::{fetch_cik_by_ticker_symbol, SecClient};
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use strum::IntoEnumIterator;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <TICKER_SYMBOL>", args[0]);
        std::process::exit(1);
    }

    let ticker_symbol = &args[1].to_uppercase(); // Ensure uppercase for consistency
    let csv_directory = "data/us-gaap/";

    // Find the CSV file matching the ticker symbol
    let csv_file_path = find_csv_for_ticker(csv_directory, ticker_symbol);

    match csv_file_path {
        Some(path) => {
            println!("Found CSV file: {}", path);
            preload_csv(&path)?;
        }
        None => {
            println!("No CSV file found for ticker '{}'.", ticker_symbol);
        }
    }

    Ok(())
}

fn find_csv_for_ticker(directory: &str, ticker: &str) -> Option<String> {
    let paths = fs::read_dir(directory).ok()?;

    for entry in paths {
        if let Ok(entry) = entry {
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();

            if file_name_str.to_uppercase().contains(ticker) && file_name_str.ends_with(".csv") {
                return Some(entry.path().to_string_lossy().to_string());
            }
        }
    }
    None
}

fn preload_csv(csv_path: &str) -> Result<(), Box<dyn Error>> {
    let fundamental_concepts: HashSet<String> = FundamentalConcept::iter()
        .map(|concept| concept.to_string())
        .collect();

    let mut reader = Reader::from_path(csv_path)?;
    let headers = reader.headers()?.clone();
    println!("CSV Headers: {:?}", headers);

    // Map column names to their indices
    let header_indices: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter(|(_, h)| fundamental_concepts.contains(*h))
        .map(|(i, _)| i)
        .collect();

    println!(
        "Filtered Headers: {:?}",
        header_indices
            .iter()
            .map(|&i| headers.get(i).unwrap_or("UNKNOWN"))
            .collect::<Vec<_>>()
    );

    for result in reader.records().take(5) {
        let record = result?;
        let filtered_record: Vec<&str> = header_indices
            .iter()
            .map(|&i| record.get(i).unwrap_or(""))
            .collect();
        println!("{:?}", filtered_record);
    }

    Ok(())
}
