use csv::{Reader, Writer};
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_cik_by_ticker_symbol, SecClient};
use sec_fetcher::transformers::distill_us_gaap_fundamental_concepts;
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
    let csv_directory = "data/orig.us-gaap/";

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
    let mut reader = Reader::from_path(csv_path)?;
    let headers = reader.headers()?.clone();
    println!("CSV Headers: {:?}", headers);

    // Collect all possible fundamental concepts mapped from the headers
    let mut fundamental_concept_headers = HashSet::new();

    for header in headers.iter() {
        if let Some(mapped_concepts) = distill_us_gaap_fundamental_concepts(header) {
            for concept in mapped_concepts {
                fundamental_concept_headers.insert(concept.to_string());
            }
        }
    }

    // Map column names to their indices based on filtered headers
    let header_indices: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter(|(_, h)| fundamental_concept_headers.contains(*h))
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

    {
        // Create CSV writer for output
        let mut writer = Writer::from_path("temp.csv")?;

        // Write header row
        writer.write_record(
            header_indices
                .iter()
                .map(|&i| headers.get(i).unwrap_or("UNKNOWN")),
        )?;

        // Write filtered rows
        for result in reader.records() {
            let record = result?;
            let filtered_record: Vec<&str> = header_indices
                .iter()
                .map(|&i| record.get(i).unwrap_or(""))
                .collect();
            writer.write_record(&filtered_record)?;
        }

        writer.flush()?;
        println!("Filtered data written to temp.csv");
    }

    Ok(())
}
