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

    let additional_fields = ["fy", "fp", "form", "filed", "accn"];
    let mut fundamental_concept_headers = HashSet::new();
    let mut header_mapping = Vec::new();

    // Identify fundamental concept mappings and track column indices
    for (index, header) in headers.iter().enumerate() {
        if additional_fields.contains(&header) {
            header_mapping.push((index, header.to_string())); // Include additional fields directly
        } else if let Some(mapped_concepts) = distill_us_gaap_fundamental_concepts(header) {
            for concept in mapped_concepts {
                fundamental_concept_headers.insert(concept.to_string());
                header_mapping.push((index, concept.to_string()));
            }
        }
    }

    println!(
        "Fundamental Concept Headers: {:?}",
        fundamental_concept_headers
    );

    let mut writer = Writer::from_path("temp.csv")?;

    // Write header row with mapped fundamental concepts + additional fields
    writer.write_record(header_mapping.iter().map(|(_, name)| name))?;

    // Write filtered records
    for result in reader.records() {
        let record = result?;
        let filtered_record: Vec<&str> = header_mapping
            .iter()
            .map(|(i, _)| record.get(*i).unwrap_or("")) // Ensure values are carried
            .collect();
        writer.write_record(&filtered_record)?;
    }

    writer.flush()?;
    println!("Filtered data written to temp.csv");

    Ok(())
}

// fn preload_csv(csv_path: &str) -> Result<(), Box<dyn Error>> {
//     let mut reader = Reader::from_path(csv_path)?;
//     let headers = reader.headers()?.clone();

//     let mut mapped_headers = Vec::new();
//     let mut column_mapping = Vec::new();

//     // Map CSV headers to FundamentalConcept using US GAAP mapping
//     for (idx, header) in headers.iter().enumerate() {
//         if let Some(mapped_concepts) = distill_us_gaap_fundamental_concepts(header) {
//             for concept in mapped_concepts {
//                 mapped_headers.push(concept.to_string());
//                 column_mapping.push((idx, concept.to_string())); // Store mapping of index -> concept
//             }
//         }
//     }

//     println!("Mapped Fundamental Concepts: {:?}", mapped_headers);

//     // Create CSV writer for output
//     let mut writer = Writer::from_path("temp.csv")?;

//     // Write header row with FundamentalConcept names
//     writer.write_record(&mapped_headers)?;

//     // Write filtered rows with mapped values
//     for result in reader.records() {
//         let record = result?;
//         let mapped_values: Vec<&str> = column_mapping
//             .iter()
//             .map(|(i, _)| record.get(*i).unwrap_or("")) // Use mapped column indices
//             .collect();
//         writer.write_record(&mapped_values)?;
//     }

//     writer.flush()?;
//     println!("Filtered data written to temp.csv");

//     Ok(())
// }
