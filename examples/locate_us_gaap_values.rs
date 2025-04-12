use csv::Reader;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <US_GAAP_TAG>", args[0]);
        std::process::exit(1);
    }

    let target_tag = &args[1];
    let data_dir = Path::new("data/us-gaap");

    let mut matches: Vec<(PathBuf, Vec<String>)> = Vec::new();

    for entry in fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|ext| ext == "csv").unwrap_or(false) {
            match search_csv_for_tag(&path, target_tag) {
                Ok(Some(values)) => matches.push((path.clone(), values)),
                Ok(None) => continue,
                Err(e) => eprintln!("⚠️ Skipped {} due to error: {}", path.display(), e),
            }
        }
    }

    if matches.is_empty() {
        println!("No files found with tag '{}'.", target_tag);
    } else {
        for (path, values) in matches {
            println!("\n📄 File: {}", path.display());
            println!("🔍 Values for '{}':", target_tag);
            for v in values.iter().take(20) {
                println!("  - {}", v);
            }
        }
    }

    Ok(())
}

fn search_csv_for_tag(
    path: &Path,
    target_tag: &str,
) -> Result<Option<Vec<String>>, Box<dyn Error>> {
    let mut reader = Reader::from_path(path)?;
    let headers = reader.headers()?.clone();

    if let Some(idx) = headers.iter().position(|h| h == target_tag) {
        let mut values = Vec::new();
        for result in reader.records() {
            let record = result?;
            if let Some(value) = record.get(idx) {
                if !value.trim().is_empty() {
                    values.push(value.to_string());
                }
            }
        }
        Ok(Some(values))
    } else {
        Ok(None)
    }
}
