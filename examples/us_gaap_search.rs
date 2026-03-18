//! Searches US GAAP CSV files for rows containing a given XBRL tag (concept name).
//!
//! Scans every `.csv` file in the target directory and finds those that have a
//! column matching the given XBRL tag name.  For each matching file, the
//! non-empty values in that column are printed, up to `--max-values` per file.
//!
//! This is useful for quickly surveying what values a particular GAAP concept
//! takes across different SEC filers (e.g. `Assets`, `Revenues`, `NetIncomeLoss`).
//!
//! The default input directory is `data/14-mar-2026-us-gaap`, which is the
//! bulk GAAP dataset produced by `pull-us-gaap-bulk`.
//!
//! # Usage
//!
//! ```text
//! cargo run --example us_gaap_search -- Assets
//! cargo run --example us_gaap_search -- Revenues --max-values 5
//! cargo run --example us_gaap_search -- NetIncomeLoss --dir data/14-mar-2026-us-gaap
//! ```

use clap::Parser;
use csv::Reader;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_DATA_DIR: &str = "data/14-mar-2026-us-gaap";

#[derive(Parser)]
#[command(about = "Search US GAAP CSV files for rows containing a given XBRL tag")]
struct Args {
    /// The XBRL tag / column header to search for (e.g. Assets, Revenues)
    tag: String,

    /// Directory of US GAAP CSV files
    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    dir: String,

    /// Maximum values to display per file (0 = no limit)
    #[arg(long, default_value_t = 20)]
    max_values: usize,
}

/// A file that contains non-empty values for the requested XBRL tag.
struct TagMatch {
    path: PathBuf,
    tag: String,
    values: Vec<String>,
    max_display: usize,
}

impl fmt::Display for TagMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\nFile: {}", self.path.display())?;
        writeln!(f, "Values for '{}':", self.tag)?;
        let limit = if self.max_display == 0 {
            self.values.len()
        } else {
            self.max_display.min(self.values.len())
        };
        for v in &self.values[..limit] {
            writeln!(f, "  - {}", v)?;
        }
        if limit < self.values.len() {
            write!(f, "  ... ({} more)", self.values.len() - limit)?;
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let data_dir = Path::new(&args.dir);

    let mut matches: Vec<TagMatch> = Vec::new();

    for entry in fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|ext| ext == "csv").unwrap_or(false) {
            match search_csv_for_tag(&path, &args.tag) {
                Ok(Some(values)) => matches.push(TagMatch {
                    path,
                    tag: args.tag.clone(),
                    values,
                    max_display: args.max_values,
                }),
                Ok(None) => continue,
                Err(e) => eprintln!("Skipped {} due to error: {}", path.display(), e),
            }
        }
    }

    if matches.is_empty() {
        println!("No files found with tag '{}'.", args.tag);
    } else {
        for m in &matches {
            println!("{}", m);
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
            if let Some(value) = record.get(idx)
                && !value.trim().is_empty()
            {
                values.push(value.to_string());
            }
        }
        Ok(Some(values))
    } else {
        Ok(None)
    }
}
