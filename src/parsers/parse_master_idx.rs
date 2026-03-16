use crate::types::MasterIndexEntry;
use chrono::NaiveDate;
use std::error::Error;

/// Parses the body of an EDGAR full-index `master.idx` file into a
/// [`Vec<MasterIndexEntry>`].
///
/// The file has a multi-line description header, a column-name row, a
/// separator row of dashes, and then pipe-delimited data rows:
///
/// ```text
/// CIK|Company Name|Form Type|Date Filed|Filename
/// --------------------------------------------------------------------------------
/// 1000045|OLD MARKET CAPITAL Corp|4|2026-01-12|edgar/data/1000045/...
/// ```
///
/// Any row that cannot be fully parsed (wrong column count, bad date) is
/// silently skipped — this keeps things robust against the occasional
/// malformed legacy entry in the archive.
pub fn parse_master_idx(text: &str) -> Result<Vec<MasterIndexEntry>, Box<dyn Error>> {
    let mut past_header = false;
    let mut entries = Vec::new();

    for line in text.lines() {
        if !past_header {
            // The separator row of dashes signals the end of the header block.
            if line.trim_start().starts_with("---") {
                past_header = true;
            }
            continue;
        }

        // Each data row has exactly 5 pipe-separated fields.
        let parts: Vec<&str> = line.splitn(5, '|').collect();
        if parts.len() != 5 {
            continue;
        }

        let date_filed = match NaiveDate::parse_from_str(parts[3].trim(), "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => continue,
        };

        entries.push(MasterIndexEntry {
            cik: parts[0].trim().to_string(),
            company_name: parts[1].trim().to_string(),
            form_type: parts[2].trim().to_string(),
            date_filed,
            filename: parts[4].trim().to_string(),
        });
    }

    Ok(entries)
}
