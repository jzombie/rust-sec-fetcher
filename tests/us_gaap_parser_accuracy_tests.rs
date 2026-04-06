use flate2::read::GzDecoder;
use polars::prelude::*;
use sec_fetcher::parsers::parse_us_gaap_fundamentals;
use sec_fetcher_shared::US_GAAP_CSV_META_COLUMNS;
use serde_json::Value;
use serde_json::json;
use std::fs::File;
use std::path::PathBuf;

/// Load a fixture by its logical name (e.g. `"AAPL_companyfacts.json"`).
/// Stored on disk as `{name}.gz` and decompressed in memory.
/// Run `cargo run --bin refresh_test_fixtures` to create or update fixtures.
fn load_fixture(name: &str) -> Value {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(format!("{}.gz", name));
    let file = File::open(&path).unwrap_or_else(|_| {
        panic!(
            "missing fixture: {} (run `cargo run --bin refresh_test_fixtures`)",
            path.display()
        )
    });
    serde_json::from_reader(GzDecoder::new(file)).expect("fixture is not valid JSON")
}

/// Helper function to validate EVERY parsed fact in the dataframe against the source JSON.
///
/// Note: This validates **accuracy** (precision), ensuring every value in the DataFrame
/// can be traced back to an exact match in the source JSON.
/// It does not strictly validate **completeness** (recall), i.e., it does not fail if the
/// DataFrame is missing a fact that exists in the JSON (unless that fact causes a mismatch in existing cells).
fn validate_dataframe_against_json(df: &DataFrame, json_data: &Value) {
    // 1. Get list of fact columns (excluding metadata)
    let meta_cols = US_GAAP_CSV_META_COLUMNS;
    // df.get_column_names() returns a slice of &PlSmallStr in recent Polars
    let all_cols = df.get_column_names();

    // Iterate and filter
    let fact_cols: Vec<String> = all_cols
        .iter()
        .map(|c| c.to_string())
        .filter(|c| !meta_cols.contains(&c.as_str()))
        .collect();

    // We need to access individual columns to iterate rows
    let fy_col = df.column("fy").expect("fy column missing");
    let fp_col = df.column("fp").expect("fp column missing");

    // Iterate over every row
    for i in 0..df.height() {
        let fy = fy_col.get(i).expect("fy val").try_extract::<u64>().unwrap(); // Polars AnyValue extract
        let fp = fp_col
            .get(i)
            .expect("fp val")
            .get_str()
            .unwrap()
            .to_string();

        // For each fact in this row
        for fact_name in &fact_cols {
            let df_col = df.column(fact_name).unwrap();
            let cell_val = df_col.get(i).unwrap();

            if let AnyValue::Null = cell_val {
                continue;
            }

            let val_str = cell_val.get_str().unwrap().to_string();
            // Value format: "1234.56::Unit"
            let parts: Vec<&str> = val_str.split("::").collect();
            assert_eq!(parts.len(), 2, "Value format should be 'val::unit'");
            let val_num: f64 = parts[0].parse().expect("Failed to parse value as f64");
            let unit = parts[1];

            // Now Find this in JSON
            // Path: facts -> "us-gaap" (or "dei"?) -> fact_name -> units -> unit

            let facts_obj = json_data["facts"].as_object().expect("No facts object");

            let mut correct_match_found = false;

            // We search through all taxonomies because we don't know which one this fact belongs to from the DF column name alone
            // (The DF fact_name is unique enough usually, but strictly it falls under a taxonomy)
            for (_taxonomy, tax_data) in facts_obj {
                if let Some(fact_node) = tax_data.get(fact_name)
                    && let Some(units_node) = fact_node["units"].as_object()
                    && let Some(observations) = units_node.get(unit)
                    && let Some(obs_array) = observations.as_array()
                {
                    // Gather valid candidates
                    let mut candidates = Vec::new();

                    for obs in obs_array {
                        // REPLICATE PARSER LOGIC FOR FY
                        let end_str = obs["end"].as_str().unwrap_or("").to_string();
                        let end_year = if end_str.len() >= 4 {
                            end_str[0..4].parse::<u64>().unwrap_or(0)
                        } else {
                            0
                        };

                        let obs_fy = if let Some(f) = obs["fy"].as_u64() {
                            // FIX LOGIC: Match the parser's mixed strictness checks
                            let obs_fp_check = obs["fp"].as_str().unwrap_or("FY");
                            if obs_fp_check == "FY" {
                                if end_year > 0 && f > end_year {
                                    continue;
                                }
                            } else if end_year > 0 && f > end_year + 1 {
                                continue;
                            }
                            f
                        } else {
                            end_year
                        };

                        let obs_fp = obs["fp"].as_str().unwrap_or("");

                        if obs_fy == fy && obs_fp == fp {
                            candidates.push(obs);
                        }
                    }

                    if candidates.is_empty() {
                        continue;
                    }

                    // Sort candidates by filed date descending, then end date descending
                    candidates.sort_by(|a, b| {
                        let filed_a = a["filed"].as_str().unwrap_or("");
                        let filed_b = b["filed"].as_str().unwrap_or("");
                        let c = filed_b.cmp(filed_a); // Descending filed
                        if c == std::cmp::Ordering::Equal {
                            let end_a = a["end"].as_str().unwrap_or("");
                            let end_b = b["end"].as_str().unwrap_or("");
                            end_b.cmp(end_a) // Descending end
                        } else {
                            c
                        }
                    });

                    // Best candidate is the first one
                    let best_match = candidates[0];
                    let best_val = best_match["val"].as_f64().unwrap_or(0.0);

                    // Compare values with epsilon for float logic? Or exact?
                    // JSON float parsing vs string parsing might have tiny diffs.
                    // But usually exact for these financial numbers.
                    if (best_val - val_num).abs() < 0.0001 {
                        correct_match_found = true;
                        break;
                    } else {
                        // Found the right FY/FP bucket, but value mismatch
                        // This implies the parser chose a different value than our logic?
                        // OR there are multiple entries with same filed date?
                        // Or the unit search found a different unit? (We are inside unit loop).
                    }
                }
            }

            assert!(
                correct_match_found,
                "Failed to verify fact '{}' for FY{} {}. Value in DF: {} ({}). Could not find matching source data in JSON following parser rules.",
                fact_name, fy, fp, val_num, unit
            );
        }
    }
}

fn run_full_validation_for_ticker(ticker: &str, filename: &str) {
    let json_data = load_fixture(filename);

    println!("Parsing data for {}...", ticker);
    let df = parse_us_gaap_fundamentals(json_data.clone()).expect("Failed to parse JSON dataframe");

    println!(
        "Validating every fact for {} (Rows: {})...",
        ticker,
        df.height()
    );
    validate_dataframe_against_json(&df, &json_data);
    println!("Validation passed for {}!", ticker);
}

#[test]
fn test_accuracy_nvda_full_exhaustive() {
    run_full_validation_for_ticker("NVDA", "NVDA_companyfacts.json");
}

#[test]
fn test_accuracy_goog_full_exhaustive() {
    run_full_validation_for_ticker("GOOG", "GOOG_companyfacts.json");
}

#[test]
fn test_accuracy_aapl_full_exhaustive() {
    run_full_validation_for_ticker("AAPL", "AAPL_companyfacts.json");
}

#[test]
fn test_accuracy_msft_full_exhaustive() {
    run_full_validation_for_ticker("MSFT", "MSFT_companyfacts.json");
}

#[test]
fn test_accuracy_off_calendar_fiscal_year() {
    let json_data = load_fixture("NVDA_companyfacts.json");

    let df = parse_us_gaap_fundamentals(json_data).expect("Failed to parse JSON dataframe");

    // Filter for FY2024 (Which ended Jan 2024)
    // The parser should correctly identify this as FY2024 because the SEC source data says "fy": 2024.
    // If it relies on the calendar year of the end date (2024-01-28), it might erroneously think it is 2024,
    // but we need to ensure it's not grabbing the *wrong* 2024 data or labeling 2023 data as 2024.
    // Actually, for NVDA:
    // FY2024 ended Jan 28, 2024. Revenue was ~60.9B.
    // FY2023 ended Jan 29, 2023. Revenue was ~26.9B.

    let fy_2024 = df
        .clone()
        .lazy()
        .filter(col("fy").eq(lit(2024)).and(col("fp").eq(lit("FY"))))
        .collect()
        .expect("Collect failed");

    // We might have multiple rows if there are multiple entities/filings, but our pivot should consolidate them per (fy, fp).
    // The 'parse_us_gaap_fundamentals' returns a pivoted info where one row = one fy/fp.
    assert!(
        fy_2024.height() >= 1,
        "Should have at least one row for FY2024. Got {}",
        fy_2024.height()
    );

    // Get the Revenue value. Note: The column name might be "Revenues" or "RevenueFromContractWithCustomerExcludingAssessedTax"
    // In the NVDA json, the tag is "Revenues" or "RevenueFromContractWithCustomerExcludingAssessedTax".
    // Let's check "Revenues" first, as that's what NVDA uses heavily.
    let col_names = fy_2024.get_column_names();
    let revenues_col = if col_names.iter().any(|&c| c == "Revenues") {
        "Revenues"
    } else {
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    };

    let revenues_str = fy_2024
        .column(revenues_col)
        .expect("Missing Revenue column in dataframe")
        .str()
        .expect("Revenues column is not string")
        .get(0)
        .expect("No value in Revenue column");

    // Validate against the Known Truth from the 10-K filing for FY2024 (filed Feb 21, 2024)
    // Value: 60,922,000,000
    println!("Found FY2024 Revenue: {}", revenues_str);
    assert!(
        revenues_str.starts_with("60922000000"),
        "Revenue for FY2024 must match the official 10-K value of 60,922,000,000. Got: {}",
        revenues_str
    );

    // Filter for FY2023 (Ended Jan 2023)
    // Value: 26,974,000,000
    let fy_2023 = df
        .lazy()
        .filter(col("fy").eq(lit(2023)).and(col("fp").eq(lit("FY"))))
        .collect()
        .expect("Collect failed");

    let revenues_2023 = fy_2023
        .column(revenues_col)
        .unwrap()
        .str()
        .unwrap()
        .get(0)
        .unwrap();
    println!("Found FY2023 Revenue: {}", revenues_2023);
    assert!(
        revenues_2023.starts_with("26974000000"),
        "Revenue for FY2023 must match the official 10-K value of 26,974,000,000. Got: {}",
        revenues_2023
    );
}

#[test]
fn test_accuracy_restatement_handling() {
    // Scenario: A company restates Q3 earnings.
    // We expect the parser to pick the LATEST filing.
    let json_data = json!({
        "cik": 999999,
        "entityName": "Restatement Corp",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "label": "Net Income",
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-30",
                                "val": 100.0,
                                "accn": "ORIG-FILING",
                                "fy": 2024,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2024-11-01"
                            },
                            {
                                "end": "2024-09-30",
                                "val": 80.0, // Corrected down
                                "accn": "AMEND-FILING",
                                "fy": 2024,
                                "fp": "Q3",
                                "form": "10-Q/A",
                                "filed": "2024-12-01" // Later date
                            }
                        ]
                    }
                }
            }
        }
    });

    let df = parse_us_gaap_fundamentals(json_data).expect("Failed to parse");

    let q3_data = df
        .lazy()
        .filter(col("fy").eq(lit(2024)).and(col("fp").eq(lit("Q3"))))
        .collect()
        .expect("Collect failed");

    // Metadata correctness check
    let accn = q3_data
        .column("accn")
        .unwrap()
        .str()
        .unwrap()
        .get(0)
        .unwrap();
    let val = q3_data
        .column("NetIncomeLoss")
        .unwrap()
        .str()
        .unwrap()
        .get(0)
        .unwrap();

    assert_eq!(
        accn, "AMEND-FILING",
        "Should pick the accession number of the latest filing"
    );
    assert!(
        val.starts_with("80"),
        "Should pick the restated value (80) not original (100). Got: {}",
        val
    );
}

#[test]
fn test_accuracy_missing_fy_fallback() {
    // Scenario: JSON lacks "fy" field (very old data or weird tagging).
    // Should extract year from the "end" date string.
    let json_data = json!({
        "cik": 11111,
        "entityName": "Old Data Corp",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "label": "Assets",
                    "units": {
                        "USD": [
                            {
                                "end": "2020-12-31",
                                "val": 5000.0,
                                "accn": "OLD-ACC",
                                // "fy" key MISSING
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2021-03-01"
                            }
                        ]
                    }
                }
            }
        }
    });

    let df = parse_us_gaap_fundamentals(json_data).expect("Parsing failed");

    let fy_col = df.column("fy").unwrap().u64().unwrap();
    assert_eq!(
        fy_col.get(0),
        Some(2020),
        "Should derive FY 2020 from end date 2020-12-31"
    );
}

#[test]
fn test_parser_prioritizes_amendments_synthetic() {
    // This synthetic test proves that if two filings claim the SAME fiscal period (fy/fp),
    // the parser uses the one with the later 'filed' date.
    let json_data = json!({
        "cik": 12345,
        "entityName": "Test Corp",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "label": "Assets",
                    "units": {
                        "USD": [
                            {
                                "end": "2020-12-31",
                                "val": 100,
                                "accn": "000-ORIGINAL",
                                "fy": 2020,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2021-03-01"
                            },
                            {
                                "end": "2020-12-31",
                                "val": 200, // Amended value
                                "accn": "000-AMENDMENT",
                                "fy": 2020, // SAME bucket
                                "fp": "FY", // SAME bucket
                                "form": "10-K/A",
                                "filed": "2021-04-01" // LATER date
                            }
                        ]
                    }
                }
            }
        }
    });

    let df = parse_us_gaap_fundamentals(json_data).expect("Parse failed");

    // Filter for FY 2020
    let row_df = df.lazy().filter(col("fy").eq(lit(2020))).collect().unwrap();

    assert_eq!(
        row_df.height(),
        1,
        "Should have collapsed into a single row"
    );

    // The value should be 200 (Amendment), not 100.
    // The parser creates strings "val::unit", so "200::USD"
    let assets_col = row_df.column("Assets").unwrap();
    // Safety: we know column exists and rows>0
    let val_any = assets_col.get(0).unwrap();
    let val_str = val_any.get_str().unwrap();

    assert!(
        val_str.starts_with("200"),
        "Expected Amended value 200, got {}",
        val_str
    );

    // Check that we captured the metadata of the amendment
    let accn_col = row_df.column("accn").unwrap();
    assert_eq!(accn_col.get(0).unwrap().get_str().unwrap(), "000-AMENDMENT");
}

#[test]
fn test_canonical_order_column_exists_and_monotonic() {
    let json_data = load_fixture("AAPL_companyfacts.json");

    let df = parse_us_gaap_fundamentals(json_data).expect("Failed to parse dataframe");

    // Check column existence
    let index_col = df
        .column("canonical_order")
        .expect("canonical_order column missing");

    // Check it is u32 or u64 (Polars row index is usually u32)
    // We expect it to be 0..N
    let index_vals: Vec<u32> = index_col
        .u32()
        .expect("canonical_order should be u32")
        .into_no_null_iter()
        .collect();

    for (i, &val) in index_vals.iter().enumerate() {
        assert_eq!(
            val, i as u32,
            "Canonical order should be equal to the row index"
        );
    }

    // Check that it correlates with FY descending (Reverse Chronological)
    let fy_col = df.column("fy").unwrap().u64().unwrap();

    for i in 0..(df.height() - 1) {
        let current_fy = fy_col.get(i).unwrap();
        let next_fy = fy_col.get(i + 1).unwrap();

        // Since canonical order 0 is latest, 1 is older...
        // FY[0] should be >= FY[1]
        assert!(
            current_fy >= next_fy,
            "Rows should be strict reverse chronological (FY desc) but row {} is FY{} and row {} is FY{}",
            i,
            current_fy,
            i + 1,
            next_fy
        );

        // Logic check: Canonical Order 0 < Canonical Order 1
        // Real Time 0 > Real Time 1
    }
}
