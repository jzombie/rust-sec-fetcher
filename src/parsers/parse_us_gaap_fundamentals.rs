use polars::prelude::pivot::pivot;
use polars::prelude::*;
use serde_json::Value;
use std::error::Error;

pub type TickerFundamentalsDataFrame = DataFrame;

// TODO: Include potential support for Form 10-SA or whatever will be used for semi-annual reporting

/// Parses a US GAAP fundamentals JSON object into a Polars DataFrame.
///
/// # Sorting and Deduplication Logic
/// The parser processes the JSON facts list and performs the following steps to ensure
/// the DataFrame contains the most accurate and up-to-date information:
///
/// 1. **Extraction**: All facts are extracted with their metadata (fy, fp, filed, accn).
/// 2. **Chronological Sorting (Filings)**: The intermediate DataFrame is sorted by the 'filed'
///    date in descending order (`.sort(["filed"], descending=true)`).
/// 3. **Deduplication (Last-in Wins)**: When multiple records exist for the same fiscal period
///    (same `fy` and `fp` keys), the `pivot` operation aggregates using the `.first()` function.
///    Because the data was pre-sorted by `filed` descending, `.first()` selects the record
///    from the most recent filing (e.g., an amendment `10-Q/A` filed later will overwrite
///    the original `10-Q`).
/// 4. **Row Ordering**: The final DataFrame is sorted by Fiscal Year (`fy`) descending, and
///    then by Fiscal Period (`fp`) descending (FY > Q3 > Q2 > Q1).
pub fn parse_us_gaap_fundamentals(
    serde_json_value: Value,
) -> Result<TickerFundamentalsDataFrame, Box<dyn Error>> {
    let data = serde_json_value;

    let mut fact_category_values = Vec::new();
    let mut fact_name_values = Vec::new();
    let mut label_values = Vec::new();
    let mut end_values = Vec::new();
    let mut value_values = Vec::new();
    let mut form_values = Vec::new();
    let mut filed_values = Vec::new();
    let mut fy_values = Vec::new();
    let mut fp_values = Vec::new();
    let mut accn_values = Vec::new();

    // Parse the "facts" section
    if let Some(facts) = data["facts"].as_object() {
        for (fact_category, fact_data) in facts {
            if let Some(fact_details) = fact_data.as_object() {
                for (fact_name, fact_info) in fact_details {
                    let label = fact_info["label"].as_str().unwrap_or("").to_string();

                    if let Some(units) = fact_info["units"].as_object() {
                        for (unit, observations) in units {
                            for obs in observations.as_array().unwrap_or(&Vec::new()) {
                                let end_str = obs["end"].as_str().unwrap_or("").to_string();
                                let end_year = if end_str.len() >= 4 {
                                    end_str[0..4].parse::<u64>().unwrap_or(0)
                                } else {
                                    0
                                };

                                // Use the 'fy' field from the observation if available.
                                // Otherwise, derive it from the 'end' date.
                                let fy_derived = if let Some(fy) = obs["fy"].as_u64() {
                                    // Sanity check: fy should not be greater than end_year
                                    // This filters out historical rows incorrectly tagged with the current filing's FY.
                                    if end_year > 0 && fy > end_year {
                                        continue;
                                    }
                                    fy
                                } else {
                                    end_year
                                };

                                fact_category_values.push(fact_category.clone());
                                fact_name_values.push(fact_name.clone());
                                label_values.push(label.clone());

                                // Joins the unit of measure with the value, using a double-colon (`::`) separator
                                let value = obs["val"]
                                    .as_f64()
                                    .map(|v| format!("{}::{}", v, unit)) // Append unit to value
                                    .unwrap_or_else(|| format!("0::{}", unit)); // Handle missing values

                                value_values.push(value);
                                end_values.push(end_str);
                                form_values.push(obs["form"].as_str().unwrap_or("").to_string());
                                filed_values.push(obs["filed"].as_str().unwrap_or("").to_string());
                                fy_values.push(fy_derived);
                                fp_values.push(obs["fp"].as_str().unwrap_or("").to_string());
                                accn_values.push(obs["accn"].as_str().unwrap_or("").to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    // Create a DataFrame from the extracted data
    let mut df = df!(
        "fact_category" => &fact_category_values,
        "fact_name" => &fact_name_values,
        "label" => &label_values,
        "end" => &end_values,
        "value" => &value_values, // Values now include units
        "form" => &form_values,
        "filed" => &filed_values,
        "fy" => &fy_values,
        "fp" => &fp_values,
        "accn" => &accn_values
    )?;

    // Sort by 'filed' date descending so that when we pivot and aggregate using .first(),
    // we take the most recently filed data (amendments/restatements) over older data.
    df = df.sort(
        ["filed"],
        SortMultipleOptions::default()
            .with_order_descending(true)
            .with_nulls_last(true),
    )?;

    // Extract metadata (filed, form, accn) for the latest filing per (fy, fp)
    // We clone because we need to reuse df for the pivot
    let meta_df = df
        .clone()
        .lazy()
        .select([col("fy"), col("fp"), col("filed"), col("form"), col("accn")])
        .unique(
            Some(vec!["fy".to_string(), "fp".to_string()]),
            UniqueKeepStrategy::First,
        )
        .collect()?;

    // Perform a single pivot (values include units)
    let mut pivot_df = pivot(
        &df,
        ["fact_name"]
            .iter()
            .map(|&s| s.to_string())
            .collect::<Vec<String>>(),
        Some(
            ["fy", "fp"]
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<String>>(),
        ),
        Some(
            ["value"]
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<String>>(),
        ),
        false,
        Some(col("value").first()),
        None,
    )?;

    // Join metadata back to the pivoted dataframe
    pivot_df = pivot_df
        .lazy()
        .join(
            meta_df.lazy(),
            [col("fy"), col("fp")],
            [col("fy"), col("fp")],
            JoinArgs::new(JoinType::Left),
        )
        .collect()?;

    // Create a ranking column for 'fp' to ensure correct chronological sorting
    // Order: FY (latest) -> Q3 -> Q2 -> Q1
    // We assign higher numbers to later periods and sort descending.
    let fp_rank_expr = when(col("fp").eq(lit("FY")))
        .then(lit(4))
        .when(col("fp").eq(lit("Q3")))
        .then(lit(3))
        .when(col("fp").eq(lit("Q2")))
        .then(lit(2))
        .when(col("fp").eq(lit("Q1")))
        .then(lit(1))
        .otherwise(lit(0))
        .alias("fp_rank");

    pivot_df = pivot_df
        .lazy()
        .with_column(fp_rank_expr)
        .collect()?;

    // Sort by FY descending, then by our custom fp rank descending
    pivot_df = pivot_df.sort(
        ["fy", "fp_rank"],
        SortMultipleOptions::default()
            .with_order_descending(true)
            .with_nulls_last(true),
    )?;

    // Drop the helper rank column
    let _ = pivot_df.drop_in_place("fp_rank");

    // Reorder columns to place metadata (fy, fp, filed, form, accn) at the start
    let mut desired_cols = vec![
        "fy".to_string(),
        "fp".to_string(),
        "filed".to_string(),
        "form".to_string(),
        "accn".to_string(),
    ];
    let existing_cols = pivot_df.get_column_names();

    // Append all other fact columns
    for c in existing_cols {
        let c_str = c.to_string();
        if !desired_cols.contains(&c_str) {
            desired_cols.push(c_str);
        }
    }

    // Select in the desired order
    pivot_df = pivot_df.select(&desired_cols)?;

    Ok(pivot_df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_us_gaap_fundamentals_uses_source_fy() {
        // Test ensuring that we respect the SEC's 'fy' field when provided,
        // rather than blindly parsing the year from the 'end' date.
        // This is critical for companies with fiscal years ending in Jan/Feb/Mar
        // where the 'end' calendar year is +1 from the 'fy'.
        let mock_json = json!({
            "cik": 123456,
            "entityName": "Fiscal Year Edge Case Corp",
            "facts": {
                "us-gaap": {
                    "Revenue": {
                        "label": "Revenue",
                        "units": {
                            "USD": [
                                {
                                    "val": 1000.0,
                                    "end": "2024-01-31", // Calendar 2024
                                    "fy": 2023,          // Fiscal 2023
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2024-03-15",
                                    "accn": "000-2023-FY"
                                }
                            ]
                        }
                    }
                }
            }
        });

        let df = parse_us_gaap_fundamentals(mock_json).expect("Failed to parse mock JSON");

        let fy_col = df.column("fy").unwrap().u64().unwrap();
        // Old logic would produce 2024 (from "2024-01-31").
        // New logic should produce 2023 (from "fy": 2023).
        assert_eq!(fy_col.get(0), Some(2023));
    }

    #[test]
    fn test_parse_us_gaap_fundamentals_sorting_and_consolidation() {
        // We include data for:
        // - 2024 FY, 2024 Q3, 2024 Q2 (to test intra-year sorting: FY > Q3 > Q2)
        // - 2 amendments for 2024 Q3: One filed later than the other (to test "latest filing" logic)
        let mock_json = json!({
            "cik": 123456,
            "entityName": "Artificial Tech Company",
            "facts": {
                "us-gaap": {
                    "Revenue": {
                        "label": "Revenue",
                        "units": {
                            "USD": [
                                // 2024 Q2: Filed 2024-05-01
                                {
                                    "val": 1000.0,
                                    "end": "2024-06-30", // Derived FY 2024
                                    "fy": 2024,
                                    "fp": "Q2",
                                    "form": "10-Q",
                                    "filed": "2024-08-01",
                                    "accn": "000-2024-Q2"
                                },
                                // 2024 Q3 (Original): Filed 2024-11-01
                                {
                                    "val": 1500.0,
                                    "end": "2024-09-30", // Derived FY 2024
                                    "fy": 2024,
                                    "fp": "Q3",
                                    "form": "10-Q",
                                    "filed": "2024-11-01",
                                    "accn": "000-2024-Q3-ORIG"
                                },
                                // 2024 Q3 (Amendment/Restatement/Later Filing): Filed 2025-02-01
                                // Should be picked over the original because 'filed' is later
                                {
                                    "val": 1600.0, // Restated value
                                    "end": "2024-09-30",
                                    "fy": 2024,
                                    "fp": "Q3",
                                    "form": "10-Q/A",
                                    "filed": "2025-02-01",
                                    "accn": "000-2024-Q3-AMEND"
                                },
                                // 2024 FY: Filed 2025-03-01
                                {
                                    "val": 5000.0,
                                    "end": "2024-12-31",
                                    "fy": 2024,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2025-03-01",
                                    "accn": "000-2024-FY"
                                }
                            ]
                        }
                    }
                }
            }
        });

        let df = parse_us_gaap_fundamentals(mock_json).expect("Failed to parse mock JSON");

        // 1. Validate Columns: Metadata columns must be first
        let cols = df.get_column_names();
        assert_eq!(cols[0].as_str(), "fy");
        assert_eq!(cols[1].as_str(), "fp");
        assert_eq!(cols[2].as_str(), "filed");
        assert_eq!(cols[3].as_str(), "form");
        assert_eq!(cols[4].as_str(), "accn");
        assert!(cols.iter().any(|c| c.as_str() == "Revenue"));

        // 2. Validate row count: Should be 3 rows (2024 FY, 2024 Q3, 2024 Q2).
        // The duplicate Q3 should be consolidated.
        assert_eq!(df.height(), 3, "DataFrame should have 3 rows after consolidation");

        // 3. Validate Sorting: Reverse chronological (FY > Q3 > Q2) within the year
        // Row 0: 2024 FY
        let fy_col = df.column("fy").unwrap().u64().unwrap();
        let fp_col = df.column("fp").unwrap().str().unwrap();
        let revenue_col = df.column("Revenue").unwrap().str().unwrap();
        let filed_col = df.column("filed").unwrap().str().unwrap();
        let accn_col = df.column("accn").unwrap().str().unwrap();

        // Check Row 0 (Should be 2024 FY)
        assert_eq!(fy_col.get(0), Some(2024));
        assert_eq!(fp_col.get(0), Some("FY"));
        assert_eq!(revenue_col.get(0), Some("5000::USD")); // 5000 from 10-K

        // Check Row 1 (Should be 2024 Q3)
        assert_eq!(fy_col.get(1), Some(2024));
        assert_eq!(fp_col.get(1), Some("Q3"));
        // Check consolidation logic: Should pick the amendment (1600.0) over original (1500.0)
        // because the amendment has a later 'filed' date (2025-02-01 vs 2024-11-01)
        // and we sort by filed descending before pivoting.
        assert_eq!(revenue_col.get(1), Some("1600::USD"));
        assert_eq!(accn_col.get(1), Some("000-2024-Q3-AMEND")); // Should be the amendment accn
        assert_eq!(filed_col.get(1), Some("2025-02-01"));

        // Check Row 2 (Should be 2024 Q2)
        assert_eq!(fy_col.get(2), Some(2024));
        assert_eq!(fp_col.get(2), Some("Q2"));
        assert_eq!(revenue_col.get(2), Some("1000::USD"));
    }
}
