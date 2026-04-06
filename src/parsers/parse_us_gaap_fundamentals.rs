use crate::enums::Url;
use crate::models::{AccessionNumber, Cik};
use polars::prelude::pivot::pivot;
use polars::prelude::*;
use sec_fetcher_shared::{US_GAAP_CSV_META_COLUMNS, normalize_fp_label, parse_period_slot_token};
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

    // Extract CIK from the top-level JSON field for use in filing_url construction.
    let cik_value: Option<u64> = data["cik"].as_u64();

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
                                    let fp_str = obs["fp"].as_str().unwrap_or("FY");

                                    // Sanity check logic based on period type
                                    if fp_str == "FY" {
                                        // Annual (FY): Strict Check.
                                        // Fiscal Year usually matches the Calendar Year of the end date.
                                        // e.g. Nvidia FY2024 ends Jan 2024 (fy=2024, end_year=2024).
                                        // If fy > end_year, it's likely a mislabeled future year (e.g. 2023 data labeled as 2024).
                                        if end_year > 0 && fy > end_year {
                                            continue;
                                        }
                                    } else {
                                        // Interim (Q1-Q3): Relaxed Check.
                                        // Quarters often fall in the calendar year prior to the FY assignment.
                                        // e.g. Nvidia FY2024 Q1 ends April 2023 (fy=2024, end_year=2023).
                                        // We allow fy to be end_year + 1.
                                        if end_year > 0 && fy > end_year + 1 {
                                            continue;
                                        }
                                    }
                                    fy
                                } else {
                                    end_year
                                };

                                fact_category_values.push(fact_category.clone());
                                fact_name_values.push(fact_name.clone());
                                label_values.push(label.clone());

                                // DESIGN: The SEC EDGAR companyfacts API is designed to resolve
                                // the XBRL `scale` / `decimals` attribute server-side and return
                                // `val` as the full, un-truncated number.  If a filer correctly
                                // tags revenues as "34,820" with scale="6" (millions), the API
                                // should return 34,820,000,000.  The `decimals` field is consumed
                                // during this process and does NOT appear in the API response.
                                //
                                // REALITY (GIGO): The SEC does not audit or correct XBRL tags
                                // before serving them.  Filer tagging errors are common: a company
                                // may submit 34820 for revenue but forget the scale tag, causing
                                // the API to faithfully return 34820 instead of 34,820,000,000.
                                // The SEC's own DERA division regularly warns filers about this
                                // exact problem (companies reporting a public float of $800M in
                                // their HTML, but $800 in their XBRL structured data).
                                //
                                // IMPLICATION: `val` is stored as-is from the API.  Downstream
                                // consumers must apply magnitude sanity checks (e.g. cross-
                                // referencing against known market-cap or revenue ranges) before
                                // treating `val` as a reliable dollar figure.
                                //
                                // Contrast with Form 13F-HR, where the `<value>` element had a
                                // known era-based scale convention that required systematic
                                // correction (see `normalize_13f_value_usd` in
                                // `normalize/thirteenf.rs`). Here there is no systematic rule —
                                // each filer error must be detected case-by-case.

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
                                // Normalize Q4 → FY: year-end quarters are always canonical FY
                                // regardless of whether the SEC tagged them via a 10-Q or 10-K.
                                fp_values
                                    .push(normalize_fp_label(obs["fp"].as_str().unwrap_or("")));
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

    // Sort Logic:
    // 1. 'filed' descending: Prioritize the latest filing date (Amendments > Original).
    // 2. 'end' descending: Within the same filing (or same filed date), prioritize the Latest Instant/Period End.
    //    This is CRITICAL for Balance Sheet (Instant) items in quarterly filings.
    //    A Q1 filing contains both Current Q1 End (2025-03-31) and Prior Year End (2024-12-31).
    //    Both are tagged 'Q1' by the SEC in some contexts, but we want the '2025-03-31' value for the Q1 row.
    //    Sorting by 'end' descending ensures the latest date comes first, so .first() picks the actual Q1 balance.
    df = df.sort(
        ["filed", "end"],
        SortMultipleOptions::default()
            .with_order_descending(true) // Descending for both
            .with_nulls_last(true),
    )?;

    // Extract metadata (filed, form, accn, end) for the latest filing per (fy, fp)
    // We clone because we need to reuse df for the pivot
    let meta_df = df
        .clone()
        .lazy()
        .select([
            col("fy"),
            col("fp"),
            col("filed"),
            col("form"),
            col("accn"),
            col("end").alias("period_end"),
        ])
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

    // Normalize the `form` column: strip the `/A` amendment suffix so that the
    // base form type is stored (e.g. `"10-Q/A"` → `"10-Q"`).  The amendment fact
    // is surfaced as a separate `is_amendment` boolean column so callers can still
    // distinguish amended rows from originals.
    //
    // This runs after the join so it operates on the already-deduplicated winning
    // row (the latest-filed filing per (fy, fp) pair).
    let (is_amendment_values, normalized_form_values): (Vec<bool>, Vec<Option<String>>) = {
        let form_col = pivot_df.column("form")?.str()?;
        form_col
            .into_iter()
            .map(|opt| match opt {
                Some(s) if s.ends_with("/A") => (true, Some(s[..s.len() - 2].to_string())),
                Some(s) => (false, Some(s.to_string())),
                None => (false, None),
            })
            .unzip()
    };
    pivot_df.with_column(Series::new("is_amendment".into(), is_amendment_values))?;
    pivot_df.with_column(Series::new("form".into(), normalized_form_values))?;

    // Compute fp_rank using the shared sec-fetcher-shared crate so that all period
    // token aliases (H1, H2, SA1, SA2, 6M, 12M, Q4, …) sort correctly.
    // The old hardcoded Polars when/then only handled Q1–Q3 + FY, leaving Q4
    // and all semi-annual/monthly tokens at rank 0 (wrong order).
    let fp_ranks: Vec<i32> = pivot_df
        .column("fp")?
        .str()?
        .into_iter()
        .map(|opt| {
            opt.and_then(parse_period_slot_token)
                .map(|r| r as i32)
                .unwrap_or(0)
        })
        .collect();
    pivot_df.with_column(Series::new("fp_rank".into(), fp_ranks))?;

    // Sort by FY descending, then by our custom fp rank descending
    pivot_df = pivot_df.sort(
        ["fy", "fp_rank"],
        SortMultipleOptions::default()
            .with_order_descending(true)
            .with_nulls_last(true),
    )?;

    // Add a canonical order index (0 = Latest Time Period, 1 = Previous, etc.)
    // This provides a stable, integer-based reverse chronological sort key.
    pivot_df = pivot_df
        .lazy()
        .with_row_index("canonical_order", None)
        .collect()?;

    // Drop the helper rank column
    let _ = pivot_df.drop_in_place("fp_rank");

    // Add filing_url column: initial values are the EDGAR filing index page.
    // fetch_us_gaap_fundamentals will overwrite these with primary document URLs
    // for any accession numbers found in the submissions API.
    let filing_urls: Vec<Option<String>> = pivot_df
        .column("accn")?
        .str()?
        .into_iter()
        .map(|opt| {
            opt.map(|accn| {
                let cik_struct = cik_value.and_then(|v| Cik::from_u64(v).ok());
                let accn_struct = AccessionNumber::from_str(accn).ok();
                match (cik_struct, accn_struct) {
                    (Some(c), Some(a)) => Url::CikAccessionIndex(c, a).value(),
                    _ => String::new(),
                }
            })
        })
        .collect();
    pivot_df.with_column(Series::new("filing_url".into(), filing_urls))?;

    // Reorder columns to place metadata (canonical_order, fy, fp, period_end, filed, form, accn, filing_url) at the start
    let mut desired_cols: Vec<String> = US_GAAP_CSV_META_COLUMNS
        .iter()
        .map(|s| s.to_string())
        .collect();
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
        assert_eq!(cols[0].as_str(), "canonical_order");
        assert_eq!(cols[1].as_str(), "fy");
        assert_eq!(cols[2].as_str(), "fp");
        assert_eq!(cols[3].as_str(), "period_end");
        assert_eq!(cols[4].as_str(), "filed");
        assert_eq!(cols[5].as_str(), "form");
        assert_eq!(cols[6].as_str(), "is_amendment");
        assert_eq!(cols[7].as_str(), "accn");
        assert!(cols.iter().any(|c| c.as_str() == "Revenue"));

        // 2. Validate row count: Should be 3 rows (2024 FY, 2024 Q3, 2024 Q2).
        // The duplicate Q3 should be consolidated.
        assert_eq!(
            df.height(),
            3,
            "DataFrame should have 3 rows after consolidation"
        );

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
        assert_eq!(accn_col.get(1), Some("000-2024-Q3-AMEND"));
        assert_eq!(filed_col.get(1), Some("2025-02-01"));

        // Amendment normalization: form must be the base type, is_amendment must reflect origin.
        let form_col = df.column("form").unwrap().str().unwrap();
        let is_amend_col = df.column("is_amendment").unwrap().bool().unwrap();
        assert_eq!(form_col.get(0), Some("10-K")); // FY row, not amended
        assert_eq!(is_amend_col.get(0), Some(false));
        assert_eq!(form_col.get(1), Some("10-Q")); // Q3 row, came from 10-Q/A — stripped
        assert_eq!(is_amend_col.get(1), Some(true));
        assert_eq!(form_col.get(2), Some("10-Q")); // Q2 row, not amended
        assert_eq!(is_amend_col.get(2), Some(false));

        // Check Row 2 (Should be 2024 Q2)
        assert_eq!(fy_col.get(2), Some(2024));
        assert_eq!(fp_col.get(2), Some("Q2"));
        assert_eq!(revenue_col.get(2), Some("1000::USD"));
    }

    #[test]
    fn test_off_calendar_fiscal_year_logic() {
        // Scenario:
        // 1. Standard Calendar Year: FY 2023 matches End 2023.
        // 2. Off-Calendar Year (e.g. Nvidia): FY 2024 ends in early 2023 (or late 2023).
        //    Logic allows fy <= end_year + 1.
        // 3. Invalid/Data Error: FY 2025 tagged for End 2023. (fy > end_year + 1). Should be dropped.

        let mock_json = json!({
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenues",
                        "description": "Revenues",
                        "units": {
                            "USD": [
                                // Case 1: Standard (Keep)
                                {
                                    "val": 100.0,
                                    "end": "2023-12-31",
                                    "fy": 2023,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2024-03-01",
                                    "accn": "standard-2023"
                                },
                                // Case 2: Off-Calendar / FY Forward (Keep)
                                // e.g. FY 2024 Q1 ending in April 2023
                                {
                                    "val": 200.0,
                                    "end": "2023-04-30",
                                    "fy": 2024,
                                    "fp": "Q1",
                                    "form": "10-Q",
                                    "filed": "2023-06-01",
                                    "accn": "off-calendar-2024-q1"
                                },
                                // Case 3: Invalid Gap (Drop)
                                // FY 2026 tagged on 2023 data. Gap of > 1 year.
                                {
                                    "val": 300.0,
                                    "end": "2023-12-31",
                                    "fy": 2026,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2024-03-01",
                                    "accn": "invalid-gap"
                                }
                            ]
                        }
                    }
                }
            }
        });

        let df = parse_us_gaap_fundamentals(mock_json)
            .expect("Failed to parse mock JSON with off-calendar dates");

        assert_eq!(
            df.height(),
            2,
            "Should reserve 2 rows, dropping the invalid gap row"
        );

        let accn_col = df.column("accn").unwrap().str().unwrap();
        let accn_values: Vec<&str> = accn_col.into_iter().flatten().collect();

        assert!(accn_values.contains(&"standard-2023"));
        assert!(accn_values.contains(&"off-calendar-2024-q1"));
        assert!(!accn_values.contains(&"invalid-gap"));
    }

    #[test]
    fn test_q4_fp_normalized_to_fy() {
        // Some companies file a 10-Q/A for period Q4. The SEC tags fp="Q4" in
        // companyfacts, but we normalize to "FY" so that the fp column is
        // consistent for downstream consumers.  If the same company also has a
        // genuine 10-K (fp="FY") for the same fiscal year, the amendment logic
        // (filed DESC → UniqueKeepStrategy::First) already picks the latest one —
        // after normalization both rows compete on the same (fy, fp="FY") key.
        let mock_json = json!({
            "cik": 9999,
            "entityName": "Q4 Filer Corp",
            "facts": {
                "us-gaap": {
                    "Revenue": {
                        "label": "Revenue",
                        "units": {
                            "USD": [
                                // Q3 — normal quarterly row, must not be touched.
                                {
                                    "val": 500.0,
                                    "end": "2022-09-30",
                                    "fy": 2022,
                                    "fp": "Q3",
                                    "form": "10-Q",
                                    "filed": "2022-11-01",
                                    "accn": "000-2022-Q3"
                                },
                                // Year-end tagged Q4 by the SEC — must be normalized to FY.
                                {
                                    "val": 2000.0,
                                    "end": "2022-12-31",
                                    "fy": 2022,
                                    "fp": "Q4",
                                    "form": "10-Q/A",
                                    "filed": "2023-02-01",
                                    "accn": "000-2022-Q4"
                                }
                            ]
                        }
                    }
                }
            }
        });

        let df = parse_us_gaap_fundamentals(mock_json).expect("parse failed");

        // Two distinct periods: FY (was Q4) and Q3.
        assert_eq!(
            df.height(),
            2,
            "Q4 and FY must not produce two separate rows"
        );

        let fp_col = df.column("fp").unwrap().str().unwrap();
        let fp_values: Vec<&str> = fp_col.into_iter().flatten().collect();

        // Q4 must have been renamed to FY.
        assert!(
            fp_values.contains(&"FY"),
            "Q4 must be normalized to FY, got {:?}",
            fp_values
        );
        assert!(
            !fp_values.contains(&"Q4"),
            "raw Q4 must not appear in fp column, got {:?}",
            fp_values
        );

        // The normalized FY row must be row 0 (newest-first ordering).
        assert_eq!(fp_col.get(0), Some("FY"));
        assert_eq!(fp_col.get(1), Some("Q3"));

        // is_amendment must be true for the Q4/FY row (came from a 10-Q/A).
        let is_amend_col = df.column("is_amendment").unwrap().bool().unwrap();
        assert_eq!(is_amend_col.get(0), Some(true));
    }
}
