use polars::prelude::pivot::pivot;
use polars::prelude::*;
use serde_json::Value;
use std::error::Error;

pub type TickerFundamentalsDataFrame = DataFrame;

// TODO: Document how sorting is performed internally
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
                                fact_category_values.push(fact_category.clone());
                                fact_name_values.push(fact_name.clone());
                                label_values.push(label.clone());

                                // Joins the unit of measure with the value, using a double-colon (`::`) separator
                                let value = obs["val"]
                                    .as_f64()
                                    .map(|v| format!("{}::{}", v, unit)) // Append unit to value
                                    .unwrap_or_else(|| format!("0::{}", unit)); // Handle missing values

                                value_values.push(value);
                                end_values.push(obs["end"].as_str().unwrap_or("").to_string());
                                form_values.push(obs["form"].as_str().unwrap_or("").to_string());
                                filed_values.push(obs["filed"].as_str().unwrap_or("").to_string());
                                fy_values.push(obs["fy"].as_u64().unwrap_or(0));
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
    let df = df!(
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

    // Perform a single pivot (values include units)
    let mut pivot_df = pivot(
        &df,
        ["fact_name"]
            .iter()
            .map(|&s| s.to_string())
            .collect::<Vec<String>>(),
        Some(
            ["fy", "fp", "form", "filed", "accn"]
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
        Some(col("value").first()), // Just take the first occurrence
        None,
    )?;

    let filed_date_series: Series = pivot_df
        .column("filed")?
        .str()?
        .as_date(Some("%Y-%m-%d"), false)?
        .into_series();

    pivot_df.with_column(filed_date_series)?;

    pivot_df = pivot_df.sort(
        ["filed"],
        SortMultipleOptions::default()
            .with_order_descending(true)
            .with_nulls_last(true),
    )?;

    Ok(pivot_df)
}
