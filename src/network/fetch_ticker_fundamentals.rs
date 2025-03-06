use crate::accessor::get_cik_by_ticker_symbol;
use crate::network::{SecClient, SecTickersDataFrame};
use polars::prelude::pivot::pivot;
use polars::prelude::*;
use serde_json::Value;
use std::error::Error;

pub type TickerFundamentalsDataFrame = DataFrame;

/// Fetches US-GAAP SEC fundamentals for a given ticker symbol
pub async fn fetch_ticker_fundamentals(
    client: &SecClient,
    df_tickers: &SecTickersDataFrame,
    ticker_symbol: &str,
) -> Result<TickerFundamentalsDataFrame, Box<dyn Error>> {
    // Get the formatted CIK for the ticker
    let cik = get_cik_by_ticker_symbol(df_tickers, ticker_symbol)?;
    // Define the URL for the SEC API

    let url = format!("https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json", cik);

    // TODO: Debug log
    println!("Using URL: {}", url);

    let data: Value = client.fetch_json(&url).await?;

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
                                fy_values.push(obs["fy"].as_i64().unwrap_or(0));
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
        ["fy", "fp", "form", "filed", "accn"]
            .iter()
            .map(|&s| s.to_string())
            .collect::<Vec<String>>(),
        Some(
            ["fact_name"]
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

    // Transpose to flip rows & columns
    pivot_df = pivot_df.transpose(Some("fact_name"), None)?;

    // Extract first row as column names
    let new_column_names = pivot_df
        .head(Some(1))
        .transpose(None, None)?
        .column("column_0")?
        .str()?
        .into_iter()
        .map(|opt| opt.unwrap_or("Unknown").to_string()) // Handle missing values
        .collect::<Vec<String>>();

    // Drop the first row and assign new column names
    let mut pivot_df = pivot_df.tail(Some(pivot_df.height() - 1));
    pivot_df.set_column_names(&new_column_names)?;

    // Extract fact_name details as separate columns
    let fact_name_series = pivot_df
        .column("fact_name")?
        .str()?
        .into_iter()
        .map(|opt| {
            opt.unwrap_or("{0,0,0,0,0}")
                .trim_matches(&['{', '}'][..])
                .split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<String>>()
        })
        .collect::<Vec<Vec<String>>>();

    let fy_series = Series::new(
        "fy".into(),
        fact_name_series
            .iter()
            .map(|x| {
                x.get(0)
                    .map(|s| s.trim_matches('"').to_string())
                    .unwrap_or_default()
            }) // Safe indexing & cleanup
            .collect::<Vec<String>>(),
    );
    let fp_series = Series::new(
        "fp".into(),
        fact_name_series
            .iter()
            .map(|x| {
                x.get(1)
                    .map(|s| s.trim_matches('"').to_string())
                    .unwrap_or_default()
            }) // Safe indexing & cleanup
            .collect::<Vec<String>>(),
    );
    let form_series = Series::new(
        "form".into(),
        fact_name_series
            .iter()
            .map(|x| {
                x.get(2)
                    .map(|s| s.trim_matches('"').to_string())
                    .unwrap_or_default()
            }) // Safe indexing & cleanup
            .collect::<Vec<String>>(),
    );
    let filed_series = Series::new(
        "filed".into(),
        fact_name_series
            .iter()
            .map(|x| {
                x.get(3)
                    .map(|s| s.trim_matches('"').to_string())
                    .unwrap_or_default()
            }) // Safe indexing & cleanup
            .collect::<Vec<String>>(),
    );
    let accn_series = Series::new(
        "accn".into(),
        fact_name_series
            .iter()
            .map(|x| {
                x.get(4)
                    .map(|s| s.trim_matches('"').to_string())
                    .unwrap_or_default()
            }) // Safe indexing & cleanup
            .collect::<Vec<String>>(),
    );

    pivot_df.drop_in_place("fact_name")?;
    pivot_df
        .with_column(fy_series)?
        .with_column(fp_series)?
        .with_column(form_series)?
        .with_column(filed_series)?
        .with_column(accn_series)?;

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
