use csv::ReaderBuilder;
use sec_fetcher::network::{
    fetch_investment_company_series_and_class_dataset,
    SecClient,
};
use sec_fetcher::config::ConfigManager;
use std::error::Error;
use std::io::{self, Cursor, Write};
use tokio;

/// Prompt user for input
fn prompt_user(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager, 1, 1000, Some(5))?;

    // Fetch CSV byte array
    let byte_array = fetch_investment_company_series_and_class_dataset(&client, 2024).await?;

    // Convert byte array to a CSV reader
    let cursor = Cursor::new(&byte_array);
    let mut reader = ReaderBuilder::new().from_reader(cursor);

    // Extract headers first (before iterating)
    let headers = reader.headers()?.clone();
    let ticker_index = headers
        .iter()
        .position(|h| h == "Class Ticker")
        .ok_or("Column 'Class Ticker' not found")?;

    // Prompt user for the stock ticker symbol
    let target_ticker = prompt_user("Enter the stock ticker symbol: ");

    // Iterate through CSV rows
    let mut found = false;
    for result in reader.records() {
        let record = result?;
        if record.get(ticker_index) == Some(target_ticker.as_str()) {
            println!("{}", record.iter().collect::<Vec<&str>>().join(", "));
            found = true;
        }
    }

    if !found {
        println!("No matching record found for ticker '{}'.", target_ticker);
    }

    Ok(())
}
