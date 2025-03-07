use std::collections::HashMap;
use std::error::Error;
use tokio;
mod network;
use network::{
    fetch_sec_tickers, fetch_us_gaap_fundamentals, CredentialManager, CredentialProvider, SecClient,
};
mod accessors;
mod enums;
mod parsers;
mod transformers;
mod utils;
use polars::prelude::{CsvWriter, SerWriter};
use std::fs::File;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let credential_manager = CredentialManager::from_prompt()?;

    let client = SecClient::from_credential_manager(&credential_manager, 1, 1000, Some(5))?;

    let tickers_df = fetch_sec_tickers(&client).await?;
    println!("Total records: {}", tickers_df.height());
    println!("{}", tickers_df.head(Some(60)));

    let ticker_series = tickers_df.column("ticker")?.str()?;
    let mut error_log: HashMap<String, String> = HashMap::new();

    for i in 0..tickers_df.height() {
        let ticker = ticker_series.get(i).unwrap_or("UNKNOWN").to_string();
        println!(
            "Processing ticker: {} ({} of {})",
            ticker,
            i + 1,
            tickers_df.height()
        );

        // print!(
        //     "{}",
        //     fetch_us_gaap_fundamentals(&client, &tickers_df, &ticker).await?
        // );
        // break;

        match fetch_us_gaap_fundamentals(&client, &tickers_df, &ticker).await {
            Ok(mut fundamentals_df) => {
                let file_path = format!("data/us-gaap/{}.csv", ticker);
                match File::create(&file_path) {
                    Ok(mut file) => {
                        if let Err(e) = CsvWriter::new(&mut file)
                            .include_header(true)
                            .finish(&mut fundamentals_df)
                        {
                            error_log.insert(ticker.clone(), format!("CSV write error: {}", e));
                        }
                    }
                    Err(e) => {
                        error_log.insert(ticker.clone(), format!("File creation error: {}", e));
                    }
                }
            }
            Err(e) => {
                error_log.insert(ticker.clone(), format!("Fetch error: {}", e));
            }
        }
    }

    // Print summary report
    if !error_log.is_empty() {
        println!("\nSummary of errors:");
        for (ticker, err) in &error_log {
            println!("- {}: {}", ticker, err);
        }
    } else {
        println!("\nAll tickers processed successfully.");
    }

    Ok(())
}
