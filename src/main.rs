use polars::prelude::{CsvWriter, SerWriter};
use sec_fetcher::{
    config::ConfigManager,
    network::{fetch_company_tickers, fetch_us_gaap_fundamentals, SecClient},
    utils::VecExtensions,
};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config_manager = ConfigManager::load()?;

    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client).await?;
    println!("Total records: {}", company_tickers.len());
    println!("{:?}", company_tickers.head(60));

    // let ticker_series = tickers_df.column("ticker")?.str()?;
    let mut error_log: HashMap<String, String> = HashMap::new();

    for (i, company_ticker) in company_tickers.iter().enumerate() {
        let ticker_symbol = &company_ticker.ticker_symbol;

        println!(
            "Processing ticker: {} ({} of {})",
            company_ticker.ticker_symbol,
            i + 1,
            company_tickers.len()
        );

        // print!(
        //     "{}",
        //     fetch_us_gaap_fundamentals(&client, &tickers_df, &ticker).await?
        // );
        // break;

        match fetch_us_gaap_fundamentals(&client, &company_tickers, &ticker_symbol).await {
            Ok(mut fundamentals_df) => {
                let file_path = format!("data/us-gaap/{}.csv", &ticker_symbol);
                match File::create(&file_path) {
                    Ok(mut file) => {
                        if let Err(e) = CsvWriter::new(&mut file)
                            .include_header(true)
                            .finish(&mut fundamentals_df)
                        {
                            error_log
                                .insert(ticker_symbol.clone(), format!("CSV write error: {}", e));
                        }
                    }
                    Err(e) => {
                        eprintln!("File creation error: {}", e);
                        error_log
                            .insert(ticker_symbol.clone(), format!("File creation error: {}", e));
                    }
                }
            }
            Err(e) => {
                error_log.insert(ticker_symbol.clone(), format!("Fetch error: {}", e));
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
