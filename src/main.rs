use polars::prelude::{CsvWriter, SerWriter};
use sec_fetcher::{
    config::ConfigManager,
    network::{
        fetch_company_tickers, fetch_investment_company_series_and_class_dataset,
        fetch_nport_filing_by_ticker_symbol, fetch_us_gaap_fundamentals, SecClient,
    },
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

    let investment_companies = fetch_investment_company_series_and_class_dataset(&client).await?;

    for fund in investment_companies {
        if let Some(ticker_symbol) = fund.class_ticker {
            let latest_nport_filing =
                fetch_nport_filing_by_ticker_symbol(&client, &ticker_symbol).await?;

            // println!("Latest NPORT filing: {:?}", latest_nport_filing);
            // println!("Ticker symbol: {}", ticker_symbol);
            for investment in &latest_nport_filing {
                println!("{:?}", investment);
                println!("");
            }
            println!("Ticker symbol: {}", ticker_symbol);

            println!("Total records: {}", latest_nport_filing.len());

            latest_nport_filing
                .write_to_csv(&format!("data/fund-holdings/{}.csv", ticker_symbol))?;
        }
    }

    Ok(())
}

// Prototype iterator for US-GAAP fundamentals
// #[tokio::main]
// async fn main() -> Result<(), Box<dyn Error>> {
//     let config_manager = ConfigManager::load()?;

//     let client = SecClient::from_config_manager(&config_manager)?;

//     let company_tickers = fetch_company_tickers(&client).await?;
//     println!("Total records: {}", company_tickers.len());
//     println!("{:?}", company_tickers.head(60));

//     // let ticker_series = tickers_df.column("ticker")?.str()?;
//     let mut error_log: HashMap<String, String> = HashMap::new();

//     for (i, company_ticker) in company_tickers.iter().enumerate() {
//         let ticker_symbol = &company_ticker.ticker_symbol;

//         println!(
//             "Processing ticker: {} ({} of {})",
//             company_ticker.ticker_symbol,
//             i + 1,
//             company_tickers.len()
//         );

//         // print!(
//         //     "{}",
//         //     fetch_us_gaap_fundamentals(&client, &tickers_df, &ticker).await?
//         // );
//         // break;

//         match fetch_us_gaap_fundamentals(&client, &company_tickers, &ticker_symbol).await {
//             Ok(mut fundamentals_df) => {
//                 let file_path = format!("data/us-gaap/{}.csv", &ticker_symbol);
//                 match File::create(&file_path) {
//                     Ok(mut file) => {
//                         if let Err(e) = CsvWriter::new(&mut file)
//                             .include_header(true)
//                             .finish(&mut fundamentals_df)
//                         {
//                             error_log
//                                 .insert(ticker_symbol.clone(), format!("CSV write error: {}", e));
//                         }
//                     }
//                     Err(e) => {
//                         eprintln!("File creation error: {}", e);
//                         error_log
//                             .insert(ticker_symbol.clone(), format!("File creation error: {}", e));
//                     }
//                 }
//             }
//             Err(e) => {
//                 error_log.insert(ticker_symbol.clone(), format!("Fetch error: {}", e));
//             }
//         }
//     }

//     // Print summary report
//     if !error_log.is_empty() {
//         println!("\nSummary of errors:");
//         for (ticker, err) in &error_log {
//             println!("- {}: {}", ticker, err);
//         }
//     } else {
//         println!("\nAll tickers processed successfully.");
//     }

//     Ok(())
// }
