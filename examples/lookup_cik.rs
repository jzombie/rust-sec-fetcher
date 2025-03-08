use csv::ReaderBuilder;
use sec_fetcher::accessors::get_cik_by_ticker_symbol;
use sec_fetcher::network::{
    fetch_cik_submissions, fetch_investment_company_series_and_class_dataset, fetch_sec_tickers,
    CikSubmission, SecClient,
};
use sec_fetcher::config::{CredentialManager, CredentialProvider};
use sec_fetcher::models::Cik;
use std::env;
use std::error::Error;
use std::io::Cursor;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let credential_manager = CredentialManager::from_prompt()?;

    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <TICKER_SYMBOL>", args[0]);
        std::process::exit(1);
    }

    let ticker_symbol = &args[1];

    let client = SecClient::from_credential_manager(&credential_manager, 1, 1000, Some(5))?;

    let mut result_cik: Option<Cik> = None;

    // First, try the primary search method
    let tickers_df = fetch_sec_tickers(&client).await?;

    if let Ok(cik) = get_cik_by_ticker_symbol(&tickers_df, ticker_symbol) {
        println!("Ticker: {}, CIK: {} (reg. stocks)", ticker_symbol, cik.to_string());
        result_cik = Some(cik);
    } else {
        // If not found, try searching in the investment company dataset
        println!("No match found in primary search. Searching in investment company dataset...");

        let byte_array = fetch_investment_company_series_and_class_dataset(&client, 2024).await?;
        let cursor = Cursor::new(&byte_array);
        let mut reader = ReaderBuilder::new().from_reader(cursor);

        // Extract headers first
        let headers = reader.headers()?.clone();
        let ticker_index = headers
            .iter()
            .position(|h| h == "Class Ticker")
            .ok_or("Column 'Class Ticker' not found")?;
        let cik_index = headers
            .iter()
            .position(|h| h == "CIK Number")
            .ok_or("Column 'CIK Number' not found")?;

        for result in reader.records() {
            let record = result?;
            if record.get(ticker_index) == Some(ticker_symbol.as_str()) {
                if let Some(cik_str) = record.get(cik_index) {    
                    println!("Ticker: {}, CIK: {} (fund)", ticker_symbol, cik_str);

                    let cik = Cik::from_str(cik_str)?;
                    result_cik = Some(cik);
                }
            }
        }
    }

    if result_cik.is_none() {
        println!("No matching record found for ticker '{}'.", ticker_symbol);
    } else {
        let cik = result_cik.unwrap();

        println!(
            "Submissions URL: https://data.sec.gov/submissions/CIK{}.json",
            cik.to_string()
        );

        // // TODO: Lookup filings -> recent -> accessionNumber, strip out the dahes, and paste in
        // // `primaryDocument` indices which contain NPORT-P, likely have a primary_doc.xml attached
        // // in which case holdings can be parsed from this
        // println!(
        //     "Edgar Data (prefix): https://www.sec.gov/Archives/edgar/data/{}/XXXX/",
        //     &result_cik.unwrap()
        // )

        let cik_submissions = fetch_cik_submissions(&client, cik).await?;

        if let Some(most_recent_nport_p_submission) =
            CikSubmission::most_recent_nport_p_submission(&cik_submissions)
        {
            println!(
                "Most recent NPORT-P submission: {:?}",
                &most_recent_nport_p_submission
            );
            println!(
                "EDGAR archive URL: {}",
                most_recent_nport_p_submission.as_edgar_archive_url()
            );
        } else {
            for cik_submission in cik_submissions {
                println!("{:?}", cik_submission);

                println!(
                    "EDGAR archive URL: {}",
                    cik_submission.as_edgar_archive_url()
                );

                // TODO: Remove
                if cik_submission.form == "10-K" {
                    break;
                }
            }
        }
    }

    Ok(())
}
