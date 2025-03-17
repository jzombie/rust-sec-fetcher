use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::CikSubmission;
use sec_fetcher::network::{fetch_cik_by_ticker_symbol, fetch_cik_submissions, SecClient};
use std::env;
use std::error::Error;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <TICKER_SYMBOL>", args[0]);
        std::process::exit(1);
    }

    let ticker_symbol = &args[1];

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let result_cik = fetch_cik_by_ticker_symbol(&client, ticker_symbol)
        .await
        .ok();

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
