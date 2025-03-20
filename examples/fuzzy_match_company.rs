use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::{CikSubmission, Ticker};
use sec_fetcher::network::{
    fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_company_tickers, SecClient,
};
use std::env;
use std::error::Error;
use tokio;

// TODO: Add unit tests for (at least)
//  - Container: SPYV
//  ---- ZION:  Investment 4347: NportInvestment { company_ticker: None, name: "ZIONS BANCORP NA"
//  ---- SHW: Investment 4551: NportInvestment { company_ticker: Some(CompanyTicker { cik: Cik { value: 89800 }, ticker_symbol: "SHW", company_name: "SHERWIN WILLIAMS CO" }),
//  ---- HSY: Investment 4522: NportInvestment { company_ticker: None, name: "HERSHEY COMPANY", lei: "21X2CX66SU2BR6QTAD08", title: "Hershey Co/The", cusip: "427866AX6",
//  ---- PGR: Investment 4391: NportInvestment { company_ticker: None, name: "PROGRESSIVE CORP", lei: "529900TACNVLY9DCR586", title: "Progressive Corp/The",
//  ---- [subsidiary; probably ok] DE: Investment 4814: NportInvestment { company_ticker: None, name: "JOHN DEERE CAPITAL CORP", lei: "E0KSF7PFQ210NWI8Z391", title: "John Deere Capital Corp",
// ---- [Incorrect stock symbol for PNC] PNC: Investment 109: NportInvestment { company_ticker: Some(CompanyTicker { cik: Cik { value: 1393612 }, ticker_symbol: "DFS", company_name: "Discover Financial Services" }), name: "PNC FINANCIAL SERVICES",
// ---- PCG: Investment 487: NportInvestment { company_ticker: None, name: "PACIFIC GAS & ELECTRIC", lei: "1HNPXZSMMB7HMBMVBS46", title: "Pacific Gas and Electric Co"

// - Container: SPY
//  ---- LOW: Investment 67: NportInvestment { company_ticker: None, name: "Lowe's Cos Inc", lei: "WAFCR4OKGSC504WU3E95", title: "Lowe's Cos Inc", cu
//  ---- TJX: Investment 69: NportInvestment { company_ticker: None, name: "TJX Cos Inc/The", lei: "V167QI9I69W364E2DY52", title: "TJX Cos Inc/The",
//  ---- WMB: Investment 145: NportInvestment { company_ticker: None, name: "Williams Cos Inc/The", lei: "D71FAKCBLFS2O0RBPG08", title: "Williams Cos Inc/The",
//  ---- DHI: Investment 218: NportInvestment { company_ticker: None, name: "DR Horton Inc", lei: "529900ZIUEYVSB8QDD25", title: "DR Horton Inc", cusip: "23331A109", isin: "US23331A1097",
//  ---- LYB: Investment 367: NportInvestment { company_ticker: None, name: "LyondellBasell Industries NV", lei: "BN6WCCZ8OVP3ITUUVN49", title: "LyondellBasell Industries NV",
//  ---- KEY: Investment 390: NportInvestment { company_ticker: None, name: "KeyCorp", lei: "RKPI3RZGV1V1FJTH5T61", title: "KeyCorp",
//  ---- DPZ: Investment 424: NportInvestment { company_ticker: None, name: "Domino's Pizza Inc", lei: "25490005ZWM1IF9UXU57", title: "Domino's Pizza Inc",
//  ---- CPB: Investment 483: NportInvestment { company_ticker: None, name: "The Campbell's Company", lei: "5493007JDSMX8Z5Z1902", title: "The Campbell's Company", cusip: "134429109",
//  ---- JNJ: Investment 22: NportInvestment { company_ticker: None, name: "Johnson & Johnson", lei: "549300G0CFPGEF6X2043", title: "Johnson & Johnson",

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} \"<SEARCH_STRING>\"", args[0]);
        std::process::exit(1);
    }

    let search_string = &args[1];

    println!("Searching for: {}", search_string);

    println!(
        "Tokenized: {:?}",
        Ticker::tokenize_company_name(search_string)
    );

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let company_tickers = fetch_company_tickers(&client).await.unwrap();

    // Override search string with company name if using direct symbol
    let search_string = {
        let exact_company_ticker = company_tickers.iter().find(|p| {
            p.symbol.to_lowercase() == search_string.to_lowercase()
                || p.company_name.to_lowercase() == search_string.to_lowercase()
        });

        // Make it easier to test by doing symbol lookup to get the company name
        let search_string = match exact_company_ticker {
            Some(ticker) => {
                println!("Exact match: {:?}", ticker);

                let company_name = ticker.company_name.to_string();

                company_name
            }
            None => search_string.to_string(),
        };

        Box::new(search_string)
    };

    println!("Using search string: {}", search_string);

    let fuzzy_matched = Ticker::get_by_fuzzy_matched_name(&company_tickers, &search_string, false);

    println!("Fuzzy matched: {:?}", fuzzy_matched);

    Ok(())
}
