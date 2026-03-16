/// Look up the SEC company profile (name, SIC industry, exchange, etc.) for one
/// or more ticker symbols.
///
/// # What this returns
///
/// The SEC EDGAR submissions endpoint (`/submissions/CIK{cik}.json`) exposes
/// all the company-level metadata SEC filers must maintain:
///
/// | Field                  | Example                         |
/// |------------------------|---------------------------------|
/// | Name                   | Apple Inc.                      |
/// | SIC code               | 3571                            |
/// | SIC description        | Electronic Computers            |

/// | Owner-org sector       | 06 Technology                   |
/// | Exchange(s)            | Nasdaq                          |
/// | Filer category         | Large accelerated filer         |
/// | State of incorporation | CA                              |
/// | Fiscal year end        | 0926 (= September 26)           |
/// | Website / IR site      | https://www.apple.com           |
///
/// The SIC description is the SEC's own industry classification. The
/// `owner_org` field provides a coarser grouping (sector level) that the SEC
/// uses internally. Neither maps directly to GICS or SIC standard tables, but
/// they are stable, authoritative, and free.
///
/// # Cache note
///
/// `fetch_company_profile` and `fetch_cik_submissions` both call the same URL,
/// so the second call for a given CIK is served from the local cache at no
/// cost.
///
/// # Usage
///
///   cargo run --example company_profile -- AAPL
///   cargo run --example company_profile -- AAPL MSFT NVDA
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_cik_by_ticker_symbol, fetch_company_description, fetch_company_profile, SecClient};
use chrono::Datelike;
use std::env;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let tickers: Vec<String> = env::args().skip(1).map(|s| s.to_uppercase()).collect();

    if tickers.is_empty() {
        eprintln!("Usage: company_profile <TICKER> [TICKER ...]");
        eprintln!("  e.g. cargo run --example company_profile -- AAPL MSFT NVDA");
        std::process::exit(1);
    }

    let cfg = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&cfg)?;

    for ticker in &tickers {
        match lookup(&client, ticker).await {
            Ok(()) => {}
            Err(e) => eprintln!("Error for {ticker}: {e}"),
        }
        println!();
    }

    Ok(())
}

async fn lookup(client: &SecClient, ticker: &str) -> Result<(), Box<dyn Error>> {
    let cik = fetch_cik_by_ticker_symbol(client, ticker).await?;
    let profile = fetch_company_profile(client, cik).await?;

    println!("Ticker:          {ticker}");
    println!("CIK:             {}", profile.cik.to_string());
    println!("Name:            {}", profile.name);
    if let Some(desc) = fetch_company_description(client, profile.cik.clone()).await? {
        println!("Description:     {desc}");
    }
    println!("SIC:             {}", profile.sic.as_deref().unwrap_or("n/a"));
    println!("Industry:        {}", profile.sic_description.as_deref().unwrap_or("n/a"));
    println!("Sector:          {}", profile.sector().unwrap_or("n/a"));
    let unique_exchanges: Vec<&str> = {
        let mut seen = std::collections::HashSet::new();
        profile
            .exchanges
            .iter()
            .filter(|e| seen.insert(e.as_str()))
            .map(|e| e.as_str())
            .collect()
    };
    println!(
        "Exchange(s):     {}",
        if unique_exchanges.is_empty() {
            "n/a".to_string()
        } else {
            unique_exchanges.join(", ")
        }
    );
    println!(
        "Filer category:  {}",
        profile.category.as_deref().unwrap_or("n/a")
    );
    println!(
        "Incorporated in: {}",
        profile.state_of_incorporation.as_deref().unwrap_or("n/a")
    );
    println!(
        "Fiscal year end: {}",
        profile
            .fiscal_year_end_date()
            .map(|d| format!("{} {}", d.format("%b"), d.day()))
            .unwrap_or_else(|| "n/a".to_string())
    );
    if let Some(url) = &profile.website {
        println!("Website:         {url}");
    }
    if let Some(url) = &profile.investor_website {
        println!("Investor site:   {url}");
    }
    if let Some(phone) = &profile.phone {
        println!("Phone:           {phone}");
    }

    Ok(())
}
