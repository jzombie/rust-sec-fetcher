//! Fetches and displays the SEC company profile for one or more ticker symbols.
//!
//! Demonstrates [`fetch_company_profile`] and [`fetch_company_description`], which
//! pull from the EDGAR submissions endpoint and the SEC company facts API respectively.
//! The profile includes SIC code, sector, exchange listing, fiscal year end, and
//! website metadata. Multiple tickers can be supplied in one invocation.
//!
//! # Usage
//!
//! ```text
//! cargo run --example company_show -- AAPL
//! cargo run --example company_show -- AAPL MSFT NVDA
//! cargo run --example company_show -- SPY
//! ```

use chrono::Datelike;
use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{
    fetch_cik_by_ticker_symbol, fetch_company_description, fetch_company_profile, SecClient,
};
use std::error::Error;
use std::fmt;

#[derive(Parser)]
#[command(
    about = "Look up the SEC company profile for one or more ticker symbols",
    long_about = None
)]
struct Args {
    /// One or more ticker symbols (e.g. AAPL MSFT NVDA)
    #[arg(required = true)]
    tickers: Vec<String>,
}

/// All company-level metadata for a single filer, ready for display.
struct CompanyProfileDisplay {
    ticker: String,
    cik: String,
    name: String,
    description: Option<String>,
    sic: String,
    industry: String,
    sector: String,
    exchanges: String,
    category: String,
    state_of_incorporation: String,
    fiscal_year_end: String,
    website: Option<String>,
    investor_website: Option<String>,
    phone: Option<String>,
}

impl fmt::Display for CompanyProfileDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Ticker:          {}", self.ticker)?;
        writeln!(f, "CIK:             {}", self.cik)?;
        writeln!(f, "Name:            {}", self.name)?;
        if let Some(ref desc) = self.description {
            writeln!(f, "Description:     {}", desc)?;
        }
        writeln!(f, "SIC:             {}", self.sic)?;
        writeln!(f, "Industry:        {}", self.industry)?;
        writeln!(f, "Sector:          {}", self.sector)?;
        writeln!(f, "Exchange(s):     {}", self.exchanges)?;
        writeln!(f, "Filer category:  {}", self.category)?;
        writeln!(f, "Incorporated in: {}", self.state_of_incorporation)?;
        writeln!(f, "Fiscal year end: {}", self.fiscal_year_end)?;
        if let Some(ref url) = self.website {
            writeln!(f, "Website:         {}", url)?;
        }
        if let Some(ref url) = self.investor_website {
            writeln!(f, "Investor site:   {}", url)?;
        }
        if let Some(ref phone) = self.phone {
            write!(f, "Phone:           {}", phone)?;
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let tickers: Vec<TickerSymbol> = args.tickers.iter().map(|s| TickerSymbol::new(s)).collect();

    let cfg = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&cfg)?;

    for ticker in &tickers {
        match fetch_profile_display(&client, ticker).await {
            Ok(profile) => println!("{}", profile),
            Err(e) => eprintln!("Error for {ticker}: {e}"),
        }
        println!();
    }

    Ok(())
}

async fn fetch_profile_display(
    client: &SecClient,
    ticker: &TickerSymbol,
) -> Result<CompanyProfileDisplay, Box<dyn Error>> {
    let cik = fetch_cik_by_ticker_symbol(client, ticker).await?;
    assert!(cik.value > 0, "CIK must be a positive non-zero integer");
    let profile = fetch_company_profile(client, cik).await?;
    let description = fetch_company_description(client, profile.cik.clone()).await?;

    // Compute all values that borrow `profile` before consuming any fields.
    let unique_exchanges: String = {
        let mut seen = std::collections::HashSet::new();
        let v: Vec<&str> = profile
            .exchanges
            .iter()
            .filter(|e| seen.insert(e.as_str()))
            .map(|e| e.as_str())
            .collect();
        if v.is_empty() {
            "n/a".to_string()
        } else {
            v.join(", ")
        }
    };
    let sector = profile.sector().unwrap_or("n/a").to_string();
    let fiscal_year_end = profile
        .fiscal_year_end_date()
        .map(|d| format!("{} {}", d.format("%b"), d.day()))
        .unwrap_or_else(|| "n/a".to_string());

    Ok(CompanyProfileDisplay {
        ticker: ticker.to_string(),
        cik: profile.cik.to_string(),
        name: profile.name,
        description,
        sic: profile.sic.unwrap_or_else(|| "n/a".to_string()),
        industry: profile.sic_description.unwrap_or_else(|| "n/a".to_string()),
        sector,
        exchanges: unique_exchanges,
        category: profile.category.unwrap_or_else(|| "n/a".to_string()),
        state_of_incorporation: profile
            .state_of_incorporation
            .unwrap_or_else(|| "n/a".to_string()),
        fiscal_year_end,
        website: profile.website,
        investor_website: profile.investor_website,
        phone: profile.phone,
    })
}
