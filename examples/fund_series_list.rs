//! Lists SEC-registered investment company series and class records.
//!
//! Downloads the SEC's investment company series/class dataset, which covers
//! all registered mutual fund series (each fund product) and their share classes
//! (each fee structure). This dataset is used internally to resolve fund CIKs
//! and validate NPORT-P holdings.
//!
//! # Usage
//!
//! ```text
//! cargo run --example fund_series_list
//! cargo run --example fund_series_list -- --limit 50
//! cargo run --example fund_series_list -- --limit 0   # no limit
//! ```

use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_investment_company_series_and_class_dataset, SecClient};
use std::error::Error;

#[derive(Parser)]
#[command(about = "List SEC-registered investment company series and class records")]
struct Args {
    /// Maximum number of funds to display (0 = no limit)
    #[arg(long, default_value_t = 20)]
    limit: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let funds = fetch_investment_company_series_and_class_dataset(&client).await?;

    // The SEC investment company dataset is always non-empty; thousands of
    // registered funds are active at any given time.
    assert!(
        !funds.is_empty(),
        "Expected at least one investment company series in the SEC dataset"
    );

    let display_iter: Box<dyn Iterator<Item = _>> = if args.limit == 0 {
        Box::new(funds.iter())
    } else {
        Box::new(funds.iter().take(args.limit))
    };

    for fund in display_iter {
        println!("{:#?}\n", fund);
    }

    if args.limit > 0 && funds.len() > args.limit {
        eprintln!(
            "(showing {} of {} total; use --limit 0 for all)",
            args.limit,
            funds.len()
        );
    }

    Ok(())
}
