use clap::Parser;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::{CikSubmission, NportInvestment, TickerSymbol};
use sec_fetcher::network::{
    fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_nport, SecClient,
};
use sec_fetcher::utils::VecExtensions;
use std::error::Error;
use std::fmt;

#[derive(Parser)]
#[command(about = "Fetch and display the most recent NPORT-P holdings for a fund ticker")]
struct Args {
    /// Fund ticker symbol (e.g. SPY, QQQ)
    ticker: String,
}

/// Display wrapper for a single N-PORT investment position.
struct NportInvestmentRow<'a> {
    index: usize,
    investment: &'a NportInvestment,
}

impl<'a> fmt::Display for NportInvestmentRow<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inv = self.investment;
        write!(
            f,
            "{:>4}.  {:<9}  {:<45}  {:>10}  {:>8.4}%  {} {}",
            self.index,
            inv.cusip,
            inv.name.chars().take(45).collect::<String>(),
            fmt_usd(inv.val_usd),
            inv.pct_val,
            inv.cur_cd,
            inv.inv_country,
        )
    }
}

fn fmt_usd(v: rust_decimal::Decimal) -> String {
    use rust_decimal_macros::dec;
    if v >= dec!(1_000_000_000) {
        format!("${:.2}B", v / dec!(1_000_000_000))
    } else if v >= dec!(1_000_000) {
        format!("${:.2}M", v / dec!(1_000_000))
    } else {
        format!("${:.0}", v)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let ticker_symbol = TickerSymbol::new(&args.ticker);

    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let cik = fetch_cik_by_ticker_symbol(&client, &ticker_symbol).await?;
    let submissions = fetch_cik_submissions(&client, cik).await?;
    let latest = CikSubmission::by_form(&submissions, "NPORT-P")
        .into_iter()
        .next()
        .ok_or("No NPORT-P filings found")?;

    let investments = fetch_nport(&client, latest).await?;

    println!(
        "{:>4}   {:<9}  {:<45}  {:>10}  {:>9}  CCY Country",
        "#", "CUSIP", "Name", "Value USD", "Weight"
    );
    println!("{}", "-".repeat(100));
    for (i, investment) in investments.head(510).iter().enumerate() {
        println!(
            "{}",
            NportInvestmentRow {
                index: i + 1,
                investment
            }
        );
    }

    Ok(())
}
