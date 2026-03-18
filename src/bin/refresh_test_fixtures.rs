//! # refresh_test_fixtures
//!
//! Downloads the latest version of every test fixture from SEC EDGAR and
//! saves each one as `<name>.json.gz` in `tests/fixtures/`.
//!
//! ## When to run
//!
//! - After cloning the repository (compressed fixtures are checked in; this
//!   utility created them and can recreate or refresh them at any time).
//! - When you want to update fixtures to the latest EDGAR data.
//!
//! ```sh
//! cargo run --bin refresh_test_fixtures
//! ```
//!
//! ## Configuration
//!
//! Uses the standard `sec_fetcher_config.toml` — the same config file every
//! other binary in this crate uses.  Rate-limiting, the User-Agent email,
//! and concurrency settings are all respected automatically.  Configure once,
//! and this binary picks it up for free.
//!
//! ## Adding a new fixture
//!
//! 1. Add a `Fixture { ... }` entry to the `FIXTURES` slice using the ticker
//!    symbol and a `FixtureKind` variant.  No CIK look-up, no URL copy-paste.
//! 2. Call `load_fixture("your_name.json")` in the test that needs it.
//!
//! That is all.

use flate2::write::GzEncoder;
use flate2::Compression;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::enums::Url;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{
    fetch_8k_filings, fetch_cik_by_ticker_symbol, fetch_filing_index, SecClient,
};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

// ── Fixture manifest ──────────────────────────────────────────────────────────
//
// Each entry drives one HTTP download → one `.json.gz` file on disk.
//
// Use ticker symbols — CIK look-up is handled automatically.  Never paste a
// raw CIK number or URL string here; use the FixtureKind variants below,
// which delegate URL construction entirely to `sec_fetcher::enums::Url`.
//
// To add a fixture:
//   1. Append a Fixture { output, ticker, kind } to the slice.
//   2. Reference the file in a test with load_fixture("my_output.json").

struct Fixture {
    /// Base filename written to `tests/fixtures/`.  Stored as `{output}.gz`.
    /// Pass this exact string to `load_fixture(...)` in test files.
    output: &'static str,
    /// Ticker symbol used to resolve the CIK at runtime.
    ticker: &'static str,
    /// Which EDGAR endpoint to download for this ticker.
    kind: FixtureKind,
}

enum FixtureKind {
    /// `https://data.sec.gov/submissions/CIK{}.json`
    Submissions,
    /// `https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json`
    CompanyFacts,
    /// Primary HTML document of the latest 8-K for this ticker.
    EightKPrimary,
    /// First renderable (non-binary) EX-* exhibit of the latest 8-K.
    EightKFirstHtmlExhibit,
}

const FIXTURES: &[Fixture] = &[
    // ── submissions (CIK metadata + recent filing history) ────────────────
    Fixture {
        output: "AAPL_submissions.json",
        ticker: "AAPL",
        kind: FixtureKind::Submissions,
    },
    Fixture {
        output: "RDDT_submissions.json",
        ticker: "RDDT",
        kind: FixtureKind::Submissions,
    },
    Fixture {
        output: "BA_submissions.json",
        ticker: "BA",
        kind: FixtureKind::Submissions,
    },
    // ── companyfacts (full XBRL-tagged financials) ────────────────────────
    Fixture {
        output: "AAPL_companyfacts.json",
        ticker: "AAPL",
        kind: FixtureKind::CompanyFacts,
    },
    Fixture {
        output: "GOOG_companyfacts.json",
        ticker: "GOOG",
        kind: FixtureKind::CompanyFacts,
    },
    Fixture {
        output: "MSFT_companyfacts.json",
        ticker: "MSFT",
        kind: FixtureKind::CompanyFacts,
    },
    Fixture {
        output: "NVDA_companyfacts.json",
        ticker: "NVDA",
        kind: FixtureKind::CompanyFacts,
    },
    // ── rendering HTML (8-K primary doc + first exhibit, stored compressed) ─
    Fixture {
        output: "AAPL_8k_primary.html",
        ticker: "AAPL",
        kind: FixtureKind::EightKPrimary,
    },
    Fixture {
        output: "AAPL_8k_exhibit.html",
        ticker: "AAPL",
        kind: FixtureKind::EightKFirstHtmlExhibit,
    },
];

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_manager = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config_manager)?;

    let fixtures_dir: PathBuf = {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures");
        p
    };
    fs::create_dir_all(&fixtures_dir)?;

    println!(
        "Downloading {} JSON fixtures into {}",
        FIXTURES.len(),
        fixtures_dir.display()
    );
    println!();

    for fixture in FIXTURES {
        let gz_path = fixtures_dir.join(format!("{}.gz", fixture.output));

        print!("  {} ({}) ... ", fixture.output, fixture.ticker);
        std::io::stdout().flush()?;

        let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new(fixture.ticker)).await?;

        let url: String = match fixture.kind {
            FixtureKind::Submissions => Url::CikSubmission(cik).value(),
            FixtureKind::CompanyFacts => Url::CompanyFacts(cik).value(),
            FixtureKind::EightKPrimary => {
                let filings = fetch_8k_filings(&client, cik).await?;
                let latest = filings
                    .first()
                    .ok_or_else(|| format!("No 8-K filings for '{}'", fixture.ticker))?;
                latest.as_primary_document_url()
            }
            FixtureKind::EightKFirstHtmlExhibit => {
                let filings = fetch_8k_filings(&client, cik).await?;
                let latest = filings
                    .first()
                    .ok_or_else(|| format!("No 8-K filings for '{}'", fixture.ticker))?;
                let index = fetch_filing_index(&client, latest).await?;
                let base = latest.as_edgar_archive_url();
                let skip = [".pdf", ".xsd", ".zip", ".xlsx", ".png", ".jpg", ".gif"];
                let exhibit = index
                    .exhibits()
                    .into_iter()
                    .find(|ex| {
                        let n = ex.name.to_ascii_lowercase();
                        !skip.iter().any(|e| n.ends_with(e))
                    })
                    .ok_or_else(|| format!("No renderable exhibit for '{}'", fixture.ticker))?;
                format!("{}/{}", base, exhibit.name)
            }
        };

        let response = client
            .raw_request(reqwest::Method::GET, &url, None, None)
            .await?;

        let status = response.status();
        if !status.is_success() {
            eprintln!("FAILED (HTTP {})", status);
            eprintln!("  URL: {}", url);
            return Err(format!("HTTP {} downloading {}", status, url).into());
        }

        let bytes = response.bytes().await?;
        let uncompressed_kb = bytes.len() / 1024;

        let gz_file = File::create(&gz_path)?;
        let mut encoder = GzEncoder::new(gz_file, Compression::best());
        encoder.write_all(&bytes)?;
        encoder.finish()?;

        let compressed_kb = fs::metadata(&gz_path)?.len() / 1024;

        println!(
            "ok  ({} KB → {} KB gz, {:.0}% reduction)",
            uncompressed_kb,
            compressed_kb,
            100.0 - (compressed_kb as f64 / uncompressed_kb as f64) * 100.0
        );
    }

    println!("Done.  All fixtures are up to date.");
    Ok(())
}
