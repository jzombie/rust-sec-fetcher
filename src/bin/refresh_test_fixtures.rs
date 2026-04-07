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

use chrono::Datelike;
use flate2::Compression;
use flate2::write::GzEncoder;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::enums::Url;
use sec_fetcher::models::{AccessionNumber, CikSubmission, TickerSymbol};
use sec_fetcher::network::{
    SecClient, fetch_8k_filings, fetch_10k_filings, fetch_cik_by_ticker_symbol, fetch_filing_index,
    fetch_related_ciks,
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
    /// The `informationTable.xml` from a specific 13F-HR filing.
    ///
    /// `accession` must be a full formatted accession number string
    /// (e.g. `"0000950123-22-012275"`).  The filing's CIK is resolved from
    /// the `ticker` field of the enclosing [`Fixture`] at runtime.
    ///
    /// ## Era fixtures
    ///
    /// | Label | Accession | Filed | Era |
    /// |---|---|---|---|
    /// | ancient | `0000950123-22-012275` | 2022-11-14 | `<value>` in **thousands** |
    /// | transition | `0000950123-23-002585` | 2023-02-14 | `<value>` in **actual USD** (first modern) |
    /// | modern | `0001193125-26-054580` | 2026-02-17 | `<value>` in **actual USD** |
    ThirteenF { accession: &'static str },
    /// Raw Form 4 ownership-document XML (not the XSL-rendered version under `xslF345X05/`).
    ///
    /// `accession` must be a full formatted accession number string
    /// (e.g. `"0001214128-26-000004"`).  The issuer's CIK is resolved from
    /// the `ticker` field; the filing index is then fetched to discover
    /// the raw XML filename.
    Form4Xml { accession: &'static str },
    /// Raw text of the EDGAR full-index `master.idx` for the given quarter.
    /// The `ticker` field is not used for URL construction; pass any valid ticker.
    MasterIdx { year: u16, quarter: u8 },
    /// `https://www.sec.gov/files/company_tickers.json`.
    /// The `ticker` field is not used for URL construction; pass any valid ticker.
    CompanyTickersJson,
    /// The raw primary document (HTML, SGML `.txt`, or inline iXBRL) for a
    /// specific year's 10-K filing.
    ///
    /// When `primary_document` is empty (pre-2000 EDGAR SGML bundles) the
    /// submission's root SGML `.txt` file is downloaded instead.
    ///
    /// Used by `tests/tenk_sections_tests.rs` to validate that
    /// `extract_sections_from_document` successfully locates Item 1 and Item 7
    /// across every major EDGAR filing era without live network access.
    TenKRaw { year: u16 },
    /// JSON array of zero-padded 10-digit CIK strings returned by
    /// [`fetch_related_ciks`] for this ticker's primary CIK.
    ///
    /// Written as `[]` for companies with no holding-company reorganisation and
    /// as a non-empty array for those that have one.  For Alphabet Inc. (GOOG)
    /// the expected value is `["0001288776"]` (Google Inc., predecessor CIK).
    ///
    /// Tests in `tests/cik_lineage_tests.rs` assert these expected contents,
    /// verifying that [`fetch_related_ciks`] is both correct (finds the real
    /// predecessor) and conservative (returns empty when no reorganisation
    /// occurred).
    RelatedCiks,
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
    // ── 13F-HR information tables (era cross-over fixtures) ────────────────
    //
    // Two BRK-B filings bracket the 2023-01-01 crossover date where the
    // <value> field changed from thousands-of-USD to actual-USD.  A third
    // recent filing confirms the modern schema.  Integration tests load these
    // fixtures and call parse_13f_xml with the known filing_date to verify
    // that normalize_13f_value_usd produces plausible per-share prices in
    // both eras.
    Fixture {
        output: "BRK_B_13f_ancient.xml",
        ticker: "BRK-B",
        kind: FixtureKind::ThirteenF {
            // Q3-2022: filed 2022-11-14.  AAPL raw=95634 → thousands era
            // → value_usd = $95,634,000 → ~$138/share for 692,000 shares.
            accession: "0000950123-22-012275",
        },
    },
    Fixture {
        output: "BRK_B_13f_transition.xml",
        ticker: "BRK-B",
        kind: FixtureKind::ThirteenF {
            // Q4-2022: filed 2023-02-14.  First filing in the actual-USD era.
            // AAPL raw=133289470 → value_usd = $133,289,470 → ~$130/share for 1,025,856 shares.
            accession: "0000950123-23-002585",
        },
    },
    Fixture {
        output: "BRK_B_13f_modern.xml",
        ticker: "BRK-B",
        kind: FixtureKind::ThirteenF {
            // Q4-2025: filed 2026-02-17.  Recent filing confirming modern schema.
            accession: "0001193125-26-054580",
        },
    },
    // ── Form 4 XML (ownership document) ──────────────────────────────────────
    //
    // AAPL director Arthur D. Levinson (CIK 0001214128), filed 2026-02-26.
    // Contains:
    //   • nonDerivativeTransaction: Common Stock, gift code G, 1113 sh disposed, 4 069 576 after
    //   • derivativeTransaction:    RSU grant code A, 1011 sh acquired, is_derivative = true
    Fixture {
        output: "AAPL_form4_levinson.xml",
        ticker: "AAPL",
        kind: FixtureKind::Form4Xml {
            accession: "0001214128-26-000004",
        },
    },
    // ── company_tickers.json ──────────────────────────────────────────────────
    //
    // Full operating-company ticker listing (~10 000 entries).  Anchored against
    // AAPL (CIK 320193), MSFT (CIK 789019), and NVDA (CIK 1045810).
    Fixture {
        output: "company_tickers.json",
        ticker: "AAPL", // ticker not used for URL construction
        kind: FixtureKind::CompanyTickersJson,
    },
    // ── master.idx quarterly snapshot ────────────────────────────────────────
    //
    // Q4 2025 (Oct–Dec 2025).  Known anchors:
    //   • AAPL 10-K   accn=0000320193-25-000079  filed 2025-10-31
    //   • AAPL  8-K   accn=0000320193-25-000077  filed 2025-10-30
    //   • MSFT 10-Q   accn=0001193125-25-256321  filed 2025-10-29
    Fixture {
        output: "master_idx_2025_Q4.idx",
        ticker: "AAPL", // ticker not used for URL construction
        kind: FixtureKind::MasterIdx {
            year: 2025,
            quarter: 4,
        },
    },
    // ── 10-K raw document fixtures ────────────────────────────────────────────
    //
    // These fixtures exercise `extract_sections_from_document` across every
    // major EDGAR filing era:
    //
    //   SGML era (pre-2000):     AAPL 1994, MSFT 1995, INTC 1996
    //   Early HTML (2002–2006):  AAPL 2003, MSFT 2005, XOM 2006
    //   Mid HTML (2008–2013):    ORCL 2009, JNJ 2010, GOOG 2012
    //   iXBRL era (2014–2018):   AAPL 2016, NVDA 2017
    //   Modern inline iXBRL:     AMZN 2019, JPM 2022, BRK-B 2023, COST 2024
    //
    // Companies that were replaced due to incorporation-by-reference (their
    // EDGAR 10-K shell pointed to a physical annual report, so no extractable
    // content existed in the filing):
    //   GE 1994, KO 2002, WMT 2005, JPM 2009 — all used inc-by-ref to annual
    //   reports; replaced with tech companies that file inline.
    //
    // Tests in `tests/tenk_sections_tests.rs` load each fixture, call
    // `extract_sections_from_document`, and assert both Item 1 and Item 7 are
    // present with at least the minimum expected character counts.
    //
    // ── SGML era ─────────────────────────────────────────────────────────────
    Fixture {
        output: "AAPL_10k_1994.raw",
        ticker: "AAPL",
        kind: FixtureKind::TenKRaw { year: 1994 },
    },
    Fixture {
        output: "MSFT_10k_1995.raw",
        ticker: "MSFT",
        kind: FixtureKind::TenKRaw { year: 1995 },
    },
    Fixture {
        output: "INTC_10k_1996.raw",
        ticker: "INTC",
        kind: FixtureKind::TenKRaw { year: 1996 },
    },
    // ── Early HTML era ────────────────────────────────────────────────────────
    Fixture {
        output: "AAPL_10k_2003.raw",
        ticker: "AAPL",
        kind: FixtureKind::TenKRaw { year: 2003 },
    },
    Fixture {
        output: "MSFT_10k_2005.raw",
        ticker: "MSFT",
        kind: FixtureKind::TenKRaw { year: 2005 },
    },
    // ── Mid HTML era ──────────────────────────────────────────────────────────
    Fixture {
        output: "ORCL_10k_2009.raw",
        ticker: "ORCL",
        kind: FixtureKind::TenKRaw { year: 2009 },
    },
    Fixture {
        output: "GOOG_10k_2012.raw",
        ticker: "GOOG",
        kind: FixtureKind::TenKRaw { year: 2012 },
    },
    // ── iXBRL transition era ──────────────────────────────────────────────────
    Fixture {
        output: "AAPL_10k_2016.raw",
        ticker: "AAPL",
        kind: FixtureKind::TenKRaw { year: 2016 },
    },
    Fixture {
        output: "NVDA_10k_2017.raw",
        ticker: "NVDA",
        kind: FixtureKind::TenKRaw { year: 2017 },
    },
    // ── Early HTML additions ─────────────────────────────────────────────────
    //
    // ExxonMobil 2006: large energy company in the early-HTML era.  XOM has a
    // single stable CIK from the 1999 Exxon/Mobil merger; a stable entity that
    // exercises the non-restructured code path.
    Fixture {
        output: "XOM_10k_2006.raw",
        ticker: "XOM",
        kind: FixtureKind::TenKRaw { year: 2006 },
    },
    // ── Mid HTML additions ────────────────────────────────────────────────────
    //
    // Johnson & Johnson 2010: large diversified healthcare company.  Fills the
    // healthcare sector gap in the era matrix.  JNJ files fully inline — no
    // incorporation-by-reference to a physical annual report.
    Fixture {
        output: "JNJ_10k_2010.raw",
        ticker: "JNJ",
        kind: FixtureKind::TenKRaw { year: 2010 },
    },
    // ── Modern inline iXBRL ───────────────────────────────────────────────────
    //
    // Amazon 2019: first year with modern inline iXBRL for Amazon; large
    // e-commerce / cloud company that files a detailed Item 1 and Item 7.
    Fixture {
        output: "AMZN_10k_2019.raw",
        ticker: "AMZN",
        kind: FixtureKind::TenKRaw { year: 2019 },
    },
    // JPMorgan Chase 2022: large US bank, modern inline iXBRL.  JPM 2009 was
    // previously excluded because its MD&A was incorporated by reference.
    // Modern (2022) JPM filings are fully inline — confirms the financial-
    // sector modern path works end to end.
    Fixture {
        output: "JPM_10k_2022.raw",
        ticker: "JPM",
        kind: FixtureKind::TenKRaw { year: 2022 },
    },
    Fixture {
        output: "BRK_B_10k_2023.raw",
        ticker: "BRK-B",
        kind: FixtureKind::TenKRaw { year: 2023 },
    },
    Fixture {
        output: "COST_10k_2024.raw",
        ticker: "COST",
        kind: FixtureKind::TenKRaw { year: 2024 },
    },
    // ── CIK lineage (RelatedCiks) ─────────────────────────────────────────────
    //
    // These fixtures capture the output of `fetch_related_ciks` for a range of
    // companies.  Tests in `tests/cik_lineage_tests.rs` check expected contents:
    //
    //   GOOG   → ["0001288776"]   (Google Inc., predecessor before Alphabet)
    //   GOOGL  → ["0001288776"]   (same predecessor — different ticker, same CIK)
    //   Others → []              (no holding-company reorganisation on record)
    //
    // Having BOTH a positive case (GOOG) and multiple negative cases (AAPL,
    // AMZN, META, MSFT, NVDA) proves the function is selective, not permissive.
    Fixture {
        output: "GOOG_related_ciks.json",
        ticker: "GOOG",
        kind: FixtureKind::RelatedCiks,
    },
    Fixture {
        output: "GOOGL_related_ciks.json",
        ticker: "GOOGL",
        kind: FixtureKind::RelatedCiks,
    },
    Fixture {
        output: "AAPL_related_ciks.json",
        ticker: "AAPL",
        kind: FixtureKind::RelatedCiks,
    },
    Fixture {
        output: "META_related_ciks.json",
        ticker: "META",
        kind: FixtureKind::RelatedCiks,
    },
    Fixture {
        output: "MSFT_related_ciks.json",
        ticker: "MSFT",
        kind: FixtureKind::RelatedCiks,
    },
    Fixture {
        output: "NVDA_related_ciks.json",
        ticker: "NVDA",
        kind: FixtureKind::RelatedCiks,
    },
    Fixture {
        output: "AMZN_related_ciks.json",
        ticker: "AMZN",
        kind: FixtureKind::RelatedCiks,
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

        // ── RelatedCiks: no URL — call fetch_related_ciks and serialise result ──
        if matches!(fixture.kind, FixtureKind::RelatedCiks) {
            let related = fetch_related_ciks(&client, &cik).await?;
            let entries: Vec<String> = related.iter().map(|c| format!("{:010}", c.value)).collect();
            let json_bytes = serde_json::to_vec(&entries)?;
            let gz_file = File::create(&gz_path)?;
            let mut encoder = GzEncoder::new(gz_file, Compression::best());
            encoder.write_all(&json_bytes)?;
            encoder.finish()?;
            if entries.is_empty() {
                println!("ok  (no predecessor CIKs)");
            } else {
                println!("ok  (predecessor CIKs: {})", entries.join(", "));
            }
            continue;
        }

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
            FixtureKind::ThirteenF { accession } => {
                // Build a minimal CikSubmission with the known accession number
                // so we can reuse fetch_filing_index to discover the filename.
                let acc_num = AccessionNumber::from_str(accession)
                    .map_err(|e| format!("Invalid accession '{}': {}", accession, e))?;
                let sub = CikSubmission {
                    cik: cik.clone(),
                    entity_type: None,
                    accession_number: acc_num,
                    form: "13F-HR".to_string(),
                    primary_document: "primary_doc.xml".to_string(),
                    filing_date: None,
                    items: vec![],
                };
                let index = fetch_filing_index(&client, &sub).await?;
                let info_doc = index
                    .documents
                    .iter()
                    .find(|d| d.document_type.to_uppercase().contains("INFORMATION TABLE"))
                    .ok_or_else(|| {
                        format!(
                            "No INFORMATION TABLE document in 13F index for accession {}",
                            accession
                        )
                    })?;
                Url::CikAccessionDocument(sub.cik, sub.accession_number, info_doc.name.clone())
                    .value()
            }
            FixtureKind::Form4Xml { accession } => {
                let acc_num = AccessionNumber::from_str(accession)
                    .map_err(|e| format!("Invalid accession '{}': {}", accession, e))?;
                let sub = CikSubmission {
                    cik: cik.clone(),
                    entity_type: None,
                    accession_number: acc_num,
                    form: "4".to_string(),
                    primary_document: "primary_doc.xml".to_string(),
                    filing_date: None,
                    items: vec![],
                };
                let index = fetch_filing_index(&client, &sub).await?;
                // The raw ownership XML sits in the filing root; the XSL-rendered
                // copy is under an xslF345X05/ subdirectory and is not listed in
                // the filing index documents array.
                let xml_doc = index
                    .documents
                    .iter()
                    .find(|d| d.name.ends_with(".xml"))
                    .ok_or_else(|| {
                        format!(
                            "No XML document in Form 4 filing index for accession {}",
                            accession
                        )
                    })?;
                Url::CikAccessionDocument(sub.cik, sub.accession_number, xml_doc.name.clone())
                    .value()
            }
            FixtureKind::MasterIdx { year, quarter } => {
                Url::EdgarFullIndex { year, quarter }.value()
            }
            FixtureKind::CompanyTickersJson => Url::CompanyTickersJson.value(),
            FixtureKind::TenKRaw { year } => {
                // Locate the filing for the requested year, then return its
                // best document URL using the same tiered strategy as
                // `fetch_10k_sections_for_filing`.
                let filings = fetch_10k_filings(&client, cik.clone()).await?;
                let filing = filings
                    .iter()
                    .find(|f| {
                        f.filing_date
                            .map(|d| d.year() == year as i32)
                            .unwrap_or(false)
                    })
                    .ok_or_else(|| {
                        format!(
                            "No 10-K filing found for '{}' in year {}",
                            fixture.ticker, year
                        )
                    })?;

                // For pre-2000 SGML bundles, primary_document is empty; fall
                // back to the root SGML .txt file at the CIK level.
                if filing.primary_document.is_empty() {
                    Url::EdgarArchive(format!(
                        "edgar/data/{}/{}.txt",
                        filing.cik, filing.accession_number
                    ))
                    .value()
                } else {
                    Url::CikAccessionDocument(
                        filing.cik.clone(),
                        filing.accession_number.clone(),
                        filing.primary_document.clone(),
                    )
                    .value()
                }
            }
            // Handled above before this match — unreachable at runtime.
            FixtureKind::RelatedCiks => unreachable!("RelatedCiks handled above"),
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
