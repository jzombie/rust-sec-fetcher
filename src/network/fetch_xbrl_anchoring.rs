use std::error::Error;

use polars::prelude::*;

use crate::enums::Url;
use crate::models::{Cik, Ticker, TickerSymbol};
use crate::network::{SecClient, fetch_filing_index};
use crate::parsers::parse_definition_linkbase;

/// Maximum number of recent periodic filings to scan for definition linkbases.
/// Each filing requires 2 HTTP requests (index HTML + EX-101.DEF), so this
/// cap keeps runtime reasonable.
const MAX_FILINGS_TO_SCAN: usize = 10;

/// Periodic form types that carry full financial statements with XBRL tagging.
const PERIODIC_FORM_TYPES: &[&str] = &[
    "10-K", "10-K/A", "10-K405", "10-K405/A",
    "10-Q", "10-Q/A",
    "20-F", "20-F/A",
    "40-F", "40-F/A",
];

/// Fetches custom→standard tag anchoring for a ticker by downloading and
/// parsing the XBRL definition linkbases from its periodic filings.
///
/// # How it works
///
/// 1. Resolves ticker → CIK.
/// 2. Fetches the CIK's filing history to find recent periodic filings
///    (10-K, 10-Q, 20-F, and amendments).
/// 3. For each filing, fetches the EDGAR filing index JSON to discover the
///    EX-101.DEF (definition linkbase) and EX-101.SCH (extension schema)
///    documents.
/// 4. Fetches and parses EX-101.DEF to find `definitionArc` elements that
///    link standard concepts (e.g. `us-gaap:Revenues`) to custom extension
///    concepts (e.g. `aapl:MyCustomRevenue`).
/// 5. Returns a DataFrame with one row per anchoring relationship.
///
/// # Returns
///
/// A [`DataFrame`] with columns:
/// `ticker`, `cik`, `accn`, `filing_date`, `form`,
/// `custom_namespace`, `custom_tag`, `standard_namespace`, `standard_tag`,
/// `arcrole`.
///
/// Returns an empty DataFrame (0 rows) if the ticker has no periodic filings
/// with extension schemas or definition linkbases.
pub async fn fetch_custom_tag_anchoring(
    client: &SecClient,
    company_tickers: &[Ticker],
    ticker: &TickerSymbol,
) -> Result<DataFrame, Box<dyn Error>> {
    let cik = Cik::get_company_cik_by_ticker_symbol(company_tickers, ticker)?;

    // Step 1: Fetch CIK submissions to find periodic filings.
    let submissions = crate::network::fetch_cik_submissions(client, cik.clone()).await?;

    // Filter to periodic forms, newest-first.
    let periodic: Vec<_> = submissions
        .iter()
        .filter(|s| {
            PERIODIC_FORM_TYPES
                .iter()
                .any(|ft| s.form.eq_ignore_ascii_case(ft))
        })
        .take(MAX_FILINGS_TO_SCAN)
        .collect();

    if periodic.is_empty() {
        return Ok(empty_anchoring_df());
    }

    // Step 2: For each periodic filing, try to fetch and parse the definition linkbase.
    let mut anchoring_rows: Vec<AnchoringRow> = Vec::new();

    for filing in &periodic {
        let accn_str = filing.accession_number.to_string();
        let form = filing.form.clone();
        let filing_date = filing.filing_date.map(|d| d.to_string());

        // Fetch filing index HTML to discover EX-101.DEF.
        let index = match fetch_filing_index(client, filing).await {
            Ok(idx) => idx,
            Err(_) => continue,
        };

        let def_doc = match index.definition_linkbase() {
            Some(doc) => doc.clone(),
            None => continue,
        };

        // Fetch the definition linkbase XML.
        let def_url = Url::CikAccessionDocument(
            cik.clone(),
            filing.accession_number.clone(),
            def_doc.name,
        )
        .value();
        let def_xml = match fetch_text(client, &def_url).await {
            Some(x) => x,
            None => continue,
        };

        // Parse the definition linkbase.
        let arcs = match parse_definition_linkbase(&def_xml) {
            Ok(a) => a,
            Err(_) => continue,
        };

        for arc in &arcs {
            if arc.from.is_standard && !arc.to.is_standard {
                anchoring_rows.push(AnchoringRow {
                    ticker: ticker.to_string(),
                    cik: cik.to_string(),
                    accn: accn_str.clone(),
                    filing_date: filing_date.clone(),
                    form: form.clone(),
                    custom_namespace: arc.to.namespace.clone().unwrap_or_default(),
                    custom_tag: arc.to.name.clone(),
                    standard_namespace: arc.from.namespace.clone().unwrap_or_default(),
                    standard_tag: arc.from.name.clone(),
                    arcrole: arc.arcrole.clone(),
                });
            } else if !arc.from.is_standard && arc.to.is_standard {
                anchoring_rows.push(AnchoringRow {
                    ticker: ticker.to_string(),
                    cik: cik.to_string(),
                    accn: accn_str.clone(),
                    filing_date: filing_date.clone(),
                    form: form.clone(),
                    custom_namespace: arc.from.namespace.clone().unwrap_or_default(),
                    custom_tag: arc.from.name.clone(),
                    standard_namespace: arc.to.namespace.clone().unwrap_or_default(),
                    standard_tag: arc.to.name.clone(),
                    arcrole: arc.arcrole.clone(),
                });
            }
        }
    }

    build_anchoring_dataframe(anchoring_rows)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

struct AnchoringRow {
    ticker: String,
    cik: String,
    accn: String,
    filing_date: Option<String>,
    form: String,
    custom_namespace: String,
    custom_tag: String,
    standard_namespace: String,
    standard_tag: String,
    arcrole: String,
}

/// Fetches a URL and returns the response body as text.
async fn fetch_text(client: &SecClient, url: &str) -> Option<String> {
    let response = client.raw_request(reqwest::Method::GET, url, None, None).await.ok()?;
    response.text().await.ok()
}

fn empty_anchoring_df() -> DataFrame {
    let empty_col: polars::prelude::Column =
        Series::new_empty("ticker".into(), &DataType::String).into();
    DataFrame::new(vec![empty_col]).unwrap()
}

fn build_anchoring_dataframe(rows: Vec<AnchoringRow>) -> Result<DataFrame, Box<dyn Error>> {
    let mut tickers = Vec::with_capacity(rows.len());
    let mut ciks = Vec::with_capacity(rows.len());
    let mut accns = Vec::with_capacity(rows.len());
    let mut filing_dates = Vec::with_capacity(rows.len());
    let mut forms = Vec::with_capacity(rows.len());
    let mut custom_nss = Vec::with_capacity(rows.len());
    let mut custom_tags = Vec::with_capacity(rows.len());
    let mut standard_nss = Vec::with_capacity(rows.len());
    let mut standard_tags = Vec::with_capacity(rows.len());
    let mut arcroles = Vec::with_capacity(rows.len());

    for row in rows {
        tickers.push(Some(row.ticker));
        ciks.push(Some(row.cik));
        accns.push(Some(row.accn));
        filing_dates.push(row.filing_date);
        forms.push(Some(row.form));
        custom_nss.push(Some(row.custom_namespace));
        custom_tags.push(Some(row.custom_tag));
        standard_nss.push(Some(row.standard_namespace));
        standard_tags.push(Some(row.standard_tag));
        arcroles.push(Some(row.arcrole));
    }

    let df = df!(
        "ticker" => &tickers,
        "cik" => &ciks,
        "accn" => &accns,
        "filing_date" => &filing_dates,
        "form" => &forms,
        "custom_namespace" => &custom_nss,
        "custom_tag" => &custom_tags,
        "standard_namespace" => &standard_nss,
        "standard_tag" => &standard_tags,
        "arcrole" => &arcroles,
    )?;

    Ok(df)
}
