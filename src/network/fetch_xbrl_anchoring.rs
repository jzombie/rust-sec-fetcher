use std::collections::HashMap;
use std::error::Error;

use polars::prelude::*;

use crate::enums::Url;
use crate::models::{Cik, Ticker, TickerSymbol};
use crate::network::{SecClient, fetch_filing_index};
use crate::parsers::{parse_calculation_linkbase, parse_label_linkbase};

/// Maximum number of recent periodic filings to scan for anchoring.
const MAX_FILINGS_TO_SCAN: usize = 10;

/// Periodic form types that carry full financial statements with XBRL tagging.
const PERIODIC_FORM_TYPES: &[&str] = &[
    "10-K", "10-K/A", "10-K405", "10-K405/A",
    "10-Q", "10-Q/A",
    "20-F", "20-F/A",
    "40-F", "40-F/A",
];

/// Fetches custom→standard polarity anchoring for a ticker by downloading and
/// parsing the XBRL calculation linkbases from its periodic filings.
///
/// # How it works
///
/// 1. Resolves ticker → CIK.
/// 2. Fetches the CIK's filing history to find recent periodic filings.
/// 3. For each filing, fetches the filing index to discover `EX-101.CAL`
///    (calculation linkbase) and `EX-101.LAB` (label linkbase).
/// 4. Parses `EX-101.CAL` to find `summation-item` arcs where a standard
///    GAAP concept (e.g. `us-gaap:NoninterestIncome`) is the parent/total
///    and a custom extension concept (e.g. `bac:FeesAndCommissions1`) is
///    the child/detail — this is the polarity anchoring.
/// 5. Returns a DataFrame with one row per anchoring relationship.
///
/// # Returns
///
/// A [`DataFrame`] with columns:
/// `ticker`, `cik`, `accn`, `filed`, `form`,
/// `ext_concept`, `std_concept`, `arcrole`, `label`.
pub async fn fetch_custom_tag_anchoring(
    client: &SecClient,
    company_tickers: &[Ticker],
    ticker: &TickerSymbol,
) -> Result<DataFrame, Box<dyn Error>> {
    let cik = Cik::get_company_cik_by_ticker_symbol(company_tickers, ticker)?;

    let submissions = crate::network::fetch_cik_submissions(client, cik.clone()).await?;

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

    let mut anchoring_rows: Vec<AnchoringRow> = Vec::new();

    for filing in &periodic {
        let accn_str = filing.accession_number.to_string();
        let form = filing.form.clone();
        let filed = filing.filing_date.map(|d| d.to_string());

        let index = match fetch_filing_index(client, filing).await {
            Ok(idx) => idx,
            Err(_) => continue,
        };

        // Fetch calculation linkbase (EX-101.CAL).
        let cal_doc = match index.calculation_linkbase() {
            Some(doc) => doc.clone(),
            None => continue,
        };
        let cal_url = Url::CikAccessionDocument(
            cik.clone(),
            filing.accession_number.clone(),
            cal_doc.name,
        )
        .value();
        let cal_xml = match fetch_text(client, &cal_url).await {
            Some(x) => x,
            None => continue,
        };

        // Fetch label linkbase (EX-101.LAB) for human-readable labels.
        let label_map = if let Some(lab_doc) = index.label_linkbase() {
            let lab_url = Url::CikAccessionDocument(
                cik.clone(),
                filing.accession_number.clone(),
                lab_doc.name.clone(),
            )
            .value();
            fetch_text(client, &lab_url)
                .await
                .and_then(|xml| parse_label_linkbase(&xml).ok())
                .unwrap_or_default()
        } else {
            HashMap::new()
        };

        // Parse the calculation linkbase.
        let arcs = match parse_calculation_linkbase(&cal_xml) {
            Ok(a) => a,
            Err(_) => continue,
        };

        for arc in &arcs {
            // We want: from=standard(total), to=custom(detail)
            if !arc.from.is_standard || arc.to.is_standard {
                continue;
            }

            let ext_concept = format_concept(&arc.to.namespace, &arc.to.name);
            let std_concept = format_concept(&arc.from.namespace, &arc.from.name);

            // Look up the label for the custom concept.
            let concept_id = match &arc.to.namespace {
                Some(ns) => format!("{}_{}", ns, arc.to.name),
                None => arc.to.name.clone(),
            };
            let label = label_map.get(&concept_id).cloned().unwrap_or_default();

            anchoring_rows.push(AnchoringRow {
                ticker: ticker.to_string(),
                cik: cik.to_string(),
                accn: accn_str.clone(),
                filed: filed.clone(),
                form: form.clone(),
                ext_concept,
                std_concept,
                label,
            });
        }
    }

    build_anchoring_dataframe(anchoring_rows)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

struct AnchoringRow {
    ticker: String,
    cik: String,
    accn: String,
    filed: Option<String>,
    form: String,
    ext_concept: String,
    std_concept: String,
    label: String,
}

fn format_concept(ns: &Option<String>, name: &str) -> String {
    match ns {
        Some(prefix) => format!("{}:{}", prefix, name),
        None => name.to_string(),
    }
}

async fn fetch_text(client: &SecClient, url: &str) -> Option<String> {
    let response = client
        .raw_request(reqwest::Method::GET, url, None, None)
        .await
        .ok()?;
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
    let mut fileds = Vec::with_capacity(rows.len());
    let mut forms = Vec::with_capacity(rows.len());
    let mut ext_concepts = Vec::with_capacity(rows.len());
    let mut std_concepts = Vec::with_capacity(rows.len());
    let mut labels = Vec::with_capacity(rows.len());

    for row in rows {
        tickers.push(Some(row.ticker));
        ciks.push(Some(row.cik));
        accns.push(Some(row.accn));
        fileds.push(row.filed);
        forms.push(Some(row.form));
        ext_concepts.push(Some(row.ext_concept));
        std_concepts.push(Some(row.std_concept));
        labels.push(Some(row.label));
    }

    let df = df!(
        "ticker" => &tickers,
        "cik" => &ciks,
        "accn" => &accns,
        "filed" => &fileds,
        "form" => &forms,
        "ext_concept" => &ext_concepts,
        "std_concept" => &std_concepts,
        "label" => &labels,
    )?;

    Ok(df)
}
