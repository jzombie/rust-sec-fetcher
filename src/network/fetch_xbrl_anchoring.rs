use std::collections::{HashMap, HashSet};
use std::error::Error;

use polars::prelude::*;
use quick_xml::Reader;

use crate::enums::Url;
use crate::models::{Cik, Ticker, TickerSymbol};
use crate::network::{SecClient, fetch_filing_index};
use crate::parsers::{parse_calculation_linkbase, parse_extension_schema, parse_label_linkbase};

/// Periodic form types that carry full financial statements with XBRL tagging.
const PERIODIC_FORM_TYPES: &[&str] = &[
    "10-K",
    "10-K/A",
    "10-K405",
    "10-K405/A",
    "10-Q",
    "10-Q/A",
    "20-F",
    "20-F/A",
    "40-F",
    "40-F/A",
];

/// Stable per-tag polarity contract for downstream consumers (e.g. trend-describer).
///
/// Scope: custom extension tags ONLY (e.g. `bac:FeesAndCommissions1`).
/// Standard taxonomy tags (e.g. `us-gaap:Revenues`) have fixed FASB-defined
/// debit/credit balances and never need anchoring. This function exists
/// solely because filer-defined extension tags have no fixed polarity — it
/// must be inferred per filing from `EX-101.CAL` summation arcs + `EX-101.SCH`
/// `xbrli:balance` declarations. Companies with no extension tags (e.g. AAPL:
/// 0 custom elements, dimensions only) correctly return an empty frame.
///
/// One row per `ext_concept` (deduplicated across filings). The representative
/// `std_concept`/`weight`/`label` comes from the most-recent filing containing
/// the tag. `conflict=true` means the same custom tag anchored to different
/// parents or weights (or redeclared with a different `xbrli:balance`) across
/// the scanned filings — downstream should distrust/pin it, not flip signs.
///
/// Columns:
/// `ticker`, `cik`, `ext_concept`, `std_concept`, `weight`, `label`,
/// `ext_balance`, `std_balance`, `effective_balance`, `consistent`,
/// `n_filings`, `conflict`, `latest_accn`, `latest_filed`, `latest_form`.
///
/// `max_filings`: `None` = scan every periodic filing for the ticker (full
/// history, required to guarantee `conflict` detection back to inception).
/// `Some(n)` = only the `n` most-recent periodic filings (faster updates).
pub async fn fetch_custom_tag_anchoring(
    client: &SecClient,
    company_tickers: &[Ticker],
    ticker: &TickerSymbol,
    max_filings: Option<usize>,
) -> Result<DataFrame, Box<dyn Error>> {
    let cik = Cik::get_company_cik_by_ticker_symbol(company_tickers, ticker)?;

    let submissions = crate::network::fetch_cik_submissions(client, cik.clone()).await?;

    let mut periodic: Vec<_> = submissions
        .iter()
        .filter(|s| {
            PERIODIC_FORM_TYPES
                .iter()
                .any(|ft| s.form.eq_ignore_ascii_case(ft))
        })
        .collect();
    // Newest first so `Some(n)` means most-recent n. Submissions are already
    // newest-first from EDGAR; sort by filing date desc to be explicit.
    periodic.sort_by_key(|a| std::cmp::Reverse(a.filing_date));
    let periodic: Vec<_> = match max_filings {
        Some(n) => periodic.into_iter().take(n).collect(),
        None => periodic,
    };

    if periodic.is_empty() {
        return Ok(empty_anchoring_df());
    }

    // (ext_concept -> occurrences across filings)
    let mut by_ext: HashMap<String, Vec<Occurrence>> = HashMap::new();
    // us-gaap taxonomy years actually referenced by the CALs we parsed
    // (e.g. "2025" from .../us-gaap/2025/...). No guessing: if a CAL pins a
    // version we use it, otherwise that parent stays "unknown".
    let mut taxonomy_years: HashSet<String> = HashSet::new();

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
        let cal_url =
            Url::CikAccessionDocument(cik.clone(), filing.accession_number.clone(), cal_doc.name)
                .value();
        let cal_xml = match fetch_text(client, &cal_url).await {
            Some(x) => x,
            None => continue,
        };

        if let Some(y) = extract_us_gaap_year(&cal_xml) {
            taxonomy_years.insert(y);
        }

        // Fetch extension schema (EX-101.SCH) for declared custom balances.
        // Missing SCH is non-fatal: ext_balance becomes "unknown".
        let ext_balances: HashMap<String, String> = if let Some(sch_doc) = index.extension_schema()
        {
            let sch_url = Url::CikAccessionDocument(
                cik.clone(),
                filing.accession_number.clone(),
                sch_doc.name.clone(),
            )
            .value();
            match fetch_text(client, &sch_url).await {
                Some(xml) => parse_extension_schema(&xml)
                    .map(|s| schema_balance_map(&s))
                    .unwrap_or_default(),
                None => HashMap::new(),
            }
        } else {
            HashMap::new()
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

            let concept_id = match &arc.to.namespace {
                Some(ns) => format!("{}_{}", ns, arc.to.name),
                None => arc.to.name.clone(),
            };
            let label = label_map.get(&concept_id).cloned().unwrap_or_default();
            let ext_balance = ext_balances
                .get(&arc.to.name)
                .or_else(|| ext_balances.get(&concept_id))
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());

            by_ext.entry(ext_concept).or_default().push(Occurrence {
                std_concept,
                weight: arc.weight,
                label,
                ext_balance,
                accn: accn_str.clone(),
                filed: filed.clone(),
                form: form.clone(),
            });
        }
    }

    if by_ext.is_empty() {
        return Ok(empty_anchoring_df());
    }

    // Resolve standard-parent balances (one taxonomy fetch per year seen).
    let std_parents: HashSet<String> = by_ext
        .values()
        .flat_map(|v| v.iter().map(|o| o.std_concept.clone()))
        .collect();
    let std_balances = resolve_std_balances(client, &std_parents, &taxonomy_years).await;

    build_stable_dataframe(ticker, &cik, by_ext, &std_balances)
}

// ── Internal types ────────────────────────────────────────────────────────────

struct Occurrence {
    std_concept: String,
    weight: f64,
    label: String,
    ext_balance: String,
    accn: String,
    filed: Option<String>,
    form: String,
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

/// Map SCH custom-element name (and id) -> declared balance (`credit`/`debit`).
fn schema_balance_map(schema: &crate::parsers::ExtensionSchema) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for el in &schema.elements {
        if let Some(b) = &el.balance {
            let b = b.to_lowercase();
            map.insert(el.name.clone(), b.clone());
            if let Some(id) = &el.id {
                map.insert(id.clone(), b);
            }
        }
    }
    map
}

/// Extract `us-gaap/{YYYY}` from CAL `xlink:href`s for taxonomy versioning.
fn extract_us_gaap_year(cal_xml: &str) -> Option<String> {
    // e.g. https://xbrl.fasb.org/us-gaap/2025/elts/us-gaap-2025.xsd
    let marker = "/us-gaap/";
    let pos = cal_xml.find(marker)?;
    let rest = &cal_xml[pos + marker.len()..];
    if rest.len() >= 4 && rest[..4].chars().all(|c| c.is_ascii_digit()) {
        Some(rest[..4].to_string())
    } else {
        None
    }
}

fn flip_balance(b: &str) -> String {
    match b {
        "credit" => "debit".to_string(),
        "debit" => "credit".to_string(),
        _ => "unknown".to_string(),
    }
}

/// Resolve `us-gaap:Concept` -> `credit`/`debit`/`unknown`.
///
/// Fetches only the FASB taxonomy versions actually referenced by the parsed
/// CALs (e.g. `{"2025", "2024"}` from `.../us-gaap/2025/...` hrefs) and parses
/// `<xsd:element name=... xbrli:balance=...>` streaming. Newest year wins when
/// a concept exists in several versions. If `years` is empty (no version found
/// in any CAL) nothing is fetched and every parent stays `"unknown"`.
/// Never fails the caller: on any fetch/parse error returns `unknown`.
async fn resolve_std_balances(
    client: &SecClient,
    parents: &HashSet<String>,
    years: &HashSet<String>,
) -> HashMap<String, String> {
    // Only resolve us-gaap parents; others stay unknown.
    let wanted: HashSet<String> = parents
        .iter()
        .filter_map(|p| p.strip_prefix("us-gaap:").map(|s| s.to_string()))
        .collect();
    let mut out: HashMap<String, String> = HashMap::new();
    for p in parents {
        out.insert(p.clone(), "unknown".to_string());
    }
    if wanted.is_empty() || years.is_empty() {
        return out;
    }

    // Newest taxonomy first so it wins on overlap.
    let mut sorted: Vec<&String> = years.iter().collect();
    sorted.sort();
    sorted.reverse();
    let mut remaining = wanted;
    for year in sorted {
        if remaining.is_empty() {
            break;
        }
        let url = format!(
            "https://xbrl.fasb.org/us-gaap/{}/elts/us-gaap-{}.xsd",
            year, year
        );
        let xml = match fetch_text(client, &url).await {
            Some(x) => x,
            None => continue,
        };
        for (local, bal) in parse_taxonomy_balances(&xml, &remaining) {
            out.insert(format!("us-gaap:{}", local), bal.clone());
            remaining.remove(&local);
        }
    }
    out
}

/// Full standard-tag polarity table for dumb downstream consumers.
///
/// `trend-describer` turns time series into English and must not infer FASB
/// semantics itself. Call this once per taxonomy year (e.g. `"2025"`) and join
/// on concept name to get `credit`/`debit` for every `us-gaap:*` column in
/// `us-gaap-fundamentals/*.csv`. Custom tags are NOT in here — see
/// `fetch_custom_tag_anchoring` for those.
///
/// Columns: `concept` (`us-gaap:Revenues`), `balance` (`credit`/`debit`).
pub async fn fetch_us_gaap_polarity_table(
    client: &SecClient,
    year: &str,
) -> Result<DataFrame, Box<dyn Error>> {
    let url = format!(
        "https://xbrl.fasb.org/us-gaap/{}/elts/us-gaap-{}.xsd",
        year, year
    );
    let xml = fetch_text(client, &url)
        .await
        .ok_or_else(|| format!("failed to fetch taxonomy {}", url))?;
    let map = parse_all_taxonomy_balances(&xml);
    let mut concepts: Vec<String> = map.keys().map(|k| format!("us-gaap:{}", k)).collect();
    concepts.sort();
    let balances: Vec<String> = concepts
        .iter()
        .map(|c| {
            map.get(c.strip_prefix("us-gaap:").unwrap_or(c))
                .cloned()
                .unwrap_or_default()
        })
        .collect();
    let concepts_opt: Vec<Option<String>> = concepts.into_iter().map(Some).collect();
    let balances_opt: Vec<Option<String>> = balances.into_iter().map(Some).collect();
    let df = df!(
        "concept" => &concepts_opt,
        "balance" => &balances_opt,
    )?;
    Ok(df)
}

/// Streaming parse of taxonomy XSD: ALL element local-name -> balance.
fn parse_all_taxonomy_balances(xml: &str) -> HashMap<String, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut map = HashMap::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Start(ref e))
            | Ok(quick_xml::events::Event::Empty(ref e)) => {
                let name = local_name(e.name().as_ref());
                if name == "element" {
                    let mut el_name: Option<String> = None;
                    let mut balance: Option<String> = None;
                    for attr in e.attributes().flatten() {
                        let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
                        let lkey = key.rsplit(':').next().unwrap_or(&key).to_string();
                        let val = String::from_utf8_lossy(attr.value.as_ref()).into_owned();
                        if lkey == "name" {
                            el_name = Some(val);
                        } else if lkey == "balance" {
                            balance = Some(val.to_lowercase());
                        }
                    }
                    if let (Some(n), Some(b)) = (el_name, balance)
                        && (b == "credit" || b == "debit")
                    {
                        map.insert(n, b);
                    }
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    map
}

/// Streaming parse of taxonomy XSD: element local-name -> balance.
fn parse_taxonomy_balances(xml: &str, wanted: &HashSet<String>) -> HashMap<String, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut map = HashMap::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Start(ref e))
            | Ok(quick_xml::events::Event::Empty(ref e)) => {
                let name = local_name(e.name().as_ref());
                if name == "element" {
                    let mut el_name: Option<String> = None;
                    let mut balance: Option<String> = None;
                    for attr in e.attributes().flatten() {
                        let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
                        let lkey = key.rsplit(':').next().unwrap_or(&key).to_string();
                        let val = String::from_utf8_lossy(attr.value.as_ref()).into_owned();
                        if lkey == "name" {
                            el_name = Some(val);
                        } else if lkey == "balance" {
                            balance = Some(val.to_lowercase());
                        }
                    }
                    if let (Some(n), Some(b)) = (el_name, balance)
                        && wanted.contains(&n)
                        && (b == "credit" || b == "debit")
                    {
                        map.insert(n, b);
                    }
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    map
}

fn local_name(name: &[u8]) -> String {
    let s = String::from_utf8_lossy(name).into_owned();
    s.rsplit(':').next().unwrap_or(&s).to_string()
}

fn empty_anchoring_df() -> DataFrame {
    let empty_col: polars::prelude::Column =
        Series::new_empty("ticker".into(), &DataType::String).into();
    DataFrame::new(vec![empty_col]).unwrap()
}

/// Aggregate per-occurrence data into one stable row per ext_concept.
///
/// Representative = occurrence from latest filing (lexicographic `filed`
/// fallback to insertion order). `conflict` = >1 distinct `(std, weight-sign)`
/// or >1 distinct `ext_balance` seen across filings.
fn build_stable_dataframe(
    ticker: &TickerSymbol,
    cik: &Cik,
    by_ext: HashMap<String, Vec<Occurrence>>,
    std_balances: &HashMap<String, String>,
) -> Result<DataFrame, Box<dyn Error>> {
    let mut exts: Vec<String> = by_ext.keys().cloned().collect();
    exts.sort();

    let mut tickers = Vec::with_capacity(exts.len());
    let mut ciks = Vec::with_capacity(exts.len());
    let mut ext_concepts = Vec::with_capacity(exts.len());
    let mut std_concepts = Vec::with_capacity(exts.len());
    let mut weights: Vec<f64> = Vec::with_capacity(exts.len());
    let mut labels = Vec::with_capacity(exts.len());
    let mut ext_balances = Vec::with_capacity(exts.len());
    let mut std_bal_col = Vec::with_capacity(exts.len());
    let mut effective_bal = Vec::with_capacity(exts.len());
    let mut consistents: Vec<bool> = Vec::with_capacity(exts.len());
    let mut n_filings: Vec<i64> = Vec::with_capacity(exts.len());
    let mut conflicts: Vec<bool> = Vec::with_capacity(exts.len());
    let mut latest_accns = Vec::with_capacity(exts.len());
    let mut latest_fileds: Vec<Option<String>> = Vec::with_capacity(exts.len());
    let mut latest_forms = Vec::with_capacity(exts.len());

    for ext in &exts {
        let occs = &by_ext[ext];
        // Distinct filings.
        let distinct_accn: HashSet<&str> = occs.iter().map(|o| o.accn.as_str()).collect();
        // Distinct (std, weight-sign) and ext balances for conflict detection.
        let distinct_key: HashSet<(String, i32)> = occs
            .iter()
            .map(|o| (o.std_concept.clone(), if o.weight >= 0.0 { 1 } else { -1 }))
            .collect();
        let distinct_ext_bal: HashSet<&str> = occs.iter().map(|o| o.ext_balance.as_str()).collect();
        let conflict = distinct_key.len() > 1 || distinct_ext_bal.len() > 1;

        // Latest occurrence: max filed (lexicographic YYYY-MM-DD works), tie -> last.
        let mut best_idx = 0usize;
        let mut best_filed: Option<&str> = None;
        for (i, o) in occs.iter().enumerate() {
            match (&o.filed, best_filed) {
                (Some(f), Some(b)) if f.as_str() <= b => {}
                (Some(f), _) => {
                    best_filed = Some(f.as_str());
                    best_idx = i;
                }
                (None, None) => {
                    best_idx = i;
                }
                _ => {}
            }
        }
        let rep = &occs[best_idx];

        let std_bal = std_balances
            .get(&rep.std_concept)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let effective = if std_bal == "unknown" {
            // Fall back to filer-declared balance when taxonomy is unavailable.
            rep.ext_balance.clone()
        } else if rep.weight >= 0.0 {
            std_bal.clone()
        } else {
            flip_balance(&std_bal)
        };
        let consistent =
            rep.ext_balance != "unknown" && effective != "unknown" && rep.ext_balance == effective;

        tickers.push(Some(ticker.to_string()));
        ciks.push(Some(cik.to_string()));
        ext_concepts.push(Some(ext.clone()));
        std_concepts.push(Some(rep.std_concept.clone()));
        weights.push(rep.weight);
        labels.push(Some(rep.label.clone()));
        ext_balances.push(Some(rep.ext_balance.clone()));
        std_bal_col.push(Some(std_bal));
        effective_bal.push(Some(effective));
        consistents.push(consistent);
        n_filings.push(distinct_accn.len() as i64);
        conflicts.push(conflict);
        latest_accns.push(Some(rep.accn.clone()));
        latest_fileds.push(rep.filed.clone());
        latest_forms.push(Some(rep.form.clone()));
    }

    let df = df!(
        "ticker" => &tickers,
        "cik" => &ciks,
        "ext_concept" => &ext_concepts,
        "std_concept" => &std_concepts,
        "weight" => &weights,
        "label" => &labels,
        "ext_balance" => &ext_balances,
        "std_balance" => &std_bal_col,
        "effective_balance" => &effective_bal,
        "consistent" => &consistents,
        "n_filings" => &n_filings,
        "conflict" => &conflicts,
        "latest_accn" => &latest_accns,
        "latest_filed" => &latest_fileds,
        "latest_form" => &latest_forms,
    )?;

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_balance_sign() {
        assert_eq!(flip_balance("credit"), "debit");
        assert_eq!(flip_balance("debit"), "credit");
        assert_eq!(flip_balance("unknown"), "unknown");
    }

    #[test]
    fn extract_year_from_cal_href() {
        let xml =
            r#"href="https://xbrl.fasb.org/us-gaap/2025/elts/us-gaap-2025.xsd#us-gaap_Revenues""#;
        assert_eq!(extract_us_gaap_year(xml).as_deref(), Some("2025"));
        assert_eq!(extract_us_gaap_year("no href here"), None);
    }

    #[test]
    fn aggregation_latest_wins_and_conflict() {
        let mut by_ext: HashMap<String, Vec<Occurrence>> = HashMap::new();
        by_ext.insert(
            "bac:X".to_string(),
            vec![
                Occurrence {
                    std_concept: "us-gaap:A".to_string(),
                    weight: 1.0,
                    label: "X old".to_string(),
                    ext_balance: "credit".to_string(),
                    accn: "accn-old".to_string(),
                    filed: Some("2025-01-01".to_string()),
                    form: "10-Q".to_string(),
                },
                Occurrence {
                    std_concept: "us-gaap:B".to_string(),
                    weight: 1.0,
                    label: "X new".to_string(),
                    ext_balance: "credit".to_string(),
                    accn: "accn-new".to_string(),
                    filed: Some("2026-03-31".to_string()),
                    form: "10-Q".to_string(),
                },
            ],
        );
        let mut std_bal = HashMap::new();
        std_bal.insert("us-gaap:A".to_string(), "credit".to_string());
        std_bal.insert("us-gaap:B".to_string(), "credit".to_string());
        let df = build_stable_dataframe(
            &TickerSymbol::new("BAC"),
            &Cik::from_u64(70858).unwrap(),
            by_ext,
            &std_bal,
        )
        .unwrap();
        assert_eq!(df.height(), 1);
        let std_col = df.column("std_concept").unwrap();
        assert_eq!(std_col.get(0).unwrap().to_string(), "\"us-gaap:B\"");
        let conflict = df.column("conflict").unwrap();
        assert_eq!(conflict.bool().unwrap().get(0), Some(true));
        let n = df.column("n_filings").unwrap();
        assert_eq!(n.i64().unwrap().get(0), Some(2));
    }

    #[test]
    fn negative_weight_flips_effective() {
        let mut by_ext: HashMap<String, Vec<Occurrence>> = HashMap::new();
        by_ext.insert(
            "bac:Y".to_string(),
            vec![Occurrence {
                std_concept: "us-gaap:Parent".to_string(),
                weight: -1.0,
                label: "Y".to_string(),
                ext_balance: "debit".to_string(),
                accn: "a".to_string(),
                filed: Some("2026-03-31".to_string()),
                form: "10-Q".to_string(),
            }],
        );
        let mut std_bal = HashMap::new();
        std_bal.insert("us-gaap:Parent".to_string(), "credit".to_string());
        let df = build_stable_dataframe(
            &TickerSymbol::new("BAC"),
            &Cik::from_u64(70858).unwrap(),
            by_ext,
            &std_bal,
        )
        .unwrap();
        let eff = df.column("effective_balance").unwrap();
        assert_eq!(eff.get(0).unwrap().to_string(), "\"debit\"");
        let cons = df.column("consistent").unwrap();
        assert_eq!(cons.bool().unwrap().get(0), Some(true));
    }
}
