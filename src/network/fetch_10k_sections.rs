use crate::enums::Url;
use crate::models::{Cik, CikSubmission};
use crate::network::{SecClient, fetch_cik_submissions, fetch_filing_index};
use crate::parsers::{TenKSections, extract_sections_from_document};
use bytes::Bytes;
use regex::Regex;
use std::error::Error;
use std::sync::LazyLock as Lazy;

// Non-prose file extensions — skip in the filing index fallback.
static BINARY_EXT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\.(jpg|jpeg|png|gif|pdf|xlsx|zip|xsd|xml|js|css)$").unwrap());

// ── Public API ────────────────────────────────────────────────────────────────

/// Returns the raw bytes of the best 10-K document for a filing.
///
/// Pass 1 — downloads the primary document.  If that document passes
/// [`TenKSections::is_adequate`], returns its bytes immediately.
///
/// Pass 2 — if the primary document is a stub (incorporates sections by
/// reference to a separate annual report), walks the filing index in two
/// tiers:
///
/// - **Tier 0** — explicitly typed `10-K`, `10-K405`, `10-K/A`, `ARS`, or
///   `EX-13*` documents (Exhibit 13 annual reports).
/// - **Tier 1** — any remaining `.htm`, `.html`, or `.txt` file.
///
/// The first document whose parsed sections pass `is_adequate` wins.  If no
/// document satisfies the check, the primary document bytes are returned as a
/// last-resort fallback.
///
/// This is the single source of truth for 10-K document selection; both the
/// network-layer [`fetch_10k_sections_for_filing`] and the test-fixture
/// refresh binary call this.
pub async fn fetch_best_10k_document(
    sec_client: &SecClient,
    filing: &CikSubmission,
) -> Result<Bytes, Box<dyn Error>> {
    let primary_url = if filing.primary_document.is_empty() {
        Url::SgmlSubmissionTxt(filing.cik.clone(), filing.accession_number.clone()).value()
    } else {
        Url::CikAccessionDocument(
            filing.cik.clone(),
            filing.accession_number.clone(),
            filing.primary_document.clone(),
        )
        .value()
    };

    let prim_resp = sec_client
        .raw_request(reqwest::Method::GET, &primary_url, None, None)
        .await?;
    if !prim_resp.status().is_success() {
        return Err(format!("HTTP {} for {}", prim_resp.status(), primary_url).into());
    }
    let prim_bytes = prim_resp.bytes().await?;

    // Fast path: primary document is self-contained.
    if {
        let t = String::from_utf8_lossy(&prim_bytes);
        extract_sections_from_document(&t).is_adequate()
    } {
        return Ok(prim_bytes);
    }

    // Slow path: walk the filing index for an adequate document.
    let index = match fetch_filing_index(sec_client, filing).await {
        Ok(idx) => idx,
        Err(_) => return Ok(prim_bytes),
    };

    for tier in 0..2u8 {
        for doc in &index.documents {
            if doc.name == filing.primary_document {
                continue;
            }
            if BINARY_EXT_RE.is_match(&doc.name) {
                continue;
            }
            let lower = doc.name.to_ascii_lowercase();
            let type_upper = doc.document_type.to_ascii_uppercase();
            let matches_tier = match tier {
                0 => {
                    type_upper == "10-K"
                        || type_upper == "10-K405"
                        || type_upper == "10-K/A"
                        || type_upper == "ARS"
                        || type_upper.starts_with("EX-13")
                }
                1 => lower.ends_with(".htm") || lower.ends_with(".html") || lower.ends_with(".txt"),
                _ => false,
            };
            if !matches_tier {
                continue;
            }

            let url = Url::CikAccessionDocument(
                filing.cik.clone(),
                filing.accession_number.clone(),
                doc.name.clone(),
            )
            .value();

            if let Ok(r) = sec_client
                .raw_request(reqwest::Method::GET, &url, None, None)
                .await
            {
                if r.status().is_success() {
                    let b = r.bytes().await?;
                    if {
                        let t = String::from_utf8_lossy(&b);
                        extract_sections_from_document(&t).is_adequate()
                    } {
                        return Ok(b);
                    }
                }
            }
        }
    }

    Ok(prim_bytes)
}

/// Fetches and extracts **all sections** from the most recent 10-K filing for
/// the given [`Cik`].
///
/// Returns a [`TenKSections`] map keyed by normalized item name (`"item_1"`,
/// `"item_7"`, `"item_1a"`, …).  Use [`TenKSections::get`],
/// [`TenKSections::item7`], etc. to access individual sections.
///
/// # Example
///
/// ```no_run
/// # use sec_fetcher::network::{SecClient, fetch_cik_by_ticker_symbol, fetch_10k_sections};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&cfg)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
/// let sections = fetch_10k_sections(&client, cik).await?;
/// if let Some(text) = sections.item7() {
///     println!("=== Item 7 ({} chars) ===\n{text}", text.len());
/// }
/// # Ok(()) }
/// ```
pub async fn fetch_10k_sections(
    sec_client: &SecClient,
    cik: Cik,
) -> Result<TenKSections, Box<dyn Error>> {
    let submissions = fetch_cik_submissions(sec_client, cik.clone()).await?;
    let filing = match CikSubmission::most_recent_10k(&submissions) {
        Some(f) => f,
        None => return Ok(TenKSections::empty()),
    };
    fetch_10k_sections_for_filing(sec_client, filing).await
}

/// Extracts all sections from a **specific** 10-K filing you already hold.
///
/// Use this when iterating over historical filings.  Obtain the list with
/// [`fetch_10k_filings`], pick the filing you want, then pass it here.
///
/// [`fetch_10k_filings`]: crate::network::fetch_10k_filings
///
/// # Strategy
///
/// 1. Try the primary document.
/// 2. If Item 1 and Item 7 are not yet substantial, fetch the filing index and
///    try candidate documents in priority order: typed 10-K docs first, then
///    `.htm` files, then `.txt` files.  Stops as soon as both core sections
///    are substantial.
///
/// # Example
///
/// ```no_run
/// # use sec_fetcher::network::{SecClient, fetch_cik_by_ticker_symbol, fetch_10k_filings,
/// #                             fetch_10k_sections_for_filing};
/// # use sec_fetcher::config::ConfigManager;
/// # use sec_fetcher::models::TickerSymbol;
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = ConfigManager::load()?;
/// let client = SecClient::from_config_manager(&cfg)?;
/// let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
/// let filings = fetch_10k_filings(&client, cik).await?;
/// for filing in &filings {
///     let date = filing.filing_date.map(|d| d.to_string()).unwrap_or_default();
///     let sections = fetch_10k_sections_for_filing(&client, filing).await?;
///     for (key, text) in sections.iter() {
///         println!("{} {}: {} chars", date, key, text.len());
///     }
/// }
/// # Ok(()) }
/// ```
pub async fn fetch_10k_sections_for_filing(
    sec_client: &SecClient,
    filing: &CikSubmission,
) -> Result<TenKSections, Box<dyn Error>> {
    let old_format = filing.primary_document.is_empty();

    // ── Pass 1: primary document ──────────────────────────────────────────────
    let primary_url = if old_format {
        Url::SgmlSubmissionTxt(filing.cik.clone(), filing.accession_number.clone()).value()
    } else {
        Url::CikAccessionDocument(
            filing.cik.clone(),
            filing.accession_number.clone(),
            filing.primary_document.clone(),
        )
        .value()
    };

    let mut best = fetch_sections_from_url(sec_client, &primary_url)
        .await
        .unwrap_or_else(|_| TenKSections::empty());

    if best.is_adequate() {
        return Ok(best);
    }

    // ── Pass 2: filing index fallback ─────────────────────────────────────────
    let index = match fetch_filing_index(sec_client, filing).await {
        Ok(idx) => idx,
        Err(_) => return Ok(best),
    };

    // Candidate priority:
    //   Tier 0 — explicitly typed "10-K" / "10-K405" / "10-K/A"
    //   Tier 1 — any .htm / .html
    //   Tier 2 — any .txt
    let mut candidates: Vec<&str> = Vec::new();

    for tier in 0..3u8 {
        for doc in &index.documents {
            let name = doc.name.as_str();
            if name == filing.primary_document.as_str() {
                continue;
            }
            if BINARY_EXT_RE.is_match(name) {
                continue;
            }
            let lower = name.to_ascii_lowercase();
            let doc_type = doc.document_type.to_ascii_uppercase();
            let matches_tier = match tier {
                0 => doc_type == "10-K" || doc_type == "10-K405" || doc_type == "10-K/A",
                1 => lower.ends_with(".htm") || lower.ends_with(".html"),
                2 => lower.ends_with(".txt"),
                _ => false,
            };
            if matches_tier && !candidates.contains(&name) {
                candidates.push(name);
            }
        }
    }

    for name in candidates {
        let url = if old_format {
            Url::EdgarArchive(format!("edgar/data/{}/{}", filing.cik, name)).value()
        } else {
            Url::CikAccessionDocument(
                filing.cik.clone(),
                filing.accession_number.clone(),
                name.to_string(),
            )
            .value()
        };

        if let Ok(sections) = fetch_sections_from_url(sec_client, &url).await {
            best.merge_with(sections);
        }

        if best.is_adequate() {
            break;
        }
    }

    Ok(best)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

async fn fetch_sections_from_url(
    sec_client: &SecClient,
    url: &str,
) -> Result<TenKSections, Box<dyn Error>> {
    let response = sec_client
        .raw_request(reqwest::Method::GET, url, None, None)
        .await?;

    if !response.status().is_success() {
        return Err(format!("HTTP {} for {}", response.status(), url).into());
    }

    let raw = response.text().await?;
    Ok(extract_sections_from_document(&raw))
}
