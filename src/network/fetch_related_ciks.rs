use crate::enums::Url;
use crate::models::Cik;
use crate::network::{SecClient, fetch_cik_submissions};
use serde_json::Value;
use std::error::Error;

/// Returns all CIKs that have ever co-registered a 10-K alongside `primary_cik`.
///
/// # Why this exists
///
/// The SEC assigns a permanent CIK to every registrant.  When a company
/// restructures as a holding company (e.g. Google Inc. → Alphabet Inc. in
/// 2015), the SEC creates a **new CIK** for the successor entity.  Historical
/// filings remain under the old CIK, and the new entity's submission history
/// starts from the reorganization date.
///
/// A ticker-to-CIK lookup (e.g. `GOOG` → `1652044` for Alphabet) therefore
/// only surfaces post-reorganization filings.  Calling `fetch_10k_filings` on
/// that CIK alone silently returns an incomplete filing history.
///
/// # How the link is discovered — no text search
///
/// EDGAR's Full-Text Search index (EFTS) records a `ciks` co-registrant array
/// on every indexed filing.  During the mandatory joint-filing period required
/// by SEC regulations for holding company reorganizations, both the predecessor
/// and successor entity CIKs appear in that array.  For Alphabet's 2016 10-K,
/// EDGAR itself records:
///
/// ```json
/// "ciks": ["0001652044", "0001288776"]
/// ```
///
/// This function queries EFTS using the **numeric CIK** via the `entity=`
/// parameter — the same exact-CIK lookup that the EDGAR search UI uses when
/// you enter a CIK number.  There is no name-based text matching; the result
/// set is the same set of 10-K filings returned by any other CIK-keyed EDGAR
/// lookup.
///
/// # Scope and limitations
///
/// - Covers **holding company reorganizations** where the SEC required a
///   co-registration transition filing (e.g. Google → Alphabet).
/// - Does **not** cover mergers where the target simply stops filing; in that
///   case the target's CIK goes dark and there is no co-registration record.
/// - Name changes within the same CIK are already transparent (same CIK).
/// - Spin-offs get their own CIK; no co-registration link exists to the parent.
///
/// When no co-registration was ever filed this function returns an empty `Vec`
/// (not an error) — for the vast majority of entities the result is empty and
/// the caller proceeds with the primary CIK's data unchanged.
pub async fn fetch_related_ciks(
    client: &SecClient,
    primary_cik: &Cik,
) -> Result<Vec<Cik>, Box<dyn Error>> {
    // ── Step 1: resolve the canonical entity name from the submissions API ────
    // The submissions endpoint is keyed by CIK — exact, deterministic, no text
    // matching.  We use the returned `name` field purely to build the EFTS URL
    // in step 2; it is never exposed to user input.
    let subs_url = Url::CikSubmission(primary_cik.clone()).value();
    let subs_data: Value = client.fetch_json(&subs_url, None).await?;
    let entity_name = match subs_data["name"].as_str() {
        Some(n) if !n.is_empty() => n.to_string(),
        _ => return Ok(vec![]),
    };

    // ── Step 2: quoted-phrase EFTS search for 10-K filings ───────────────────
    // EDGAR's EFTS indexes every filing document.  A quoted phrase query for
    // the entity's registered name returns filings that prominently name this
    // entity, including any transition-era co-registration filings where both
    // the predecessor and successor CIK appear in the `ciks` array.
    let search_url = Url::EftsCoRegistrantsByName {
        entity_name: entity_name.clone(),
    }
    .value();
    let search_data: Value = client.fetch_json(&search_url, None).await?;

    let primary_str = format!("{:010}", primary_cik.value);
    let mut related: Vec<Cik> = Vec::new();

    if let Some(hits) = search_data["hits"]["hits"].as_array() {
        for hit in hits {
            let ciks_arr = match hit["_source"]["ciks"].as_array() {
                Some(a) => a,
                None => continue,
            };

            // Ignore filings where our CIK is not listed — should not happen
            // for a by-CIK query, but guard defensively.
            if !ciks_arr
                .iter()
                .any(|v| v.as_str().map_or(false, |s| s == primary_str))
            {
                continue;
            }

            for cik_val in ciks_arr {
                if let Some(s) = cik_val.as_str() {
                    if let Ok(n) = s.trim_start_matches('0').parse::<u64>() {
                        if n != primary_cik.value && !related.iter().any(|c: &Cik| c.value == n) {
                            related.push(Cik { value: n });
                        }
                    }
                }
            }
        }
    }

    Ok(related)
}

/// Fetches the complete, merged filing history for an entity across all
/// CIKs it has ever been registered under.
///
/// For the vast majority of entities (those that have never reorganised as a
/// holding company) the result is identical to calling [`fetch_cik_submissions`]
/// directly — one extra cached EFTS lookup is the only overhead.
///
/// For entities that *have* reorganised (e.g. `GOOG` → Alphabet Inc.), the
/// returned list merges filings from both the current CIK and every predecessor
/// CIK discovered via [`fetch_related_ciks`], sorted newest-first.
pub async fn fetch_all_entity_submissions(
    client: &SecClient,
    primary_cik: Cik,
) -> Result<Vec<crate::models::CikSubmission>, Box<dyn Error>> {
    let related = fetch_related_ciks(client, &primary_cik).await?;

    let mut all = fetch_cik_submissions(client, primary_cik).await?;

    for related_cik in related {
        match fetch_cik_submissions(client, related_cik).await {
            Ok(mut subs) => all.append(&mut subs),
            Err(e) => eprintln!("Warning: could not fetch submissions for related CIK: {}", e),
        }
    }

    // Primary submissions are already newest-first; predecessors may not be.
    all.sort_by(|a, b| b.filing_date.cmp(&a.filing_date));

    Ok(all)
}

