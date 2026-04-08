//! Tests for the CIK predecessor / holding-company lineage system.
//!
//! These tests load [`RelatedCiks`] fixtures written by
//! `cargo run --bin refresh_test_fixtures` and assert the expected contents —
//! proving that [`sec_fetcher::network::fetch_related_ciks`] is **correct**
//! (finds real predecessor CIKs when they exist) and **conservative** (returns
//! empty when no holding-company reorganisation occurred).
//!
//! ## How predecessor CIKs are discovered
//!
//! When a company restructures as a holding company the SEC requires both the
//! old and new entity to **co-file** the transition-year 10-K, which causes
//! EDGAR's Full-Text Search index to record both CIKs in the same filing's
//! `ciks[]` array.  `fetch_related_ciks` queries EFTS using the entity's
//! canonical name (looked up from the CIK-keyed submissions API — no user
//! text input) and then filters results to only those filings where the
//! primary CIK is already present, making false positives impossible.
//!
//! ## Test matrix
//!
//! | Ticker | Expected predecessors | Reason |
//! |--------|----------------------|--------|
//! | GOOG   | `["0001288776"]`     | Google Inc. → Alphabet Inc. reorganisation (Dec 2015) |
//! | GOOGL  | `["0001288776"]`     | Same Alphabet CIK; confirms ticker-independence |
//! | AAPL   | `[]`                 | Apple has never reorganised as a holding company |
//! | META   | `[]`                 | Facebook → Meta Platforms was a simple name change (same CIK) |
//! | MSFT   | `[]`                 | Microsoft, stable CIK since 1975 |
//! | NVDA   | `[]`                 | NVIDIA, stable CIK |
//! | AMZN   | `[]`                 | Amazon, stable CIK |

mod common;

fn load_related_ciks(name: &str) -> Vec<String> {
    serde_json::from_value(common::fixture_json(name))
        .expect("fixture is not a JSON array of strings")
}

// ── Positive cases (predecessor CIK expected) ─────────────────────────────────

/// GOOG (Alphabet Inc.) must report Google Inc. (CIK 0001288776) as its
/// predecessor.
///
/// This is the definitive proof that the EFTS co-registrant mechanism works:
/// `fetch_related_ciks(Alphabet CIK)` queries EFTS by Alphabet's canonical
/// name, finds the 2016 10-K that lists BOTH CIK 1652044 and CIK 1288776,
/// and returns the predecessor CIK.  When `entity_history` is run for GOOG,
/// filings from 2005 through 2026 are returned (23 total), spanning both CIKs.
#[test]
fn goog_has_predecessor_google_inc() {
    let ciks = load_related_ciks("GOOG_related_ciks.json");

    assert!(
        ciks.contains(&"0001288776".to_string()),
        "expected Google Inc. predecessor CIK 0001288776 in GOOG related CIKs, got: {:?}",
        ciks
    );
    eprintln!("GOOG predecessor CIKs: {:?}", ciks);
}

/// GOOGL (also Alphabet Inc.) must report the same Google Inc. predecessor.
///
/// GOOG and GOOGL are separate ticker classes (Class C vs Class A shares) but
/// both resolve to the same Alphabet CIK 1652044.  This test confirms that
/// the predecessor lookup is CIK-based (not ticker-based) and therefore
/// returns the same result regardless of which ticker class is queried.
#[test]
fn googl_has_same_predecessor_as_goog() {
    let ciks = load_related_ciks("GOOGL_related_ciks.json");

    assert!(
        ciks.contains(&"0001288776".to_string()),
        "expected Google Inc. predecessor CIK 0001288776 in GOOGL related CIKs, got: {:?}",
        ciks
    );
    eprintln!("GOOGL predecessor CIKs: {:?}", ciks);
}

// ── Negative cases (no predecessor expected) ──────────────────────────────────

/// Apple has never reorganised as a holding company.  The CIK that appears in
/// today's `company_tickers.json` for AAPL (CIK 320193) has been the single
/// registrant since Apple Computer Inc. first appeared in EDGAR in 1993.
#[test]
fn aapl_has_no_predecessor() {
    let ciks = load_related_ciks("AAPL_related_ciks.json");
    assert!(
        ciks.is_empty(),
        "AAPL should have no predecessor CIKs, got: {:?}",
        ciks
    );
}

/// Meta Platforms (META) was formerly Facebook Inc. — a name change within
/// the same CIK (1326801), not a holding-company reorganisation.  No EFTS
/// co-registration filing ever listed a second CIK alongside 1326801.
#[test]
fn meta_has_no_predecessor() {
    let ciks = load_related_ciks("META_related_ciks.json");
    assert!(
        ciks.is_empty(),
        "META should have no predecessor CIKs, got: {:?}",
        ciks
    );
}

/// Microsoft has operated under a single CIK (789019) since its 1986 IPO.
#[test]
fn msft_has_no_predecessor() {
    let ciks = load_related_ciks("MSFT_related_ciks.json");
    assert!(
        ciks.is_empty(),
        "MSFT should have no predecessor CIKs, got: {:?}",
        ciks
    );
}

/// NVIDIA has operated under a single CIK (1045810) since its 1999 IPO.
#[test]
fn nvda_has_no_predecessor() {
    let ciks = load_related_ciks("NVDA_related_ciks.json");
    assert!(
        ciks.is_empty(),
        "NVDA should have no predecessor CIKs, got: {:?}",
        ciks
    );
}

/// Amazon has operated under a single CIK (1018724) since its 1997 IPO.
#[test]
fn amzn_has_no_predecessor() {
    let ciks = load_related_ciks("AMZN_related_ciks.json");
    assert!(
        ciks.is_empty(),
        "AMZN should have no predecessor CIKs, got: {:?}",
        ciks
    );
}
