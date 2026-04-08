//! Robustness tests for 10-K section extraction across every major EDGAR era.
//!
//! These tests load real filing documents saved by
//! `cargo run --bin refresh-test-fixtures` and assert that
//! [`sec_fetcher::network::extract_sections_from_document`] successfully
//! locates both **Item 1 (Business)** and **Item 7 (MD&A)** from the raw bytes.
//!
//! ## EDGAR eras covered
//!
//! | Era | Years | Format | Fixtures |
//! |---|---|---|---|
//! | SGML | pre-2000 | SGML `.txt` bundle | AAPL 1994, MSFT 1995, GE 1994 |
//! | Early HTML | 2002–2006 | plain HTML | KO 2002, WMT 2005 |
//! | Mid HTML | 2008–2013 | HTML w/ some XBRL | JPM 2009, GOOG 2012 |
//! | iXBRL | 2014–2018 | inline-XBRL HTML | AAPL 2016, NVDA 2017 |
//! | Modern | 2019+ | inline-XBRL HTML | BRK-B 2023, COST 2024 |
//!
//! ## Minimum-length thresholds
//!
//! Real section bodies are tens-of-thousands of characters; TOC stubs are
//! fewer than 400 characters.  The 1 500-char floor used here is deliberately
//! generous for the smallest valid older filings while comfortably sitting
//! above any TOC stub.
//!
//! If a fixture is missing, the test panics directing the developer to run
//! the fixture refresh binary.

use sec_fetcher::network::extract_sections_from_document;

mod common;

fn load_raw_fixture(name: &str) -> String {
    common::fixture_string(name)
}

/// Asserts that `extract_sections_from_document` extracts both Item 1 and
/// Item 7 from the fixture with at least `min_chars` characters each.
fn assert_sections_extracted(fixture_name: &str, min_chars: usize) {
    let raw = load_raw_fixture(fixture_name);

    let sections = extract_sections_from_document(&raw)
        .expect("html2text panicked on fixture — this fixture may be malformed");

    let item1_len = sections.item1().map_or(0, |s| s.len());
    let item7_len = sections.item7().map_or(0, |s| s.len());

    assert!(
        sections.item1().is_some(),
        "[{}] Item 1 not found (expected >= {} chars)",
        fixture_name,
        min_chars
    );
    assert!(
        item1_len >= min_chars,
        "[{}] Item 1 too short: {} chars (expected >= {})",
        fixture_name,
        item1_len,
        min_chars
    );

    assert!(
        sections.item7().is_some(),
        "[{}] Item 7 not found (expected >= {} chars)",
        fixture_name,
        min_chars
    );
    assert!(
        item7_len >= min_chars,
        "[{}] Item 7 too short: {} chars (expected >= {})",
        fixture_name,
        item7_len,
        min_chars
    );

    eprintln!(
        "[{}] Item 1: {} chars  Item 7: {} chars",
        fixture_name, item1_len, item7_len
    );
}

// ── SGML era (pre-2000) ───────────────────────────────────────────────────────

/// Apple 10-K filed for fiscal year ending September 1994.
///
/// SGML bundle with empty `primary_document` — the full submission is a
/// single `.txt` file at the CIK root.  The SGML path uses regex tag-strip
/// instead of html2text to avoid the html5ever `<head>` content-loss bug.
#[test]
fn sgml_aapl_1994_extracts_both_sections() {
    assert_sections_extracted("AAPL_10k_1994.raw", 1_500);
}

/// Microsoft 10-K filed for fiscal year ending June 1995.
///
/// Same SGML era as AAPL 1994 — validates that non-Apple tickers in the
/// SGML era also work correctly.
#[test]
fn sgml_msft_1995_extracts_both_sections() {
    assert_sections_extracted("MSFT_10k_1995.raw", 1_500);
}

/// GE (General Electric) 10-K filed for fiscal year ending December 1994.
///
/// Large conglomerate with a more complex SGML structure — stress-tests the
/// max-gap section selector against a document with many Item references.
///
/// NOTE: GE's 1994 EDGAR filing incorporated MD&A by reference to the
/// physical Annual Report to Share Owners.  No extractable Item 7 body
/// exists in the filing.  The fixture slot is occupied by INTC 1996 instead.
///
/// Intel (INTC) 10-K filed for fiscal year 1995 (filing date in 1996).
///
/// Large tech chipmaker — SGML era, inline filer.  Tests the SGML tag-strip
/// path on a high-volume industrial filer different from AAPL and MSFT.
#[test]
fn sgml_intc_1996_extracts_both_sections() {
    assert_sections_extracted("INTC_10k_1996.raw", 1_500);
}

// ── Early HTML era (2002–2006) ────────────────────────────────────────────────

/// Apple 10-K for fiscal year ending September 2003.
///
/// First generation of plain-HTML filings for Apple — heading markup varies
/// across documents and often includes HTML entities in section titles.  Tests
/// html2text entity decoding and section extraction in the early-HTML era.
/// Apple is used instead of Coca-Cola (KO 2002) which incorporated all
/// substantive sections by reference to its physical annual report.
#[test]
fn early_html_aapl_2003_extracts_both_sections() {
    assert_sections_extracted("AAPL_10k_2003.raw", 3_000);
}

/// Microsoft 10-K for fiscal year ending June 2005.
///
/// Large software company with a complex filing structure and many
/// cross-references.  Tests that "see Item 7" references in Item 1 body text
/// do not cause the max-gap strategy to mis-identify the section boundary.
/// Microsoft is used instead of Walmart (WMT 2005) which incorporated its
/// MD&A by reference to the Annual Report to Shareholders.
#[test]
fn early_html_msft_2005_extracts_both_sections() {
    assert_sections_extracted("MSFT_10k_2005.raw", 3_000);
}

// ── Mid HTML era (2008–2013) ──────────────────────────────────────────────────

/// Oracle 10-K for fiscal year ending May 2009.
///
/// Large enterprise-software company filed during the post-crisis period.
/// MD&A (Item 7) in tech 10-Ks is detailed; this test validates that the
/// extractor captures the full section without premature truncation.
/// Oracle is used instead of JPMorgan Chase (JPM 2009) which (a) incorporated
/// MD&A by reference to its physical annual report and (b) used `ITEM 7:`
/// (colon separator) which our regex also now accepts.
#[test]
fn mid_html_orcl_2009_extracts_both_sections() {
    assert_sections_extracted("ORCL_10k_2009.raw", 5_000);
}

/// Johnson & Johnson 10-K for fiscal year 2010.
///
/// Large diversified healthcare / consumer-goods company in the mid-HTML era.
/// JNJ historically incorporated Items 6–8 by reference to Exhibit 13 (the
/// Annual Report to Shareholders).  The fixture is the Exhibit-13 annual
/// report, discovered via the filing-index multi-pass strategy in
/// `refresh_test_fixtures`; it contains the full inline MD&A.
#[test]
fn mid_html_jnj_2010_extracts_both_sections() {
    assert_sections_extracted("JNJ_10k_2010.raw", 5_000);
}

/// Alphabet (Google) 10-K for fiscal year 2012.
///
/// Tech company with a well-structured mid-era HTML filing.  Confirms
/// robustness for technology companies in the pre-iXBRL era.
#[test]
fn mid_html_goog_2012_extracts_both_sections() {
    assert_sections_extracted("GOOG_10k_2012.raw", 5_000);
}

// ── iXBRL transition era (2014–2018) ─────────────────────────────────────────

/// Apple 10-K for fiscal year ending September 2016.
///
/// iXBRL wrapper filing where `primary_document` is a shell containing only
/// viewer HTML.  The fixture stores the primary document; the test confirms
/// that useful content is still present in this particular year (2016 Apple
/// began embedding the full narrative in the primary iXBRL document, unlike
/// the 2015 wrapper structure).
#[test]
fn ixbrl_aapl_2016_extracts_both_sections() {
    assert_sections_extracted("AAPL_10k_2016.raw", 5_000);
}

/// NVIDIA 10-K for fiscal year ending January 2017.
///
/// Semiconductors / GPU company at the cusp of the iXBRL rollout.  Tests
/// the transition era for a non-Apple tech filing.
#[test]
fn ixbrl_nvda_2017_extracts_both_sections() {
    assert_sections_extracted("NVDA_10k_2017.raw", 3_000);
}

// ── Early HTML additions ────────────────────────────────────────────────────

/// ExxonMobil 10-K for fiscal year 2006.
///
/// Large energy company operating as ExxonMobil Corporation — formed from the
/// 1999 Exxon/Mobil merger under a single stable CIK (no predecessor lookup
/// required even though the underlying companies merged).  Filed as early-HTML;
/// tests that the extractor handles an energy-sector filing with extensive
/// operational MD&A distinct from tech filings.
#[test]
fn early_html_xom_2006_extracts_both_sections() {
    assert_sections_extracted("XOM_10k_2006.raw", 3_000);
}

// ── Modern inline iXBRL (2019+) ───────────────────────────────────────────

/// Amazon 10-K for fiscal year 2019.
///
/// Large e-commerce and cloud (AWS) company, first full year under modern
/// inline iXBRL.  Amazon's Item 1 is concise but Item 7 is extremely detailed
/// (multi-segment financials, risk factors, FX discussion).  Confirms the
/// extractor handles modern e-commerce / cloud sector filings.
#[test]
fn modern_amzn_2019_extracts_both_sections() {
    assert_sections_extracted("AMZN_10k_2019.raw", 5_000);
}

/// JPMorgan Chase 10-K for fiscal year 2022.
///
/// Large US bank in the modern iXBRL era.  JPM 2009 was previously excluded
/// from the matrix because its MD&A was incorporated by reference to a
/// physical annual report.  By 2022, all large accelerated filers submit fully
/// inline documents.  This test validates the financial-sector modern path and
/// checks that a large, multi-page banking MD&A is captured in full.
#[test]
fn modern_jpm_2022_extracts_both_sections() {
    assert_sections_extracted("JPM_10k_2022.raw", 10_000);
}

/// Berkshire Hathaway (BRK-B) 10-K for fiscal year 2023.
///
/// Large conglomerate with complex business description (many subsidiaries)
/// and an extremely detailed MD&A.  The large file size stress-tests the
/// 1 MB html2text budget in `extract_sections_from_document`.
#[test]
fn modern_brk_b_2023_extracts_both_sections() {
    assert_sections_extracted("BRK_B_10k_2023.raw", 10_000);
}

/// Costco 10-K for fiscal year ending September 2024.
///
/// Straightforward modern retailer filing — confirms the baseline modern
/// inline-iXBRL path works for a non-tech, non-financial company.
#[test]
fn modern_cost_2024_extracts_both_sections() {
    assert_sections_extracted("COST_10k_2024.raw", 5_000);
}
