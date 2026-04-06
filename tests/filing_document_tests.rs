/// Unit tests for [`sec_fetcher::models::FilingDocument`] and
/// [`sec_fetcher::models::FilingIndex`].
use sec_fetcher::models::{FilingDocument, FilingIndex};

fn doc(name: &str, doc_type: &str) -> FilingDocument {
    FilingDocument {
        name: name.to_string(),
        document_type: doc_type.to_string(),
    }
}

// ── FilingDocument::is_exhibit ────────────────────────────────────────────────

#[test]
fn is_exhibit_true_for_ex_prefix() {
    assert!(doc("file.htm", "EX-99.1").is_exhibit());
    assert!(doc("file.htm", "EX-10.1").is_exhibit());
    assert!(doc("file.htm", "EX-31.1").is_exhibit());
    assert!(doc("file.htm", "EX-32.2").is_exhibit());
    assert!(doc("file.htm", "EX-23.1").is_exhibit());
    assert!(doc("file.htm", "EX-101.INS").is_exhibit());
}

#[test]
fn is_exhibit_false_for_primary_forms() {
    assert!(!doc("form.htm", "8-K").is_exhibit());
    assert!(!doc("form.htm", "10-K").is_exhibit());
    assert!(!doc("graphic.jpg", "GRAPHIC").is_exhibit());
}

#[test]
fn is_exhibit_case_insensitive() {
    assert!(doc("file.htm", "ex-99.1").is_exhibit());
    assert!(doc("file.htm", "Ex-99.1").is_exhibit());
}

// ── FilingDocument::is_press_release ─────────────────────────────────────────

#[test]
fn is_press_release_for_ex_99() {
    assert!(doc("pr.htm", "EX-99.1").is_press_release());
    assert!(doc("pr.htm", "EX-99.2").is_press_release());
    assert!(doc("pr.htm", "EX-99").is_press_release());
}

#[test]
fn is_press_release_false_for_other_exhibits() {
    assert!(!doc("k.htm", "EX-10.1").is_press_release());
    assert!(!doc("k.htm", "EX-31.1").is_press_release());
    assert!(!doc("k.htm", "EX-23.1").is_press_release());
}

// ── FilingDocument::is_sarbanes_oxley_cert ───────────────────────────────────

#[test]
fn is_sox_cert_for_ex_31_and_32() {
    assert!(doc("cert.htm", "EX-31.1").is_sarbanes_oxley_cert());
    assert!(doc("cert.htm", "EX-31.2").is_sarbanes_oxley_cert());
    assert!(doc("cert.htm", "EX-32.1").is_sarbanes_oxley_cert());
    assert!(doc("cert.htm", "EX-32.2").is_sarbanes_oxley_cert());
}

#[test]
fn is_sox_cert_false_for_others() {
    assert!(!doc("file.htm", "EX-99.1").is_sarbanes_oxley_cert());
    assert!(!doc("file.htm", "EX-23.1").is_sarbanes_oxley_cert());
    assert!(!doc("file.htm", "10-K").is_sarbanes_oxley_cert());
}

// ── FilingDocument::is_auditor_consent ───────────────────────────────────────

#[test]
fn is_auditor_consent_for_ex_23() {
    assert!(doc("consent.htm", "EX-23.1").is_auditor_consent());
    assert!(doc("consent.htm", "EX-23.2").is_auditor_consent());
}

#[test]
fn is_auditor_consent_false_for_others() {
    assert!(!doc("file.htm", "EX-31.1").is_auditor_consent());
    assert!(!doc("file.htm", "EX-99.1").is_auditor_consent());
}

// ── FilingDocument::is_xbrl_data ─────────────────────────────────────────────

#[test]
fn is_xbrl_data_for_ex_101() {
    assert!(doc("xbrl.xml", "EX-101.INS").is_xbrl_data());
    assert!(doc("xbrl.xml", "EX-101.SCH").is_xbrl_data());
    assert!(doc("xbrl.xml", "EX-101.PRE").is_xbrl_data());
}

#[test]
fn is_xbrl_data_false_for_others() {
    assert!(!doc("file.htm", "EX-99.1").is_xbrl_data());
    assert!(!doc("file.htm", "EX-31.1").is_xbrl_data());
}

// ── FilingDocument::is_substantive_exhibit ───────────────────────────────────

#[test]
fn press_release_is_substantive() {
    assert!(doc("pr.htm", "EX-99.1").is_substantive_exhibit());
}

#[test]
fn material_contract_is_substantive() {
    assert!(doc("contract.htm", "EX-10.1").is_substantive_exhibit());
}

#[test]
fn sox_cert_is_not_substantive() {
    assert!(!doc("cert.htm", "EX-31.1").is_substantive_exhibit());
    assert!(!doc("cert.htm", "EX-32.1").is_substantive_exhibit());
}

#[test]
fn auditor_consent_is_not_substantive() {
    assert!(!doc("consent.htm", "EX-23.1").is_substantive_exhibit());
}

#[test]
fn xbrl_data_is_not_substantive() {
    assert!(!doc("xbrl.xml", "EX-101.INS").is_substantive_exhibit());
}

#[test]
fn graphic_is_not_substantive() {
    assert!(!doc("logo.jpg", "GRAPHIC").is_substantive_exhibit());
    // GRAPHIC is also not an EX- type, so is_exhibit is false first
}

#[test]
fn primary_form_is_not_substantive() {
    assert!(!doc("form.htm", "8-K").is_substantive_exhibit());
}

// ── FilingDocument::is_html / is_text ────────────────────────────────────────

#[test]
fn is_html_for_htm_and_html_extensions() {
    assert!(doc("report.htm", "8-K").is_html());
    assert!(doc("report.html", "8-K").is_html());
    assert!(doc("REPORT.HTM", "8-K").is_html());
}

#[test]
fn is_html_false_for_other_extensions() {
    assert!(!doc("report.txt", "8-K").is_html());
    assert!(!doc("report.xml", "8-K").is_html());
    assert!(!doc("report.pdf", "8-K").is_html());
}

#[test]
fn is_text_for_txt_extension() {
    assert!(doc("doc.txt", "8-K").is_text());
    assert!(doc("DOC.TXT", "8-K").is_text());
}

#[test]
fn is_text_false_for_other_extensions() {
    assert!(!doc("doc.htm", "8-K").is_text());
    assert!(!doc("doc.xml", "8-K").is_text());
}

// ── FilingIndex ───────────────────────────────────────────────────────────────

fn sample_index() -> FilingIndex {
    FilingIndex {
        documents: vec![
            doc("form8k.htm", "8-K"),
            doc("ex991.htm", "EX-99.1"),
            doc("ex311.htm", "EX-31.1"),
            doc("ex321.htm", "EX-32.1"),
            doc("ex231.htm", "EX-23.1"),
            doc("xbrl.xml", "EX-101.INS"),
            doc("logo.jpg", "GRAPHIC"),
            doc("contract.htm", "EX-10.1"),
        ],
    }
}

#[test]
fn exhibits_returns_all_ex_documents() {
    let idx = sample_index();
    let exhibits = idx.exhibits();
    // EX-99.1, EX-31.1, EX-32.1, EX-23.1, EX-101.INS, EX-10.1  (not 8-K or GRAPHIC)
    assert_eq!(exhibits.len(), 6);
}

#[test]
fn substantive_exhibits_excludes_noise() {
    let idx = sample_index();
    let substantive = idx.substantive_exhibits();
    // Only EX-99.1 and EX-10.1 are substantive
    assert_eq!(substantive.len(), 2);
    let types: Vec<&str> = substantive
        .iter()
        .map(|d| d.document_type.as_str())
        .collect();
    assert!(types.contains(&"EX-99.1"));
    assert!(types.contains(&"EX-10.1"));
}

#[test]
fn press_releases_returns_only_ex_99() {
    let idx = sample_index();
    let prs = idx.press_releases();
    assert_eq!(prs.len(), 1);
    assert_eq!(prs[0].document_type, "EX-99.1");
}

#[test]
fn empty_index_returns_empty_slices() {
    let idx = FilingIndex { documents: vec![] };
    assert!(idx.exhibits().is_empty());
    assert!(idx.substantive_exhibits().is_empty());
    assert!(idx.press_releases().is_empty());
}
