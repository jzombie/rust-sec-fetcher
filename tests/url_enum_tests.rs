/// Unit tests for [`sec_fetcher::enums::Url`].
///
/// Verifies that every `Url` variant produces the expected fully-qualified URL
/// string.  CIK and AccessionNumber values are kept minimal for readability.
use sec_fetcher::enums::Url;
use sec_fetcher::models::{AccessionNumber, Cik};

fn cik(n: u64) -> Cik {
    Cik::from_u64(n).unwrap()
}

fn accn(cik_val: u64, year: u16, seq: u32) -> AccessionNumber {
    AccessionNumber {
        cik: cik(cik_val),
        year,
        sequence: seq,
    }
}

#[test]
fn investment_company_dataset_url() {
    let url = Url::InvestmentCompanySeriesAndClassDataset(2025).value();
    assert_eq!(
        url,
        "https://www.sec.gov/files/investment/data/other/investment-company-series-and-class-information/investment-company-series-class-2025.csv"
    );
    // Year is embedded — a different year must produce a different URL.
    assert_ne!(
        url,
        Url::InvestmentCompanySeriesAndClassDataset(2024).value()
    );
}

#[test]
fn cik_submission_url() {
    let url = Url::CikSubmission(cik(320193)).value();
    assert_eq!(url, "https://data.sec.gov/submissions/CIK0000320193.json");
}

#[test]
fn cik_submission_page_url() {
    let url = Url::CikSubmissionPage("CIK0000320193-submissions-001.json".to_string()).value();
    assert_eq!(
        url,
        "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json"
    );
}

#[test]
fn cik_accession_url() {
    let url = Url::CikAccession(cik(320193), accn(320193, 24, 123456)).value();
    assert_eq!(
        url,
        "https://www.sec.gov/Archives/edgar/data/0000320193/000032019324123456"
    );
    // Plain accession URL must not include the index-page suffix.
    assert!(!url.ends_with("-index.htm"));
}

#[test]
fn cik_accession_index_url() {
    let a = accn(320193, 24, 123456);
    let url = Url::CikAccessionIndex(cik(320193), a).value();
    assert_eq!(
        url,
        "https://www.sec.gov/Archives/edgar/data/0000320193/000032019324123456/0000320193-24-123456-index.htm"
    );
}

#[test]
fn cik_accession_primary_document_url() {
    let a = accn(320193, 24, 123456);
    let url = Url::CikAccessionPrimaryDocument(cik(320193), a).value();
    assert_eq!(
        url,
        "https://www.sec.gov/Archives/edgar/data/0000320193/000032019324123456/primary_doc.xml"
    );
}

#[test]
fn cik_accession_document_url() {
    let a = accn(320193, 24, 123456);
    let url = Url::CikAccessionDocument(cik(320193), a, "report.htm".to_string()).value();
    assert_eq!(
        url,
        "https://www.sec.gov/Archives/edgar/data/0000320193/000032019324123456/report.htm"
    );
    // Named-document URL must not collide with the primary-document URL.
    assert!(!url.ends_with("/primary_doc.xml"));
}

#[test]
fn company_tickers_json_url() {
    let url = Url::CompanyTickersJson.value();
    assert_eq!(url, "https://www.sec.gov/files/company_tickers.json");
}

#[test]
fn company_tickers_txt_url() {
    let url = Url::CompanyTickersTxt.value();
    assert_eq!(url, "https://www.sec.gov/include/ticker.txt");
}

#[test]
fn company_facts_url() {
    let url = Url::CompanyFacts(cik(320193)).value();
    assert_eq!(
        url,
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    );
}

#[test]
fn edgar_current_feed_url() {
    let url = Url::EdgarCurrentFeed {
        form_type: "8-K".to_string(),
        count: 40,
        before: String::new(),
    }
    .value();
    assert_eq!(
        url,
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=&output=atom"
    );
    // Must use getcurrent action, not getcompany, and must not contain a CIK.
    assert!(!url.contains("action=getcompany"));
    assert!(!url.contains("CIK="));
}

#[test]
fn edgar_company_feed_url() {
    let url = Url::EdgarCompanyFeed {
        cik: cik(320193),
        form_type: "10-K".to_string(),
        count: 10,
        before: String::new(),
    }
    .value();
    assert_eq!(
        url,
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K&dateb=&owner=include&count=10&search_text=&output=atom"
    );
    // Must use getcompany action, not getcurrent.
    assert!(!url.contains("action=getcurrent"));
}

#[test]
fn edgar_full_index_url() {
    let url = Url::EdgarFullIndex {
        year: 2025,
        quarter: 3,
    }
    .value();
    assert_eq!(
        url,
        "https://www.sec.gov/Archives/edgar/full-index/2025/QTR3/master.idx"
    );
}

#[test]
fn edgar_archive_url() {
    let url = Url::EdgarArchive("edgar/data/320193/0000320193-24-000006.txt".to_string()).value();
    assert_eq!(
        url,
        "https://www.sec.gov/Archives/edgar/data/320193/0000320193-24-000006.txt"
    );
}

#[test]
fn sic_codes_url() {
    let url = Url::SicCodes.value();
    assert_eq!(url, "https://www.sec.gov/info/edgar/siccodes.htm");
}

#[test]
fn sgml_submission_txt_url() {
    // AAPL 1994 10-K — matches the path in master.idx:
    // edgar/data/320193/0000320193-94-000016.txt
    let url = Url::SgmlSubmissionTxt(cik(320193), accn(320193, 94, 16)).value();
    assert_eq!(
        url,
        "https://www.sec.gov/Archives/edgar/data/0000320193/0000320193-94-000016.txt"
    );
    // Must use the dashed (formatted) accession number as the filename.
    assert!(url.ends_with("0000320193-94-000016.txt"));
}
