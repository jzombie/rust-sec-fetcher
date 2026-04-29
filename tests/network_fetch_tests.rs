/// Unit tests for the publicly exported **parse functions** in the
/// [`sec_fetcher::network`] and [`sec_fetcher::parsers`] modules.
///
/// These tests operate on static JSON / text bodies so they never hit the
/// real SEC EDGAR network.
use sec_fetcher::models::Cik;
use sec_fetcher::parsers::{
    parse_cik_submissions_json, parse_company_tickers_json, parse_master_idx, parse_ticker_txt,
};
use serde_json::json;

// ---------------------------------------------------------------------------
// parse_cik_submissions_json
// ---------------------------------------------------------------------------

#[test]
fn test_parse_cik_submissions_json_basic() {
    let data = json!({
        "entityType": "operating",
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-24-000123", "0000320193-24-000045"],
                "filingDate": ["2024-11-01", "2024-10-15"],
                "form": ["10-K", "8-K"],
                "primaryDocument": ["primary-document.htm", "primary-document.htm"],
                "primaryDocDescription": ["Annual Report", "Current Report"],
                "act": ["34", "34"],
                "items": ["", "2.02,9.01"],
                "fileNumber": ["", ""],
                "isXBRL": [1, 0],
                "isInlineXBRL": [1, 0]
            }
        }
    });

    let cik = Cik::from_str("0000320193").unwrap();
    let submissions = parse_cik_submissions_json(&data, cik.clone());

    assert_eq!(submissions.len(), 2);
    assert_eq!(submissions[0].cik.to_string(), "0000320193");
    assert_eq!(submissions[0].form_type().to_string(), "10-K");
    assert_eq!(
        submissions[0].filing_date.unwrap().to_string(),
        "2024-11-01"
    );
    assert_eq!(submissions[0].items.len(), 0); // empty items

    assert_eq!(submissions[1].form_type().to_string(), "8-K");
    assert_eq!(submissions[1].items, vec!["2.02", "9.01"]);
    assert_eq!(
        submissions[1].filing_date.unwrap().to_string(),
        "2024-10-15"
    );
}

#[test]
fn test_parse_cik_submissions_json_missing_recent() {
    let data = json!({
        "entityType": "operating",
        "filings": {}
    });
    let cik = Cik::from_str("0000320193").unwrap();
    let submissions = parse_cik_submissions_json(&data, cik);
    assert!(
        submissions.is_empty(),
        "Expected empty when filings.recent is missing"
    );
}

#[test]
fn test_parse_cik_submissions_json_empty_recent() {
    let data = json!({
        "filings": {
            "recent": {
                "accessionNumber": [],
                "filingDate": [],
                "form": [],
                "primaryDocument": [],
                "primaryDocDescription": [],
                "act": [],
                "items": [],
                "fileNumber": [],
                "isXBRL": [],
                "isInlineXBRL": []
            }
        }
    });
    let cik = Cik::from_str("0000320193").unwrap();
    let submissions = parse_cik_submissions_json(&data, cik);
    assert!(
        submissions.is_empty(),
        "Expected empty when filings.recent has no entries"
    );
}

#[test]
fn test_parse_cik_submissions_json_no_entity_type() {
    let data = json!({
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-24-000123"],
                "filingDate": ["2024-11-01"],
                "form": ["10-K"],
                "primaryDocument": ["primary-document.htm"],
                "primaryDocDescription": ["Annual Report"],
                "act": ["34"],
                "items": [""],
                "fileNumber": [""],
                "isXBRL": [1],
                "isInlineXBRL": [1]
            }
        }
    });
    let cik = Cik::from_str("0000320193").unwrap();
    let submissions = parse_cik_submissions_json(&data, cik);
    assert_eq!(submissions.len(), 1);
    assert!(submissions[0].entity_type.is_none());
}

// ---------------------------------------------------------------------------
// parse_company_tickers_json
// ---------------------------------------------------------------------------

#[test]
fn test_parse_company_tickers_json_basic() {
    let data = json!({
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"}
    });

    let tickers = parse_company_tickers_json(&data).unwrap();
    assert_eq!(tickers.len(), 2);

    let aapl = tickers
        .iter()
        .find(|t| t.symbol.to_string() == "AAPL")
        .unwrap();
    assert_eq!(aapl.company_name, "Apple Inc.");
    assert_eq!(aapl.cik.to_string(), "0000320193");

    let msft = tickers
        .iter()
        .find(|t| t.symbol.to_string() == "MSFT")
        .unwrap();
    assert_eq!(msft.company_name, "Microsoft Corporation");
    assert_eq!(msft.cik.to_string(), "0000789019");
}

#[test]
fn test_parse_company_tickers_json_empty() {
    let data = json!({});
    let tickers = parse_company_tickers_json(&data).unwrap();
    assert!(tickers.is_empty());
}

#[test]
fn test_parse_company_tickers_json_non_object() {
    let data = json!("not an object");
    let tickers = parse_company_tickers_json(&data).unwrap();
    assert!(tickers.is_empty(), "Non-object should yield an empty vec");
}

#[test]
fn test_parse_company_tickers_json_handles_missing_fields() {
    let data = json!({
        "0": {"cik_str": 320193, "ticker": "AAPL"},
        "1": {"cik_str": 789019, "title": "Microsoft Corporation"}
    });

    let tickers = parse_company_tickers_json(&data).unwrap();
    // Entry 0 has no title -> empty company_name
    // Entry 1 has no ticker -> empty symbol
    assert_eq!(tickers.len(), 2);
    assert_eq!(tickers[0].company_name, "");
    assert_eq!(tickers[1].symbol.to_string(), "");
}

// ---------------------------------------------------------------------------
// parse_ticker_txt
// ---------------------------------------------------------------------------

#[test]
fn test_parse_ticker_txt_basic() {
    let text = "AAPL\t320193\nMSFT\t789019\nGOOGL\t1652044\n";
    let tickers = parse_ticker_txt(text);
    assert_eq!(tickers.len(), 3);
    assert_eq!(tickers[0].symbol.to_string(), "AAPL");
    assert_eq!(tickers[0].cik.to_string(), "0000320193");
    assert_eq!(tickers[0].company_name, "");
}

#[test]
fn test_parse_ticker_txt_empty() {
    let tickers = parse_ticker_txt("");
    assert!(tickers.is_empty());
}

#[test]
fn test_parse_ticker_txt_skips_malformed_lines() {
    let text = "AAPL\t320193\nBAD_LINE\nMSFT\t789019\n\t\nNO_CIK\t\n";
    let tickers = parse_ticker_txt(text);
    assert_eq!(
        tickers.len(),
        2,
        "Should skip lines without tab, empty lines, and lines with missing CIK"
    );
    assert_eq!(tickers[0].symbol.to_string(), "AAPL");
    assert_eq!(tickers[1].symbol.to_string(), "MSFT");
}

#[test]
fn test_parse_ticker_txt_skips_invalid_cik() {
    let text = "AAPL\tnot-a-number\nMSFT\t789019\n";
    let tickers = parse_ticker_txt(text);
    assert_eq!(tickers.len(), 1);
    assert_eq!(tickers[0].symbol.to_string(), "MSFT");
}

#[test]
fn test_parse_ticker_txt_handles_whitespace() {
    let text = "  AAPL  \t  320193  \n";
    let tickers = parse_ticker_txt(text);
    assert_eq!(tickers.len(), 1);
    assert_eq!(tickers[0].symbol.to_string(), "AAPL");
    assert_eq!(tickers[0].cik.to_string(), "0000320193");
}

#[test]
fn test_parse_ticker_txt_handles_cik_overflow() {
    // CIK > 10 digits should be rejected by Cik::from_u64
    let text = "TEST\t12345678901\n";
    let tickers = parse_ticker_txt(text);
    assert!(tickers.is_empty(), "CIK > 10 digits should be rejected");
}

// ---------------------------------------------------------------------------
// parse_master_idx
// ---------------------------------------------------------------------------

#[test]
fn test_parse_master_idx_basic() {
    let text = "Description line\nCIK|Company Name|Form Type|Date Filed|Filename\n\
                --------------------------------------------------------------------------------\n\
                320193|APPLE INC|10-K|2024-11-01|edgar/data/320193/1234/primary-document.htm\n\
                789019|MICROSOFT CORP|10-Q|2024-10-29|edgar/data/789019/5678/primary-document.htm\n";

    let entries = parse_master_idx(text).unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].company_name, "APPLE INC");
    assert_eq!(entries[0].form_type, "10-K");
    assert_eq!(entries[0].date_filed.to_string(), "2024-11-01");
    assert_eq!(
        entries[0].filename,
        "edgar/data/320193/1234/primary-document.htm"
    );

    assert_eq!(entries[1].company_name, "MICROSOFT CORP");
    assert_eq!(entries[1].form_type, "10-Q");
    assert_eq!(entries[1].date_filed.to_string(), "2024-10-29");
}

#[test]
fn test_parse_master_idx_empty_after_header() {
    let text = "CIK|Company Name|Form Type|Date Filed|Filename\n\
                --------------------------------------------------------------------------------\n";
    let entries = parse_master_idx(text).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn test_parse_master_idx_skips_malformed_rows() {
    let text = "CIK|Company Name|Form Type|Date Filed|Filename\n\
                --------------------------------------------------------------------------------\n\
                320193|APPLE INC|10-K|2024-11-01|edgar/data/320193/1234/primary-document.htm\n\
                only|two|fields\n\
                789019|MICROSOFT CORP|10-Q|2024-10-29|edgar/data/789019/5678/primary-document.htm\n";

    let entries = parse_master_idx(text).unwrap();
    assert_eq!(entries.len(), 2, "Should skip rows with wrong column count");
}

#[test]
fn test_parse_master_idx_skips_bad_date() {
    let text = "CIK|Company Name|Form Type|Date Filed|Filename\n\
                --------------------------------------------------------------------------------\n\
                320193|APPLE INC|10-K|not-a-date|edgar/data/320193/1234/primary-document.htm\n";

    let entries = parse_master_idx(text).unwrap();
    assert!(
        entries.is_empty(),
        "Should skip rows with unparseable dates"
    );
}

#[test]
fn test_parse_master_idx_no_header_dashes() {
    // If there's no separator row of dashes, nothing gets parsed
    let text = "CIK|Company Name|Form Type|Date Filed|Filename\n\
                320193|APPLE INC|10-K|2024-11-01|edgar/data/320193/1234/primary-document.htm\n";

    let entries = parse_master_idx(text).unwrap();
    assert!(entries.is_empty(), "No dash separator means no data rows");
}

#[test]
fn test_parse_master_idx_handles_empty_lines() {
    let text = "CIK|Company Name|Form Type|Date Filed|Filename\n\
                --------------------------------------------------------------------------------\n\
                320193|APPLE INC|10-K|2024-11-01|edgar/data/320193/1234/primary-document.htm\n\
                \n\
                789019|MICROSOFT CORP|10-Q|2024-10-29|edgar/data/789019/5678/primary-document.htm\n";

    let entries = parse_master_idx(text).unwrap();
    assert_eq!(
        entries.len(),
        2,
        "Empty lines between data should be handled"
    );
}
