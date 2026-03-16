use crate::models::Form4Transaction;
use chrono::NaiveDate;
use quick_xml::events::Event;
use quick_xml::Reader;
use rust_decimal::Decimal;
use std::error::Error;
use std::str::FromStr;

/// Parses a Form 4 (or Form 4/A) XML document and returns one
/// [`Form4Transaction`] per row in the `<nonDerivativeTable>` and
/// `<derivativeTable>` sections, sorted by `transaction_date` descending.
///
/// Fetch the XML via [`crate::network::fetch_form4_filing`].
pub fn parse_form4_xml(
    xml: &str,
    filing_date: Option<NaiveDate>,
) -> Result<Vec<Form4Transaction>, Box<dyn Error>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    // ── Filer state (one per document) ───────────────────────────────────────
    let mut filer_name = String::new();
    let mut filer_cik = String::new();
    let mut is_director = false;
    let mut is_officer = false;
    let mut officer_title: Option<String> = None;
    let mut is_ten_pct_owner = false;

    // ── Transaction row state (reset per transaction) ─────────────────────────
    let mut security_title = String::new();
    let mut transaction_date_str = String::new();
    let mut transaction_code = String::new();
    let mut shares_str = String::new();
    let mut price_str = String::new();
    let mut acq_disp = String::new();
    let mut shares_after_str = String::new();

    let mut transactions: Vec<Form4Transaction> = Vec::new();
    let mut tag_stack: Vec<String> = Vec::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let local = local_name_str(e.name().as_ref());
                match local.as_str() {
                    "nonDerivativeTransaction" | "derivativeTransaction" => {
                        reset_txn(
                            &mut security_title,
                            &mut transaction_date_str,
                            &mut transaction_code,
                            &mut shares_str,
                            &mut price_str,
                            &mut acq_disp,
                            &mut shares_after_str,
                        );
                    }
                    _ => {}
                }
                tag_stack.push(local);
            }

            Ok(Event::Text(ref e)) => {
                let text = e.unescape().unwrap_or_default().trim().to_string();
                if text.is_empty() {
                    buf.clear();
                    continue;
                }
                let current = tag_stack.last().map(|s| s.as_str()).unwrap_or("");
                let parent = tag_stack
                    .len()
                    .checked_sub(2)
                    .and_then(|i| tag_stack.get(i))
                    .map(|s| s.as_str())
                    .unwrap_or("");

                // Fields that wrap their text in a <value> child element
                // form an exhaustive else branch so `text` is moved exactly once.
                if current == "value" {
                    match parent {
                        "securityTitle" => security_title = text,
                        "transactionDate" => transaction_date_str = text,
                        "transactionShares" => shares_str = text,
                        "transactionPricePerShare" => price_str = text,
                        "transactionAcquiredDisposedCode" => acq_disp = text,
                        "sharesOwnedFollowingTransaction" => shares_after_str = text,
                        _ => {}
                    }
                } else {
                    // Fields with direct text (no <value> wrapper).
                    match current {
                        "transactionCode" => transaction_code = text,
                        "rptOwnerName" => filer_name = text,
                        "rptOwnerCik" => filer_cik = text,
                        "officerTitle" => officer_title = Some(text),
                        "isDirector" => is_director = text == "1",
                        "isOfficer" => is_officer = text == "1",
                        "is10PercentOwner" => is_ten_pct_owner = text == "1",
                        _ => {}
                    }
                }
            }

            Ok(Event::End(ref e)) => {
                let local = local_name_str(e.name().as_ref());
                match local.as_str() {
                    "nonDerivativeTransaction" | "derivativeTransaction" => {
                        let is_derivative = local == "derivativeTransaction";
                        transactions.push(Form4Transaction {
                            filer_name: filer_name.clone(),
                            filer_cik: filer_cik.clone(),
                            is_director,
                            is_officer,
                            officer_title: officer_title.clone(),
                            is_ten_pct_owner,
                            filing_date,
                            security_title: std::mem::take(&mut security_title),
                            transaction_date: NaiveDate::parse_from_str(
                                &transaction_date_str,
                                "%Y-%m-%d",
                            )
                            .ok(),
                            transaction_code: std::mem::take(&mut transaction_code),
                            shares: Decimal::from_str(&shares_str).unwrap_or_default(),
                            price_per_share: Decimal::from_str(&price_str)
                                .ok()
                                .filter(|d| !d.is_zero()),
                            acquired_disposed: std::mem::take(&mut acq_disp),
                            shares_owned_after: Decimal::from_str(&shares_after_str).ok(),
                            is_derivative,
                        });
                        // Clear row state so stale values don't bleed into the next row.
                        reset_txn(
                            &mut security_title,
                            &mut transaction_date_str,
                            &mut transaction_code,
                            &mut shares_str,
                            &mut price_str,
                            &mut acq_disp,
                            &mut shares_after_str,
                        );
                    }
                    _ => {}
                }
                tag_stack.pop();
            }

            Ok(Event::Eof) => break,
            Err(e) => return Err(Box::new(e)),
            _ => {}
        }
        buf.clear();
    }

    // Sort newest transaction first.
    transactions.sort_by(|a, b| b.transaction_date.cmp(&a.transaction_date));
    Ok(transactions)
}

fn reset_txn(
    security_title: &mut String,
    transaction_date_str: &mut String,
    transaction_code: &mut String,
    shares_str: &mut String,
    price_str: &mut String,
    acq_disp: &mut String,
    shares_after_str: &mut String,
) {
    security_title.clear();
    transaction_date_str.clear();
    transaction_code.clear();
    shares_str.clear();
    price_str.clear();
    acq_disp.clear();
    shares_after_str.clear();
}

fn local_name_str(name: &[u8]) -> String {
    let s = std::str::from_utf8(name).unwrap_or("");
    s.rfind(':')
        .map(|i| s[i + 1..].to_string())
        .unwrap_or_else(|| s.to_string())
}
