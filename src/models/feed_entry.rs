use crate::models::Cik;
use chrono::{DateTime, FixedOffset, NaiveDate};

/// A single entry from an SEC EDGAR filing Atom feed.
///
/// Parsed from the global "current filings" feed or a per-company feed.
/// Fetch via [`crate::network::fetch_edgar_feed`].
///
/// # Event-driven / delta polling
///
/// Entries are returned **newest-first** by EDGAR. [`FeedEntry::updated`] is
/// the EDGAR acceptance timestamp (ISO 8601 with UTC offset). Use it as a
/// **high-water mark**:
///
/// 1. After each poll, store the `updated` value of the first (newest) entry.
/// 2. On the next poll, iterate entries and stop when
///    `entry.updated <= last_stored_updated`.
/// 3. Everything before that boundary is the **delta** — filings that are new
///    since your last check.
///
/// EDGAR caps the feed at 40 entries per request. Polling every few minutes
/// is sufficient to stay current with the normal filing rate. If you suspect
/// high volume (e.g. earnings season), reduce your poll interval.
#[derive(Debug, Clone)]
pub struct FeedEntry {
    /// Accession number string (e.g. `"0001104659-26-027766"`).
    /// Use with [`crate::models::AccessionNumber::from_str`] if you need the
    /// structured form, or pass directly to [`crate::network::fetch_filing_index`]
    /// via a [`crate::models::CikSubmission`].
    pub accession_number: String,

    /// CIK of the filing entity, extracted from the filing index URL.
    /// `None` only if the URL is malformed (should not happen in practice).
    pub cik: Option<Cik>,

    /// Company (or person) name as it appears in the feed title.
    pub company_name: String,

    /// SEC form type (e.g. `"8-K"`, `"10-K"`, `"4"`, `"SC 13G"`).
    pub form_type: String,

    /// Date the filing was filed (from the feed summary).
    pub filing_date: Option<NaiveDate>,

    /// URL to the filing's EDGAR index page (`-index.htm`).
    /// Pass this directly to a browser or use it to construct document URLs.
    pub filing_href: String,

    /// EDGAR acceptance timestamp.
    ///
    /// Use this as your high-water mark for delta polling — see the type-level
    /// docs above. Pass it to [`crate::network::fetch_edgar_feed_since`] as
    /// `since` to receive only entries filed after this point.
    pub updated: DateTime<FixedOffset>,

    /// 8-K item codes parsed from the feed summary (e.g. `["1.01", "9.01"]`).
    /// Empty for non-8-K forms.
    pub items: Vec<String>,
}

impl FeedEntry {
    /// Returns `true` if this is an earnings release (8-K Item 2.02 or
    /// legacy pre-2004 Item 12).
    pub fn is_earnings_release(&self) -> bool {
        self.items.iter().any(|i| i == "2.02" || i == "12")
    }

    /// Returns `true` if this is a mid-quarter event (non-earnings 8-K with
    /// at least one meaningful item tag other than 9.01).
    pub fn is_mid_quarter_event(&self) -> bool {
        !self.is_earnings_release()
            && self.form_type.to_uppercase() == "8-K"
            && self.items.iter().any(|i| i != "9.01" && !i.is_empty())
    }

    /// Converts [`FeedEntry::updated`] to EDGAR's `dateb` format (`"YYYYMMDDHHmmss"`).
    ///
    /// Pass the result to [`crate::network::fetch_edgar_feed_page`] as the
    /// `before` argument to fetch the next page of older entries.
    pub fn updated_as_dateb(&self) -> String {
        self.updated.format("%Y%m%d%H%M%S").to_string()
    }
}
