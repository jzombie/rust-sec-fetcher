use crate::enums::Url;
use crate::models::{AccessionNumber, Cik};
use chrono::NaiveDate;

#[derive(Clone, Debug)]
pub struct CikSubmission {
    pub cik: Cik,
    // TODO: Add these fields (and more), but not per submission; capture as a single separate entity
    // pub name: Option<String>,            // i.e. "Apple"
    pub entity_type: Option<String>, // i.e. "operating"
    // pub sic: Option<u64>,                // i.e. 3571
    // pub sic_description: Option<String>, // i.e. "Electronic Computers"
    // pub owner_org: Option<String>,       // i.e. "06 Technology"
    // insiderTransactionForOwnerExists
    // insiderTransactionForIssuerExists
    pub accession_number: AccessionNumber,
    pub form: String,
    pub primary_document: String,
    pub filing_date: Option<NaiveDate>,
    /// SEC 8-K item tags reported with this filing, e.g. `["2.02", "9.01"]`.
    /// Empty for non-8-K forms. Item 9.01 ("Financial Statements and Exhibits")
    /// is present on nearly every 8-K and is not meaningful for routing.
    ///
    /// # Item numbering schema history
    ///
    /// The SEC overhauled the 8-K disclosure framework in **August 2004**.
    ///
    /// - **Pre-2004 (before Aug 2004):** Items were plain integers `1` through `13`.
    ///   Filings from this era will have items like `"5"`, `"7"`, `"12"` in this
    ///   field. The old Item 12 ("Results of Operations and Financial Condition")
    ///   is the direct predecessor of the modern Item 2.02 and should be treated
    ///   identically for earnings-pipeline routing.
    ///
    /// - **Post-2004 (Aug 2004 onwards):** Items use the current dotted
    ///   "Section.Item" format, e.g. `"2.02"`, `"5.02"`, `"9.01"`.
    ///
    /// When processing historical filings, check for both the old integer form
    /// and the modern dotted form — see `is_earnings_release()` for an example.
    // TODO: Add full pre-2004 item 1–13 definitions and cross-walk table here.
    pub items: Vec<String>,
}

impl CikSubmission {
    pub fn filter_nport_p_submissions(cik_submissions: &[Self]) -> Vec<Self> {
        cik_submissions
            .iter()
            .filter(|submission| submission.form.to_uppercase() == "NPORT-P")
            .cloned()
            .collect()
    }

    pub fn most_recent_nport_p_submission(cik_submissions: &[Self]) -> Option<Self> {
        let nport_p_submissions = Self::filter_nport_p_submissions(cik_submissions);

        nport_p_submissions.first().cloned()
    }

    pub fn filter_8k_submissions(cik_submissions: &[Self]) -> Vec<Self> {
        cik_submissions
            .iter()
            .filter(|submission| submission.form.to_uppercase() == "8-K")
            .cloned()
            .collect()
    }

    /// Returns the most recent 10-K (or 10-K405) submission, if any.
    ///
    /// The list returned by [`crate::network::fetch_cik_submissions`] is
    /// already ordered newest-first, so the first match is the latest annual
    /// report.
    pub fn most_recent_10k(cik_submissions: &[Self]) -> Option<&Self> {
        cik_submissions.iter().find(|s| {
            let f = s.form.to_uppercase();
            f == "10-K" || f == "10-K405"
        })
    }

    /// Returns `true` if this 8-K is an **earnings release** (Item 2.02).
    ///
    /// Also matches the pre-August-2004 equivalent: old **Item 12**
    /// ("Results of Operations and Financial Condition"), which was the
    /// predecessor to 2.02 before the SEC's numbering overhaul.
    ///
    /// # Triplet training — Path 1: Earnings Pipeline
    ///
    /// **Anchor:**   The EX-99.1 press release attached to this 8-K
    ///               (fetch via `fetch_filing_index` → `exhibits()`).
    ///
    /// **Positive:** The earnings call transcript for the *same* quarter.
    ///               Transcripts are published by services such as Motley Fool,
    ///               Seeking Alpha, or The Motley Fool Transcribing team within
    ///               hours of the call. The filing date on this 8-K is your
    ///               temporal anchor — the transcript will share the same date.
    ///               Semantically: both documents describe the same financial
    ///               results in complementary registers (legal/tabular vs.
    ///               executive/analyst Q&A).
    ///
    /// **Negative:** The earnings call transcript from the *previous* quarter.
    ///               Same company, same document type, but different period —
    ///               makes the model learn period-specificity rather than just
    ///               company-identity. Use the immediately preceding `2.02`
    ///               filing's transcript.
    pub fn is_earnings_release(&self) -> bool {
        // "2.02" = post-Aug-2004 form; "12" = pre-Aug-2004 form (old Item 12)
        self.items.iter().any(|item| item == "2.02" || item == "12")
    }

    /// Returns `true` if this 8-K is a **mid-quarter event** — everything
    /// *except* an earnings release.
    ///
    /// # Triplet training — Path 2: Mid-Quarter Event Pipeline
    ///
    /// ## Anchor → Positive → Negative
    ///
    /// **Anchor:**   The primary 8-K document itself (the dense legal disclosure).
    ///               Fetch via `as_primary_document_url()`.
    ///
    /// **Positive:** A CNBC/Reuters/Bloomberg news article that covers *this
    ///               specific event* — same company, same date window (±2 days),
    ///               and explicitly references the disclosed event. This maps the
    ///               legal register of the 8-K to the journalistic natural-language
    ///               description of the same event. Filtering by item type helps
    ///               match article topics:
    ///               - `1.01` / `2.01` → M&A, partnership, or contract articles
    ///               - `5.02`          → executive appointment/departure articles
    ///               - `5.07`          → shareholder meeting / vote articles
    ///               - `8.01`          → product approval, government deal, etc.
    ///
    /// **Negative (easy):**  An article about a *different* event for the same
    ///               company — e.g. an M&A article used as the negative for a
    ///               5.02 (executive change) anchor. Same company, different
    ///               event semantics.
    ///
    /// **Negative (hard):**  The *next* earnings call transcript for the same
    ///               company. This proves to the model that an earnings call is
    ///               fundamentally different in structure and content from a
    ///               mid-quarter event disclosure, even though both discuss the
    ///               same company.
    ///
    /// ## 10-K / 10-Q as structural context ("Investigator" vs. "Historian")
    ///
    /// When pairing an 8-K with a periodic filing (10-K or 10-Q) for structural
    /// analysis, always use the **most recent filing before the 8-K date**, never
    /// the next one.
    ///
    /// **Why the previous filing — the "Investigator" model:**
    ///   The previous 10-K is the only map of the company that exists at the
    ///   moment the 8-K drops. It describes the current structure — warehouses,
    ///   suppliers, risk factors, critical materials, concentration of revenue.
    ///   Training on (8-K, previous 10-K) teaches the model to ask:
    ///   *"What part of the existing structure does this event break?"*
    ///   This leads to discovering **clusters** — e.g. an 8-K disclosing a banned
    ///   chemical can be matched against every other company that listed that same
    ///   chemical as a "Critical Raw Material" in their most recent 10-K, surfacing
    ///   all 20+ at-risk companies the instant the 8-K is filed.
    ///
    /// **Why NOT the next filing — the "Historian" trap:**
    ///   The next 10-Q does not exist at the time of the 8-K. Training on it
    ///   teaches the model to correlate events with outcomes (e.g. "warehouse
    ///   fire → $200M loss"), not with structure. The model learns price
    ///   correlation, not structural risk — and it can only do so in hindsight,
    ///   months after the opportunity has passed.
    ///
    /// **In practice:**
    ///   Given an 8-K filing date, find the most recent 10-K or 10-Q filed by
    ///   the same CIK *before* that date. Use its full text as the structural
    ///   context document. The 8-K is the signal; the previous 10-K is the map.
    ///
    /// Common mid-quarter items:
    /// - `1.01` — Entry into a Material Definitive Agreement (contracts, M&A)
    /// - `1.02` — Termination of a Material Agreement
    /// - `2.01` — Completion of Acquisition or Disposition
    /// - `5.02` — Departure/Election of Directors or Officers
    /// - `5.07` — Submission of Matters to a Vote of Security Holders
    /// - `7.01` — Regulation FD Disclosure
    /// - `8.01` — Other Events
    ///
    /// Item 9.01 (Financial Statements and Exhibits) is excluded from the check
    /// because it appears on nearly every 8-K and carries no routing meaning.
    pub fn is_mid_quarter_event(&self) -> bool {
        !self.is_earnings_release()
            && self.form.to_uppercase() == "8-K"
            && self
                .items
                .iter()
                .any(|item| item != "9.01" && !item.is_empty())
    }

    /// Returns the meaningful (non-9.01) item tags for this filing.
    pub fn significant_items(&self) -> Vec<&str> {
        self.items
            .iter()
            .filter(|item| item.as_str() != "9.01" && !item.is_empty())
            .map(|s| s.as_str())
            .collect()
    }

    pub fn as_edgar_archive_url(&self) -> String {
        Url::CikAccession(self.cik.clone(), self.accession_number.clone()).value()
    }

    /// Returns the URL of the primary document for this filing.
    ///
    /// This points directly to the main document (e.g., the HTML 8-K or 10-K body),
    /// rather than the filing index directory.
    pub fn as_primary_document_url(&self) -> String {
        format!("{}/{}", self.as_edgar_archive_url(), self.primary_document)
    }
}
