use chrono::NaiveDate;
use rust_decimal::Decimal;

/// One transaction row from a Form 4 (or Form 4/A) filing.
///
/// A single Form 4 XML can produce multiple `Form4Transaction` values —
/// one per row in either the `<nonDerivativeTable>` or `<derivativeTable>`.
/// Filer identity fields (`filer_name`, `is_officer`, etc.) are the same
/// across all rows from the same filing and are denormalised here for
/// convenience.
#[derive(Debug, Clone)]
pub struct Form4Transaction {
    // ── Filer identity (same for all rows in one filing) ─────────────────────
    /// Full name of the reporting insider (e.g. `"Cook Timothy D"`).
    pub filer_name: String,
    /// CIK of the reporting insider (not the issuing company).
    pub filer_cik: String,
    pub is_director: bool,
    pub is_officer: bool,
    /// Job title when `is_officer` is true (e.g. `"CEO"`, `"CFO"`).
    pub officer_title: Option<String>,
    pub is_ten_pct_owner: bool,
    /// Date the filing was accepted by EDGAR (from `CikSubmission`).
    pub filing_date: Option<NaiveDate>,

    // ── Transaction row ───────────────────────────────────────────────────────
    /// Human-readable security name (e.g. `"Common Stock"`).
    pub security_title: String,
    /// Date the transaction actually occurred (may differ from filing date).
    pub transaction_date: Option<NaiveDate>,
    /// SEC transaction type code.
    ///
    /// Common values:
    /// - `P` — open-market purchase
    /// - `S` — open-market sale
    /// - `A` — grant / award
    /// - `F` — tax withholding (shares surrendered to issuer)
    /// - `M` — exercise of derivative security
    /// - `G` — gift
    pub transaction_code: String,
    /// Number of shares or units transacted.
    pub shares: Decimal,
    /// Per-share price at time of transaction. `None` when not applicable
    /// (e.g. gifts, grants, vesting).
    pub price_per_share: Option<Decimal>,
    /// `"A"` = acquired, `"D"` = disposed.
    pub acquired_disposed: String,
    /// Shares owned by this insider after the transaction.
    pub shares_owned_after: Option<Decimal>,
    /// `true` when this row came from `<derivativeTable>` (options, RSUs, etc.)
    /// rather than `<nonDerivativeTable>` (direct share transactions).
    pub is_derivative: bool,
}

impl Form4Transaction {
    /// Returns a short human-readable description of the transaction code.
    pub fn code_description(&self) -> &str {
        match self.transaction_code.as_str() {
            "P" => "Purchase",
            "S" => "Sale",
            "A" => "Award/Grant",
            "F" => "Tax withholding",
            "M" => "Option exercise",
            "X" => "In-the-money exercise",
            "G" => "Gift",
            "D" => "Disposition to issuer",
            "V" => "Voluntary report",
            "J" => "Other",
            _   => "Transaction",
        }
    }
}
