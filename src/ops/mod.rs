//! High-level operations over SEC EDGAR data.
//!
//! This module provides composed, orchestrated operations built on top of the
//! lower-level [`crate::network`] fetch functions.  Each operation encapsulates
//! a multi-step workflow — deduplication, multi-form-type retrieval, rendering
//! pipelines, portfolio position normalisation — so that callers (examples,
//! CLIs, applications) can work at a higher level of abstraction.
//!
//! # Module layout
//!
//! | Sub-module | Purpose |
//! |---|---|
//! | [`filing`] | Filing body + exhibit rendering pipeline |
//! | [`holdings`] | Portfolio position normalisation and diff (N-PORT / 13F) |
//! | [`ipo`] | IPO registration filing timeline and feed polling (S-1 / F-1 family) |
//!
//! # Naming conventions
//!
//! Functions in this module follow a `get_*` / `render_*` convention:
//!
//! - `get_*` — retrieves, combines, and/or processes data, returning domain
//!   types ready for display or further analysis.
//! - `render_*` — fetches remote documents and converts them to rendered text.
//!
//! Low-level `fetch_*` functions remain in [`crate::network`].
//!
//! Form-type group constants (e.g. `IPO_REGISTRATION_FORM_TYPES`) are defined
//! as associated constants on [`crate::enums::FormType`], placing them adjacent
//! to the enum they describe.

pub mod filing;
pub mod holdings;
pub mod ipo;

pub use filing::{
    RenderedExhibit, RenderedFiling, render_all_exhibits, render_exhibit_doc, render_filing,
};
pub use holdings::{
    Diff, Position, WEIGHT_CHANGE_THRESHOLD, diff_holdings, positions_from_13f,
    positions_from_nport,
};
pub use ipo::{get_ipo_feed_entries, get_ipo_registration_filings};
