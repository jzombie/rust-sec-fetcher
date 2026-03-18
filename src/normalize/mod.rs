//! # Canonical value normalization for SEC EDGAR data
//!
//! All numeric values ingested from EDGAR XML that require **any** scale
//! conversion, unit adjustment, or derived computation before storage pass
//! through this module.  There are **no inline conversions elsewhere** in this
//! codebase — the parsers call functions defined here and store whatever those
//! functions return.
//!
//! ## Why a single module?
//!
//! SEC EDGAR has accumulated inconsistencies across filing types and schema
//! versions (different units, different scales, era cutoffs).  Scattering the
//! conversion logic across parsers and op layers makes each inconsistency
//! invisible, duplicatable, and untestable in isolation.  This module is the
//! single place where every conversion decision is recorded, cited, and tested.
//!
//! ## Modules
//!
//! | Module | Responsibility |
//! |--------|----------------|
//! | [`thirteenf`] | 13F-HR `<value>` unit crossover (thousands-era vs. actual-USD era) and weight computation |
//! | [`pct`] | [`Pct`] — scale-enforcing (0–100), unbounded percentage newtype |
//!
//! ## Adding a new filing type
//!
//! If you add a parser for a new SEC form that contains numeric fields
//! requiring any normalization, add the conversion logic here — do **not** put
//! it in the parser file or the ops layer.

pub mod pct;
pub mod thirteenf;

pub use pct::Pct;
pub use thirteenf::{
    compute_13f_weight_pct, is_13f_thousands_era, normalize_13f_value_usd,
    THIRTEENF_THOUSANDS_ERA_CUTOFF,
};
