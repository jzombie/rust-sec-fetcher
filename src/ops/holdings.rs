use crate::models::{NportInvestment, ThirteenfHolding};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

/// Minimum absolute weight change (percentage points) for a position to appear
/// in the [`Diff::changed`] list.
pub const WEIGHT_CHANGE_THRESHOLD: Decimal = dec!(0.10);

/// A normalised single-position snapshot.
///
/// Produced from both [`NportInvestment`] (N-PORT filings) and
/// [`ThirteenfHolding`] (13F filings) via [`positions_from_nport`] and
/// [`positions_from_13f`] respectively.  Used as the unit of comparison in
/// [`diff_holdings`].
#[derive(Clone)]
pub struct Position {
    pub cusip: String,
    pub name: String,
    /// Market value in USD.
    pub val_usd: Decimal,
    /// Portfolio weight as a percentage (0–100 scale).
    ///
    /// For N-PORT this comes directly from the `pct_val` field (converted from
    /// 0–1 to 0–100); for 13F it is derived from the position's share of the
    /// total reported value across all holdings.
    pub weight: Decimal,
}

/// Changes between two consecutive portfolio snapshots.
///
/// Produced by [`diff_holdings`].  All lists are sorted by absolute weight
/// change descending.
pub struct Diff {
    /// Positions present in `new` but absent in `old` (new buys).
    pub added: Vec<Position>,
    /// Positions present in `old` but absent in `new` (full sells).
    pub removed: Vec<Position>,
    /// Positions present in both where the absolute weight delta meets or
    /// exceeds [`WEIGHT_CHANGE_THRESHOLD`].  Each tuple is `(old, new)`.
    pub changed: Vec<(Position, Position)>,
}

/// Converts N-PORT investments into normalised [`Position`] snapshots.
pub fn positions_from_nport(investments: &[NportInvestment]) -> Vec<Position> {
    investments
        .iter()
        .map(|h| Position {
            cusip: h.cusip.clone(),
            name: h.name.clone(),
            val_usd: h.val_usd,
            // pct_val is already a percentage (0–100 scale) parsed directly
            // from the N-PORT <pctVal> XML element.
            weight: h.pct_val,
        })
        .collect()
}

/// Converts 13F holdings into normalised [`Position`] snapshots.
///
/// Portfolio weight is derived from each position's share of the total
/// reported value, since 13F filings do not include a percentage field.
pub fn positions_from_13f(holdings: &[ThirteenfHolding]) -> Vec<Position> {
    let total: Decimal = holdings.iter().map(|h| h.value_usd).sum();
    holdings
        .iter()
        .map(|h| Position {
            cusip: h.cusip.clone(),
            name: h.name.clone(),
            val_usd: h.value_usd,
            weight: if total.is_zero() {
                dec!(0)
            } else {
                (h.value_usd / total * dec!(100)).round_dp(4)
            },
        })
        .collect()
}

/// Computes the difference between two portfolio snapshots.
///
/// Positions are matched by CUSIP.  A position appears in [`Diff::changed`]
/// only when the absolute weight change is ≥ [`WEIGHT_CHANGE_THRESHOLD`].
///
/// The `changed` list is sorted by absolute weight change descending.
pub fn diff_holdings(old: &[Position], new: &[Position]) -> Diff {
    let old_map: HashMap<&str, &Position> = old.iter().map(|p| (p.cusip.as_str(), p)).collect();
    let new_map: HashMap<&str, &Position> = new.iter().map(|p| (p.cusip.as_str(), p)).collect();

    let added: Vec<Position> = new
        .iter()
        .filter(|p| !old_map.contains_key(p.cusip.as_str()))
        .cloned()
        .collect();

    let removed: Vec<Position> = old
        .iter()
        .filter(|p| !new_map.contains_key(p.cusip.as_str()))
        .cloned()
        .collect();

    let mut changed: Vec<(Position, Position)> = new
        .iter()
        .filter_map(|n| {
            old_map.get(n.cusip.as_str()).and_then(|o| {
                let delta = (n.weight - o.weight).abs();
                if delta >= WEIGHT_CHANGE_THRESHOLD {
                    Some(((*o).clone(), n.clone()))
                } else {
                    None
                }
            })
        })
        .collect();

    changed.sort_by(|(o1, n1), (o2, n2)| {
        let d1 = (n1.weight - o1.weight).abs();
        let d2 = (n2.weight - o2.weight).abs();
        d2.cmp(&d1)
    });

    Diff {
        added,
        removed,
        changed,
    }
}
