use serde::{Deserialize, Serialize};

/// A normalized SEC ticker symbol.
///
/// Normalization rules (applied once on construction):
/// 1. Trim leading/trailing ASCII whitespace.
/// 2. Convert to uppercase.
/// 3. Replace `.` and `/` with `-` (e.g. `BRK.B` → `BRK-B`, `BRK/B` → `BRK-B`).
///
/// All SEC data processing — parsing, storage, and lookup — uses this type
/// so that `BRK.B`, `BRK/B`, `brk-b`, and `BRK-B` all resolve to the same
/// canonical key `"BRK-B"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct TickerSymbol(String);

impl TickerSymbol {
    /// Create a normalized `TickerSymbol` from any string-like value.
    pub fn new(s: &str) -> Self {
        TickerSymbol(s.trim().to_uppercase().replace(['.', '/'], "-"))
    }

    /// Returns the normalized symbol as a `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TickerSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::ops::Deref for TickerSymbol {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for TickerSymbol {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TickerSymbol {
    fn from(s: &str) -> Self {
        TickerSymbol::new(s)
    }
}

impl From<String> for TickerSymbol {
    fn from(s: String) -> Self {
        TickerSymbol::new(&s)
    }
}

/// Custom deserialization always normalizes the raw string value.
/// This ensures cached data round-trips correctly even if the source
/// stored un-normalized symbols.
impl<'de> Deserialize<'de> for TickerSymbol {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(TickerSymbol::new(&s))
    }
}
