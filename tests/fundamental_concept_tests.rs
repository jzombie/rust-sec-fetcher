/// Unit tests for [`sec_fetcher::enums::FundamentalConcept`].
///
/// This is a 67-variant enum backed by `strum` derives
/// (`EnumString`, `EnumIter`, `Display`).  These tests verify that every
/// variant round-trips through the strum derive macros.
use sec_fetcher::enums::FundamentalConcept;
use std::str::FromStr;

/// Verifies that every variant round-trips through Display and FromStr.
#[test]
fn test_fundamental_concept_round_trips() {
    use strum::IntoEnumIterator;
    for concept in FundamentalConcept::iter() {
        let display = concept.to_string();
        let parsed = FundamentalConcept::from_str(&display).unwrap_or_else(|_| {
            panic!("Failed to parse FundamentalConcept from '{display}'")
        });
        assert_eq!(concept, parsed, "Round-trip failed for variant {concept}");
    }
}

/// Verifies that specific well-known concepts parse correctly.
#[test]
fn test_fundamental_concept_known_variants() {
    assert_eq!(
        FundamentalConcept::from_str("Assets").unwrap(),
        FundamentalConcept::Assets
    );
    assert_eq!(
        FundamentalConcept::from_str("Revenues").unwrap(),
        FundamentalConcept::Revenues
    );
    assert_eq!(
        FundamentalConcept::from_str("NetIncomeLoss").unwrap(),
        FundamentalConcept::NetIncomeLoss
    );
    assert_eq!(
        FundamentalConcept::from_str("OperatingIncomeLoss").unwrap(),
        FundamentalConcept::OperatingIncomeLoss
    );
}

/// Verifies that an unknown string returns an error.
#[test]
fn test_fundamental_concept_invalid_variant() {
    let result = FundamentalConcept::from_str("NonExistentConcept");
    assert!(result.is_err(), "Expected Err for unknown variant");
}

/// Verifies that parsing is case-sensitive (as expected from strum EnumString).
#[test]
fn test_fundamental_concept_case_sensitive() {
    let result = FundamentalConcept::from_str("assets");
    assert!(result.is_err(), "strum EnumString is case-sensitive by default");
}

/// Verifies Debug output for a few variants.
#[test]
fn test_fundamental_concept_debug() {
    let debug = format!("{:?}", FundamentalConcept::Assets);
    assert_eq!(debug, "Assets");
    let debug = format!("{:?}", FundamentalConcept::NetIncomeLoss);
    assert_eq!(debug, "NetIncomeLoss");
}

/// Verifies Clone and Eq semantics.
#[test]
fn test_fundamental_concept_clone_eq() {
    let a = FundamentalConcept::Assets;
    let b = FundamentalConcept::Assets;
    let c = FundamentalConcept::Revenues;
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(a.clone(), b);
}
