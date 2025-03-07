use sec_fetcher::accessor::get_us_gaap_human_readable_mapping;
use std::collections::HashSet;

#[test]
fn test_assets() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("Assets").map(|v| v.into_iter().collect::<HashSet<_>>()),
        Some(HashSet::from(["Assets"]))
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("AssetsCurrent")
            .map(|v| v.into_iter().collect::<HashSet<_>>()),
        Some(HashSet::from(["Assets", "CurrentAssets",]))
    );
}

#[test]
fn test_benefits_costs_expenses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("BenefitsLossesAndExpenses")
            .map(|v| v.into_iter().collect::<HashSet<_>>()),
        Some(HashSet::from(["BenefitsCostsExpenses", "CostsAndExpenses"]))
    );
}

#[test]
fn test_commitments_and_contingencies() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("CommitmentsAndContingencies")
            .map(|v| v.into_iter().collect::<HashSet<_>>()),
        Some(HashSet::from(["CommitmentsAndContingencies"]))
    );
}

#[test]
fn test_comprehensive_income_loss() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest"
        )
        .map(|v| v.into_iter().collect::<HashSet<_>>()),
        Some(HashSet::from(["ComprehensiveIncomeLoss"]))
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("ComprehensiveIncomeNetOfTax")
            .map(|v| v.into_iter().collect::<HashSet<_>>()),
        Some(HashSet::from([
            "ComprehensiveIncomeLoss",
            "ComprehensiveIncomeLossAttributableToParent"
        ]))
    );
}

#[test]
fn test_comprehensive_income_loss_attributable_to_noncontrolling_interest() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "ComprehensiveIncomeNetOfTaxAttributableToNoncontrollingInterest"
        )
        .map(|v| v.into_iter().collect::<HashSet<_>>()),
        Some(HashSet::from([
            "ComprehensiveIncomeLossAttributableToNoncontrollingInterest"
        ]))
    );
}

#[test]
fn test_comprehensive_income_loss_attributable_to_parent() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("ComprehensiveIncomeNetOfTax")
            .map(|v| v.into_iter().collect::<HashSet<_>>()),
        Some(HashSet::from([
            "ComprehensiveIncomeLossAttributableToParent",
            "ComprehensiveIncomeLoss"
        ]))
    );
}
