use sec_fetcher::accessor::get_us_gaap_human_readable_mapping;

#[test]
fn test_assets() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("Assets"),
        Some(vec!["Assets"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("AssetsCurrent"),
        Some(vec!["Assets", "CurrentAssets"])
    );
}

// #[test]
// fn test_benefits_costs_expenses() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("BenefitsLossesAndExpenses")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["BenefitsCostsExpenses", "CostsAndExpenses"]))
//     );
// }

// #[test]
// fn test_commitments_and_contingencies() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CommitmentsAndContingencies")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["CommitmentsAndContingencies"]))
//     );
// }

// #[test]
// fn test_comprehensive_income_loss() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest"
//         )
//         .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["ComprehensiveIncomeLoss"]))
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ComprehensiveIncomeNetOfTax")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from([
//             "ComprehensiveIncomeLoss",
//             "ComprehensiveIncomeLossAttributableToParent"
//         ]))
//     );
// }

// #[test]
// fn test_comprehensive_income_loss_attributable_to_noncontrolling_interest() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "ComprehensiveIncomeNetOfTaxAttributableToNoncontrollingInterest"
//         )
//         .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from([
//             "ComprehensiveIncomeLossAttributableToNoncontrollingInterest"
//         ]))
//     );
// }

// #[test]
// fn test_comprehensive_income_loss_attributable_to_parent() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ComprehensiveIncomeNetOfTax")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from([
//             "ComprehensiveIncomeLossAttributableToParent",
//             "ComprehensiveIncomeLoss"
//         ]))
//     );
// }

// #[test]
// fn test_cost_of_revenue() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CostOfRevenue")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["CostOfRevenue", "CostsAndExpenses"]))
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CostOfGoodsAndServicesSold")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["CostOfRevenue"]))
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CostOfServices")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["CostOfRevenue"]))
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CostOfGoodsSold")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["CostOfRevenue"]))
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization"
//         )
//         .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["CostOfRevenue"]))
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CostOfGoodsSoldElectric")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["CostOfRevenue"]))
//     );
// }

// #[test]
// fn test_costs_and_expenses() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CostsAndExpenses")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["CostsAndExpenses"]))
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("BenefitsLossesAndExpenses")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["CostsAndExpenses", "BenefitsCostsExpenses"]))
//     );
// }

// #[test]
// fn test_current_assets() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("AssetsCurrent")
//             .map(|v| v.into_iter().collect::<HashSet<_>>()),
//         Some(HashSet::from(["CurrentAssets", "Assets"]))
//     );
// }
