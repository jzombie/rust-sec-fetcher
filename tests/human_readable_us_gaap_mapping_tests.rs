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

#[test]
fn test_benefits_costs_expenses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("BenefitsLossesAndExpenses"),
        Some(vec!["BenefitsCostsExpenses", "CostsAndExpenses"])
    );
}

#[test]
fn test_commitments_and_contingencies() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("CommitmentsAndContingencies"),
        Some(vec!["CommitmentsAndContingencies"])
    );
}

#[test]
fn test_comprehensive_income_loss() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest"
        ),
        Some(vec!["ComprehensiveIncomeLoss"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("ComprehensiveIncomeNetOfTax"),
        Some(vec![
            "ComprehensiveIncomeLoss",
            "ComprehensiveIncomeLossAttributableToParent"
        ])
    );
}

#[test]
fn test_comprehensive_income_loss_attributable_to_noncontrolling_interest() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "ComprehensiveIncomeNetOfTaxAttributableToNoncontrollingInterest"
        ),
        Some(vec![
            "ComprehensiveIncomeLossAttributableToNoncontrollingInterest"
        ])
    );
}

#[test]
fn test_comprehensive_income_loss_attributable_to_parent() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("ComprehensiveIncomeNetOfTax"),
        Some(vec![
            "ComprehensiveIncomeLoss",
            "ComprehensiveIncomeLossAttributableToParent",
        ])
    );
}

#[test]
fn test_cost_of_revenue() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("CostOfRevenue"),
        Some(vec!["CostOfRevenue"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CostOfGoodsAndServicesSold"),
        Some(vec!["CostOfRevenue"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CostOfServices"),
        Some(vec!["CostOfRevenue"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CostOfGoodsSold"),
        Some(vec!["CostOfRevenue"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization"
        ),
        Some(vec!["CostOfRevenue"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CostOfGoodsSoldElectric"),
        Some(vec!["CostOfRevenue"])
    );
}

#[test]
fn test_costs_and_expenses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("CostsAndExpenses"),
        Some(vec!["CostsAndExpenses"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("BenefitsLossesAndExpenses"),
        Some(vec!["BenefitsCostsExpenses", "CostsAndExpenses"])
    );
}

#[test]
fn test_current_assets() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("AssetsCurrent"),
        Some(vec!["Assets", "CurrentAssets"])
    );
}
