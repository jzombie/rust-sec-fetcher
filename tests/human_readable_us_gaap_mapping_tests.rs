use sec_fetcher::{accessor::get_us_gaap_human_readable_mapping, enums::FundamentalConcept};

#[test]
fn test_assets() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("Assets"),
        Some(vec![FundamentalConcept::Assets])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("AssetsCurrent"),
        Some(vec![
            FundamentalConcept::CurrentAssets,
            FundamentalConcept::Assets
        ])
    );
}

#[test]
fn test_benefits_costs_expenses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("BenefitsLossesAndExpenses"),
        Some(vec![
            FundamentalConcept::BenefitsCostsExpenses,
            FundamentalConcept::CostsAndExpenses
        ])
    );
}

#[test]
fn test_commitments_and_contingencies() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("CommitmentsAndContingencies"),
        Some(vec![FundamentalConcept::CommitmentsAndContingencies])
    );
}

#[test]
fn test_comprehensive_income_loss() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest"
        ),
        Some(vec![FundamentalConcept::ComprehensiveIncomeLoss])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("ComprehensiveIncomeNetOfTax"),
        Some(vec![
            FundamentalConcept::ComprehensiveIncomeLossAttributableToParent,
            FundamentalConcept::ComprehensiveIncomeLoss,
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
            FundamentalConcept::ComprehensiveIncomeLossAttributableToNoncontrollingInterest
        ])
    );
}

#[test]
fn test_comprehensive_income_loss_attributable_to_parent() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("ComprehensiveIncomeNetOfTax"),
        Some(vec![
            FundamentalConcept::ComprehensiveIncomeLossAttributableToParent,
            FundamentalConcept::ComprehensiveIncomeLoss,
        ])
    );
}

#[test]
fn test_cost_of_revenue() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("CostOfRevenue"),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CostOfGoodsAndServicesSold"),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CostOfServices"),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CostOfGoodsSold"),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization"
        ),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CostOfGoodsSoldElectric"),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );
}

#[test]
fn test_costs_and_expenses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("CostsAndExpenses"),
        Some(vec![FundamentalConcept::CostsAndExpenses])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("BenefitsLossesAndExpenses"),
        Some(vec![
            FundamentalConcept::BenefitsCostsExpenses,
            FundamentalConcept::CostsAndExpenses
        ])
    );
}

#[test]
fn test_current_assets() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("AssetsCurrent"),
        Some(vec![
            FundamentalConcept::CurrentAssets,
            FundamentalConcept::Assets
        ])
    );
}

#[test]
fn test_current_liabilities() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("LiabilitiesCurrent"),
        Some(vec![FundamentalConcept::CurrentLiabilities])
    );
}

#[test]
fn test_equity() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        ),
        Some(vec![FundamentalConcept::Equity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("StockholdersEquity"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "PartnersCapitalIncludingPortionAttributableToNoncontrollingInterest"
        ),
        Some(vec![FundamentalConcept::Equity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("PartnersCapital"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CommonStockholdersEquity"),
        Some(vec![FundamentalConcept::Equity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MembersEquity"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );
}

#[test]
fn test_equity_attributable_to_noncontrolling_interest() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("MinorityInterest"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("PartnersCapitalAttributableToNoncontrollingInterest"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MinorityInterestInLimitedPartnerships"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MinorityInterestInOperatingPartnerships"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MinorityInterestInPreferredUnitHolders"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MinorityInterestInJointVentures"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("OtherMinorityInterests"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("NonredeemableNoncontrollingInterest"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("NoncontrollingInterestInVariableInterestEntity"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );
}

#[test]
fn test_equity_attributable_to_parent() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("StockholdersEquity"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("PartnersCapital"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MembersEquity"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );
}

#[test]
fn test_exchange_gains_losses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("EffectOfExchangeRateOnCashAndCashEquivalents"),
        Some(vec![FundamentalConcept::ExchangeGainsLosses])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "EffectOfExchangeRateOnCashAndCashEquivalentsContinuingOperations"
        ),
        Some(vec![FundamentalConcept::ExchangeGainsLosses])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("EffectOfExchangeRateOnCashContinuingOperations"),
        Some(vec![FundamentalConcept::ExchangeGainsLosses])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "EffectOfExchangeRateOnCashAndCashEquivalentsDiscontinuedOperations"
        ),
        Some(vec![FundamentalConcept::ExchangeGainsLosses])
    );
}

// #[test]
// fn test_extraordinary_items_of_income_expense_net_of_tax() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ExtraordinaryItemNetOfTax"),
//         Some(vec!["ExtraordinaryItemsOfIncomeExpenseNetOfTax"])
//     );
// }

// #[test]
// fn test_gain_loss_on_sale_properties_net_tax() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("GainLossOnSaleOfPropertiesNetOfApplicableIncomeTaxes"),
//         Some(vec!["GainLossOnSalePropertiesNetTax"])
//     );
// }

// #[test]
// fn test_gross_profit() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("GrossProfit"),
//         Some(vec!["GrossProfit"])
//     );
// }

// #[test]
// fn test_income_loss_before_equity_method_investments() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
//         Some(vec!["IncomeLossBeforeEquityMethodInvestments", "IncomeLossFromContinuingOperationsBeforeTax"])
//     );
// }

// #[test]
// fn test_income_loss_from_continuing_operations_after_tax() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"),
//         Some(vec!["IncomeLossFromContinuingOperationsAfterTax", "NetIncomeLoss"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple"
//         ),
//         Some(vec!["IncomeLossFromContinuingOperationsAfterTax"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperations"),
//         Some(vec![
//             "IncomeLossFromContinuingOperationsAfterTax",
//             "NetIncomeLoss"
//         ])
//     );
// }

// #[test]
// fn test_income_loss_from_continuing_operations_before_tax() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
//         Some(vec![
//             "IncomeLossFromContinuingOperationsBeforeTax",
//         ])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
//         Some(vec![
//             "IncomeLossBeforeEquityMethodInvestments", "IncomeLossFromContinuingOperationsBeforeTax",
//         ])
//     );
// }

// #[test]
// fn test_income_loss_from_discontinued_operations_net_of_tax() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeLossFromDiscontinuedOperationsNetOfTax"),
//         Some(vec!["IncomeLossFromDiscontinuedOperationsNetOfTax",])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "DiscontinuedOperationGainLossOnDisposalOfDiscontinuedOperationNetOfTax"
//         ),
//         Some(vec!["IncomeLossFromDiscontinuedOperationsNetOfTax",])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity"
//         ),
//         Some(vec!["IncomeLossFromDiscontinuedOperationsNetOfTax",])
//     );
// }

// #[test]
// fn test_income_loss_from_equity_method_investments() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeLossFromEquityMethodInvestments"),
//         Some(vec!["IncomeLossFromEquityMethodInvestments",])
//     );
// }

// #[test]
// fn test_income_tax_expense_benefit() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeTaxExpenseBenefit"),
//         Some(vec![
//             "IncomeStatementStartPeriodYearToDate",
//             "IncomeTaxExpenseBenefit"
//         ])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeTaxExpenseBenefitContinuingOperations"),
//         Some(vec![
//             "IncomeStatementStartPeriodYearToDate",
//             "IncomeTaxExpenseBenefit"
//         ])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("FederalHomeLoanBankAssessments"),
//         Some(vec![
//             "IncomeStatementStartPeriodYearToDate",
//             "IncomeTaxExpenseBenefit"
//         ])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CurrentIncomeTaxExpenseBenefit"),
//         Some(vec![
//             "IncomeStatementStartPeriodYearToDate",
//             "IncomeTaxExpenseBenefit"
//         ])
//     );
// }

// #[test]
// fn test_interest_and_debt_expense() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("InterestAndDebtExpense"),
//         Some(vec!["InterestAndDebtExpense"])
//     );
// }

// #[test]
// fn test_interest_and_divident_income_operating() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("InterestAndDividendIncomeOperating"),
//         Some(vec!["InterestAndDividendIncomeOperating", "Revenues"])
//     );
// }

// #[test]
// fn test_interest_income_expense_after_provision_for_losses() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("InterestIncomeExpenseAfterProvisionForLoanLoss"),
//         Some(vec!["InterestIncomeExpenseAfterProvisionForLosses"])
//     );
// }

// #[test]
// fn test_interest_income_expense_operating_net() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("InterestIncomeExpenseNet"),
//         Some(vec!["InterestIncomeExpenseOperatingNet"])
//     );
// }

// #[test]
// fn test_liabilities() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("Liabilities"),
//         Some(vec!["Liabilities"])
//     );
// }

// #[test]
// fn test_liabilities_and_equity() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("LiabilitiesAndStockholdersEquity"),
//         Some(vec!["LiabilitiesAndEquity"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("LiabilitiesAndPartnersCapital"),
//         Some(vec!["LiabilitiesAndEquity"])
//     );
// }

// #[test]
// fn test_nature_of_operations() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NatureOfOperations"),
//         Some(vec!["NatureOfOperations"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("BusinessDescriptionAndBasisOfPresentationTextBlock"),
//         Some(vec!["NatureOfOperations"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "OrganizationConsolidationAndPresentationOfFinancialStatementsDisclosureTextBlock"
//         ),
//         Some(vec!["NatureOfOperations"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("OrganizationConsolidationBasisOfPresentationBusinessDescriptionAndAccountingPoliciesTextBlock"),
//         Some(vec!["NatureOfOperations"])
//     );
// }

// #[test]
// fn test_net_cash_flow() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CashAndCashEquivalentsPeriodIncreaseDecrease"),
//         Some(vec!["NetCashFlow"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CashPeriodIncreaseDecrease"),
//         Some(vec!["NetCashFlow"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInContinuingOperations"),
//         Some(vec!["NetCashFlowContinuing", "NetCashFlow"])
//     );
// }

// #[test]
// fn test_net_cash_flow_continuing() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInContinuingOperations"),
//         Some(vec!["NetCashFlowContinuing", "NetCashFlow"])
//     );
// }

// #[test]
// fn test_net_cash_flow_discontinued() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInDiscontinuedOperations"),
//         Some(vec!["NetCashFlowDiscontinued"])
//     );
// }

// #[test]
// fn test_net_cash_flow_from_financing_activities() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInFinancingActivities"),
//         Some(vec!["NetCashFlowFromFinancingActivities"])
//     );
// }

// #[test]
// fn test_net_cash_flow_from_financing_activities_continuing() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"
//         ),
//         Some(vec!["NetCashFlowFromFinancingActivitiesContinuing"])
//     );
// }

// #[test]
// fn test_net_cash_flow_from_financing_and_activities_discontinued() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "CashProvidedByUsedInFinancingActivitiesDiscontinuedOperations"
//         ),
//         Some(vec!["NetCashFlowFromFinancingActivitiesDiscontinued"])
//     );
// }

// #[test]
// fn test_net_cash_flow_from_investing_activities() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInInvestingActivities"),
//         Some(vec!["NetCashFlowFromInvestingActivities"])
//     );
// }

// #[test]
// fn test_net_cash_flow_from_investing_activities_continuing() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"
//         ),
//         Some(vec!["NetCashFlowFromInvestingActivitiesContinuing"])
//     );
// }

// #[test]
// fn test_net_cash_flow_from_investing_activities_discontinued() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "CashProvidedByUsedInInvestingActivitiesDiscontinuedOperations"
//         ),
//         Some(vec!["NetCashFlowFromInvestingActivitiesDiscontinued"])
//     );
// }

// #[test]
// fn test_net_cash_flow_from_operating_activities() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInOperatingActivities"),
//         Some(vec!["NetCashFlowFromOperatingActivities"])
//     );
// }

// #[test]
// fn test_net_cash_flow_from_operating_activities_continuing() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
//         ),
//         Some(vec!["NetCashFlowFromOperatingActivitiesContinuing"])
//     );
// }

// #[test]
// fn test_net_cash_flow_from_operating_activities_discontinued() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations"
//         ),
//         Some(vec!["NetCashFlowFromOperatingActivitiesDiscontinued"])
//     );
// }

// #[test]
// fn test_net_income_loss() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ProfitLoss"),
//         Some(vec!["NetIncomeLoss"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetIncomeLoss"),
//         Some(vec!["NetIncomeLossAttributableToParent", "NetIncomeLoss"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetIncomeLossAvailableToCommonStockholdersBasic"),
//         Some(vec![
//             "NetIncomeLossAvailableToCommonStockholdersBasic",
//             "NetIncomeLoss"
//         ])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperations"),
//         Some(vec![
//             "IncomeLossFromContinuingOperationsAfterTax",
//             "NetIncomeLoss"
//         ])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeLossAttributableToParent"),
//         Some(vec!["NetIncomeLoss"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"),
//         Some(vec!["IncomeLossFromContinuingOperationsAfterTax", "NetIncomeLoss"])
//     );
// }

// #[test]
// fn test_net_income_loss_attributable_to_noncontrolling_interest() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetIncomeLossAttributableToNoncontrollingInterest"),
//         Some(vec!["NetIncomeLossAttributableToNoncontrollingInterest"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "NetIncomeLossAttributableToNonredeemableNoncontrollingInterest"
//         ),
//         Some(vec!["NetIncomeLossAttributableToNoncontrollingInterest"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "NetIncomeLossAttributableToRedeemableNoncontrollingInterest"
//         ),
//         Some(vec!["NetIncomeLossAttributableToNoncontrollingInterest"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity"
//         ),
//         Some(vec!["NetIncomeLossAttributableToNoncontrollingInterest"])
//     );
// }

// #[test]
// fn test_net_income_loss_attributable_to_parent() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetIncomeLoss"),
//         Some(vec!["NetIncomeLossAttributableToParent", "NetIncomeLoss"])
//     );
// }

// #[test]
// fn test_net_income_loss_available_to_common_stockholders_basic() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetIncomeLossAvailableToCommonStockholdersBasic"),
//         Some(vec![
//             "NetIncomeLossAvailableToCommonStockholdersBasic",
//             "NetIncomeLoss"
//         ])
//     );
// }

// #[test]
// fn test_assets_non_current() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("AssetsNoncurrent"),
//         Some(vec!["NoncurrentAssets"])
//     );
// }

// #[test]
// fn test_noncurrent_liabilities() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("LiabilitiesNoncurrent"),
//         Some(vec!["NoncurrentLiabilities"])
//     );
// }

// #[test]
// fn test_non_interest_expense() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NoninterestExpense"),
//         Some(vec!["NoninterestExpense"])
//     );
// }

// #[test]
// fn test_non_interest_income() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NoninterestIncome"),
//         Some(vec!["NoninterestIncome"])
//     );
// }

// #[test]
// fn test_non_operating_income_loss() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NonoperatingIncomeExpense"),
//         Some(vec!["NonoperatingIncomeLoss"])
//     );
// }

// #[test]
// fn test_operating_expenses() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("OperatingExpenses"),
//         Some(vec!["OperatingExpenses"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("OperatingCostsAndExpenses"),
//         Some(vec!["OperatingExpenses"])
//     );
// }

// #[test]
// fn test_operating_income_loss() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("OperatingIncomeLoss"),
//         Some(vec!["OperatingIncomeLoss"])
//     );
// }

// #[test]
// fn test_other_comprehensive_income_loss() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("OtherComprehensiveIncomeLossNetOfTax"),
//         Some(vec!["OtherComprehensiveIncomeLoss"])
//     );
// }

// #[test]
// fn test_other_operating_income_expenses() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("OtherOperatingIncome"),
//         Some(vec!["OtherOperatingIncomeExpenses"])
//     );
// }

// #[test]
// fn test_preferred_stock_dividends_and_other_adjustments() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("PreferredStockDividendsAndOtherAdjustments"),
//         Some(vec!["PreferredStockDividendsAndOtherAdjustments"])
//     );
// }

// #[test]
// fn test_provision_for_loan_lease_and_other_losses() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ProvisionForLoanLeaseAndOtherLosses"),
//         Some(vec!["ProvisionForLoanLeaseAndOtherLosses"])
//     );
// }

// #[test]
// fn test_redeemable_noncontrolling_interest() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RedeemableNoncontrollingInterestEquityCarryingAmount"),
//         Some(vec!["RedeemableNoncontrollingInterest"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "RedeemableNoncontrollingInterestEquityCommonCarryingAmount"
//         ),
//         Some(vec!["RedeemableNoncontrollingInterest"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "RedeemableNoncontrollingInterestEquityPreferredCarryingAmount"
//         ),
//         Some(vec!["RedeemableNoncontrollingInterest"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "RedeemableNoncontrollingInterestEquityOtherCarryingAmount"
//         ),
//         Some(vec!["RedeemableNoncontrollingInterest"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RedeemableNoncontrollingInterestEquityFairValue"),
//         Some(vec!["RedeemableNoncontrollingInterest"])
//     );
// }

// #[test]
// fn test_research_and_development() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ResearchAndDevelopmentExpense"),
//         Some(vec!["ResearchAndDevelopment"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"
//         ),
//         Some(vec!["ResearchAndDevelopment"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost"
//         ),
//         Some(vec!["ResearchAndDevelopment"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ResearchAndDevelopmentInProcess"),
//         Some(vec!["ResearchAndDevelopment"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "ResearchAndDevelopmentAssetAcquiredOtherThanThroughBusinessCombinationWrittenOff"
//         ),
//         Some(vec!["ResearchAndDevelopment"])
//     );
// }

// #[test]
// fn test_revenues() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("Revenues"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("SalesRevenueNet"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("SalesRevenueServicesNet"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("SalesRevenueGoodsNet"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RevenuesNetOfInterestExpense"),
//         Some(vec!["RevenuesNetInterestExpense", "Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("HealthCareOrganizationRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("InterestAndDividendIncomeOperating"),
//         Some(vec!["InterestAndDividendIncomeOperating", "Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RealEstateRevenueNet"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RevenueMineralSales"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("OilAndGasRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("FinancialServicesRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RegulatedAndUnregulatedOperatingRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ShippingAndHandlingRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("SalesRevenueFromEnergyCommoditiesAndServices"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("PhaseInPlanAmountOfCapitalizedCostsRecovered"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("SecondaryProcessingRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RevenueSteamProductsAndServices"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RevenueFromLeasedAndOwnedHotels"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("FranchisorRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("SubscriptionRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("AdvertisingRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("AdmissionsRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "RevenueFromEnrollmentAndRegistrationFeesExcludingHospitalityEnterprises"
//         ),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("MembershipDuesRevenueOnGoing"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("LicensesRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RoyaltyRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("SalesOfOilAndGasProspects"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ClearingFeesRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ReimbursementRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RevenueFromGrants"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RevenueOtherManufacturedProducts"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ConstructionMaterialsRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("TimberRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RecyclingRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("OtherSalesRevenueNet"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("SaleOfTrustAssetsToPayExpenses"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("PassengerRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("VehicleTollRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("CargoAndFreightRevenue"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("NetInvestmentIncome"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RevenuesExcludingInterestAndDividends"),
//         Some(vec!["RevenuesExcludingInterestAndDividends", "Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("InvestmentBankingRevenue"),
//         Some(vec!["RevenuesExcludingInterestAndDividends", "Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("UnderwritingIncomeLoss"),
//         Some(vec!["Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("ElectricUtilityRevenue"),
//         Some(vec!["Revenues"])
//     );
// }

// #[test]
// fn test_revenues_excluding_interest_and_dividends() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RevenuesExcludingInterestAndDividends"),
//         Some(vec!["RevenuesExcludingInterestAndDividends", "Revenues"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("BrokerageCommissionsRevenue"),
//         Some(vec!["RevenuesExcludingInterestAndDividends"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("InvestmentBankingRevenue"),
//         Some(vec!["RevenuesExcludingInterestAndDividends", "Revenues"])
//     );
// }

// #[test]
// fn test_revenues_net_interest_expense() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RevenuesNetOfInterestExpense"),
//         Some(vec!["RevenuesNetInterestExpense", "Revenues"])
//     );
// }

// #[test]
// fn test_temporary_equity() {
//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests"
//         ),
//         Some(vec!["TemporaryEquity"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("TemporaryEquityRedemptionValue"),
//         Some(vec!["TemporaryEquity"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("RedeemablePreferredStockCarryingAmount"),
//         Some(vec!["TemporaryEquity"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("TemporaryEquityCarryingAmount"),
//         Some(vec!["TemporaryEquity"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("TemporaryEquityValueExcludingAdditionalPaidInCapital"),
//         Some(vec!["TemporaryEquity"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("TemporaryEquityCarryingAmountAttributableToParent"),
//         Some(vec!["TemporaryEquity"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping(
//             "TemporaryEquityCarryingAmountAttributableToNoncontrollingInterest"
//         ),
//         Some(vec!["TemporaryEquity"])
//     );

//     assert_eq!(
//         get_us_gaap_human_readable_mapping("TemporaryEquityLiquidationPreference"),
//         Some(vec!["TemporaryEquity"])
//     );
// }
