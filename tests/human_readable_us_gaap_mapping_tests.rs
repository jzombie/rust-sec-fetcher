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

#[test]
fn test_extraordinary_items_of_income_expense_net_of_tax() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("ExtraordinaryItemNetOfTax"),
        Some(vec![
            FundamentalConcept::ExtraordinaryItemsOfIncomeExpenseNetOfTax
        ])
    );
}

#[test]
fn test_gain_loss_on_sale_properties_net_tax() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("GainLossOnSaleOfPropertiesNetOfApplicableIncomeTaxes"),
        Some(vec![FundamentalConcept::GainLossOnSalePropertiesNetTax])
    );
}

#[test]
fn test_gross_profit() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("GrossProfit"),
        Some(vec![FundamentalConcept::GrossProfit])
    );
}

#[test]
fn test_income_loss_before_equity_method_investments() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
        Some(vec![FundamentalConcept::IncomeLossBeforeEquityMethodInvestments, FundamentalConcept::IncomeLossFromContinuingOperationsBeforeTax])
    );
}

#[test]
fn test_income_loss_from_continuing_operations_after_tax() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"),
        Some(vec![FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax, FundamentalConcept::NetIncomeLoss])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple"
        ),
        Some(vec![
            FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperations"),
        Some(vec![
            FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax,
            FundamentalConcept::NetIncomeLoss
        ])
    );
}

#[test]
fn test_income_loss_from_continuing_operations_before_tax() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
        Some(vec![
            FundamentalConcept::IncomeLossFromContinuingOperationsBeforeTax,
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
        Some(vec![
            FundamentalConcept::IncomeLossBeforeEquityMethodInvestments, FundamentalConcept::IncomeLossFromContinuingOperationsBeforeTax,
        ])
    );
}

#[test]
fn test_income_loss_from_discontinued_operations_net_of_tax() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromDiscontinuedOperationsNetOfTax"),
        Some(vec![
            FundamentalConcept::IncomeLossFromDiscontinuedOperationsNetOfTax
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "DiscontinuedOperationGainLossOnDisposalOfDiscontinuedOperationNetOfTax"
        ),
        Some(vec![
            FundamentalConcept::IncomeLossFromDiscontinuedOperationsNetOfTax
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity"
        ),
        Some(vec![
            FundamentalConcept::IncomeLossFromDiscontinuedOperationsNetOfTax
        ])
    );
}

#[test]
fn test_income_loss_from_equity_method_investments() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromEquityMethodInvestments"),
        Some(vec![
            FundamentalConcept::IncomeLossFromEquityMethodInvestments
        ])
    );
}

#[test]
fn test_income_tax_expense_benefit() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeTaxExpenseBenefit"),
        Some(vec![
            FundamentalConcept::IncomeStatementStartPeriodYearToDate,
            FundamentalConcept::IncomeTaxExpenseBenefit
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeTaxExpenseBenefitContinuingOperations"),
        Some(vec![
            FundamentalConcept::IncomeStatementStartPeriodYearToDate,
            FundamentalConcept::IncomeTaxExpenseBenefit
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("FederalHomeLoanBankAssessments"),
        Some(vec![
            FundamentalConcept::IncomeStatementStartPeriodYearToDate,
            FundamentalConcept::IncomeTaxExpenseBenefit
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CurrentIncomeTaxExpenseBenefit"),
        Some(vec![
            FundamentalConcept::IncomeStatementStartPeriodYearToDate,
            FundamentalConcept::IncomeTaxExpenseBenefit
        ])
    );
}

#[test]
fn test_interest_and_debt_expense() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("InterestAndDebtExpense"),
        Some(vec![FundamentalConcept::InterestAndDebtExpense])
    );
}

#[test]
fn test_interest_and_divident_income_operating() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("InterestAndDividendIncomeOperating"),
        Some(vec![
            FundamentalConcept::InterestAndDividendIncomeOperating,
            FundamentalConcept::Revenues
        ])
    );
}

#[test]
fn test_interest_income_expense_after_provision_for_losses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("InterestIncomeExpenseAfterProvisionForLoanLoss"),
        Some(vec![
            FundamentalConcept::InterestIncomeExpenseAfterProvisionForLosses
        ])
    );
}

#[test]
fn test_interest_income_expense_operating_net() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("InterestIncomeExpenseNet"),
        Some(vec![FundamentalConcept::InterestIncomeExpenseOperatingNet])
    );
}

#[test]
fn test_liabilities() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("Liabilities"),
        Some(vec![FundamentalConcept::Liabilities])
    );
}

#[test]
fn test_liabilities_and_equity() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("LiabilitiesAndStockholdersEquity"),
        Some(vec![FundamentalConcept::LiabilitiesAndEquity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("LiabilitiesAndPartnersCapital"),
        Some(vec![FundamentalConcept::LiabilitiesAndEquity])
    );
}

#[test]
fn test_nature_of_operations() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NatureOfOperations"),
        Some(vec![FundamentalConcept::NatureOfOperations])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("BusinessDescriptionAndBasisOfPresentationTextBlock"),
        Some(vec![FundamentalConcept::NatureOfOperations])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "OrganizationConsolidationAndPresentationOfFinancialStatementsDisclosureTextBlock"
        ),
        Some(vec![FundamentalConcept::NatureOfOperations])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("OrganizationConsolidationBasisOfPresentationBusinessDescriptionAndAccountingPoliciesTextBlock"),
        Some(vec![FundamentalConcept::NatureOfOperations])
    );
}

#[test]
fn test_net_cash_flow() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("CashAndCashEquivalentsPeriodIncreaseDecrease"),
        Some(vec![FundamentalConcept::NetCashFlow])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CashPeriodIncreaseDecrease"),
        Some(vec![FundamentalConcept::NetCashFlow])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInContinuingOperations"),
        Some(vec![
            FundamentalConcept::NetCashFlowContinuing,
            FundamentalConcept::NetCashFlow
        ])
    );
}

#[test]
fn test_net_cash_flow_continuing() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInContinuingOperations"),
        Some(vec![
            FundamentalConcept::NetCashFlowContinuing,
            FundamentalConcept::NetCashFlow
        ])
    );
}

#[test]
fn test_net_cash_flow_discontinued() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInDiscontinuedOperations"),
        Some(vec![FundamentalConcept::NetCashFlowDiscontinued])
    );
}

#[test]
fn test_net_cash_flow_from_financing_activities() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInFinancingActivities"),
        Some(vec![FundamentalConcept::NetCashFlowFromFinancingActivities])
    );
}

#[test]
fn test_net_cash_flow_from_financing_activities_continuing() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"
        ),
        Some(vec![
            FundamentalConcept::NetCashFlowFromFinancingActivitiesContinuing
        ])
    );
}

#[test]
fn test_net_cash_flow_from_financing_and_activities_discontinued() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "CashProvidedByUsedInFinancingActivitiesDiscontinuedOperations"
        ),
        Some(vec![
            FundamentalConcept::NetCashFlowFromFinancingActivitiesDiscontinued
        ])
    );
}

#[test]
fn test_net_cash_flow_from_investing_activities() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInInvestingActivities"),
        Some(vec![FundamentalConcept::NetCashFlowFromInvestingActivities])
    );
}

#[test]
fn test_net_cash_flow_from_investing_activities_continuing() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"
        ),
        Some(vec![
            FundamentalConcept::NetCashFlowFromInvestingActivitiesContinuing
        ])
    );
}

#[test]
fn test_net_cash_flow_from_investing_activities_discontinued() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "CashProvidedByUsedInInvestingActivitiesDiscontinuedOperations"
        ),
        Some(vec![
            FundamentalConcept::NetCashFlowFromInvestingActivitiesDiscontinued
        ])
    );
}

#[test]
fn test_net_cash_flow_from_operating_activities() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInOperatingActivities"),
        Some(vec![FundamentalConcept::NetCashFlowFromOperatingActivities])
    );
}

#[test]
fn test_net_cash_flow_from_operating_activities_continuing() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
        ),
        Some(vec![
            FundamentalConcept::NetCashFlowFromOperatingActivitiesContinuing
        ])
    );
}

#[test]
fn test_net_cash_flow_from_operating_activities_discontinued() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations"
        ),
        Some(vec![
            FundamentalConcept::NetCashFlowFromOperatingActivitiesDiscontinued
        ])
    );
}

#[test]
fn test_net_income_loss() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("ProfitLoss"),
        Some(vec![FundamentalConcept::NetIncomeLoss])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("NetIncomeLoss"),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToParent,
            FundamentalConcept::NetIncomeLoss
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("NetIncomeLossAvailableToCommonStockholdersBasic"),
        Some(vec![
            FundamentalConcept::NetIncomeLossAvailableToCommonStockholdersBasic,
            FundamentalConcept::NetIncomeLoss
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperations"),
        Some(vec![
            FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax,
            FundamentalConcept::NetIncomeLoss
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossAttributableToParent"),
        Some(vec![FundamentalConcept::NetIncomeLoss])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"),
        Some(vec![FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax, FundamentalConcept::NetIncomeLoss])
    );
}

#[test]
fn test_net_income_loss_attributable_to_noncontrolling_interest() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NetIncomeLossAttributableToNoncontrollingInterest"),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "NetIncomeLossAttributableToNonredeemableNoncontrollingInterest"
        ),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "NetIncomeLossAttributableToRedeemableNoncontrollingInterest"
        ),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity"
        ),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToNoncontrollingInterest
        ])
    );
}

#[test]
fn test_net_income_loss_attributable_to_parent() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NetIncomeLoss"),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToParent,
            FundamentalConcept::NetIncomeLoss
        ])
    );
}

#[test]
fn test_net_income_loss_available_to_common_stockholders_basic() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NetIncomeLossAvailableToCommonStockholdersBasic"),
        Some(vec![
            FundamentalConcept::NetIncomeLossAvailableToCommonStockholdersBasic,
            FundamentalConcept::NetIncomeLoss
        ])
    );
}

#[test]
fn test_assets_non_current() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("AssetsNoncurrent"),
        Some(vec![FundamentalConcept::NoncurrentAssets])
    );
}

#[test]
fn test_noncurrent_liabilities() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("LiabilitiesNoncurrent"),
        Some(vec![FundamentalConcept::NoncurrentLiabilities])
    );
}

#[test]
fn test_non_interest_expense() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NoninterestExpense"),
        Some(vec![FundamentalConcept::NoninterestExpense])
    );
}

#[test]
fn test_non_interest_income() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NoninterestIncome"),
        Some(vec![FundamentalConcept::NoninterestIncome])
    );
}

#[test]
fn test_non_operating_income_loss() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NonoperatingIncomeExpense"),
        Some(vec![FundamentalConcept::NonoperatingIncomeLoss])
    );
}

#[test]
fn test_operating_expenses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("OperatingExpenses"),
        Some(vec![FundamentalConcept::OperatingExpenses])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("OperatingCostsAndExpenses"),
        Some(vec![FundamentalConcept::OperatingExpenses])
    );
}

#[test]
fn test_operating_income_loss() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("OperatingIncomeLoss"),
        Some(vec![FundamentalConcept::OperatingIncomeLoss])
    );
}

#[test]
fn test_other_comprehensive_income_loss() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("OtherComprehensiveIncomeLossNetOfTax"),
        Some(vec![FundamentalConcept::OtherComprehensiveIncomeLoss])
    );
}

#[test]
fn test_other_operating_income_expenses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("OtherOperatingIncome"),
        Some(vec![FundamentalConcept::OtherOperatingIncomeExpenses])
    );
}

#[test]
fn test_preferred_stock_dividends_and_other_adjustments() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("PreferredStockDividendsAndOtherAdjustments"),
        Some(vec![
            FundamentalConcept::PreferredStockDividendsAndOtherAdjustments
        ])
    );
}

#[test]
fn test_provision_for_loan_lease_and_other_losses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("ProvisionForLoanLeaseAndOtherLosses"),
        Some(vec![
            FundamentalConcept::ProvisionForLoanLeaseAndOtherLosses
        ])
    );
}

#[test]
fn test_redeemable_noncontrolling_interest() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("RedeemableNoncontrollingInterestEquityCarryingAmount"),
        Some(vec![FundamentalConcept::RedeemableNoncontrollingInterest])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "RedeemableNoncontrollingInterestEquityCommonCarryingAmount"
        ),
        Some(vec![FundamentalConcept::RedeemableNoncontrollingInterest])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "RedeemableNoncontrollingInterestEquityPreferredCarryingAmount"
        ),
        Some(vec![FundamentalConcept::RedeemableNoncontrollingInterest])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "RedeemableNoncontrollingInterestEquityOtherCarryingAmount"
        ),
        Some(vec![FundamentalConcept::RedeemableNoncontrollingInterest])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RedeemableNoncontrollingInterestEquityFairValue"),
        Some(vec![FundamentalConcept::RedeemableNoncontrollingInterest])
    );
}

#[test]
fn test_research_and_development() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("ResearchAndDevelopmentExpense"),
        Some(vec![FundamentalConcept::ResearchAndDevelopment])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"
        ),
        Some(vec![FundamentalConcept::ResearchAndDevelopment])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost"
        ),
        Some(vec![FundamentalConcept::ResearchAndDevelopment])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("ResearchAndDevelopmentInProcess"),
        Some(vec![FundamentalConcept::ResearchAndDevelopment])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "ResearchAndDevelopmentAssetAcquiredOtherThanThroughBusinessCombinationWrittenOff"
        ),
        Some(vec![FundamentalConcept::ResearchAndDevelopment])
    );
}

#[test]
fn test_revenues() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("Revenues"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("SalesRevenueNet"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("SalesRevenueServicesNet"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("SalesRevenueGoodsNet"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RevenuesNetOfInterestExpense"),
        Some(vec![
            FundamentalConcept::RevenuesNetInterestExpense,
            FundamentalConcept::Revenues
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("HealthCareOrganizationRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("InterestAndDividendIncomeOperating"),
        Some(vec![
            FundamentalConcept::InterestAndDividendIncomeOperating,
            FundamentalConcept::Revenues
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RealEstateRevenueNet"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RevenueMineralSales"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("OilAndGasRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("FinancialServicesRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RegulatedAndUnregulatedOperatingRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("ShippingAndHandlingRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("SalesRevenueFromEnergyCommoditiesAndServices"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("PhaseInPlanAmountOfCapitalizedCostsRecovered"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("SecondaryProcessingRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RevenueSteamProductsAndServices"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RevenueFromLeasedAndOwnedHotels"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("FranchisorRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("SubscriptionRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("AdvertisingRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("AdmissionsRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "RevenueFromEnrollmentAndRegistrationFeesExcludingHospitalityEnterprises"
        ),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MembershipDuesRevenueOnGoing"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("LicensesRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RoyaltyRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("SalesOfOilAndGasProspects"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("ClearingFeesRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("ReimbursementRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RevenueFromGrants"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RevenueOtherManufacturedProducts"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("ConstructionMaterialsRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("TimberRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RecyclingRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("OtherSalesRevenueNet"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("SaleOfTrustAssetsToPayExpenses"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("PassengerRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("VehicleTollRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CargoAndFreightRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("NetInvestmentIncome"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RevenuesExcludingInterestAndDividends"),
        Some(vec![
            FundamentalConcept::RevenuesExcludingInterestAndDividends,
            FundamentalConcept::Revenues
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("InvestmentBankingRevenue"),
        Some(vec![
            FundamentalConcept::RevenuesExcludingInterestAndDividends,
            FundamentalConcept::Revenues
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("UnderwritingIncomeLoss"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("ElectricUtilityRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );
}

#[test]
fn test_revenues_excluding_interest_and_dividends() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("RevenuesExcludingInterestAndDividends"),
        Some(vec![
            FundamentalConcept::RevenuesExcludingInterestAndDividends,
            FundamentalConcept::Revenues
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("BrokerageCommissionsRevenue"),
        Some(vec![
            FundamentalConcept::RevenuesExcludingInterestAndDividends
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("InvestmentBankingRevenue"),
        Some(vec![
            FundamentalConcept::RevenuesExcludingInterestAndDividends,
            FundamentalConcept::Revenues
        ])
    );
}

#[test]
fn test_revenues_net_interest_expense() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("RevenuesNetOfInterestExpense"),
        Some(vec![
            FundamentalConcept::RevenuesNetInterestExpense,
            FundamentalConcept::Revenues
        ])
    );
}

#[test]
fn test_temporary_equity() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests"
        ),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("TemporaryEquityRedemptionValue"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("RedeemablePreferredStockCarryingAmount"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("TemporaryEquityCarryingAmount"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("TemporaryEquityValueExcludingAdditionalPaidInCapital"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("TemporaryEquityCarryingAmountAttributableToParent"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "TemporaryEquityCarryingAmountAttributableToNoncontrollingInterest"
        ),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("TemporaryEquityLiquidationPreference"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );
}
