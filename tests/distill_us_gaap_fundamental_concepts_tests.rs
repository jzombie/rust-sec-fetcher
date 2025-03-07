use sec_fetcher::{transformers::distill_us_gaap_fundamental_concepts, enums::FundamentalConcept};

#[test]
fn test_assets() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("Assets"),
        Some(vec![FundamentalConcept::Assets])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("AssetsCurrent"),
        Some(vec![
            FundamentalConcept::CurrentAssets,
            FundamentalConcept::Assets
        ])
    );
}

#[test]
fn test_benefits_costs_expenses() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("BenefitsLossesAndExpenses"),
        Some(vec![
            FundamentalConcept::BenefitsCostsExpenses,
            FundamentalConcept::CostsAndExpenses
        ])
    );
}

#[test]
fn test_commitments_and_contingencies() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("CommitmentsAndContingencies"),
        Some(vec![FundamentalConcept::CommitmentsAndContingencies])
    );
}

#[test]
fn test_comprehensive_income_loss() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest"
        ),
        Some(vec![FundamentalConcept::ComprehensiveIncomeLoss])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("ComprehensiveIncomeNetOfTax"),
        Some(vec![
            FundamentalConcept::ComprehensiveIncomeLossAttributableToParent,
            FundamentalConcept::ComprehensiveIncomeLoss,
        ])
    );
}

#[test]
fn test_comprehensive_income_loss_attributable_to_noncontrolling_interest() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts(
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
        distill_us_gaap_fundamental_concepts("ComprehensiveIncomeNetOfTax"),
        Some(vec![
            FundamentalConcept::ComprehensiveIncomeLossAttributableToParent,
            FundamentalConcept::ComprehensiveIncomeLoss,
        ])
    );
}

#[test]
fn test_cost_of_revenue() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("CostOfRevenue"),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("CostOfGoodsAndServicesSold"),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("CostOfServices"),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("CostOfGoodsSold"),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization"
        ),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("CostOfGoodsSoldElectric"),
        Some(vec![FundamentalConcept::CostOfRevenue])
    );
}

#[test]
fn test_costs_and_expenses() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("CostsAndExpenses"),
        Some(vec![FundamentalConcept::CostsAndExpenses])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("BenefitsLossesAndExpenses"),
        Some(vec![
            FundamentalConcept::BenefitsCostsExpenses,
            FundamentalConcept::CostsAndExpenses
        ])
    );
}

#[test]
fn test_current_assets() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("AssetsCurrent"),
        Some(vec![
            FundamentalConcept::CurrentAssets,
            FundamentalConcept::Assets
        ])
    );
}

#[test]
fn test_current_liabilities() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("LiabilitiesCurrent"),
        Some(vec![FundamentalConcept::CurrentLiabilities])
    );
}

#[test]
fn test_equity() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        ),
        Some(vec![FundamentalConcept::Equity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("StockholdersEquity"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "PartnersCapitalIncludingPortionAttributableToNoncontrollingInterest"
        ),
        Some(vec![FundamentalConcept::Equity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("PartnersCapital"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("CommonStockholdersEquity"),
        Some(vec![FundamentalConcept::Equity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("MembersEquity"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );
}

#[test]
fn test_equity_attributable_to_noncontrolling_interest() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("MinorityInterest"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("PartnersCapitalAttributableToNoncontrollingInterest"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("MinorityInterestInLimitedPartnerships"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("MinorityInterestInOperatingPartnerships"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("MinorityInterestInPreferredUnitHolders"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("MinorityInterestInJointVentures"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("OtherMinorityInterests"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("NonredeemableNoncontrollingInterest"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("NoncontrollingInterestInVariableInterestEntity"),
        Some(vec![
            FundamentalConcept::EquityAttributableToNoncontrollingInterest
        ])
    );
}

#[test]
fn test_equity_attributable_to_parent() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("StockholdersEquity"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("PartnersCapital"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("MembersEquity"),
        Some(vec![
            FundamentalConcept::EquityAttributableToParent,
            FundamentalConcept::Equity
        ])
    );
}

#[test]
fn test_exchange_gains_losses() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("EffectOfExchangeRateOnCashAndCashEquivalents"),
        Some(vec![FundamentalConcept::ExchangeGainsLosses])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "EffectOfExchangeRateOnCashAndCashEquivalentsContinuingOperations"
        ),
        Some(vec![FundamentalConcept::ExchangeGainsLosses])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("EffectOfExchangeRateOnCashContinuingOperations"),
        Some(vec![FundamentalConcept::ExchangeGainsLosses])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "EffectOfExchangeRateOnCashAndCashEquivalentsDiscontinuedOperations"
        ),
        Some(vec![FundamentalConcept::ExchangeGainsLosses])
    );
}

#[test]
fn test_extraordinary_items_of_income_expense_net_of_tax() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("ExtraordinaryItemNetOfTax"),
        Some(vec![
            FundamentalConcept::ExtraordinaryItemsOfIncomeExpenseNetOfTax
        ])
    );
}

#[test]
fn test_gain_loss_on_sale_properties_net_tax() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "GainLossOnSaleOfPropertiesNetOfApplicableIncomeTaxes"
        ),
        Some(vec![FundamentalConcept::GainLossOnSalePropertiesNetTax])
    );
}

#[test]
fn test_gross_profit() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("GrossProfit"),
        Some(vec![FundamentalConcept::GrossProfit])
    );
}

#[test]
fn test_income_loss_before_equity_method_investments() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
        Some(vec![FundamentalConcept::IncomeLossBeforeEquityMethodInvestments, FundamentalConcept::IncomeLossFromContinuingOperationsBeforeTax])
    );
}

#[test]
fn test_income_loss_from_continuing_operations_after_tax() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"),
        Some(vec![FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax, FundamentalConcept::NetIncomeLoss])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple"
        ),
        Some(vec![
            FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeLossFromContinuingOperations"),
        Some(vec![
            FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax,
            FundamentalConcept::NetIncomeLoss
        ])
    );
}

#[test]
fn test_income_loss_from_continuing_operations_before_tax() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
        Some(vec![
            FundamentalConcept::IncomeLossFromContinuingOperationsBeforeTax,
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
        Some(vec![
            FundamentalConcept::IncomeLossBeforeEquityMethodInvestments, FundamentalConcept::IncomeLossFromContinuingOperationsBeforeTax,
        ])
    );
}

#[test]
fn test_income_loss_from_discontinued_operations_net_of_tax() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeLossFromDiscontinuedOperationsNetOfTax"),
        Some(vec![
            FundamentalConcept::IncomeLossFromDiscontinuedOperationsNetOfTax
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "DiscontinuedOperationGainLossOnDisposalOfDiscontinuedOperationNetOfTax"
        ),
        Some(vec![
            FundamentalConcept::IncomeLossFromDiscontinuedOperationsNetOfTax
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
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
        distill_us_gaap_fundamental_concepts("IncomeLossFromEquityMethodInvestments"),
        Some(vec![
            FundamentalConcept::IncomeLossFromEquityMethodInvestments
        ])
    );
}

#[test]
fn test_income_tax_expense_benefit() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeTaxExpenseBenefit"),
        Some(vec![
            FundamentalConcept::IncomeStatementStartPeriodYearToDate,
            FundamentalConcept::IncomeTaxExpenseBenefit
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeTaxExpenseBenefitContinuingOperations"),
        Some(vec![
            FundamentalConcept::IncomeStatementStartPeriodYearToDate,
            FundamentalConcept::IncomeTaxExpenseBenefit
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("FederalHomeLoanBankAssessments"),
        Some(vec![
            FundamentalConcept::IncomeStatementStartPeriodYearToDate,
            FundamentalConcept::IncomeTaxExpenseBenefit
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("CurrentIncomeTaxExpenseBenefit"),
        Some(vec![
            FundamentalConcept::IncomeStatementStartPeriodYearToDate,
            FundamentalConcept::IncomeTaxExpenseBenefit
        ])
    );
}

#[test]
fn test_interest_and_debt_expense() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("InterestAndDebtExpense"),
        Some(vec![FundamentalConcept::InterestAndDebtExpense])
    );
}

#[test]
fn test_interest_and_divident_income_operating() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("InterestAndDividendIncomeOperating"),
        Some(vec![
            FundamentalConcept::InterestAndDividendIncomeOperating,
            FundamentalConcept::Revenues
        ])
    );
}

#[test]
fn test_interest_income_expense_after_provision_for_losses() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("InterestIncomeExpenseAfterProvisionForLoanLoss"),
        Some(vec![
            FundamentalConcept::InterestIncomeExpenseAfterProvisionForLosses
        ])
    );
}

#[test]
fn test_interest_income_expense_operating_net() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("InterestIncomeExpenseNet"),
        Some(vec![FundamentalConcept::InterestIncomeExpenseOperatingNet])
    );
}

#[test]
fn test_liabilities() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("Liabilities"),
        Some(vec![FundamentalConcept::Liabilities])
    );
}

#[test]
fn test_liabilities_and_equity() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("LiabilitiesAndStockholdersEquity"),
        Some(vec![FundamentalConcept::LiabilitiesAndEquity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("LiabilitiesAndPartnersCapital"),
        Some(vec![FundamentalConcept::LiabilitiesAndEquity])
    );
}

#[test]
fn test_nature_of_operations() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("NatureOfOperations"),
        Some(vec![FundamentalConcept::NatureOfOperations])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("BusinessDescriptionAndBasisOfPresentationTextBlock"),
        Some(vec![FundamentalConcept::NatureOfOperations])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "OrganizationConsolidationAndPresentationOfFinancialStatementsDisclosureTextBlock"
        ),
        Some(vec![FundamentalConcept::NatureOfOperations])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("OrganizationConsolidationBasisOfPresentationBusinessDescriptionAndAccountingPoliciesTextBlock"),
        Some(vec![FundamentalConcept::NatureOfOperations])
    );
}

#[test]
fn test_net_cash_flow() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("CashAndCashEquivalentsPeriodIncreaseDecrease"),
        Some(vec![FundamentalConcept::NetCashFlow])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("CashPeriodIncreaseDecrease"),
        Some(vec![FundamentalConcept::NetCashFlow])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("NetCashProvidedByUsedInContinuingOperations"),
        Some(vec![
            FundamentalConcept::NetCashFlowContinuing,
            FundamentalConcept::NetCashFlow
        ])
    );
}

#[test]
fn test_net_cash_flow_continuing() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("NetCashProvidedByUsedInContinuingOperations"),
        Some(vec![
            FundamentalConcept::NetCashFlowContinuing,
            FundamentalConcept::NetCashFlow
        ])
    );
}

#[test]
fn test_net_cash_flow_discontinued() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("NetCashProvidedByUsedInDiscontinuedOperations"),
        Some(vec![FundamentalConcept::NetCashFlowDiscontinued])
    );
}

#[test]
fn test_net_cash_flow_from_financing_activities() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("NetCashProvidedByUsedInFinancingActivities"),
        Some(vec![FundamentalConcept::NetCashFlowFromFinancingActivities])
    );
}

#[test]
fn test_net_cash_flow_from_financing_activities_continuing() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts(
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
        distill_us_gaap_fundamental_concepts(
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
        distill_us_gaap_fundamental_concepts("NetCashProvidedByUsedInInvestingActivities"),
        Some(vec![FundamentalConcept::NetCashFlowFromInvestingActivities])
    );
}

#[test]
fn test_net_cash_flow_from_investing_activities_continuing() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts(
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
        distill_us_gaap_fundamental_concepts(
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
        distill_us_gaap_fundamental_concepts("NetCashProvidedByUsedInOperatingActivities"),
        Some(vec![FundamentalConcept::NetCashFlowFromOperatingActivities])
    );
}

#[test]
fn test_net_cash_flow_from_operating_activities_continuing() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts(
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
        distill_us_gaap_fundamental_concepts(
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
        distill_us_gaap_fundamental_concepts("ProfitLoss"),
        Some(vec![FundamentalConcept::NetIncomeLoss])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("NetIncomeLoss"),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToParent,
            FundamentalConcept::NetIncomeLoss
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("NetIncomeLossAvailableToCommonStockholdersBasic"),
        Some(vec![
            FundamentalConcept::NetIncomeLossAvailableToCommonStockholdersBasic,
            FundamentalConcept::NetIncomeLoss
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeLossFromContinuingOperations"),
        Some(vec![
            FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax,
            FundamentalConcept::NetIncomeLoss
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeLossAttributableToParent"),
        Some(vec![FundamentalConcept::NetIncomeLoss])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"),
        Some(vec![FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax, FundamentalConcept::NetIncomeLoss])
    );
}

#[test]
fn test_net_income_loss_attributable_to_noncontrolling_interest() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("NetIncomeLossAttributableToNoncontrollingInterest"),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "NetIncomeLossAttributableToNonredeemableNoncontrollingInterest"
        ),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "NetIncomeLossAttributableToRedeemableNoncontrollingInterest"
        ),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToNoncontrollingInterest
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
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
        distill_us_gaap_fundamental_concepts("NetIncomeLoss"),
        Some(vec![
            FundamentalConcept::NetIncomeLossAttributableToParent,
            FundamentalConcept::NetIncomeLoss
        ])
    );
}

#[test]
fn test_net_income_loss_available_to_common_stockholders_basic() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("NetIncomeLossAvailableToCommonStockholdersBasic"),
        Some(vec![
            FundamentalConcept::NetIncomeLossAvailableToCommonStockholdersBasic,
            FundamentalConcept::NetIncomeLoss
        ])
    );
}

#[test]
fn test_assets_non_current() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("AssetsNoncurrent"),
        Some(vec![FundamentalConcept::NoncurrentAssets])
    );
}

#[test]
fn test_noncurrent_liabilities() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("LiabilitiesNoncurrent"),
        Some(vec![FundamentalConcept::NoncurrentLiabilities])
    );
}

#[test]
fn test_non_interest_expense() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("NoninterestExpense"),
        Some(vec![FundamentalConcept::NoninterestExpense])
    );
}

#[test]
fn test_non_interest_income() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("NoninterestIncome"),
        Some(vec![FundamentalConcept::NoninterestIncome])
    );
}

#[test]
fn test_non_operating_income_loss() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("NonoperatingIncomeExpense"),
        Some(vec![FundamentalConcept::NonoperatingIncomeLoss])
    );
}

#[test]
fn test_operating_expenses() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("OperatingExpenses"),
        Some(vec![FundamentalConcept::OperatingExpenses])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("OperatingCostsAndExpenses"),
        Some(vec![FundamentalConcept::OperatingExpenses])
    );
}

#[test]
fn test_operating_income_loss() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("OperatingIncomeLoss"),
        Some(vec![FundamentalConcept::OperatingIncomeLoss])
    );
}

#[test]
fn test_other_comprehensive_income_loss() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("OtherComprehensiveIncomeLossNetOfTax"),
        Some(vec![FundamentalConcept::OtherComprehensiveIncomeLoss])
    );
}

#[test]
fn test_other_operating_income_expenses() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("OtherOperatingIncome"),
        Some(vec![FundamentalConcept::OtherOperatingIncomeExpenses])
    );
}

#[test]
fn test_preferred_stock_dividends_and_other_adjustments() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("PreferredStockDividendsAndOtherAdjustments"),
        Some(vec![
            FundamentalConcept::PreferredStockDividendsAndOtherAdjustments
        ])
    );
}

#[test]
fn test_provision_for_loan_lease_and_other_losses() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("ProvisionForLoanLeaseAndOtherLosses"),
        Some(vec![
            FundamentalConcept::ProvisionForLoanLeaseAndOtherLosses
        ])
    );
}

#[test]
fn test_redeemable_noncontrolling_interest() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "RedeemableNoncontrollingInterestEquityCarryingAmount"
        ),
        Some(vec![FundamentalConcept::RedeemableNoncontrollingInterest])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "RedeemableNoncontrollingInterestEquityCommonCarryingAmount"
        ),
        Some(vec![FundamentalConcept::RedeemableNoncontrollingInterest])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "RedeemableNoncontrollingInterestEquityPreferredCarryingAmount"
        ),
        Some(vec![FundamentalConcept::RedeemableNoncontrollingInterest])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "RedeemableNoncontrollingInterestEquityOtherCarryingAmount"
        ),
        Some(vec![FundamentalConcept::RedeemableNoncontrollingInterest])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RedeemableNoncontrollingInterestEquityFairValue"),
        Some(vec![FundamentalConcept::RedeemableNoncontrollingInterest])
    );
}

#[test]
fn test_research_and_development() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("ResearchAndDevelopmentExpense"),
        Some(vec![FundamentalConcept::ResearchAndDevelopment])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"
        ),
        Some(vec![FundamentalConcept::ResearchAndDevelopment])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost"
        ),
        Some(vec![FundamentalConcept::ResearchAndDevelopment])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("ResearchAndDevelopmentInProcess"),
        Some(vec![FundamentalConcept::ResearchAndDevelopment])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "ResearchAndDevelopmentAssetAcquiredOtherThanThroughBusinessCombinationWrittenOff"
        ),
        Some(vec![FundamentalConcept::ResearchAndDevelopment])
    );
}

#[test]
fn test_revenues() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("Revenues"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("SalesRevenueNet"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("SalesRevenueServicesNet"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("SalesRevenueGoodsNet"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RevenuesNetOfInterestExpense"),
        Some(vec![
            FundamentalConcept::RevenuesNetInterestExpense,
            FundamentalConcept::Revenues
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("HealthCareOrganizationRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("InterestAndDividendIncomeOperating"),
        Some(vec![
            FundamentalConcept::InterestAndDividendIncomeOperating,
            FundamentalConcept::Revenues
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RealEstateRevenueNet"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RevenueMineralSales"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("OilAndGasRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("FinancialServicesRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RegulatedAndUnregulatedOperatingRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("ShippingAndHandlingRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("SalesRevenueFromEnergyCommoditiesAndServices"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("PhaseInPlanAmountOfCapitalizedCostsRecovered"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("SecondaryProcessingRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RevenueSteamProductsAndServices"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RevenueFromLeasedAndOwnedHotels"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("FranchisorRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("SubscriptionRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("AdvertisingRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("AdmissionsRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "RevenueFromEnrollmentAndRegistrationFeesExcludingHospitalityEnterprises"
        ),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("MembershipDuesRevenueOnGoing"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("LicensesRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RoyaltyRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("SalesOfOilAndGasProspects"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("ClearingFeesRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("ReimbursementRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RevenueFromGrants"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RevenueOtherManufacturedProducts"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("ConstructionMaterialsRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("TimberRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RecyclingRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("OtherSalesRevenueNet"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("SaleOfTrustAssetsToPayExpenses"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("PassengerRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("VehicleTollRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("CargoAndFreightRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("NetInvestmentIncome"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RevenuesExcludingInterestAndDividends"),
        Some(vec![
            FundamentalConcept::RevenuesExcludingInterestAndDividends,
            FundamentalConcept::Revenues
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("InvestmentBankingRevenue"),
        Some(vec![
            FundamentalConcept::RevenuesExcludingInterestAndDividends,
            FundamentalConcept::Revenues
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("UnderwritingIncomeLoss"),
        Some(vec![FundamentalConcept::Revenues])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("ElectricUtilityRevenue"),
        Some(vec![FundamentalConcept::Revenues])
    );
}

#[test]
fn test_revenues_excluding_interest_and_dividends() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("RevenuesExcludingInterestAndDividends"),
        Some(vec![
            FundamentalConcept::RevenuesExcludingInterestAndDividends,
            FundamentalConcept::Revenues
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("BrokerageCommissionsRevenue"),
        Some(vec![
            FundamentalConcept::RevenuesExcludingInterestAndDividends
        ])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("InvestmentBankingRevenue"),
        Some(vec![
            FundamentalConcept::RevenuesExcludingInterestAndDividends,
            FundamentalConcept::Revenues
        ])
    );
}

#[test]
fn test_revenues_net_interest_expense() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts("RevenuesNetOfInterestExpense"),
        Some(vec![
            FundamentalConcept::RevenuesNetInterestExpense,
            FundamentalConcept::Revenues
        ])
    );
}

#[test]
fn test_temporary_equity() {
    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests"
        ),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("TemporaryEquityRedemptionValue"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("RedeemablePreferredStockCarryingAmount"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("TemporaryEquityCarryingAmount"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "TemporaryEquityValueExcludingAdditionalPaidInCapital"
        ),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("TemporaryEquityCarryingAmountAttributableToParent"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts(
            "TemporaryEquityCarryingAmountAttributableToNoncontrollingInterest"
        ),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );

    assert_eq!(
        distill_us_gaap_fundamental_concepts("TemporaryEquityLiquidationPreference"),
        Some(vec![FundamentalConcept::TemporaryEquity])
    );
}
