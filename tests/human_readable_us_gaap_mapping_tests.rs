use sec_fetcher::accessor::get_us_gaap_human_readable_mapping;

#[test]
fn test_assets() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("Assets"),
        Some(vec!["Assets"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("AssetsCurrent"),
        Some(vec!["CurrentAssets", "Assets"])
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
            "ComprehensiveIncomeLossAttributableToParent",
            "ComprehensiveIncomeLoss",
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
        Some(vec!["CurrentAssets", "Assets"])
    );
}

#[test]
fn test_current_liabilities() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("LiabilitiesCurrent"),
        Some(vec!["CurrentLiabilities"])
    );
}

#[test]
fn test_equity() {
    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        ),
        Some(vec!["Equity"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("StockholdersEquity"),
        Some(vec!["Equity", "EquityAttributableToParent"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "PartnersCapitalIncludingPortionAttributableToNoncontrollingInterest"
        ),
        Some(vec!["Equity"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("PartnersCapital"),
        Some(vec!["Equity", "EquityAttributableToParent"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CommonStockholdersEquity"),
        Some(vec!["Equity"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MembersEquity"),
        Some(vec!["Equity", "EquityAttributableToParent"])
    );
}

#[test]
fn test_equity_attributable_to_noncontrolling_interest() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("MinorityInterest"),
        Some(vec!["EquityAttributableToNoncontrollingInterest"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("PartnersCapitalAttributableToNoncontrollingInterest"),
        Some(vec!["EquityAttributableToNoncontrollingInterest"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MinorityInterestInLimitedPartnerships"),
        Some(vec!["EquityAttributableToNoncontrollingInterest"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MinorityInterestInOperatingPartnerships"),
        Some(vec!["EquityAttributableToNoncontrollingInterest"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MinorityInterestInPreferredUnitHolders"),
        Some(vec!["EquityAttributableToNoncontrollingInterest"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MinorityInterestInJointVentures"),
        Some(vec!["EquityAttributableToNoncontrollingInterest"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("OtherMinorityInterests"),
        Some(vec!["EquityAttributableToNoncontrollingInterest"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("NonredeemableNoncontrollingInterest"),
        Some(vec!["EquityAttributableToNoncontrollingInterest"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("NoncontrollingInterestInVariableInterestEntity"),
        Some(vec!["EquityAttributableToNoncontrollingInterest"])
    );
}

#[test]
fn test_equity_attributable_to_parent() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("StockholdersEquity"),
        Some(vec!["Equity", "EquityAttributableToParent"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("PartnersCapital"),
        Some(vec!["Equity", "EquityAttributableToParent"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("MembersEquity"),
        Some(vec!["Equity", "EquityAttributableToParent"])
    );
}

#[test]
fn test_exchange_gains_losses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("EffectOfExchangeRateOnCashAndCashEquivalents"),
        Some(vec!["ExchangeGainsLosses"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "EffectOfExchangeRateOnCashAndCashEquivalentsContinuingOperations"
        ),
        Some(vec!["ExchangeGainsLosses"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("EffectOfExchangeRateOnCashContinuingOperations"),
        Some(vec!["ExchangeGainsLosses"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "EffectOfExchangeRateOnCashAndCashEquivalentsDiscontinuedOperations"
        ),
        Some(vec!["ExchangeGainsLosses"])
    );
}

#[test]
fn test_extraordinary_items_of_income_expense_net_of_tax() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("ExtraordinaryItemNetOfTax"),
        Some(vec!["ExtraordinaryItemsOfIncomeExpenseNetOfTax"])
    );
}

#[test]
fn test_gain_loss_on_sale_properties_net_tax() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("GainLossOnSaleOfPropertiesNetOfApplicableIncomeTaxes"),
        Some(vec!["GainLossOnSalePropertiesNetTax"])
    );
}

#[test]
fn test_gross_profit() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("GrossProfit"),
        Some(vec!["GrossProfit"])
    );
}

#[test]
fn test_income_loss_before_equity_method_investments() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
        Some(vec!["IncomeLossBeforeEquityMethodInvestments", "IncomeLossFromContinuingOperationsBeforeTax"])
    );
}

#[test]
fn test_income_loss_from_continuing_operations_after_tax() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"),
        Some(vec!["IncomeLossFromContinuingOperationsAfterTax", "NetIncomeLoss"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple"
        ),
        Some(vec!["IncomeLossFromContinuingOperationsAfterTax"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperations"),
        Some(vec![
            "IncomeLossFromContinuingOperationsAfterTax",
            "NetIncomeLoss"
        ])
    );
}

#[test]
fn test_income_loss_from_continuing_operations_before_tax() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
        Some(vec![
            "IncomeLossFromContinuingOperationsBeforeTax",
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
        Some(vec![
            "IncomeLossBeforeEquityMethodInvestments", "IncomeLossFromContinuingOperationsBeforeTax",
        ])
    );
}

#[test]
fn test_income_loss_from_discontinued_operations_net_of_tax() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromDiscontinuedOperationsNetOfTax"),
        Some(vec!["IncomeLossFromDiscontinuedOperationsNetOfTax",])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "DiscontinuedOperationGainLossOnDisposalOfDiscontinuedOperationNetOfTax"
        ),
        Some(vec!["IncomeLossFromDiscontinuedOperationsNetOfTax",])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity"
        ),
        Some(vec!["IncomeLossFromDiscontinuedOperationsNetOfTax",])
    );
}

#[test]
fn test_income_loss_from_equity_method_investments() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeLossFromEquityMethodInvestments"),
        Some(vec!["IncomeLossFromEquityMethodInvestments",])
    );
}

#[test]
fn test_income_tax_expense_benefit() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeTaxExpenseBenefit"),
        Some(vec![
            "IncomeStatementStartPeriodYearToDate",
            "IncomeTaxExpenseBenefit"
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("IncomeTaxExpenseBenefitContinuingOperations"),
        Some(vec![
            "IncomeStatementStartPeriodYearToDate",
            "IncomeTaxExpenseBenefit"
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("FederalHomeLoanBankAssessments"),
        Some(vec![
            "IncomeStatementStartPeriodYearToDate",
            "IncomeTaxExpenseBenefit"
        ])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CurrentIncomeTaxExpenseBenefit"),
        Some(vec![
            "IncomeStatementStartPeriodYearToDate",
            "IncomeTaxExpenseBenefit"
        ])
    );
}

#[test]
fn test_interest_and_debt_expense() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("InterestAndDebtExpense"),
        Some(vec!["InterestAndDebtExpense"])
    );
}

#[test]
fn test_interest_and_divident_income_operating() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("InterestAndDividendIncomeOperating"),
        Some(vec!["InterestAndDividendIncomeOperating", "Revenues"])
    );
}

#[test]
fn test_interest_income_expense_after_provision_for_losses() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("InterestIncomeExpenseAfterProvisionForLoanLoss"),
        Some(vec!["InterestIncomeExpenseAfterProvisionForLosses"])
    );
}

#[test]
fn test_interest_income_expense_operating_net() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("InterestIncomeExpenseNet"),
        Some(vec!["InterestIncomeExpenseOperatingNet"])
    );
}

#[test]
fn test_liabilities() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("Liabilities"),
        Some(vec!["Liabilities"])
    );
}

#[test]
fn test_liabilities_and_equity() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("LiabilitiesAndStockholdersEquity"),
        Some(vec!["LiabilitiesAndEquity"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("LiabilitiesAndPartnersCapital"),
        Some(vec!["LiabilitiesAndEquity"])
    );
}

#[test]
fn test_nature_of_operations() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NatureOfOperations"),
        Some(vec!["NatureOfOperations"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("BusinessDescriptionAndBasisOfPresentationTextBlock"),
        Some(vec!["NatureOfOperations"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping(
            "OrganizationConsolidationAndPresentationOfFinancialStatementsDisclosureTextBlock"
        ),
        Some(vec!["NatureOfOperations"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("OrganizationConsolidationBasisOfPresentationBusinessDescriptionAndAccountingPoliciesTextBlock"),
        Some(vec!["NatureOfOperations"])
    );
}

#[test]
fn test_net_cash_flow() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("CashAndCashEquivalentsPeriodIncreaseDecrease"),
        Some(vec!["NetCashFlow"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("CashPeriodIncreaseDecrease"),
        Some(vec!["NetCashFlow"])
    );

    assert_eq!(
        get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInContinuingOperations"),
        Some(vec!["NetCashFlowContinuing", "NetCashFlow"])
    );
}

#[test]
fn test_net_cash_flow_continuing() {
    assert_eq!(
        get_us_gaap_human_readable_mapping("NetCashProvidedByUsedInContinuingOperations"),
        Some(vec!["NetCashFlowContinuing", "NetCashFlow"])
    );
}
