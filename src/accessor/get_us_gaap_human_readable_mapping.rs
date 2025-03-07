use crate::utils::invert_multivalue_indexmap;
use indexmap::IndexMap;
use once_cell::sync::Lazy;

pub type FundamentalConceptName = &'static str;
pub type TaxonomyConceptName = &'static str;

// Human-readable mapping: http://www.xbrlsite.com/2014/Reference/Mapping.pdf
static US_GAAP_MAPPING: Lazy<IndexMap<FundamentalConceptName, Vec<TaxonomyConceptName>>> =
    Lazy::new(|| {
        let mut map = IndexMap::new();

        // Entries are arranged by `Try Order` ascending

        map.insert("Assets", vec!["Assets", "AssetsCurrent"]);
        map.insert("BenefitsCostsExpenses", vec!["BenefitsLossesAndExpenses"]);
        map.insert(
            "CommitmentsAndContingencies",
            vec!["CommitmentsAndContingencies"],
        );
        map.insert(
            "ComprehensiveIncomeLoss",
            vec![
                "ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest",
                "ComprehensiveIncomeNetOfTax",
            ],
        );
        map.insert(
            "ComprehensiveIncomeLossAttributableToNoncontrollingInterest",
            vec![
                "ComprehensiveIncomeNetOfTaxAttributableToNoncontrollingInterest",
                "ComprehensiveIncome",
            ],
        );
        map.insert(
            "ComprehensiveIncomeLossAttributableToParent",
            vec!["ComprehensiveIncomeNetOfTax"],
        );
        map.insert(
            "CostOfRevenue",
            vec![
                "CostOfRevenue",
                "CostOfGoodsAndServicesSold",
                "CostOfServices",
                "CostOfGoodsSold",
                "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization",
                "CostOfGoodsSoldElectric",
            ],
        );
        map.insert(
            "CostsAndExpenses",
            vec!["CostsAndExpenses", "BenefitsLossesAndExpenses"],
        );
        map.insert("CurrentAssets", vec!["AssetsCurrent"]);
        map.insert("CurrentLiabilities", vec!["LiabilitiesCurrent"]);
        map.insert(
            "Equity",
            vec![
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                "StockholdersEquity",
                "PartnersCapitalIncludingPortionAttributableToNoncontrollingInterest",
                "PartnersCapital",
                "CommonStockholdersEquity",
                "MembersEquity",
            ],
        );
        map.insert(
            "EquityAttributableToNoncontrollingInterest",
            vec![
                "MinorityInterest",
                "PartnersCapitalAttributableToNoncontrollingInterest",
                "MinorityInterestInLimitedPartnerships",
                "MinorityInterestInOperatingPartnerships",
                "MinorityInterestInPreferredUnitHolders",
                "MinorityInterestInJointVentures",
                "OtherMinorityInterests",
                "NonredeemableNoncontrollingInterest",
                "NoncontrollingInterestInVariableInterestEntity",
            ],
        );
        map.insert(
            "EquityAttributableToParent",
            vec!["StockholdersEquity", "PartnersCapital", "MembersEquity"],
        );
        map.insert(
            "ExchangeGainsLosses",
            vec![
                "EffectOfExchangeRateOnCashAndCashEquivalents",
                "EffectOfExchangeRateOnCashAndCashEquivalentsContinuingOperations",
                "EffectOfExchangeRateOnCashContinuingOperations",
                "EffectOfExchangeRateOnCashAndCashEquivalentsDiscontinuedOperations",
            ],
        );
        map.insert(
            "ExtraordinaryItemsOfIncomeExpenseNetOfTax",
            vec!["ExtraordinaryItemNetOfTax"],
        );
        map.insert(
            "GainLossOnSalePropertiesNetTax",
            vec!["GainLossOnSaleOfPropertiesNetOfApplicableIncomeTaxes"],
        );
        map.insert("GrossProfit", vec!["GrossProfit"]);
        map.insert("IncomeLossBeforeEquityMethodInvestments", vec!["IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]);
        map.insert("IncomeLossFromContinuingOperationsAfterTax", vec![
            "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
            "IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple",
            "IncomeLossFromContinuingOperations"
        ]);
        map.insert("IncomeLossFromContinuingOperationsBeforeTax", vec![
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"
        ]);
        map.insert(
            "IncomeLossFromDiscontinuedOperationsNetOfTax",
            vec![
                "IncomeLossFromDiscontinuedOperationsNetOfTax",
                "DiscontinuedOperationGainLossOnDisposalOfDiscontinuedOperationNetOfTax",
                "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
            ],
        );
        map.insert(
            "IncomeLossFromEquityMethodInvestments",
            vec!["IncomeLossFromEquityMethodInvestments"],
        );
        map.insert(
            "IncomeStatementStartPeriodYearToDate",
            vec![
                "IncomeTaxExpenseBenefit",
                "IncomeTaxExpenseBenefitContinuingOperations",
                "FederalHomeLoanBankAssessments",
                "CurrentIncomeTaxExpenseBenefit",
            ],
        );
        map.insert(
            "IncomeTaxExpenseBenefit",
            vec![
                "IncomeTaxExpenseBenefit",
                "IncomeTaxExpenseBenefitContinuingOperations",
                "FederalHomeLoanBankAssessments",
                "CurrentIncomeTaxExpenseBenefit",
            ],
        );
        map.insert("InterestAndDebtExpense", vec!["InterestAndDebtExpense"]);
        map.insert(
            "InterestAndDividendIncomeOperating",
            vec!["InterestAndDividendIncomeOperating"],
        );
        map.insert(
            "InterestAndDividendIncomeOperating",
            vec!["InterestAndDividendIncomeOperating"],
        );
        map.insert("InterestExpenseOperating", vec!["InterestExpense"]);
        map.insert(
            "InterestIncomeExpenseAfterProvisionForLosses",
            vec!["InterestIncomeExpenseAfterProvisionForLoanLoss"],
        );
        map.insert(
            "InterestIncomeExpenseNet",
            vec!["InterestIncomeExpenseOperatingNet"],
        );
        map.insert("Liabilities", vec!["InterestExpenseOperating"]);
        map.insert(
            "LiabilitiesAndEquity",
            vec!["InterestExpenseOperating", "LiabilitiesAndPartnersCapital"],
        );
        map.insert(
        "NatureOfOperations",
        vec![
            "NatureOfOperations",
            "BusinessDescriptionAndBasisOfPresentationTextBlock",
            "OrganizationConsolidationAndPresentationOfFinancialStatementsDisclosureTextBlock",
            "OrganizationConsolidationBasisOfPresentationBusinessDescriptionAndAccountingPoliciesTextBlock",
        ],
    );
        map.insert(
            "NetCashFlow",
            vec![
                "CashAndCashEquivalentsPeriodIncreaseDecrease",
                "CashPeriodIncreaseDecrease",
                "NetCashProvidedByUsedInContinuingOperations",
            ],
        );
        map.insert(
            "NetCashFlowContinuing",
            vec!["NetCashProvidedByUsedInContinuingOperations"],
        );
        map.insert(
            "NetCashFlowDiscontinued",
            vec!["NetCashProvidedByUsedInDiscontinuedOperations"],
        );
        map.insert(
            "NetCashFlowFromFinancingActivities",
            vec!["NetCashProvidedByUsedInFinancingActivities"],
        );
        map.insert(
            "NetCashFlowFromFinancingActivitiesContinuing",
            vec!["NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
        );
        map.insert(
            "NetCashFlowFromFinancingActivitiesDiscontinued",
            vec!["CashProvidedByUsedInFinancingActivitiesDiscontinuedOperations"],
        );
        map.insert(
            "NetCashFlowFromInvestingActivities",
            vec!["NetCashProvidedByUsedInInvestingActivities"],
        );
        map.insert(
            "NetCashFlowFromInvestingActivitiesContinuing",
            vec!["NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
        );
        map.insert(
            "NetCashFlowFromInvestingActivitiesDiscontinued",
            vec!["CashProvidedByUsedInInvestingActivitiesDiscontinuedOperations"],
        );
        map.insert(
            "NetCashFlowFromOperatingActivities",
            vec!["NetCashProvidedByUsedInOperatingActivities"],
        );
        map.insert(
            "NetCashFlowFromOperatingActivitiesContinuing",
            vec!["NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        );
        map.insert(
            "NetCashFlowFromOperatingActivitiesDiscontinued",
            vec!["CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations"],
        );
        map.insert(
            "NetIncomeLoss",
            vec![
            "ProfitLoss",
            "NetIncomeLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
            "IncomeLossFromContinuingOperations",
            "IncomeLossAttributableToParent",
            "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"
        ],
        );
        map.insert(
            "NetIncomeLossAttributableToNoncontrollingInterest",
            vec![
                "NetIncomeLossAttributableToNoncontrollingInterest",
                "NetIncomeLossAttributableToNonredeemableNoncontrollingInterest",
                "NetIncomeLossAttributableToRedeemableNoncontrollingInterest",
                "IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity",
            ],
        );
        map.insert("NetIncomeLossAttributableToParent", vec!["NetIncomeLoss"]);
        map.insert("NoncurrentAssets", vec!["AssetsNoncurrent"]);
        map.insert("NoncurrentLiabilities", vec!["LiabilitiesNoncurrent"]);
        map.insert("NoninterestExpense", vec!["NoninterestExpense"]);
        map.insert("NoninterestIncome", vec!["NoninterestIncome"]);
        map.insert("NonoperatingIncomeLoss", vec!["NonoperatingIncomeExpense"]);
        map.insert(
            "OperatingExpenses",
            vec!["OperatingExpenses", "OperatingCostsAndExpenses"],
        );
        map.insert("OperatingIncomeLoss", vec!["OperatingIncomeLoss"]);
        map.insert(
            "OtherComprehensiveIncomeLoss",
            vec!["OtherComprehensiveIncomeLossNetOfTax"],
        );
        map.insert("OtherOperatingIncomeExpenses", vec!["OtherOperatingIncome"]);
        map.insert(
            "PreferredStockDividendsAndOtherAdjustments",
            vec!["PreferredStockDividendsAndOtherAdjustments"],
        );
        map.insert(
            "ProvisionForLoanLeaseAndOtherLosses",
            vec!["ProvisionForLoanLeaseAndOtherLosses"],
        );
        map.insert(
            "RedeemableNoncontrollingInterest",
            vec![
                "RedeemableNoncontrollingInterestEquityCarryingAmount",
                "RedeemableNoncontrollingInterestEquityCommonCarryingAmount",
                "RedeemableNoncontrollingInterestEquityPreferredCarryingAmount",
                "RedeemableNoncontrollingInterestEquityOtherCarryingAmount",
                "RedeemableNoncontrollingInterestEquityFairValue",
            ],
        );
        map.insert(
            "ResearchAndDevelopment",
            vec![
                "ResearchAndDevelopmentExpense",
                "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
                "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
                "ResearchAndDevelopmentInProcess",
                "ResearchAndDevelopmentAssetAcquiredOtherThanThroughBusinessCombinationWrittenOff",
            ],
        );
        map.insert(
            "Revenues",
            vec![
                "Revenues",
                "SalesRevenueNet",
                "SalesRevenueServicesNet",
                "SalesRevenueGoodsNet",
                "RevenuesNetOfInterestExpense",
                "HealthCareOrganizationRevenue",
                "InterestAndDividendIncomeOperating",
                "RealEstateRevenueNet",
                "RevenueMineralSales",
                "OilAndGasRevenue",
                "FinancialServicesRevenue",
                "RegulatedAndUnregulatedOperatingRevenue",
                // [Note: The document skips from 12 to 21 here]
                "ShippingAndHandlingRevenue",
                "SalesRevenueFromEnergyCommoditiesAndServices",
                "UtilityRevenue",
                "PhaseInPlanAmountOfCapitalizedCostsRecovered",
                "SecondaryProcessingRevenue",
                "RevenueSteamProductsAndServices",
                // [Note: The document skips from 27 to 30 here]
                "RevenueFromLeasedAndOwnedHotels",
                "FranchisorRevenue",
                "SubscriptionRevenue",
                "AdvertisingRevenue",
                "AdmissionsRevenue",
                "RevenueFromEnrollmentAndRegistrationFeesExcludingHospitalityEnterprises",
                "MembershipDuesRevenueOnGoing",
                "LicensesRevenue",
                "RoyaltyRevenue",
                "SalesOfOilAndGasProspects",
                "ClearingFeesRevenue",
                "ReimbursementRevenue",
                "RevenueFromGrants",
                "RevenueOtherManufacturedProducts",
                "ConstructionMaterialsRevenue",
                "TimberRevenue",
                "RecyclingRevenue",
                "OtherSalesRevenueNet",
                "SaleOfTrustAssetsToPayExpenses",
                "PassengerRevenue",
                "VehicleTollRevenue",
                "CargoAndFreightRevenue",
                // [Note: The document skips from 51 to 52 here]
                "NetInvestmentIncome",
                "RevenuesExcludingInterestAndDividends",
                "InvestmentBankingRevenue",
                "UnderwritingIncomeLoss",
                "MarketDataRevenue",
                "ElectricUtilityRevenue",
            ],
        );
        map.insert(
            "RevenuesExcludingInterestAndDividends",
            vec![
                "RevenuesExcludingInterestAndDividends",
                "BrokerageCommissionsRevenue",
                "InvestmentBankingRevenue",
            ],
        );
        map.insert(
            "RevenuesNetInterestExpense",
            vec!["RevenuesNetOfInterestExpense"],
        );
        map.insert(
            "TemporaryEquity",
            vec![
            "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests",
            "TemporaryEquityRedemptionValue",
            "RedeemablePreferredStockCarryingAmount",
            "TemporaryEquityCarryingAmount",
            "TemporaryEquityValueExcludingAdditionalPaidInCapital",
            "TemporaryEquityCarryingAmountAttributableToParent",
            "TemporaryEquityCarryingAmountAttributableToNoncontrollingInterest",
            "TemporaryEquityLiquidationPreference",
        ],
        );

        map
    });

static US_GAAP_MAPPING_INVERTED: Lazy<IndexMap<TaxonomyConceptName, Vec<FundamentalConceptName>>> =
    Lazy::new(|| {
        let map_inverted = invert_multivalue_indexmap(&US_GAAP_MAPPING);

        map_inverted
    });

/// Retrieves the fundamental accounting concepts corresponding to a given
/// US GAAP taxonomy concept, preserving the original "try order."
///
/// # Description
/// This function looks up a **US GAAP taxonomy concept** (`us-gaap:*`) and returns
/// a list of fundamental concept names (`Vec<&'static str>`) that reference it.
/// The order in the returned list **respects the original "try order"**
/// defined in the `US_GAAP_MAPPING` structure as documented in:
///
/// - [Mappings between Fundamental Accounting Concepts and US GAAP XBRL Taxonomy Concepts (Human readable)](http://www.xbrlsite.com/2014/Reference/Mapping.pdf)
///
/// The function relies on an **inverted** [`IndexMap`](https://docs.rs/indexmap/latest/indexmap/)
/// (`US_GAAP_MAPPING_INVERTED`) to ensure that both **insertion order is preserved**
/// and **lookups remain efficient (`O(1)`)**.
///
/// # Arguments
/// - `us_gaap_key`: The US GAAP taxonomy concept name (e.g., `"Assets"`, `"NetIncomeLoss"`).
///
/// # Returns
/// - `Some(Vec<&'static str>)` if the given taxonomy concept exists in the mapping.
/// - `None` if no corresponding fundamental concept is found.
///
/// # Order Preservation
/// The returned list maintains the **original "try order"** from the mapping.
/// This means that when multiple fundamental concepts are mapped to the same
/// taxonomy concept, they will be returned in the same order they were inserted.
///
/// # Complexity
/// - **Lookup:** `O(1)`, due to `IndexMap`.
/// - **Cloning:** `O(N)`, where `N` is the number of matching fundamental concepts.
///
/// # Example
/// ```
/// use sec_fetcher::accessor::get_us_gaap_human_readable_mapping;
///
/// fn main() {
///     let result = get_us_gaap_human_readable_mapping("AssetsCurrent");
///
///     assert_eq!(
///         result,
///         Some(vec![
///             "Assets",
///             "CurrentAssets",
///         ])
///     );
/// }
/// ```
pub fn get_us_gaap_human_readable_mapping(us_gaap_key: &str) -> Option<Vec<&'static str>> {
    let map_inverted = &US_GAAP_MAPPING_INVERTED;

    if let Some(values) = map_inverted.get(us_gaap_key) {
        Some(values.clone()) // Return a clone of the found values
    } else {
        None // Return None if no mapping is found
    }
}
