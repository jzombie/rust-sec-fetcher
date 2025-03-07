use crate::{enums::FundamentalConcept, utils::invert_multivalue_indexmap};
use indexmap::IndexMap;
use once_cell::sync::Lazy;

pub type TaxonomyConceptName = &'static str;

// TODO: Move to `transform` directory
// Human-readable mapping: http://www.xbrlsite.com/2014/Reference/Mapping.pdf
static US_GAAP_MAPPING: Lazy<IndexMap<FundamentalConcept, Vec<TaxonomyConceptName>>> = Lazy::new(
    || {
        let mut map = IndexMap::new();

        // Entries are arranged by `Try Order` ascending

        map.insert(FundamentalConcept::Assets, vec!["Assets", "AssetsCurrent"]);
        map.insert(
            FundamentalConcept::BenefitsCostsExpenses,
            vec!["BenefitsLossesAndExpenses"],
        );
        map.insert(
            FundamentalConcept::CommitmentsAndContingencies,
            vec!["CommitmentsAndContingencies"],
        );
        map.insert(
            FundamentalConcept::ComprehensiveIncomeLoss,
            vec![
                "ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest",
                "ComprehensiveIncomeNetOfTax",
            ],
        );
        map.insert(
            FundamentalConcept::ComprehensiveIncomeLossAttributableToNoncontrollingInterest,
            vec![
                "ComprehensiveIncomeNetOfTaxAttributableToNoncontrollingInterest",
                "ComprehensiveIncome",
            ],
        );
        map.insert(
            FundamentalConcept::ComprehensiveIncomeLossAttributableToParent,
            vec!["ComprehensiveIncomeNetOfTax"],
        );
        map.insert(
            FundamentalConcept::CostOfRevenue,
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
            FundamentalConcept::CostsAndExpenses,
            vec!["CostsAndExpenses", "BenefitsLossesAndExpenses"],
        );
        map.insert(FundamentalConcept::CurrentAssets, vec!["AssetsCurrent"]);
        map.insert(
            FundamentalConcept::CurrentLiabilities,
            vec!["LiabilitiesCurrent"],
        );
        map.insert(
            FundamentalConcept::Equity,
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
            FundamentalConcept::EquityAttributableToNoncontrollingInterest,
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
            FundamentalConcept::EquityAttributableToParent,
            vec!["StockholdersEquity", "PartnersCapital", "MembersEquity"],
        );
        map.insert(
            FundamentalConcept::ExchangeGainsLosses,
            vec![
                "EffectOfExchangeRateOnCashAndCashEquivalents",
                "EffectOfExchangeRateOnCashAndCashEquivalentsContinuingOperations",
                "EffectOfExchangeRateOnCashContinuingOperations",
                "EffectOfExchangeRateOnCashAndCashEquivalentsDiscontinuedOperations",
            ],
        );
        map.insert(
            FundamentalConcept::ExtraordinaryItemsOfIncomeExpenseNetOfTax,
            vec!["ExtraordinaryItemNetOfTax"],
        );
        map.insert(
            FundamentalConcept::GainLossOnSalePropertiesNetTax,
            vec!["GainLossOnSaleOfPropertiesNetOfApplicableIncomeTaxes"],
        );
        map.insert(FundamentalConcept::GrossProfit, vec!["GrossProfit"]);
        map.insert(FundamentalConcept::IncomeLossBeforeEquityMethodInvestments, vec!["IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]);
        map.insert(FundamentalConcept::IncomeLossFromContinuingOperationsAfterTax, vec![
            "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
            "IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple",
            "IncomeLossFromContinuingOperations"
        ]);
        map.insert(FundamentalConcept::IncomeLossFromContinuingOperationsBeforeTax, vec![
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"
        ]);
        map.insert(
            FundamentalConcept::IncomeLossFromDiscontinuedOperationsNetOfTax,
            vec![
                "IncomeLossFromDiscontinuedOperationsNetOfTax",
                "DiscontinuedOperationGainLossOnDisposalOfDiscontinuedOperationNetOfTax",
                "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
            ],
        );
        map.insert(
            FundamentalConcept::IncomeLossFromEquityMethodInvestments,
            vec!["IncomeLossFromEquityMethodInvestments"],
        );
        map.insert(
            FundamentalConcept::IncomeStatementStartPeriodYearToDate,
            vec![
                "IncomeTaxExpenseBenefit",
                "IncomeTaxExpenseBenefitContinuingOperations",
                "FederalHomeLoanBankAssessments",
                "CurrentIncomeTaxExpenseBenefit",
            ],
        );
        map.insert(
            FundamentalConcept::IncomeTaxExpenseBenefit,
            vec![
                "IncomeTaxExpenseBenefit",
                "IncomeTaxExpenseBenefitContinuingOperations",
                "FederalHomeLoanBankAssessments",
                "CurrentIncomeTaxExpenseBenefit",
            ],
        );
        map.insert(
            FundamentalConcept::InterestAndDebtExpense,
            vec!["InterestAndDebtExpense"],
        );
        map.insert(
            FundamentalConcept::InterestAndDividendIncomeOperating,
            vec!["InterestAndDividendIncomeOperating"],
        );
        map.insert(
            FundamentalConcept::InterestAndDividendIncomeOperating,
            vec!["InterestAndDividendIncomeOperating"],
        );
        map.insert(
            FundamentalConcept::InterestExpenseOperating,
            vec!["InterestExpense"],
        );
        map.insert(
            FundamentalConcept::InterestIncomeExpenseAfterProvisionForLosses,
            vec!["InterestIncomeExpenseAfterProvisionForLoanLoss"],
        );
        map.insert(
            FundamentalConcept::InterestIncomeExpenseOperatingNet,
            vec!["InterestIncomeExpenseNet"],
        );
        map.insert(FundamentalConcept::Liabilities, vec!["Liabilities"]);
        map.insert(
            FundamentalConcept::LiabilitiesAndEquity,
            vec![
                "LiabilitiesAndStockholdersEquity",
                "LiabilitiesAndPartnersCapital",
            ],
        );
        map.insert(
        FundamentalConcept::NatureOfOperations,
        vec![
            "NatureOfOperations",
            "BusinessDescriptionAndBasisOfPresentationTextBlock",
            "OrganizationConsolidationAndPresentationOfFinancialStatementsDisclosureTextBlock",
            "OrganizationConsolidationBasisOfPresentationBusinessDescriptionAndAccountingPoliciesTextBlock",
        ],
    );
        map.insert(
            FundamentalConcept::NetCashFlow,
            vec![
                "CashAndCashEquivalentsPeriodIncreaseDecrease",
                "CashPeriodIncreaseDecrease",
                "NetCashProvidedByUsedInContinuingOperations",
            ],
        );
        map.insert(
            FundamentalConcept::NetCashFlowContinuing,
            vec!["NetCashProvidedByUsedInContinuingOperations"],
        );
        map.insert(
            FundamentalConcept::NetCashFlowDiscontinued,
            vec!["NetCashProvidedByUsedInDiscontinuedOperations"],
        );
        map.insert(
            FundamentalConcept::NetCashFlowFromFinancingActivities,
            vec!["NetCashProvidedByUsedInFinancingActivities"],
        );
        map.insert(
            FundamentalConcept::NetCashFlowFromFinancingActivitiesContinuing,
            vec!["NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
        );
        map.insert(
            FundamentalConcept::NetCashFlowFromFinancingActivitiesDiscontinued,
            vec!["CashProvidedByUsedInFinancingActivitiesDiscontinuedOperations"],
        );
        map.insert(
            FundamentalConcept::NetCashFlowFromInvestingActivities,
            vec!["NetCashProvidedByUsedInInvestingActivities"],
        );
        map.insert(
            FundamentalConcept::NetCashFlowFromInvestingActivitiesContinuing,
            vec!["NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
        );
        map.insert(
            FundamentalConcept::NetCashFlowFromInvestingActivitiesDiscontinued,
            vec!["CashProvidedByUsedInInvestingActivitiesDiscontinuedOperations"],
        );
        map.insert(
            FundamentalConcept::NetCashFlowFromOperatingActivities,
            vec!["NetCashProvidedByUsedInOperatingActivities"],
        );
        map.insert(
            FundamentalConcept::NetCashFlowFromOperatingActivitiesContinuing,
            vec!["NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        );
        map.insert(
            FundamentalConcept::NetCashFlowFromOperatingActivitiesDiscontinued,
            vec!["CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations"],
        );
        map.insert(
            FundamentalConcept::NetIncomeLoss,
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
            FundamentalConcept::NetIncomeLossAvailableToCommonStockholdersBasic,
            vec!["NetIncomeLossAvailableToCommonStockholdersBasic"],
        );
        map.insert(
            FundamentalConcept::NetIncomeLossAttributableToNoncontrollingInterest,
            vec![
                "NetIncomeLossAttributableToNoncontrollingInterest",
                "NetIncomeLossAttributableToNonredeemableNoncontrollingInterest",
                "NetIncomeLossAttributableToRedeemableNoncontrollingInterest",
                "IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity",
            ],
        );
        map.insert(
            FundamentalConcept::NetIncomeLossAttributableToParent,
            vec!["NetIncomeLoss"],
        );
        map.insert(
            FundamentalConcept::NoncurrentAssets,
            vec!["AssetsNoncurrent"],
        );
        map.insert(
            FundamentalConcept::NoncurrentLiabilities,
            vec!["LiabilitiesNoncurrent"],
        );
        map.insert(
            FundamentalConcept::NoninterestExpense,
            vec!["NoninterestExpense"],
        );
        map.insert(
            FundamentalConcept::NoninterestIncome,
            vec!["NoninterestIncome"],
        );
        map.insert(
            FundamentalConcept::NonoperatingIncomeLoss,
            vec!["NonoperatingIncomeExpense"],
        );
        map.insert(
            FundamentalConcept::OperatingExpenses,
            vec!["OperatingExpenses", "OperatingCostsAndExpenses"],
        );
        map.insert(
            FundamentalConcept::OperatingIncomeLoss,
            vec!["OperatingIncomeLoss"],
        );
        map.insert(
            FundamentalConcept::OtherComprehensiveIncomeLoss,
            vec!["OtherComprehensiveIncomeLossNetOfTax"],
        );
        map.insert(
            FundamentalConcept::OtherOperatingIncomeExpenses,
            vec!["OtherOperatingIncome"],
        );
        map.insert(
            FundamentalConcept::PreferredStockDividendsAndOtherAdjustments,
            vec!["PreferredStockDividendsAndOtherAdjustments"],
        );
        map.insert(
            FundamentalConcept::ProvisionForLoanLeaseAndOtherLosses,
            vec!["ProvisionForLoanLeaseAndOtherLosses"],
        );
        map.insert(
            FundamentalConcept::RedeemableNoncontrollingInterest,
            vec![
                "RedeemableNoncontrollingInterestEquityCarryingAmount",
                "RedeemableNoncontrollingInterestEquityCommonCarryingAmount",
                "RedeemableNoncontrollingInterestEquityPreferredCarryingAmount",
                "RedeemableNoncontrollingInterestEquityOtherCarryingAmount",
                "RedeemableNoncontrollingInterestEquityFairValue",
            ],
        );
        map.insert(
            FundamentalConcept::ResearchAndDevelopment,
            vec![
                "ResearchAndDevelopmentExpense",
                "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
                "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
                "ResearchAndDevelopmentInProcess",
                "ResearchAndDevelopmentAssetAcquiredOtherThanThroughBusinessCombinationWrittenOff",
            ],
        );
        map.insert(
            FundamentalConcept::Revenues,
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
            FundamentalConcept::RevenuesExcludingInterestAndDividends,
            vec![
                "RevenuesExcludingInterestAndDividends",
                "BrokerageCommissionsRevenue",
                "InvestmentBankingRevenue",
            ],
        );
        map.insert(
            FundamentalConcept::RevenuesNetInterestExpense,
            vec!["RevenuesNetOfInterestExpense"],
        );
        map.insert(
            FundamentalConcept::TemporaryEquity,
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
    },
);

static US_GAAP_MAPPING_INVERTED: Lazy<IndexMap<TaxonomyConceptName, Vec<FundamentalConcept>>> =
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
/// and **lookups remain efficient**.
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
/// # Example
/// ```
/// use sec_fetcher::accessor::distill_us_gaap_fundamental_concepts;
/// use sec_fetcher::enums::FundamentalConcept;
///
/// fn main() {
///     let result = distill_us_gaap_fundamental_concepts("AssetsCurrent");
///
///     assert_eq!(
///         result,
///         Some(vec![
///             FundamentalConcept::CurrentAssets,
///             FundamentalConcept::Assets,
///         ])
///     );
/// }
/// ```
pub fn distill_us_gaap_fundamental_concepts(us_gaap_key: &str) -> Option<Vec<FundamentalConcept>> {
    let map_inverted = &US_GAAP_MAPPING_INVERTED;
    let map_order = &US_GAAP_MAPPING; // Preserve Try Order from US_GAAP_MAPPING

    if let Some(mut values) = map_inverted.get(us_gaap_key).cloned() {
        if values.len() > 1 {
            // Compute Try Order by finding the first occurrence of `us_gaap_key` in **each fundamental concept's vector**
            let get_try_order = |concept: &FundamentalConcept| -> usize {
                map_order
                    .get(concept) // Get the corresponding vector for the fundamental concept
                    .and_then(|concepts| concepts.iter().position(|&c| c == us_gaap_key)) // Find index inside that vector
                    .map(|pos| pos + 1) // Convert to 1-based index
                    .unwrap_or(usize::MAX) // If never found, push to the end
            };

            {
                // Debugging: Print Correct Try Order
                for concept in &values {
                    let try_order = get_try_order(concept);
                    // TODO: Debug log
                    println!(
                        "Concept: {:?}, Try Order: {}, US-GAAP: {}",
                        concept, try_order, us_gaap_key
                    );
                }
                // TODO: Debug log
                println!("----");
            }

            // Sort by Correct Try Order
            values.sort_by_key(|concept| get_try_order(concept));
        }

        Some(values) // Returns `Vec<FundamentalConcept>` directly
    } else {
        None
    }
}
