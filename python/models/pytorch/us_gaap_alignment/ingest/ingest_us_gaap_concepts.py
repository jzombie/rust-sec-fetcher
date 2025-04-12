import csv
import logging
from db import DB
from tqdm import tqdm

# ============================================================================
# ALLOWED_CONCEPT_TYPES
# ----------------------------------------------------------------------------
# This list defines the subset of XBRL and US GAAP data types permitted for
# alignment to OFSS financial statement categories. Only concepts with one of
# these types will be considered during ingestion and mapping.
#
# These types were selected based on:
# - Relevance to numerical financial reporting (e.g., dollar values, shares)
# - Suitability for autoencoding, vector embedding, or semantic comparison
# - Exclusion of non-numeric or non-quantitative data (e.g., booleans, enums)
#
# The list includes:
# - Percentages, per-share values, monetary units, share counts
# - Volumetric and per-unit measures (srt/dtr volume)
# - Integer and decimal types where the semantics are clearly quantitative
# - Interest rates and cash/operational flow concepts over time
#
# NOTE:
# - Types like stringItemType, pureItemType, booleanItemType, etc., are excluded
#   unless explicitly whitelisted elsewhere.
ALLOWED_CONCEPT_TYPES = [
    "dtr-types:percentItemType",
    "dtr-types:perShareItemType",
    "xbrli:monetaryItemType",
    "xbrli:sharesItemType",
    "dtr-types:volumeItemType",
    "srt-types:perUnitItemType",
    "xbrli:decimalItemType",
    "xbrli:durationItemType",
    "us-types:interestRateItemType",
    "xbrli:integerItemType",
    "dtr-types:flowItemType"
]

# Note: The decision was made to use this CSV instead of the raw XBRL as it is easier to parse and to obtain
# the label and description.
#
# https://www.fasb.org/page/detail?pageId=/projects/FASB-Taxonomies/2025-gaap-financial-reporting-taxonomy.html
def upsert_us_gaap_concepts(db: DB, csv_data: list[dict]) -> None:
    """
    Upserts base-level US GAAP concepts into the database, including their
    concept type, balance type, period type, label, and documentation fields.

    Only records with a 'us-gaap' prefix and a valid concept type are imported.

    Args:
        db (DB): Database connection instance.
        csv_data (list[dict]): Parsed CSV records with fields like:
            - 'name': US GAAP tag
            - 'type': XBRL or DTR/SRT item type
            - 'balance': 'debit' or 'credit' (optional)
            - 'periodType': 'instant' or 'duration' (optional)
            - 'label': Human-readable label (optional)
            - 'documentation': Description (optional)
    """

    discarded_us_gaap_concept_types = set()

    try:
        for row in tqdm(csv_data, desc="Importing US GAAP Concepts"):
            if row['prefix'] != "us-gaap" or (
                row['type'] not in ALLOWED_CONCEPT_TYPES
            ):
                if row['prefix'] == 'us-gaap':
                    discarded_us_gaap_concept_types.add(row['type'])
                continue

            name = row['name']
            concept_type = row['type']
            balance = row['balance'] if row['balance'] else None
            period_type = row['periodType'] if row['periodType'] else None
            label = row['label'] if row['label'] else None
            documentation = row['documentation'] if row['documentation'] else None

            concept_type_id = db.upsert_entity('us_gaap_concept_type', {'concept_type': concept_type}, ['concept_type'])

            # Upsert balance type if provided
            if balance is not None:
                balance_type_id = db.upsert_entity('us_gaap_balance_type', {'balance': balance}, ['balance'])
            else:
                balance_type_id = None
            
            # Upsert period type if provided
            if period_type is not None:
                period_type_id = db.upsert_entity('us_gaap_period_type', {'period_type': period_type}, ['period_type'])
            else:
                period_type_id = None
            
            # Upsert the concept itself in the `us_gaap_concept` table
            concept_data = {
                'name': name,
                'concept_type_id': concept_type_id,
                'balance_type_id': balance_type_id,
                'period_type_id': period_type_id,
                'label': label,
                'documentation': documentation,
            }

            concept_id = db.upsert_entity('us_gaap_concept', concept_data, ['name'])

            # logging.debug(f"Upserted data for concept ({concept_id}): {name}")

        for discarded_type in discarded_us_gaap_concept_types:
            logging.warning(f"Discarded US GAAP concept type: {discarded_type}")
        logging.warning(f"Total discarded US GAAP {len(discarded_us_gaap_concept_types)} concept types.")

        logging.info('US GAAP concept data has been successfully upserted.')
    except Exception as e:
        logging.error(f"Error upserting GAAP concept data: {e}")
        raise
