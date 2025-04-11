import csv
import logging
from db import DB
from tqdm import tqdm

ALLOWED_NON_XBRLI_CONCEPT_TYPES = [
    "dtr-types:perShareItemType",
    "dtr-types:percentItemType",
    "dtr-types:volumeItemType",
    "srt-types:perUnitItemType"
]

# Note: The decision was made to use this CSV instead of the raw XBRL as it is easier to parse and to obtain
# the label and description.
#
# https://www.fasb.org/page/detail?pageId=/projects/FASB-Taxonomies/2025-gaap-financial-reporting-taxonomy.html
def upsert_us_gaap_concepts(db: DB, csv_data):
    """
    Upserts the US GAAP concept data into the database.

    Parameters:
    - db: The DB instance from your ORM.
    - csv_data
    """

    try:
        for row in tqdm(csv_data, desc="Importing US GAAP Concepts"):
            if row['prefix'] != "us-gaap" or (
                not row['type'].startswith("xbrli:") and 
                row['type'] not in ALLOWED_NON_XBRLI_CONCEPT_TYPES
            ):
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

        logging.info('US GAAP concept data has been successfully upserted.')
    except Exception as e:
        logging.error(f"Error upserting GAAP concept data: {e}")
        raise
