# Important: Ensure taxonomy is imported before running this script

import csv
import logging
from tqdm import tqdm
from db import DB

def upsert_ofss_concept_mappings(db: DB, csv_data: list[dict]) -> None:
    """
    Upserts mappings between US GAAP tags and OFSS category IDs. Skips
    concepts not found in the database and logs all unmapped entries.

    Args:
        db (DB): Database connection instance.
        csv_data (list[dict]): Parsed CSV rows with fields:
            - 'tag': US GAAP tag
            - 'balance': Optional balance type
            - 'period_type': Optional period type
            - 'ofss_id': OFSS category ID to associate
    """

    unmapped_concept_names = set()

    try:
        for row in tqdm(csv_data, desc="Importing OFSS mappings"):
            concept_name = row['tag']

            if concept_name.endswith("Text"):
                logging.debug(f"Skipping concept: {concept_name}")
                continue

            # statement_type = row['statement_type'] if row['statement_type'] else None
            balance = row['balance'] if row['balance'] else None
            period_type = row['period_type'] if row['period_type'] else None

            ofss_id = row['ofss_id'] if row['ofss_id'] else None

            if balance is not None:
                # Upsert balance type
                balance_type_id = db.upsert_entity('us_gaap_balance_type', {'balance': balance}, ['balance'])
            else:
                balance_type_id = None
            
            if period_type is not None:
                # Upsert period type
                period_type_id = db.upsert_entity('us_gaap_period_type', {'period_type': period_type}, ['period_type'])
            else:
                period_type_id = None
            
            # TODO: Remove entirely (this shouldn't be a part of the OFSS id mapping)
            # if statement_type is not None:
            #     # Upsert statement type
            #     statement_type_id = db.upsert_entity('us_gaap_statement_type', {'statement_type': statement_type}, ['statement_type'])
            # else:
            #     statement_type_id = None
            
            # Upsert us_gaap_concept (the tag itself)
            concept_data = {
                'name': concept_name,
                'balance_type_id': balance_type_id,
                'period_type_id': period_type_id,
            }
            
            # Look up the tag ID
            concept_row = db.get(
                "SELECT id FROM us_gaap_concept WHERE name = %s",
                ["id"],
                params=(concept_data['name'],)
            )

            if concept_row.empty:
                logging.warning("Skipping unmapped concept: %s", concept_name)
                unmapped_concept_names.add(concept_name)
                continue

            concept_id = concept_row.iloc[0]['id']

            if ofss_id is not None:
                # Aassociate tag with the ofss category
                db.upsert_entity('us_gaap_concept_ofss_category', {
                    'us_gaap_concept_id': concept_id,
                    'ofss_category_id': ofss_id,
                    'is_manually_mapped': True # These are considered manually mapped because the CSV was manually created
                }, ['us_gaap_concept_id', 'ofss_category_id'])

                # Note: This is ideal, but must know all category ids first
                # Upsert associations for `us_gaap_concept_ofss_category`
                # db.upsert_and_cleanup(
                #     table_name="us_gaap_concept_ofss_category",
                #     entity_id_name="us_gaap_concept_id",
                #     parent_id_name="ofss_category_id",
                #     parent_id=concept_id,
                #     associate_entities=[{'ofss_category_id': ofss_category_id} for ofss_category_id in [ofss_id]],
                #     upsert_datetime_field=None,
                #     unique_fields=["us_gaap_concept_id", "ofss_category_id"]
                # )

                # TODO: Remove entirely (this shouldn't be a part of the OFSS id mapping)
                # if statement_type_id is not None:
                #     # Associate tag with statement type
                #     db.upsert_entity('us_gaap_concept_statement_type', {
                #         'us_gaap_concept_id': concept_id,
                #         'us_gaap_statement_type_id': statement_type_id,
                #         'is_manually_mapped': False # These are *not* considered manually mapped because the original mapping was originally set in the CSV
                #     }, ['us_gaap_concept_id', 'us_gaap_statement_type_id'])

                #     # Note: This is ideal, but must know all statement ids first
                #     # db.upsert_and_cleanup(
                #     #     table_name="us_gaap_concept_statement_type",
                #     #     entity_id_name="us_gaap_concept_id",
                #     #     parent_id_name="us_gaap_statement_type_id",
                #     #     parent_id=concept_id,
                #     #     associate_entities=[{'us_gaap_statement_type_id': statement_type_id} for statement_type_id in [statement_type_id]],
                #     #     upsert_datetime_field=None,
                #     #     unique_fields=["us_gaap_concept_id", "us_gaap_statement_type_id"]
                #     # )

            # logging.debug(f"Upserted data for concept: {concept_name} with ofss_id: {ofss_id}")

        for concept_name in unmapped_concept_names:
            logging.warning(f"Unmapped concept: {concept_name}")
        logging.warning(f"Total unmapped concepts: {len(unmapped_concept_names)}")

        logging.info('Financial statement data has been successfully upserted.')
    except Exception as e:
        logging.error(f"Error upserting financial statement data: {e}")
        raise
