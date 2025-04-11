import logging
from db import DB
from tqdm import tqdm

def upsert_us_gaap_description_variations(db: DB, csv_data):
    """
    Inserts GAAP concept description variations into the database.

    Parameters:
    - db: The DB instance from your ORM.
    - csv_data (list): The parsed CSV data.
    """
    try:
        for row in tqdm(csv_data, desc="Importing variations"):
            concept_name = row['tag']
            variation_text = row['description_variation'].lower()

            if not concept_name or not variation_text:
                continue

            # Look up the concept ID
            concept_row = db.get(
                "SELECT id FROM us_gaap_concept WHERE name = %s",
                ["id"],
                params=(concept_name,)
            )

            if concept_row.empty:
                # logging.warning("Skipping unknown concept: %s", concept_name)
                continue

            concept_id = concept_row.iloc[0]['id']

            # Insert variation (no deduplication enforced by schema)
            db.upsert_entity(
                table_name="us_gaap_concept_description_variation",
                field_dict={
                    'us_gaap_concept_id': concept_id,
                    'text': variation_text
                },
                unique_fields=["us_gaap_concept_id", "text"]
            )

            # logging.debug("Inserted variation for concept: %s", concept_name)

        logging.info("All description variations imported successfully.")

    except Exception as e:
        logging.error(f"Error importing description variations: {e}")
        raise
