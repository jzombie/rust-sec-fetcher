import logging
from db import DB
from tqdm import tqdm

def upsert_us_gaap_description_variations(db: DB, csv_data: list[dict]) -> None:
    """
    Upserts description variation text for US GAAP concepts into the database.

    Each variation is lowercased and tied to its corresponding concept via
    foreign key lookup. If the concept cannot be found, the variation is skipped.

    Args:
        db (DB): Database connection instance.
        csv_data (list[dict]): Parsed CSV records with keys:
            - 'tag': GAAP concept name.
            - 'description_variation': Free-text variation to be embedded.
    """
    
    unmapped_concept_names = set()

    try:
        for row in tqdm(csv_data, desc="Importing variations"):
            concept_name = row['tag']
            variation_text = row['description_variation'].lower()

            # Skip if either field is empty
            # The hand-rolled CSV file may have skipped variation text representing the original value
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
                unmapped_concept_names.add(concept_name)
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

        for concept_name in unmapped_concept_names:
            logging.warning(f"Unmapped concept: {concept_name}")
        logging.warning("Total unmapped concepts: %d", len(unmapped_concept_names))

        logging.info("All description variations imported successfully.")

    except Exception as e:
        logging.error(f"Error importing description variations: {e}")
        raise
