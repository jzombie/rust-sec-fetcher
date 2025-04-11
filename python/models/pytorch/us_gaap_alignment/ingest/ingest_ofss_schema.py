import logging
from db import DB

def insert_ofss_data(db: DB, schema_data, parent_category_id=None):
    """
    Inserts the OFSS data into the database recursively.

    Parameters:
    - db: The DB instance from your ORM.
    - schema_data (dict): The parsed OFSS schema data.
    - parent_category_id: The ID of the parent category for hierarchical categories.
    """
    try:
        for category_name, category_data in schema_data.items():
            logging.debug(f"Processing {category_name}: {type(category_data)}")

            # If category_data is an integer, it's an item (leaf node), not a category
            if isinstance(category_data, int):  # Leaf node, should be inserted as item
                item_data = {
                    'id': category_data,
                    'group_id': parent_category_id,
                    'category_name': category_name,
                }
                logging.debug(f"Inserting item: {item_data}")
                db.upsert_entity('ofss_category', item_data, ["group_id", "category_name"])

            else:  # Otherwise, it's a category (which may have subcategories)
                # Insert the category into the 'ofss_group' table
                category_data_to_insert = {
                    'group_name': category_name,
                    'parent_group_id': parent_category_id
                }
                logging.debug(f"Inserting category: {category_data_to_insert}")

                # Insert the category into the database and get the category id
                category_id = db.upsert_entity('ofss_group', category_data_to_insert, ["group_name"])

                # If the category_data is a dictionary, recurse for subcategories
                if isinstance(category_data, dict):  # Recursion for subcategories
                    insert_ofss_data(db, category_data, parent_category_id=category_id)

        logging.info('OFSS schema data has been successfully imported.')
    except Exception as e:
        logging.error(f"Error importing OFSS schema: {e}")
        raise
