import logging
import mysql.connector
from .db import DB


# IMPORTANT: This script will drop and recreate the given schema and will DELETE ALL DATA!
def reset_schema(db: DB, schema_name: str) -> None:
    """
    Drops and recreates the given schema. WARNING: This will delete all data.

    Parameters:
        schema_name (str): The name of the schema (database) to reset.
    """
    try:
        cursor = db.conn.cursor()

        # Drop the schema
        cursor.execute(f"DROP DATABASE IF EXISTS `{schema_name}`")

        # Recreate the schema
        cursor.execute(f"CREATE DATABASE `{schema_name}`")

        logging.info("Schema '%s' has been reset successfully.", schema_name)

        cursor.close()
    except mysql.connector.Error as e:
        logging.error("Failed to reset schema '%s': %s", schema_name, e)
        raise
