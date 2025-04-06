import logging
import sys
import traceback
from datetime import datetime, date
from contextlib import contextmanager
from typing import Dict, List, Any
import mysql.connector
import pandas as pd
import numpy as np
from .db_connector import DBConnector



class DB(DBConnector):
    def get(self, query: str, columns: List[str], params: tuple = None) -> pd.DataFrame:
        """
        Execute a query and return the results as a DataFrame.

        Parameters:
            query (str): The SQL query to execute.
            columns (List[str]): List of column names for the resulting DataFrame.
            params (tuple, optional): A tuple of parameters to be used with the SQL query.
                This helps prevent SQL injection by safely inserting variables into the query.
                Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the query results, with the specified columns.

        Raises:
            mysql.connector.Error: If an error occurs during query execution.

        Example:
            To retrieve data safely using parameters:
            query = "SELECT * FROM users WHERE user_id = %s"
            params = (123,)
            df = self.get(query, ["user_id", "user_name"], params=params)
        """
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
        except mysql.connector.Error as err:
            logging.error("Failed to fetch data: %s", err)
            raise  # Re-raise the caught exception

        # TODO: Apply normalization as needed
        df = pd.DataFrame(results, columns=columns)
        return df

    @staticmethod
    def parse_datetime(d: str | date | datetime | None) -> datetime | None:
        """
        Parses a date, datetime, or string representing a datetime into a datetime object.
        If the input is None, it returns None. If the input is a string, it attempts to parse it
        into a datetime object. Handles both date-only and full datetime strings.
        """
        if d is None:
            return None
        if isinstance(d, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(d, fmt)
                except ValueError:
                    pass
            logging.error("Datetime string %s is not in the expected format.", d)
            raise ValueError(
                f"Input string {d} does not match expected datetime formats."
            )
        if isinstance(d, date) and not isinstance(d, datetime):
            return datetime(
                d.year, d.month, d.day
            )  # Convert date to datetime at midnight
        return d

    @staticmethod
    def to_sql_datetime(d: str | date | datetime | None) -> str:
        """
        Converts a datetime object to a SQL datetime string (YYYY-MM-DD HH:MM:SS).
        """
        dt = DB.parse_datetime(d)
        if dt is None:
            return None
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def to_sql_date(d: str | date | datetime | None) -> str:
        """
        Converts a datetime object to a SQL date string (YYYY-MM-DD) by parsing the input
        and then formatting it correctly as a date.
        """
        dt = DB.parse_datetime(d)
        if dt is None:
            return None
        return dt.date().strftime("%Y-%m-%d")

    @contextmanager
    def capture_errors(self, arbitrary_identifier=None):
        """
        Context manager to capture and log errors to the database.
        """
        try:
            yield
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            exc_type, exc_value, exc_traceback = sys.exc_info()

            # Extract stack trace
            stack_trace = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )

            # Get the last function where the exception occurred in user's code
            for trace in reversed(traceback.extract_tb(e.__traceback__)):
                if "site-packages" not in trace.filename:
                    last_trace = trace
                    break
            else:
                last_trace = traceback.extract_tb(e.__traceback__)[-1]

            function_name = last_trace.name
            filename = last_trace.filename
            line_number = last_trace.lineno
            timestamp = datetime.now()

            # Get the class name
            class_name = self.__class__.__name__

            # Use existing connection to log to the database
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO error_log (arbitrary_identifier, error_type, error_message, stack_trace,
                                    function_name, filename, line_number, timestamp, class_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    arbitrary_identifier,
                    error_type,
                    error_message,
                    stack_trace,
                    function_name,
                    filename,
                    line_number,
                    timestamp,
                    class_name,
                ),
            )
            self.conn.commit()
            cursor.close()

            logging.error(
                "[%s] Logged error to database: %s %s in function %s at %s:%s",
                class_name,
                error_type,
                error_message,
                function_name,
                filename,
                line_number,
            )

    @contextmanager
    def transaction(self, arbitrary_identifier=None):
        with self.capture_errors():
            try:
                # Start the transaction
                self.conn.start_transaction(arbitrary_identifier)
                yield
                # Commit the transaction
                self.conn.commit()
            except Exception as e:
                # Rollback the transaction in case of error
                self.conn.rollback()
                logging.error("Transaction failed and has been rolled back: %s", e)
                raise

    def upsert_entity(
        self,
        table_name: str,
        field_dict: dict,
        unique_fields: List[str] = None,
        retry: bool = False,
        trim_strings: bool = True,
    ) -> int | None:
        """
        Inserts or updates field values into the specified table and fields,
        then returns its ID. If the record exists, it updates the record.

        Parameters:
        - table_name (str): The name of the table to insert or update.
        - field_dict (dict): A dictionary of field_name: field_value pairs.
        - unique_fields (List[str]): A list of field names that uniquely identify a record.
        - retry (bool): A flag to indicate whether this call is a retry after handling
          specific errors (e.g., out-of-range errors). This prevents infinite recursion
          by ensuring retries happen only once per operation. Defaults to False.
        - trim_strings (bool): A flag to indicate whether or not strings should be trimmed.

        Returns:
        - int | None: The ID of the inserted or updated record.
        """

        try:
            # Ensure field_dict is provided
            if not field_dict:
                return None  # Or raise an error

            # Convert numpy types to standard Python types
            for key, value in field_dict.items():
                if isinstance(value, (np.integer, np.int64)):
                    field_dict[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    field_dict[key] = float(value)
                elif isinstance(
                    value, (np.bool_, bool)
                ):  # Handle both numpy and standard Python bool
                    field_dict[key] = (
                        1 if value else 0
                    )  # Convert boolean to integer 1 or 0
                elif value is None:
                    field_dict[key] = (
                        None  # This remains None to be interpreted as NULL in SQL
                    )
                else:
                    # Directly convert everything else to a string to handle HttpUrl or AnyUrl
                    field_dict[key] = str(value)

                    if trim_strings:
                        field_dict[key] = field_dict[key].strip()

            # Create a cursor object for executing SQL queries
            cursor = self.conn.cursor()

            field_names = ", ".join(field_dict.keys())
            field_values = tuple(field_dict.values())
            placeholders = ", ".join(["%s"] * len(field_dict))

            # If unique_fields is not provided, use all fields as unique identifiers
            if unique_fields is None:
                unique_fields = list(field_dict.keys())

            # Create the WHERE clause for unique fields
            where_clause = " AND ".join(f"{key} = %s" for key in unique_fields)
            unique_values = tuple(field_dict[key] for key in unique_fields)

            # SQL query to get ID based on the provided unique field values
            sql_select_query = f"SELECT id FROM {table_name} WHERE {where_clause};"

            # Execute the SELECT query
            cursor.execute(sql_select_query, unique_values)

            # Fetch the first result (if any)
            result = cursor.fetchone()

            # If the value already exists in the table, proceed to update it
            if result:
                set_clause = ", ".join(f"{key} = %s" for key in field_dict.keys())
                update_values = field_values + unique_values
                sql_update_query = (
                    f"UPDATE {table_name} SET {set_clause} WHERE {where_clause};"
                )
                cursor.execute(sql_update_query, update_values)
                self.conn.commit()
                cursor.close()
                return result[0]

            # If the value does not exist in the table, proceed to insert it
            sql_insert_query = (
                f"INSERT INTO {table_name} ({field_names}) VALUES ({placeholders});"
            )

            # Execute the INSERT query
            cursor.execute(sql_insert_query, field_values)

            # Commit the transaction to make the insertion permanent
            self.conn.commit()

            # Re-execute the SELECT query to fetch the ID of the newly inserted value
            cursor.execute(sql_select_query, unique_values)
            result = cursor.fetchone()

            # If we successfully inserted the value and fetched its ID, return the ID
            if result:
                return result[0]
            else:
                raise ValueError(f"Could not upsert entity into table: {table_name}")

        except mysql.connector.Error as e:
            if e.errno == 1264 and not retry:  # Handle out-of-range errors gracefully
                logging.warning(
                    "Out-of-range error for table %s: %s. Retrying with NO_ENGINE_SUBSTITUTION.",
                    table_name,
                    str(e),
                )
                # Dynamically set NO_ENGINE_SUBSTITUTION and retry the operation
                cursor.execute("SET SESSION sql_mode = 'NO_ENGINE_SUBSTITUTION';")
                cursor.close()  # Close the cursor before retrying
                return self.upsert_entity(
                    table_name,
                    field_dict,
                    unique_fields,
                    retry=True,
                    trim_strings=trim_strings,
                )
            else:
                raise  # Re-raise other errors
        finally:
            if cursor:
                # Close the cursor now that we're done with it
                cursor.close()

    def upsert_and_cleanup(
        self,
        table_name: str,
        entity_id_name: str,
        parent_id_name: str,
        parent_id: int,
        new_entities: List[Dict[str, Any]],
        unique_fields: List[str],
        datetime_field: str = "upsert_datetime",
        trim_strings: bool = True,
    ) -> None:
        """
        A utility method to upsert entities and remove those that are no longer associated.

        IMPORTANT! This assumes that a unique `id` field identifies each row.

        Parameters:
        - table_name (str): The name of the table to upsert into.
        - entity_id_name (str): The name of the entity ID field in the table.
        - parent_id_name (str): The name of the parent ID field in the table.
        - parent_id (int): The ID of the parent entity.
        - new_entities (List[Dict[str, Any]]): A list of dictionaries representing new entities.
        - unique_fields (List[str]): A list of fields that uniquely identify a record.
        - datetime_field (str): The name of the datetime field for the upsert operation (default is 'upsert_datetime').
        - trim_strings (str): A flag to indicate whether or not strings should be trimmed.
        """

        cursor = self.conn.cursor()

        # Fetch current entities
        current_entities_query = f"""
        SELECT id FROM {table_name} WHERE {parent_id_name} = %s
        """
        cursor.execute(current_entities_query, (parent_id,))
        current_entity_ids = {row[0] for row in cursor.fetchall()}

        # Track upsert entities
        upsert_entity_ids = set()
        total_new_entities = len(new_entities)
        for idx, entity in enumerate(new_entities):
            if total_new_entities > 100 and idx % 100 == 0:
                logging.info(
                    f"Performing entity upsert: {idx + 1} of {total_new_entities}"
                )

            # Ensure datetime field is set
            entity[datetime_field] = self.to_sql_datetime(datetime.now())

            # Upsert the entity
            entity_id = self.upsert_entity(
                table_name=table_name,
                field_dict={parent_id_name: parent_id, **entity},
                unique_fields=unique_fields,
                trim_strings=trim_strings,
            )
            upsert_entity_ids.add(entity_id)

        # Identify entities to remove
        entities_to_remove = current_entity_ids - upsert_entity_ids

        # Remove outdated entities
        if entities_to_remove:
            delete_query = f"""
            DELETE FROM {table_name} WHERE {parent_id_name} = %s AND {entity_id_name} IN (%s)
            """ % (
                parent_id,
                ", ".join(map(str, entities_to_remove)),
            )
            cursor.execute(delete_query)
            self.conn.commit()

        cursor.close()

    def batch_upsert_entities(
        self,
        table_name: str,
        field_dicts: List[Dict[str, any]],
        unique_fields: List[str] = None,
    ) -> None:
        """
        Batch inserts or updates field values into the specified table and fields.

        IMPORTANT: This version can lead to `id` fragmentation! If `id` fragmentation
        should be avoided, use the slower `upsert_entity` instead. Because of the
        potential for heavy `id` fragmentation, it is highly recommended to use
        the `BIGINT` type on the primary key, if auto incrementing.

        Parameters:
        - table_name (str): The name of the table to insert or update.
        - field_dicts (List[Dict[str, any]]): A list of dictionaries of field_name: field_value pairs.
        - unique_fields (List[str]): A list of field names that uniquely identify a record.
        """

        if not field_dicts:
            return

        # Escape field names with backticks
        field_names = ", ".join(f"`{key}`" for key in field_dicts[0].keys())
        placeholders = ", ".join(["%s"] * len(field_dicts[0]))

        # Create the ON DUPLICATE KEY UPDATE clause
        update_clause = ", ".join(
            [f"`{key}` = VALUES(`{key}`)" for key in field_dicts[0].keys()]
        )

        # Construct the full SQL query for batch insert/update
        sql = f"""
            INSERT INTO `{table_name}` ({field_names})
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {update_clause};
        """

        # Prepare the list of tuples for executemany
        values = [tuple(record.values()) for record in field_dicts]

        # Execute the batch upsert query
        cursor = self.conn.cursor()
        cursor.executemany(sql, values)
        self.conn.commit()
        cursor.close()
