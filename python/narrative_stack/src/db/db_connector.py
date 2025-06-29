import os
import logging
from config import init_config, DBConfig
import time
import mysql.connector
import warnings
from contextlib import contextmanager


init_config()


class DBConnector:
    # IMPORTANT: This overrides warning.showwarning handling and could likely
    # be improved further!
    def custom_warning_handler(
        self,
        message,
        category,  # filename, lineno, file=None, line=None
    ):
        if (
            category is UserWarning
            and message
            == "pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy."
        ):
            pass

    # Destructor method
    def __del__(self):
        warnings.showwarning = self.default_warning_handler

    def __init__(self, db_config: DBConfig):
        self.db_config = db_config

        # Establish the connection
        self.connect()

        # TODO: Uncomment?
        # self.keep_alive_thread = threading.Thread(target=self.keep_alive)
        # self.keep_alive_thread.daemon = (
        #     True  # Set daemon to True to exit the thread when the main program exits
        # )
        # self.keep_alive_thread.start()

        self.default_warning_handler = warnings.showwarning

        # Attach the custom warning handler
        warnings.showwarning = self.custom_warning_handler

    def connect(self):
        """Establish the connection to the database."""

        self.conn = mysql.connector.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            user=self.db_config.user,
            password=self.db_config.password,
            database=self.db_config.database,
            # connection_timeout=10,  # Timeout for the connection attempt (in seconds)
            # read_timeout=30,        # Timeout for waiting for response from server (in seconds)
            # write_timeout=30        # Timeout for sending data to server (in seconds)
        )

    def disconnect(self):
        """Disconnect from the database."""
        try:
            self.conn.close()
            logging.info("Successfully disconnected from MySQL database.")
        except mysql.connector.Error as err:
            logging.error("MySQL Error while disconnecting: %s", err)

    def keep_alive(self):
        """Ping the server at regular intervals to keep the connection alive."""
        while True:
            try:
                # TODO: Watch out for:
                # 2023-10-11 08:37:02,206 - [db_connector.py:32] - DEBUG - sending MySQL keepalive ping
                # Python(36068,0x16d2a3000) malloc: *** error for object 0x6000016f8460: pointer being freed was not allocated
                # Python(36068,0x16d2a3000) malloc: *** set a breakpoint in malloc_error_break to debug

                logging.debug("sending MySQL keepalive ping")
                self.conn.ping()
                time.sleep(30)  # Sleep for 30 seconds
            except mysql.connector.Error as err:
                logging.error("MySQL Error: %s", err)
                self.connect()  # Reconnect if an error occurs

    @contextmanager
    def temporary_cursor(self, max_execution_time=None):
        """
        Create a context-managed cursor with specified timeouts and execution time limit.

        Args:
            max_execution_time (int): Maximum execution time for queries in this session, in seconds.

        Yields:
            tuple: A tuple containing the cursor and connection for database operations within a context block.
        """
        temp_conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            port=os.getenv("MYSQL_PORT"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
        )
        try:
            temp_cursor = temp_conn.cursor()

            # Set maximum execution time for the session
            if max_execution_time is not None:
                sql_max_execution_time = max_execution_time * 1000

                logging.debug(
                    f"Setting SQL custom max_execution_time to {sql_max_execution_time}"
                )

                # https://dev.mysql.com/doc/refman/8.3/en/server-system-variables.html#sysvar_max_execution_time
                temp_cursor.execute(
                    f"SET SESSION max_execution_time = {sql_max_execution_time};"
                )  # Convert to milliseconds

            yield temp_cursor, temp_conn
        finally:
            if temp_cursor:
                temp_cursor.close()
            if temp_conn:
                temp_conn.close()
