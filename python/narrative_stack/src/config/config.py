import os
import logging
from dotenv import load_dotenv
import sys

is_config_init = False


def init_config():
    """
    Initializes the package configuration.

    Note: Subsequent invocations will be ignored.
    """

    if is_config_init:
        return

    # TODO: Run `seed_everything` from here?

    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Detect if the current process is running under pytest
    is_test = "pytest" in sys.modules

    # Choose correct .env file
    env_filename = ".env.test" if is_test else ".env"
    env_path = os.path.join(script_dir, f"../../{env_filename}")

    # Load the .env file
    load_dotenv(dotenv_path=env_path)

    class ColorizedHandler(logging.StreamHandler):
        COLORS = {
            "ERROR": "\033[91m",  # Red
            "WARNING": "\033[93m",  # Yellow
            "INFO": "\033[92m",  # Green
            "DEBUG": "\033[0m",  # Reset to default
        }

        def emit(self, record):
            # Get the ANSI escape code for this log level
            color_code = self.COLORS.get(record.levelname, "\033[0m")
            # Add the ANSI escape code
            record.msg = f"{color_code}{record.msg}\033[0m"
            super().emit(record)

    # Configure logging format to include filename and line number for better traceability
    log_format = "%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"

    # Initialize the logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create the colorized handler and set the format
    handler = ColorizedHandler()
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
