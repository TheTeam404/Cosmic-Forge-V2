# --- START OF ENHANCED FILE utils/helpers.py ---
"""
Utility functions used across the application, including project root finding,
logging setup, and simple mathematical helpers.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional

# --- Project Root Finding ---

# Cache the project root directory
_project_root: Optional[str] = None

def get_project_root() -> str:
    """
    Finds and caches the project root directory.

    The project root is crucial for locating assets, logs, and configuration files
    relative to the application's base, regardless of where it's run from.

    Strategies (in order):
    1. Checks if the directory containing this file, or its parents (up to 4 levels),
       contain common project markers ('pyproject.toml', '.git').
    2. Checks if the directory containing this file, or its parents (up to 4 levels),
       contain application-specific markers ('main.py', 'config.yaml').
    3. Falls back to the parent directory of this utils.py file, issuing a warning.

    Returns:
        str: The absolute path to the determined project root.

    Raises:
        FileNotFoundError: If the root cannot be reasonably determined and strict
                           error handling is preferred over the fallback (currently commented out).
    """
    global _project_root
    if _project_root is not None:
        return _project_root

    try:
        # Start from the directory containing this utils.py file
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ not defined (e.g., interactive mode) - use current working directory
        current_dir = os.getcwd()
        logging.warning(
            "__file__ not defined, using current working directory '%s' "
            "as starting point for project root search.", current_dir
        )

    search_dir = current_dir
    found_root = None
    max_levels = 4 # Limit search depth to prevent excessive searching

    # --- Strategy 1: Look for standard markers ---
    project_markers = ['pyproject.toml', '.git']
    logging.debug("Searching for project markers (%s) starting from '%s'", ", ".join(project_markers), search_dir)
    temp_search_dir = search_dir
    for level in range(max_levels + 1): # Check current dir + max_levels parents
        if any(os.path.exists(os.path.join(temp_search_dir, marker)) for marker in project_markers):
            found_root = temp_search_dir
            logging.debug("Found project marker in '%s' at level %d.", found_root, level)
            break
        parent = os.path.dirname(temp_search_dir)
        if parent == temp_search_dir:
            logging.debug("Reached filesystem root during marker search.")
            break # Reached filesystem root
        temp_search_dir = parent

    # --- Strategy 2: Look for application-specific markers ---
    if found_root is None:
        app_markers = ['main.py', 'config.yaml']
        logging.debug("Searching for app markers (%s) starting from '%s'", ", ".join(app_markers), search_dir)
        temp_search_dir = search_dir # Reset search start
        for level in range(max_levels + 1):
             # Check if *both* markers exist
            if all(os.path.exists(os.path.join(temp_search_dir, marker)) for marker in app_markers):
                found_root = temp_search_dir
                logging.debug("Found app markers in '%s' at level %d.", found_root, level)
                break
            parent = os.path.dirname(temp_search_dir)
            if parent == temp_search_dir:
                logging.debug("Reached filesystem root during app marker search.")
                break
            temp_search_dir = parent

    # --- Strategy 3: Fallback ---
    if found_root is None:
        # Fallback: assume utils.py is one level below root
        fallback_root = os.path.dirname(current_dir)
        logging.warning(
            "Could not definitively find project root using markers (%s or %s). "
            "Falling back to the parent directory of the utils module: '%s'. "
            "Consider adding a marker file (e.g., 'pyproject.toml') to your project root for reliability.",
            ", ".join(project_markers), ", ".join(app_markers), fallback_root
        )
        # --- Stricter Option (Uncomment to raise error instead of fallback) ---
        # raise FileNotFoundError(
        #     "Could not determine project root. Place a marker file "
        #     "('pyproject.toml', '.git') or ensure 'main.py' and 'config.yaml' "
        #     "exist in the root directory."
        # )
        # --- End Stricter Option ---
        found_root = fallback_root # Use fallback if not raising error

    _project_root = os.path.abspath(found_root) # Store absolute path
    logging.info("Project root determined as: %s", _project_root)
    return _project_root

# --- Logging Setup ---

def setup_logging(log_config: Optional[Dict[str, Any]] = None):
    """
    Configures the root logger with console and rotating file handlers.

    Reads settings from log_config or uses sensible defaults.
    Removes any previously configured handlers on the root logger to prevent duplication.

    Args:
        log_config (Optional[Dict[str, Any]]): Dictionary usually loaded from
            the 'logging' section of a config file. Expected keys include:
            'log_dir', 'log_file_name', 'log_level_console', 'log_level_file',
            'log_format', 'log_date_format', 'log_max_bytes', 'log_backup_count'.
    """
    if log_config is None:
        log_config = {}

    try:
        # --- Determine Log Directory and File ---
        try:
            project_root = get_project_root() # Ensure root is found first
        except FileNotFoundError as e_root:
            # If root finding fails critically, logging setup cannot proceed relative to root
            logging.basicConfig(level=logging.ERROR) # Basic config for this error message
            logging.error("Cannot set up logging relative to project root: %s", e_root)
            print(f"ERROR: Cannot set up logging relative to project root: {e_root}", file=sys.stderr)
            return # Abort logging setup

        log_dir_relative = log_config.get('log_dir', 'logs') # Default directory name
        log_directory = os.path.join(project_root, log_dir_relative)

        try:
            os.makedirs(log_directory, exist_ok=True) # Create log directory if it doesn't exist
        except OSError as e_dir:
            logging.basicConfig(level=logging.ERROR) # Basic config for this error message
            logging.error("Failed to create log directory '%s': %s", log_directory, e_dir, exc_info=True)
            print(f"ERROR: Failed to create log directory '{log_directory}': {e_dir}", file=sys.stderr)
            return # Abort if log directory cannot be created

        log_filename = log_config.get('log_file_name', 'app.log') # Default log file name
        log_filepath = os.path.join(log_directory, log_filename)

        # --- Get Logging Levels ---
        # Ensure defaults are valid level names
        default_console_level_str = 'INFO'
        default_file_level_str = 'DEBUG'

        console_level_str = str(log_config.get('log_level_console', default_console_level_str)).upper()
        file_level_str = str(log_config.get('log_level_file', default_file_level_str)).upper()

        # Get numeric level, falling back to default numeric level if string is invalid
        console_level = getattr(logging, console_level_str, getattr(logging, default_console_level_str))
        file_level = getattr(logging, file_level_str, getattr(logging, default_file_level_str))

        if console_level_str not in logging._nameToLevel:
             logging.warning("Invalid console log level '%s' in config. Using default '%s'.", console_level_str, default_console_level_str)
        if file_level_str not in logging._nameToLevel:
             logging.warning("Invalid file log level '%s' in config. Using default '%s'.", file_level_str, default_file_level_str)


        # Root logger level should be the *lowest* of the handler levels to allow all messages through
        root_level = min(console_level, file_level)

        # --- Get Formatting ---
        log_format_string = log_config.get(
            'log_format',
            '%(asctime)s - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s'
        )
        log_date_format_string = log_config.get('log_date_format', '%Y-%m-%d %H:%M:%S')

        try:
            log_formatter = logging.Formatter(log_format_string, datefmt=log_date_format_string)
        except ValueError as e_fmt:
            logging.basicConfig(level=logging.ERROR)
            logging.error("Invalid log format string or date format string: %s", e_fmt)
            print(f"ERROR: Invalid log format string or date format string: {e_fmt}", file=sys.stderr)
            # Fallback to basic formatting
            log_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

        # --- Configure Root Logger ---
        root_logger = logging.getLogger()
        root_logger.setLevel(root_level)

        # Remove existing handlers to prevent duplicates
        if root_logger.hasHandlers():
            logging.debug("Removing %d existing logging handlers.", len(root_logger.handlers))
            for handler in root_logger.handlers[:]: # Iterate over a copy
                try:
                    handler.close()
                except Exception: pass # Ignore errors during handler close
                root_logger.removeHandler(handler)

        # --- Create and Add Console Handler ---
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(log_formatter)
            root_logger.addHandler(console_handler)
        except Exception as e_console:
            logging.basicConfig(level=logging.ERROR)
            logging.error("Failed to set up console logging handler: %s", e_console, exc_info=True)
            print(f"ERROR: Could not setup console logging: {e_console}", file=sys.stderr)
            # Attempt to continue with file logging if possible

        # --- Create and Add File Handler ---
        try:
            # Ensure max_bytes and backup_count are sensible integers
            max_bytes = int(log_config.get('log_max_bytes', 5 * 1024 * 1024)) # Default 5MB
            backup_count = int(log_config.get('log_backup_count', 4)) # Default 4 backups
            if max_bytes <= 0:
                 logging.warning("log_max_bytes must be positive. Using default 5MB.")
                 max_bytes = 5 * 1024 * 1024
            if backup_count < 0:
                 logging.warning("log_backup_count cannot be negative. Using default 4.")
                 backup_count = 4

            file_handler = RotatingFileHandler(
                log_filepath,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8',
                delay=True # Delay opening file until first log message
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)
            # Log file handler setup details *after* handlers are added
            logging.debug(
                f"File logging configured: path='{log_filepath}', level={logging.getLevelName(file_level)}, "
                f"max_size={max_bytes} bytes, backups={backup_count}"
            )
        except Exception as e_file:
            # Log error using the console handler (if it was configured)
            logging.error("Failed to set up file logging handler: %s", e_file, exc_info=True)
            # Also print, in case console logging itself failed
            print(f"ERROR: Could not setup file logging to '{log_filepath}': {e_file}", file=sys.stderr)

        logging.info(f"Logging setup complete. Root Level: {logging.getLevelName(root_level)}, "
                     f"Console: {logging.getLevelName(console_level)}, File: {logging.getLevelName(file_level)}")

    except Exception as e_setup:
        # Catch-all for any unexpected error during the main setup logic
        logging.basicConfig(level=logging.ERROR) # Ensure *some* logging exists
        logging.exception("CRITICAL ERROR during logging setup: %s", e_setup)
        print(f"CRITICAL ERROR during logging setup: {e_setup}", file=sys.stderr)


# --- Simple Math Utility ---

def ensure_odd(value: Any, default_odd: int = 3) -> int:
    """
    Converts a value to an integer and ensures it is odd.

    Rounds floats before converting to int. If conversion fails or the value is None,
    returns the specified default odd integer.

    Args:
        value (Any): The input value (will be attempted to convert to int).
        default_odd (int): The default odd value to return on failure. Must be odd.

    Returns:
        int: An odd integer.
    """
    # Ensure the default itself is odd
    if default_odd % 2 == 0:
         # Adjust default to the next higher odd number for consistency
         logging.warning("Provided default_odd value %d was even, adjusting to %d", default_odd, default_odd + 1)
         default_odd += 1

    if value is None:
        return default_odd
    try:
        # Attempt to round if float, then convert to int
        if isinstance(value, float):
             v_int = int(round(value))
        else:
             v_int = int(value)

        # Check if already odd
        if v_int % 2 != 0:
            return v_int
        else:
            # Make it odd by adding 1 (works for positive and negative evens)
            return v_int + 1
    except (ValueError, TypeError):
        logging.warning(
            "Could not convert value '%s' (type %s) to int for ensure_odd. Returning default %d.",
            value, type(value).__name__, default_odd,
            exc_info=False # Keep log cleaner for common conversion issues
        )
        return default_odd

# --- END OF ENHANCED FILE utils/helpers.py ---