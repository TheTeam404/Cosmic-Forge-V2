"""
Main Entry Point for the LIBS Cosmic Forge Application.
Initializes the application environment, loads configuration,
and launches the main user interface window.
"""

import sys
import os
import traceback
import logging

# --- Third-party Imports ---
# Attempt critical imports first and provide informative errors
try:
    import yaml
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt, QCoreApplication, QSettings # Import QSettings explicitly if needed
except ImportError as e:
    # Minimal error message if basic Qt/YAML is missing
    print(f"CRITICAL ERROR: Missing essential libraries: {e}")
    print("This application requires Python 3.10+ and several libraries.")
    print("Please install them using the command:")
    print("pip install -r requirements.txt")
    # Attempt to show a simple message box if QApplication exists (it might not)
    try:
        app_temp = QApplication([]) # Attempt to create a temporary app
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setWindowTitle("Startup Error - Missing Library")
        error_box.setText(f"Missing essential library: {e}.\nPlease install requirements (see console).")
        error_box.exec()
    except Exception:
        pass # If even QApplication fails, console message is the best we can do
    sys.exit(1)

# --- Application Metadata ---
APP_VERSION = "1.0.0" # <-- DEFINE VERSION HERE (Update as needed)
ORG_NAME = "CosmicForgeDev"
APP_NAME = "LIBSForge"

# Set application attributes early for QSettings and platform integration
QCoreApplication.setOrganizationName(ORG_NAME)
QCoreApplication.setApplicationName(APP_NAME)
# QCoreApplication.setApplicationVersion(APP_VERSION) # Can be set here or on app instance later
# Required for High DPI display scaling (usually good to enable)
#QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
#QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

# --- Local Imports ---
# Import after essential libraries are confirmed
# We need get_project_root early if logging setup depends on it before config load
try:
    from utils.helpers import get_project_root
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import get_project_root: {e}")
    traceback.print_exc()
    sys.exit(1)

# Determine root and potential default log path early
_project_root_early = get_project_root() # Call it once here

# Setup basic logging BEFORE loading config or full app, in case those fail
try:
    from utils.helpers import setup_logging
    # Initial minimal setup, will be reconfigured after config load
    setup_logging({'log_level_console': 'INFO', 'log_level_file': 'WARNING', 'log_dir': os.path.join(_project_root_early, 'logs')})
    logging.info("Basic logging initialized.")
except Exception as log_e:
    logging.basicConfig(level=logging.INFO) # Absolute fallback
    logging.error(f"Failed basic logging setup: {log_e}", exc_info=True)
    print(f"WARNING: Failed basic logging setup: {log_e}", file=sys.stderr)

# Now import the rest
try:
    from ui.main_window import MainWindow
    from ui.theme import ThemeManager # ThemeManager needed here if used before MainWindow init
except ImportError as e:
    logging.critical(f"Failed to import application components: {e}", exc_info=True)
    print(f"CRITICAL ERROR: Failed to import application components: {e}")
    print("Ensure the project structure is correct and all modules exist.")
    print(f"Project root: {_project_root_early}")
    # Attempt to show GUI message box
    try:
        app_temp = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Application Structure Error",
                             f"Failed to import component: {e}.\nCheck project files and console output.")
    except Exception: pass
    sys.exit(1)
except Exception as e:
    logging.critical(f"Unexpected error during local imports: {e}", exc_info=True)
    print(f"CRITICAL ERROR: Unexpected error during local imports: {e}")
    sys.exit(1)


# --- Configuration Loading ---
def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    # Define robust default values for essential config sections/keys
    default_config = {
        'application': {'remember_window_state': True},
        'default_theme': 'dark_cosmic',
        'logging': {'log_level_console': 'INFO', 'log_level_file': 'DEBUG', 'log_dir': 'logs', 'log_file_name': 'app.log'},
        'plotting': {'matplotlib_style_dark': 'dark_background', 'matplotlib_style_light': 'default'},
        'database': {'online_search_timeout_s': 15, 'online_query_delay_s': 1.5},
        'file_io': {'default_delimiter': '\t', 'default_comment_char': '#'},
        # Add defaults for other sections as needed
        'processing': {}, 'peak_detection': {}, 'peak_fitting': {},
        'cflibs': {}, 'machine_learning': {}
    }
    try:
        if not os.path.exists(config_path):
            logging.warning(f"Config file not found: {config_path}. Using default settings.")
            return default_config

        with open(config_path, 'r', encoding='utf-8') as f:
            config_loaded = yaml.safe_load(f)

        if not isinstance(config_loaded, dict):
             logging.warning(f"Config file {config_path} is empty or invalid. Using default settings.")
             return default_config

        # Deep merge loaded config over defaults (simple update is shallow)
        # For nested dicts, update recursively if needed, or just use dict.update for top level
        final_config = default_config.copy()
        # A simple recursive merge helper
        def merge_dicts(target, source):
             for key, value in source.items():
                  if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                       merge_dicts(target[key], value)
                  else:
                       target[key] = value

        merge_dicts(final_config, config_loaded)

        logging.info(f"Configuration loaded successfully from {config_path}")
        return final_config

    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file {config_path}: {e}", exc_info=True)
        QMessageBox.warning(None, "Config Error", f"Error parsing config.yaml:\n{e}\nUsing default settings.")
        return default_config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}", exc_info=True)
        QMessageBox.warning(None, "Config Error", f"Failed to load config.yaml:\n{e}\nUsing default settings.")
        return default_config

# --- Global Exception Hook ---
def global_exception_hook(exctype, value, tb):
    """Catches unhandled exceptions, logs them, and shows a message."""
    # Log first, before trying to display UI
    logging.critical("Unhandled exception caught!", exc_info=(exctype, value, tb))
    # Format traceback
    tb_text = "".join(traceback.format_exception(exctype, value, tb))
    error_msg = f"A critical error occurred:\n\n{exctype.__name__}: {value}"
    # Short message for title/main text, full in details

    try:
        # Check if GUI is likely available and functional
        app_inst = QApplication.instance()
        # Additional check: ensure event loop isn't already stopped or app closing down
        if app_inst: # and QCoreApplication.startingUp(): # This might be too restrictive
             # Use a fresh QMessageBox instance
             error_box = QMessageBox()
             error_box.setIcon(QMessageBox.Icon.Critical)
             error_box.setWindowTitle("Unhandled Application Error")
             error_box.setText(error_msg) # Shorter message
             error_box.setDetailedText(tb_text) # Full traceback in details
             error_box.setStandardButtons(QMessageBox.StandardButton.Ok)
             error_box.exec()
        else: # If GUI unavailable or unreliable, print to console
            print("\n--- UNHANDLED APPLICATION ERROR (NO GUI or GUI Unresponsive) ---")
            print(error_msg)
            print("\nFull Traceback:\n", tb_text)
            print("----------------------------------------------------------------")
    except Exception as hook_e:
        logging.error(f"Error within the global exception hook itself: {hook_e}", exc_info=True)
        # Fallback print if message box fails
        print("\n--- ERROR IN EXCEPTION HOOK ---")
        print(f"Hook Error: {hook_e}")
        print(f"\nOriginal Error: {error_msg}")
        print("\nOriginal Traceback:\n", tb_text)
        print("-----------------------------")
    finally:
        # Critical: Ensure the application exits after an unhandled exception
        print("Application will now exit due to unhandled error.", file=sys.stderr)
        os._exit(1) # Force exit if sys.exit doesn't work (e.g., event loop issues)

# Assign the hook early, after basic logging is potentially available
sys.excepthook = global_exception_hook
logging.info("Global exception hook assigned.")

# --- Main Execution ---
def run_app():
    """Initializes and runs the main application."""
    app = None # Define app outside try block for potential use in except
    main_window = None
    root_dir = _project_root_early # Use the already determined root
    config_path = os.path.join(root_dir, 'config.yaml')

    try:
        # 1. Load Configuration FIRST
        config = load_config(config_path)
        # load_config logs success/warnings/errors internally

        # 2. Setup Logging AGAIN (fully configured using loaded config)
        setup_logging(config.get('logging', {})) # Pass the 'logging' section
        logging.info("Logging reconfigured using settings from config.yaml.")
        # Log essential info *after* full logging is set up
        logging.info(f"--- Starting {APP_NAME} v{APP_VERSION} ---") # Use global constant
        logging.info(f"Project Root: {root_dir}")
        logging.debug(f"Loaded Config (first 5 keys): {list(config.keys())[:5]}...") # Avoid logging full config

        # 3. Initialize QApplication (Ensure it's done before any QWidget)
        logging.debug("Initializing QApplication...")
        app = QApplication.instance() # Check if already exists
        if app is None:
            logging.debug("No QApplication instance found, creating a new one.")
            app = QApplication(sys.argv)
        else:
            logging.debug("Reusing existing QApplication instance.")

        # Set version on the instance if desired
        app.setApplicationVersion(APP_VERSION)
        logging.debug(f"QApplication instance ready. Org: {QCoreApplication.organizationName()}, App: {QCoreApplication.applicationName()}, Ver: {app.applicationVersion()}")


        # 4. Create and Show Main Window (inside a try block for window-specific errors)
        logging.debug("Creating MainWindow...")
        try:
            main_window = MainWindow(config) # Pass full config
            logging.debug("MainWindow instance created.")
            main_window.show()
            logging.debug("MainWindow shown.")
        except Exception as e_win:
            # Log the specific window creation error
            logging.critical(f"Failed to initialize or show the main window: {e_win}", exc_info=True)
            # Show a message box (QApplication should exist now)
            QMessageBox.critical(None, "Application Startup Error",
                                 f"Failed to create the main application window:\n{e_win}\n\n"
                                 "Check the logs for more details.")
            sys.exit(1) # Exit if the main window fails

        # 5. Start Event Loop
        logging.info("Application startup successful. Entering event loop.")
        exit_code = app.exec()
        logging.info(f"Application event loop finished. Exiting with code {exit_code}.")
        sys.exit(exit_code) # Use the exit code from the event loop

    except SystemExit as e:
        # Catch sys.exit() explicitly to avoid the global hook treating it as an unhandled error
        if e.code == 0:
             logging.info(f"Application exiting normally (Code: {e.code}).")
        else:
             logging.warning(f"Application exiting with non-zero code: {e.code}.")
        raise # Re-raise SystemExit to actually exit

    except Exception as e_main:
        # This catch block is a fallback *if* the global exception hook somehow fails
        # or if the error happens before the hook is assigned (unlikely now).
        logging.critical(f"Unexpected error in run_app scope: {e_main}", exc_info=True)
        print(f"FATAL ERROR (run_app guard): {e_main}\n{''.join(traceback.format_exc())}", file=sys.stderr)
        try:
            if app: # Only show message box if app was initialized
                QMessageBox.critical(None, "Fatal Error", f"Fatal error during application run:\n{e_main}\nApp will exit.")
        finally:
            sys.exit(1)


# --- Entry Point Guard ---
if __name__ == "__main__":
    # The global exception hook is already set
    run_app() # Call the main execution function