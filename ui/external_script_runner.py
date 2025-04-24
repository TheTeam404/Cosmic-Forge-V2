# --- START OF REVISED external_script_runner.py ---
"""
Utility functions and a dialog for running external scripts.

Contains:
- get_project_root: Finds the project root directory.
- setup_logging: Configures application logging.
- ensure_odd: Utility math function.
- ExternalScriptRunnerDialog: A PyQt dialog to run external commands/scripts.
"""

import os
import sys
import logging
import subprocess # QProcess uses this underlying mechanism
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional, List

# --- PyQt Imports (Needed for the Dialog) ---
try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
        QPlainTextEdit, QFileDialog, QMessageBox, QDialogButtonBox,
        QSizePolicy, QApplication, QWidget # QApplication needed for standalone testing
    )
    from PyQt6.QtCore import QProcess, Qt, pyqtSlot
    from PyQt6.QtGui import QFont , QAction
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    # Define dummy classes if PyQt is not available, so the rest of the file parses
    # This allows the utility functions to be used even if PyQt isn't installed,
    # but the Dialog cannot be instantiated.
    class QDialog: pass
    class QProcess: pass
    class QPushButton: pass
    class QPlainTextEdit: pass
    # Add others as needed if referenced directly outside the class

    logging.warning("PyQt6 not found. ExternalScriptRunnerDialog will not be available.")
    # You might want to raise an error here if the dialog is critical


# --- Project Root Finding ---

# Cache the project root directory
_project_root: Optional[str] = None

def get_project_root() -> str:
    """
    Finds and caches the project root directory.

    Strategies (in order):
    1. Looks for 'pyproject.toml' or a '.git' directory up to 4 levels up.
    2. Looks for 'main.py' AND 'config.yaml' up to 4 levels up.
    3. Falls back based on the script's location (with a warning).

    Returns:
        str: The absolute path to the determined project root.

    Raises:
        FileNotFoundError: If the root cannot be reasonably determined after fallback.
                           (Note: Current implementation falls back instead of raising)
    """
    global _project_root
    if _project_root is not None:
        return _project_root

    try:
        # Start from the directory containing this utils.py file
        # Use abspath to handle different execution contexts
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ not defined (e.g., interactive mode) - use current working directory
        current_dir = os.getcwd()
        logging.warning("__file__ not defined, using current working directory '%s' as starting point for root search.", current_dir)
    except Exception as e:
        logging.error(f"Unexpected error getting initial directory: {e}", exc_info=True)
        current_dir = os.getcwd() # Fallback to cwd
        logging.warning("Falling back to current working directory '%s' due to error.", current_dir)


    search_dir = current_dir
    found_root = None
    max_levels = 4 # Limit search depth

    # --- Strategy 1: Standard Markers ---
    logging.debug("Searching for project root markers (.git, pyproject.toml) upwards from '%s'", current_dir)
    for _ in range(max_levels):
        if os.path.exists(os.path.join(search_dir, 'pyproject.toml')) or \
           os.path.exists(os.path.join(search_dir, '.git')):
            found_root = search_dir
            logging.debug("Found marker in '%s'", found_root)
            break
        parent = os.path.dirname(search_dir)
        if parent == search_dir: # Reached filesystem root
            logging.debug("Reached filesystem root during marker search.")
            break
        search_dir = parent

    # --- Strategy 2: Specific File Combination ---
    if found_root is None:
        logging.debug("Searching for main.py + config.yaml upwards from '%s'", current_dir)
        search_dir = current_dir # Reset search start
        for _ in range(max_levels):
            main_py_path = os.path.join(search_dir, 'main.py')
            config_yaml_path = os.path.join(search_dir, 'config.yaml')
            if os.path.exists(main_py_path) and os.path.exists(config_yaml_path):
                found_root = search_dir
                logging.debug("Found main.py and config.yaml in '%s'", found_root)
                break
            parent = os.path.dirname(search_dir)
            if parent == search_dir: # Reached filesystem root
                logging.debug("Reached filesystem root during main/config search.")
                break
            search_dir = parent

    # --- Strategy 3: Fallback ---
    if found_root is None:
        # Last resort fallback: assume this file is one level below root
        # Adjust this logic if the utils file location is different
        fallback_root = os.path.dirname(current_dir)
        logging.warning(
            "Could not definitively find project root using markers or specific files "
            "up to %d levels from '%s'. Falling back to parent directory: '%s'",
            max_levels, current_dir, fallback_root
        )
        # Uncomment to make finding the root mandatory:
        # raise FileNotFoundError(
        #     f"Could not determine project root from '{current_dir}'. Place a marker file "
        #     "('pyproject.toml', '.git') or ensure 'main.py'/'config.yaml' "
        #     "exist in the root, or adjust fallback logic."
        # )
        found_root = fallback_root # Use fallback anyway for now

    _project_root = os.path.abspath(found_root) # Store absolute path
    logging.info("Project root determined as: %s", _project_root)
    return _project_root

# --- Logging Setup ---

def setup_logging(log_config: Optional[Dict[str, Any]] = None):
    """
    Configures root logger with console and rotating file handlers.

    Reads settings from log_config or uses sensible defaults.
    Removes any previously configured handlers on the root logger.

    Args:
        log_config (Optional[Dict[str, Any]]): Dictionary usually loaded from
                                               the 'logging' section of a config file.
                                               Expected keys (with defaults):
                                               - log_dir ('logs')
                                               - log_file_name ('app.log')
                                               - log_level_console ('INFO')
                                               - log_level_file ('DEBUG')
                                               - log_format (see code)
                                               - log_date_format ('%Y-%m-%d %H:%M:%S')
                                               - log_max_bytes (5*1024*1024)
                                               - log_backup_count (4)
    """
    if log_config is None:
        log_config = {}

    # Use basicConfig as a fallback if setup fails critically
    try:
        # --- Determine Log Directory and File ---
        project_root = get_project_root() # Ensure root is found first
        log_dir_relative = log_config.get('log_dir', 'logs') # Default directory name

        # Defensive check for log_dir_relative type
        if not isinstance(log_dir_relative, str) or not log_dir_relative:
             logging.warning("Invalid 'log_dir' in config, using default 'logs'.")
             log_dir_relative = 'logs'

        log_directory = os.path.join(project_root, log_dir_relative)

        try:
            os.makedirs(log_directory, exist_ok=True) # Create log directory if it doesn't exist
        except OSError as e:
            # Use basic logging since handlers aren't set up yet
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"Failed to create log directory '{log_directory}': {e}", exc_info=True)
            print(f"ERROR: Failed to create log directory '{log_directory}': {e}", file=sys.stderr)
            # Attempt to fall back to project root? Or just disable file logging?
            log_directory = project_root
            logging.warning(f"Falling back to log directory '{log_directory}'")
            # Or return here to prevent further issues

        log_filename = log_config.get('log_file_name', 'app.log') # Default log file name
        if not isinstance(log_filename, str) or not log_filename:
             logging.warning("Invalid 'log_file_name' in config, using default 'app.log'.")
             log_filename = 'app.log'
        log_filepath = os.path.join(log_directory, log_filename)

        # --- Get Logging Levels ---
        console_level_str = str(log_config.get('log_level_console', 'INFO')).upper()
        file_level_str = str(log_config.get('log_level_file', 'DEBUG')).upper()

        # getattr with default handles invalid level names
        console_level = getattr(logging, console_level_str, logging.INFO)
        file_level = getattr(logging, file_level_str, logging.DEBUG)
        if console_level_str not in logging._nameToLevel:
            logging.warning(f"Invalid console log level '{console_level_str}', using INFO.")
        if file_level_str not in logging._nameToLevel:
            logging.warning(f"Invalid file log level '{file_level_str}', using DEBUG.")


        # Root logger level should be the *lowest* of the handler levels to allow all messages through
        root_level = min(console_level, file_level)

        # --- Get Formatting ---
        default_format = '%(asctime)s - %(levelname)-8s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
        log_format_string = log_config.get('log_format', default_format)
        log_date_format_string = log_config.get('log_date_format', '%Y-%m-%d %H:%M:%S')

        try:
             log_formatter = logging.Formatter(log_format_string, datefmt=log_date_format_string)
        except Exception as e_fmt:
             logging.warning(f"Invalid log format string '{log_format_string}' or date format '{log_date_format_string}'. Using default format. Error: {e_fmt}")
             log_formatter = logging.Formatter(default_format, datefmt='%Y-%m-%d %H:%M:%S')


        # --- Configure Root Logger ---
        root_logger = logging.getLogger()
        root_logger.setLevel(root_level)

        # Remove existing handlers to prevent duplicates if setup_logging is called again
        if root_logger.hasHandlers():
            logging.log(root_level, "Removing existing logging handlers.") # Use root_level log
            for handler in root_logger.handlers[:]: # Iterate over a copy
                try:
                    handler.close()
                except Exception as e_close:
                    # Log at a lower level, might not be critical
                    logging.warning(f"Error closing handler {handler}: {e_close}")
                root_logger.removeHandler(handler)

        # --- Create and Add Console Handler ---
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(log_formatter)
            root_logger.addHandler(console_handler)
        except Exception as e_console:
             # Fallback needed here if console handler fails
             logging.basicConfig(level=logging.ERROR)
             logging.error(f"CRITICAL: Failed to set up console logging handler: {e_console}", exc_info=True)
             print(f"CRITICAL ERROR: Could not setup console logging: {e_console}", file=sys.stderr)
             # Consider exiting or returning if console logging is essential

        # --- Create and Add File Handler ---
        file_handler = None
        try:
            # Ensure max_bytes and backup_count are integers
            try:
                max_bytes = int(log_config.get('log_max_bytes', 5 * 1024 * 1024)) # Default 5MB
            except (ValueError, TypeError):
                logging.warning(f"Invalid 'log_max_bytes' value '{log_config.get('log_max_bytes')}', using default 5MB.")
                max_bytes = 5 * 1024 * 1024
            if max_bytes <= 0:
                logging.warning(f"'log_max_bytes' must be positive, using default 5MB.")
                max_bytes = 5 * 1024 * 1024

            try:
                backup_count = int(log_config.get('log_backup_count', 4)) # Default 4 backups
            except (ValueError, TypeError):
                logging.warning(f"Invalid 'log_backup_count' value '{log_config.get('log_backup_count')}', using default 4.")
                backup_count = 4
            if backup_count < 0:
                 logging.warning(f"'log_backup_count' cannot be negative, using default 4.")
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
                f"File logging configured: path='{log_filepath}', level={file_level_str}, "
                f"max_size={max_bytes} bytes, backups={backup_count}"
            )
        except OSError as e_os:
            logging.error(f"Permission or OS error setting up file logging handler for '{log_filepath}': {e_os}", exc_info=True)
            print(f"ERROR: OS error setting up file logging to '{log_filepath}': {e_os}", file=sys.stderr)
        except Exception as e_file:
            # Log error using the console handler (which should be configured by now)
            logging.error(f"Failed to set up file logging handler for '{log_filepath}': {e_file}", exc_info=True)
            # Also print, in case console logging itself is broken
            print(f"ERROR: Could not setup file logging to '{log_filepath}': {e_file}", file=sys.stderr)

        # Log final status
        file_status = f"File: {file_level_str} ({'Active' if file_handler else 'Inactive'})"
        logging.info(f"Logging setup complete. Root Level: {logging.getLevelName(root_level)}, "
                     f"Console: {console_level_str}, {file_status}")

    except Exception as e_setup:
        # Catch-all for any unexpected error during setup
        logging.basicConfig(level=logging.ERROR) # Ensure *some* logging exists
        logging.exception("CRITICAL ERROR during logging setup: %s", e_setup)
        print(f"CRITICAL ERROR during logging setup: {e_setup}", file=sys.stderr)


# --- Simple Math Utility ---

def ensure_odd(value: Any) -> int:
    """
    Converts a value to an integer and ensures it is odd.

    Useful for parameters like filter kernel sizes.

    Args:
        value (Any): The input value (will be attempted to convert to int).

    Returns:
        int: An odd integer. Returns 3 if conversion fails or input is None/invalid.
    """
    default_odd = 3
    if value is None:
        logging.debug("ensure_odd received None, returning default %d.", default_odd)
        return default_odd
    try:
        v_int = int(value)
        # Check if already odd
        if v_int % 2 != 0:
            return v_int
        else:
            # Make it odd (handle negative evens correctly too)
            # Ensure it doesn't become zero if input was 0 or -1
            new_val = v_int + 1 if v_int >= 0 else v_int - 1
            # Prevent returning 0 if input was 0 or -1
            return new_val if new_val != 0 else (1 if v_int == 0 else -1)
    except (ValueError, TypeError):
        logging.warning("Could not convert '%s' to int for ensure_odd. Returning default %d.",
                        value, default_odd, exc_info=True) # Add exc_info
        return default_odd


# --- External Script Runner Dialog ---

# Only define the class if PyQt was imported successfully
if _QT_AVAILABLE:
    class ExternalScriptRunnerDialog(QDialog):
        """
        A dialog to execute external scripts or commands using QProcess.
        """
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self.setWindowTitle("Run External Script")
            self.setMinimumSize(600, 400)

            self.process = QProcess(self)
            self.process.readyReadStandardOutput.connect(self._handle_stdout)
            self.process.readyReadStandardError.connect(self._handle_stderr)
            self.process.stateChanged.connect(self._handle_state_change)
            # finished provides exit code and status
            self.process.finished.connect(self._handle_finished)
            # errorOccurred is more detailed for launch failures etc.
            self.process.errorOccurred.connect(self._handle_error)

            self._init_ui()
            self._update_button_states(QProcess.ProcessState.NotRunning) # Initial state

        def _init_ui(self):
            """Initialize UI elements and layout."""
            layout = QVBoxLayout(self)

            # --- Command Input ---
            form_layout = QHBoxLayout()
            form_layout.addWidget(QLabel("Command/Script:"))
            self.command_input = QLineEdit()
            self.command_input.setPlaceholderText("Enter command or path to script")
            form_layout.addWidget(self.command_input)

            self.browse_button = QPushButton("Browse...")
            self.browse_button.setToolTip("Browse for an executable or script")
            self.browse_button.clicked.connect(self._browse_script)
            form_layout.addWidget(self.browse_button)
            layout.addLayout(form_layout)

            # --- Arguments Input ---
            args_layout = QHBoxLayout()
            args_layout.addWidget(QLabel("Arguments:"))
            self.args_input = QLineEdit()
            self.args_input.setPlaceholderText("Enter arguments separated by spaces (or handle quoting)")
            args_layout.addWidget(self.args_input)
            layout.addLayout(args_layout)

            # --- Output Area ---
            layout.addWidget(QLabel("Output:"))
            self.output_area = QPlainTextEdit()
            self.output_area.setReadOnly(True)
            self.output_area.setFont(QFont("Courier New", 9)) # Monospaced font
            self.output_area.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            layout.addWidget(self.output_area)

            # --- Status Label ---
            self.status_label = QLabel("Status: Ready")
            layout.addWidget(self.status_label)

            # --- Control Buttons ---
            self.button_box = QDialogButtonBox()
            self.run_button = self.button_box.addButton("Run", QDialogButtonBox.ButtonRole.ActionRole)
            self.stop_button = self.button_box.addButton("Stop", QDialogButtonBox.ButtonRole.ActionRole)
            self.close_button = self.button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole) # Use RejectRole for standard behavior

            self.run_button.clicked.connect(self.start_script)
            self.stop_button.clicked.connect(self.stop_script)
            self.close_button.clicked.connect(self.reject) # Connect Close to reject

            layout.addWidget(self.button_box)
            self.setLayout(layout)

        @pyqtSlot()
        def _browse_script(self):
            """Open a file dialog to select a script/executable."""
            # Consider adding filters for specific script types if desired
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Script or Executable", "", "All Files (*);;Python Scripts (*.py);;Batch Files (*.bat *.cmd);;Shell Scripts (*.sh)")
            if file_path:
                self.command_input.setText(file_path)

        @pyqtSlot()
        def start_script(self):
            """Starts the external process."""
            command = self.command_input.text().strip()
            if not command:
                QMessageBox.warning(self, "Input Error", "Please enter a command or script path.")
                return

            args_str = self.args_input.text().strip()
            # Basic argument splitting - consider using shlex for more robust parsing if needed
            arguments = args_str.split() if args_str else []

            self.output_area.clear()
            self.status_label.setText("Status: Starting...")
            logging.info(f"Attempting to run command: '{command}' with args: {arguments}")

            # QProcess handles quoting issues better if you pass arguments as a list
            # For simple commands entered directly, startDetached might be simpler,
            # but start() with list arguments is generally more robust.
            # If command is a python script, you might need to prepend 'python' or sys.executable
            if command.lower().endswith(".py") and not command.startswith("python"):
                 # Make sure to use the correct python executable if venvs are involved
                 executable = sys.executable # Use the same python that runs the app
                 logging.info(f"Prepending Python executable: {executable}")
                 arguments.insert(0, command) # Script path becomes the first argument
                 command = executable

            try:
                self.process.start(command, arguments)
                # State change will be handled by _handle_state_change and _handle_error
            except Exception as e:
                # This catch might be redundant if QProcess.errorOccurred handles it
                error_msg = f"Failed to initiate process start: {e}"
                logging.error(error_msg, exc_info=True)
                self.output_area.appendPlainText(f"ERROR: {error_msg}\n")
                self.status_label.setText("Status: Error")
                self._update_button_states(QProcess.ProcessState.NotRunning)


        @pyqtSlot()
        def stop_script(self):
            """Stops the running process."""
            if self.process.state() == QProcess.ProcessState.Running:
                self.status_label.setText("Status: Attempting to stop...")
                logging.info("Attempting to terminate process.")
                self.process.terminate() # Ask nicely first

                # Optionally, add a timer to forcefully kill if terminate doesn't work quickly
                # QTimer.singleShot(3000, self._force_kill)
            else:
                logging.warning("Stop clicked but process is not running.")

        # def _force_kill(self):
        #     if self.process.state() == QProcess.ProcessState.Running:
        #         logging.warning("Process did not terminate, killing forcefully.")
        #         self.process.kill()


        @pyqtSlot()
        def _handle_stdout(self):
            """Append standard output to the text area."""
            data = self.process.readAllStandardOutput()
            try:
                # Try decoding using UTF-8, fallback to locale encoding or lossy
                text = bytes(data).decode('utf-8', errors='replace')
            except Exception as e:
                logging.warning(f"Error decoding stdout: {e}")
                text = repr(bytes(data)) # Show raw representation on error
            self.output_area.appendPlainText(text)
            self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())


        @pyqtSlot()
        def _handle_stderr(self):
            """Append standard error to the text area."""
            data = self.process.readAllStandardError()
            try:
                text = bytes(data).decode('utf-8', errors='replace')
            except Exception as e:
                 logging.warning(f"Error decoding stderr: {e}")
                 text = repr(bytes(data))
            # Optionally, format stderr differently (e.g., color)
            self.output_area.appendPlainText(f"[STDERR] {text}")
            self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())


        @pyqtSlot(QProcess.ProcessState)
        def _handle_state_change(self, state: QProcess.ProcessState):
            """Update UI elements based on process state changes."""
            logging.debug(f"Process state changed to: {state}")
            self._update_button_states(state)
            if state == QProcess.ProcessState.NotRunning:
                # Status might be set more accurately by finished/error handlers
                if "stopping" not in self.status_label.text().lower() and \
                   "error" not in self.status_label.text().lower() and \
                   "finished" not in self.status_label.text().lower():
                    self.status_label.setText("Status: Ready")
            elif state == QProcess.ProcessState.Starting:
                self.status_label.setText("Status: Starting...")
            elif state == QProcess.ProcessState.Running:
                self.status_label.setText("Status: Running...")


        @pyqtSlot(int, QProcess.ExitStatus)
        def _handle_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
            """Handle process completion."""
            status_text = "normally" if exit_status == QProcess.ExitStatus.NormalExit else "with crash"
            logging.info(f"Process finished {status_text}. Exit code: {exit_code}")
            self.output_area.appendPlainText(f"\n--- Process finished ({status_text}, Exit Code: {exit_code}) ---\n")
            self.status_label.setText(f"Status: Finished (Exit Code: {exit_code})")
            # State should automatically transition to NotRunning, triggering button updates


        @pyqtSlot(QProcess.ProcessError)
        def _handle_error(self, error: QProcess.ProcessError):
            """Handle errors reported by QProcess."""
            error_string = self.process.errorString() # Get descriptive error
            logging.error(f"QProcess Error Occurred: {error} - {error_string}")
            self.output_area.appendPlainText(f"\n--- PROCESS ERROR: {error_string} ({error}) ---\n")
            self.status_label.setText(f"Status: Error ({error})")
            # State should automatically transition to NotRunning, triggering button updates


        def _update_button_states(self, state: QProcess.ProcessState):
            """Enable/disable buttons based on process state."""
            is_running = (state == QProcess.ProcessState.Running or state == QProcess.ProcessState.Starting)
            self.run_button.setEnabled(not is_running)
            self.stop_button.setEnabled(is_running)
            self.command_input.setEnabled(not is_running)
            self.args_input.setEnabled(not is_running)
            self.browse_button.setEnabled(not is_running)

        def closeEvent(self, event):
            """Ensure the process is stopped when the dialog closes."""
            if self.process.state() != QProcess.ProcessState.NotRunning:
                logging.info("Dialog closing, stopping running process.")
                self.stop_script()
                # Optionally wait briefly for termination before accepting close
                # self.process.waitForFinished(500) # Wait max 500ms
            super().closeEvent(event)

else:
    # If Qt not available, provide a dummy class or raise error on instantiation
    class ExternalScriptRunnerDialog:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot create ExternalScriptRunnerDialog: PyQt6 is not installed or available.")


# --- Standalone Test ---
if __name__ == "__main__":
    # Basic logging setup for testing utils
    setup_logging({
        'log_level_console': 'DEBUG',
        'log_level_file': 'DEBUG'
    })

    logging.info("--- Testing Utility Functions ---")
    # Test ensure_odd
    print(f"ensure_odd(4) -> {ensure_odd(4)}")
    print(f"ensure_odd(5) -> {ensure_odd(5)}")
    print(f"ensure_odd(0) -> {ensure_odd(0)}")
    print(f"ensure_odd(-1) -> {ensure_odd(-1)}")
    print(f"ensure_odd(-2) -> {ensure_odd(-2)}")
    print(f"ensure_odd(None) -> {ensure_odd(None)}")
    print(f"ensure_odd('abc') -> {ensure_odd('abc')}")
    print(f"ensure_odd(4.6) -> {ensure_odd(4.6)}")

    # Test project root finding
    root = get_project_root()
    print(f"Project Root Found: {root}")

    # Test Dialog (only if Qt is available)
    if _QT_AVAILABLE:
        app = QApplication(sys.argv)
        dialog = ExternalScriptRunnerDialog()

        # --- Example Commands to Test ---
        # Windows:
        # dialog.command_input.setText("ping")
        # dialog.args_input.setText("localhost -n 3")
        # dialog.command_input.setText("cmd") # Interactive might behave oddly
        # dialog.args_input.setText("/c dir") # Run dir and exit

        # Linux/macOS:
        # dialog.command_input.setText("ping")
        # dialog.args_input.setText("localhost -c 3")
        # dialog.command_input.setText("ls")
        # dialog.args_input.setText("-lha")

        # Python script (assuming python is in PATH):
        # Create a dummy test.py: print("Hello from Python Script!"); import time; time.sleep(2); print("Script finished.")
        # dialog.command_input.setText("test.py") # Will prepend sys.executable
        # dialog.args_input.setText("arg1 arg2")

        dialog.show()
        sys.exit(app.exec())
    else:
        print("\nPyQt6 not available, skipping ExternalScriptRunnerDialog test.")
        # Test instantiating the dummy class
        try:
            ExternalScriptRunnerDialog()
        except RuntimeError as e:
            print(f"Correctly caught expected error: {e}")


# --- END OF REVISED external_script_runner.py ---