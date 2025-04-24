# --- START OF REFACTORED FILE libs_cosmic_forge/core/session_manager.py ---
"""
Handles saving and loading the application state (session).

This allows users to save their work (loaded spectrum reference, processing settings,
analysis results like peaks, fits, plasma parameters) and resume later.

Uses JSON for storing serializable state information. Large data arrays (like
raw spectral data) are NOT stored directly; instead, file paths are saved,
and data is expected to be reloaded from the original source. Derived data
like DataFrames are converted to a serializable format.
"""

import os
import json
import logging
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import issues with MainWindow
if TYPE_CHECKING:
    from ui.main_window import MainWindow

# Import core data models (assuming they are in the same 'core' directory)
# Adjust relative import if necessary based on your project structure
from .data_models import Spectrum, Peak, FitResult, NISTMatch
# Import core file IO (not strictly needed here, but shows dependency)
# from .file_io import load_spectrum_from_file

# --- Constants ---
SESSION_FILE_EXTENSION = ".lcfses" # LIBS Cosmic Forge Session
SESSION_VERSION = "1.1" # Increment version due to DataFrame serialization change

# --- Custom JSON Encoder for NumPy/Pandas types ---
class DataEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy and potentially other data types """
    def default(self, obj):
        # NumPy Scalars
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            # Handle NaN/Inf appropriately for JSON
            if np.isnan(obj): return None       # Represent NaN as null
            elif np.isinf(obj): return str(obj) # Represent Inf as "Infinity" / "-Infinity"
            else: return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        # NumPy Arrays (explicitly prevent serialization)
        elif isinstance(obj, np.ndarray):
            logging.warning("Attempted to serialize NumPy array directly in session. Skipping.")
            return "<NumPy Array (not saved)>" # Placeholder string
        # Pandas Timestamps (if they appear in config/settings)
        elif isinstance(obj, pd.Timestamp):
             return obj.isoformat() # Convert to ISO 8601 string
        # Other non-serializable types
        elif isinstance(obj, (np.void)): # Should not occur often
            return None
        # Let the base class default method raise the TypeError for other types
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            logging.error(f"Object of type {type(obj)} is not JSON serializable. Value: {obj!r}")
            raise # Re-raise the error after logging


class SessionManager:
    """Manages saving and loading application session state."""

    SESSION_FILE_EXTENSION = SESSION_FILE_EXTENSION # Expose for use elsewhere

    def __init__(self, main_window: 'MainWindow'):
        """
        Initializes the SessionManager.

        Args:
            main_window: A reference to the main application window instance.
                         Needed to access current state and UI elements.
        """
        self.main_window = main_window
        # No need to store config separately if always accessible via main_window
        # self.config = main_window.config

    def _get_panel_state(self, panel_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Helper to safely retrieve settings from a UI panel widget.

        Checks for common methods like 'get_settings' or 'get_state'.

        Args:
            panel_identifier: The key used to identify the dock widget in
                              `main_window.docks` (e.g., 'processing', 'nist_search').

        Returns:
            A dictionary containing the panel's state, or None if retrieval fails.
        """
        panel_widget = None
        try:
            # Panels might be directly under main_window or inside docks
            if hasattr(self.main_window, f"{panel_identifier}_panel"):
                panel_widget = getattr(self.main_window, f"{panel_identifier}_panel", None)
            elif hasattr(self.main_window, f"{panel_identifier}_view"):
                 panel_widget = getattr(self.main_window, f"{panel_identifier}_view", None)
            elif panel_identifier in self.main_window.docks:
                 dock = self.main_window.docks.get(panel_identifier)
                 if dock: panel_widget = dock.widget()

            if panel_widget:
                 if hasattr(panel_widget, 'get_settings') and callable(panel_widget.get_settings):
                     return panel_widget.get_settings()
                 elif hasattr(panel_widget, 'get_state') and callable(panel_widget.get_state):
                      return panel_widget.get_state()
                 else:
                      logging.debug(f"Panel/View '{panel_identifier}' found, but has no "
                                    "'get_settings' or 'get_state' method.")
            else:
                 # It's okay if some panels don't exist or don't need saving
                 logging.debug(f"Panel/View/Dock for '{panel_identifier}' not found.")

        except Exception as e:
            logging.warning(f"Could not get state for panel '{panel_identifier}': {e}", exc_info=True)

        return None

    def _serialize_dataframe(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict]]:
        """Converts a Pandas DataFrame to a JSON-serializable list of records."""
        if df is None:
            return None
        if isinstance(df, pd.DataFrame):
            if not df.empty:
                 try:
                     # 'records' format is generally easy to reconstruct
                     return df.to_dict(orient='records')
                 except Exception as e:
                     logging.error(f"Failed to serialize DataFrame to dict: {e}", exc_info=True)
                     return None # Indicate serialization failure
            else:
                 return [] # Empty list for empty DataFrame
        else:
             logging.warning(f"Attempted to serialize non-DataFrame object: {type(df)}")
             return None


    def gather_session_data(self) -> Dict[str, Any]:
        """
        Collects the current application state into a serializable dictionary.

        This dictionary contains references (like file paths), settings,
        and serializable representations of analysis results.
        """
        state = {"session_version": SESSION_VERSION}
        mw = self.main_window # Shorthand

        # --- Main Window State ---
        try:
            state['window_geometry'] = mw.saveGeometry().toBase64().data().decode('ascii')
            state['window_state'] = mw.saveState().toBase64().data().decode('ascii')
            state['current_theme'] = mw.theme_manager.current_theme_name
            state['last_load_dir'] = mw._last_load_dir
            state['last_save_dir'] = mw._last_save_dir
        except Exception as e:
            logging.error(f"Failed to gather main window state: {e}", exc_info=True)
            # Decide if this is critical - perhaps raise error or return partial state?
            # For now, continue gathering other data.

        # --- Loaded Data References ---
        if mw.current_spectrum and mw.current_spectrum.source_filepath:
            state['current_spectrum_path'] = mw.current_spectrum.source_filepath
            # Store loader params if available in metadata
            state['current_spectrum_delimiter'] = mw.current_spectrum.metadata.get('used_delimiter')
            state['current_spectrum_comment'] = mw.current_spectrum.metadata.get('used_comment_char')
        else:
            state['current_spectrum_path'] = None

        state['multi_spectra_paths'] = [s.source_filepath for s in mw.multi_spectra if s and s.source_filepath]

        # --- UI Panel States ---
        # Use identifiers consistent with MainWindow creation/access
        panel_keys = ['processing', 'detection', 'fitting', 'nist_search',
                      'boltzmann', 'cflibs', 'ml_analysis', 'peak_list'] # Added peak_list if it has state
        for key in panel_keys:
             state[f"{key}_settings"] = self._get_panel_state(key)

        # --- Analysis Results ---
        # Serialize Peak data. Using to_dataframe_row() returns a dict.
        # NOTE: For full restoration fidelity, a Peak.to_dict() method that saves
        # more internal state (like alternative_fits if needed) and a corresponding
        # Peak.from_dict() classmethod might be preferable in the future.
        state['detected_peaks'] = [p.to_dataframe_row() for p in mw.detected_peaks] if mw.detected_peaks else []

        # Serialize NIST matches (if available)
        # NOTE: Similar to Peaks, full NISTMatch restoration might need to_dict/from_dict
        state['nist_matches'] = [m.to_dict() for m in mw.nist_matches] if mw.nist_matches else [] # Assumes m.to_dict() exists

        # Serialize DataFrames derived from analysis
        state['boltzmann_plot_data'] = self._serialize_dataframe(mw.boltzmann_plot_data)
        state['cf_libs_concentrations'] = self._serialize_dataframe(mw.cf_libs_concentrations)

        # --- Plasma Parameters (directly serializable) ---
        state['plasma_temp_k'] = mw.plasma_temp_k
        state['electron_density_cm3'] = mw.electron_density_cm3

        # --- Optional: Plot State ---
        # Saving plot limits can be fragile if axes are cleared or changed.
        # try:
        #     if mw.plot_widget and mw.plot_widget.ax and mw.plot_widget.ax.has_data():
        #         state['plot_xlim'] = mw.plot_widget.ax.get_xlim()
        #         state['plot_ylim'] = mw.plot_widget.ax.get_ylim()
        # except Exception as e:
        #      logging.debug(f"Could not get plot limits for session save: {e}")

        logging.info("Session data gathered for saving.")
        logging.debug(f"Session State Keys: {list(state.keys())}")
        return state

    def save_session(self, filepath: str) -> bool:
        """
        Saves the current application state to a JSON file using the custom encoder.

        Args:
            filepath: The full path to the file where the session will be saved.

        Returns:
            True if saving was successful, False otherwise.
        """
        logging.info(f"Attempting to save session to: {filepath}")
        mw = self.main_window # Shorthand

        # Prevent saving during critical operations (check main window's busy flag)
        if mw._is_busy:
             logging.warning("Save session request ignored: Application is busy.")
             # Avoid showing QMessageBox from here, let MainWindow handle UI feedback if needed
             # QMessageBox.warning(mw, "Busy", "Cannot save session while another operation is in progress.")
             mw.update_status("Cannot save session: Busy.", 3000)
             return False

        # 1. Gather the current state
        try:
            session_data = self.gather_session_data()
        except Exception as e:
             logging.error(f"Failed to gather session data for saving: {e}", exc_info=True)
             # Avoid showing QMessageBox from here
             mw.update_status("Failed to gather session data.", 5000)
             return False

        # 2. Serialize and write to file
        try:
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=4, cls=DataEncoder) # Use the custom encoder

            logging.info(f"Session saved successfully to {filepath}")
            mw.update_status(f"Session saved: {os.path.basename(filepath)}", 5000)
            return True

        except TypeError as e:
             # This catches errors during JSON serialization (e.g., unhandled types)
             logging.error(f"Serialization error saving session to {filepath}: {e}. Check data types.", exc_info=True)
             # Avoid showing QMessageBox from here
             mw.update_status(f"Save Error: Data type issue ({e}).", 5000)
             return False
        except IOError as e:
             logging.error(f"File I/O error saving session to {filepath}: {e}", exc_info=True)
             # Avoid showing QMessageBox from here
             mw.update_status(f"Save Error: Could not write file ({e}).", 5000)
             return False
        except Exception as e:
            # Catch other unexpected errors during saving
            logging.error(f"Unexpected error saving session state to {filepath}: {e}", exc_info=True)
            # Avoid showing QMessageBox from here
            mw.update_status(f"Save Error: Unexpected error ({e}).", 5000)
            return False


    def load_session_data(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Loads session state from a JSON file into a dictionary.

        Performs basic validation (file existence, JSON format, version check).
        Does **not** apply the state to the application; that is the responsibility
        of the caller (typically MainWindow).

        Args:
            filepath: The path to the session file (.lcfses).

        Returns:
            A dictionary containing the loaded session state, or None if loading fails.

        Raises:
            FileNotFoundError: If the specified filepath does not exist.
            ValueError: If the file is not valid JSON or missing essential keys.
            IOError: For other file reading issues.
        """
        logging.info(f"Attempting to load session data from: {filepath}")
        if not os.path.isfile(filepath): # Use isfile for clarity
             logging.error(f"Session file not found: {filepath}")
             raise FileNotFoundError(f"Session file not found: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # --- Version Check ---
            file_version = state.get("session_version")
            if file_version is None:
                logging.warning(f"Session file '{os.path.basename(filepath)}' has no version information.")
                # Decide policy: Allow loading or reject? Allow for now.
            elif file_version != SESSION_VERSION:
                 logging.warning(f"Session file version mismatch (File: {file_version}, App requires: {SESSION_VERSION}). "
                                 "Loading may be incomplete or results may be unexpected.")
                 # Future: Could add version migration logic here if needed.

            # --- Basic Validation (Check for essential keys) ---
            # These keys are fundamental for restoring the basic application context.
            essential_keys = ['window_geometry', 'window_state', 'current_theme', 'current_spectrum_path']
            missing_keys = [key for key in essential_keys if key not in state]
            if missing_keys:
                 # Raise ValueError as the file structure is invalid for our use
                 raise ValueError(f"Session file is missing essential data key(s): {', '.join(missing_keys)}")

            logging.info(f"Session data loaded successfully from {os.path.basename(filepath)} (Version: {file_version}).")
            return state

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding session file (invalid JSON) {filepath}: {e}", exc_info=True)
            # Raise ValueError for MainWindow to catch and inform user about corruption
            raise ValueError(f"Could not parse session file (invalid JSON): {e}") from e
        except ValueError as e: # Catch our own ValueError from missing keys
             logging.error(f"Validation error in session file {filepath}: {e}", exc_info=False) # Already logged details
             raise # Re-raise the specific ValueError
        except Exception as e:
            # Catch other potential errors during file reading or initial processing
            logging.error(f"Failed to load session state data from {filepath}: {e}", exc_info=True)
            # Raise a more generic IOError indicating a problem reading the file
            raise IOError(f"Failed to load session file '{os.path.basename(filepath)}': An unexpected error occurred.") from e

# --- END OF REFACTORED FILE libs_cosmic_forge/core/session_manager.py ---