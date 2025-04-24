# core/file_io.py

"""
Functions for loading and saving spectral data and analysis results.
Handles various text formats and saving to CSV.
"""

import os
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any

# --- Corrected Imports ---
# Import the ACTUAL data models from the central location
try:
    from .data_models import Spectrum, Peak, NISTMatch # Adjusted relative import
    CORE_MODELS_IMPORTED = True
except ImportError as e:
    logging.critical(f"CRITICAL ERROR in file_io.py: Cannot import core data models: {e}. File I/O will fail.")
    # Define dummy classes so the rest of the file *might* parse, but functionality is broken
    class Spectrum: pass
    class Peak: pass
    CORE_MODELS_IMPORTED = False
# --- END Corrected Imports ---


# --- REMOVED Placeholder Class Definitions ---
# class Peak: ... (Removed Placeholder)
# class Spectrum: ... (Removed Placeholder)
# --- END REMOVED Placeholder Class Definitions ---


# --- Constants ---
DEFAULT_SAVE_DELIMITER = ',' # Use comma for CSV saving by default
DEFAULT_FLOAT_FORMAT = '%.6g' # Use general format with up to 6 significant digits for saving
DEFAULT_COMMENT_CHAR = '#' # Default character for comments if none specified

# --- Loading Function ---
def load_spectrum_from_file(filepath: str,
                            delimiter: Optional[str] = None,
                            comment_char: Optional[str] = DEFAULT_COMMENT_CHAR,
                            encoding: str = 'utf-8') -> Spectrum:
    """
    Loads spectral data from a text-based file into a Spectrum object
    (using the definition from core.data_models).

    Handles common delimiters (tab, comma, space, semicolon) and comment lines.
    Attempts to coerce data to numeric types and handles basic errors.
    Sorts data by wavelength and removes duplicate wavelengths.

    Args:
        filepath (str): The full path to the data file.
        delimiter (Optional[str]): The column delimiter. If None, attempts common ones.
        comment_char (Optional[str]): Character indicating comment lines to ignore.
                                      Defaults to '#'. Use None or '' for no comment character.
        encoding (str): File encoding (default: 'utf-8').

    Returns:
        Spectrum: An initialized Spectrum object (from core.data_models).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid, data cannot be parsed, or < 2 valid points remain.
        ImportError: If the core Spectrum model couldn't be imported.
        Exception: For other unexpected errors during loading.
    """
    if not CORE_MODELS_IMPORTED:
         # Prevent further execution if the core model is missing
         raise ImportError("Core Spectrum model could not be imported. Cannot load file.")

    logging.info(f"Attempting load: {filepath} (Delimiter: {delimiter or 'Auto'}, Comment: '{comment_char}')")
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")

    used_comment = comment_char if comment_char else None
    guessed_delimiter = delimiter

    try:
        # --- Delimiter Guessing (if not provided) ---
        if delimiter is None:
            logging.debug(f"No delimiter provided, attempting to guess for {os.path.basename(filepath)}.")
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    lines_to_check = []
                    for _ in range(10):
                        line = f.readline();
                        if not line: break
                        line_strip = line.strip()
                        if line_strip and (not used_comment or not line_strip.startswith(used_comment)): lines_to_check.append(line_strip)
                        if len(lines_to_check) >= 5: break
                if not lines_to_check: raise ValueError("File appears empty or contains only comment lines.")
                first_data_line = lines_to_check[0]; logging.debug(f"First data line for guessing: '{first_data_line}'")
                if '\t' in first_data_line: guessed_delimiter = '\t'
                elif ',' in first_data_line: guessed_delimiter = ','
                elif ';' in first_data_line: guessed_delimiter = ';'
                elif ' ' in first_data_line:
                     parts = first_data_line.split();
                     try:
                         if len(parts) >= 2 and all('.' in p or p.isdigit() or ('e' in p.lower()) or ('-' in p) for p in parts[:2]): guessed_delimiter = r'\s+'
                         else: raise ValueError("Space separated but not clearly numeric pairs")
                     except ValueError: logging.debug("Space found, but parts don't look numeric. Falling back."); guessed_delimiter = '\t'
                else: guessed_delimiter = '\t'; logging.warning(f"Could not guess delimiter for {os.path.basename(filepath)}, defaulting to Tab ('\\t').")
                logging.info(f"Guessed delimiter: {repr(guessed_delimiter)} for {os.path.basename(filepath)}")
            except Exception as guess_e: logging.warning(f"Delimiter guessing failed ({guess_e}), defaulting to Tab ('\\t')."); guessed_delimiter = '\t'

        # --- Read using Pandas ---
        logging.debug(f"Reading with Pandas: sep={repr(guessed_delimiter)}, comment={repr(used_comment)}")
        try:
            df = pd.read_csv(filepath, sep=guessed_delimiter, header=None, names=["Wavelength", "Intensity"], comment=used_comment, encoding=encoding, skipinitialspace=True, skip_blank_lines=True, on_bad_lines='warn', dtype=str, engine='python')
        except pd.errors.EmptyDataError: raise ValueError("File is empty or contains only comments/blank lines.") from None
        except pd.errors.ParserError as pe: logging.error(f"Pandas parsing error: {pe}. Check delimiter/comments."); raise ValueError(f"Failed to parse file structure. Check delimiter/comments. Error: {pe}") from pe
        except Exception as read_e: logging.error(f"Error reading file with Pandas: {read_e}", exc_info=True); raise IOError(f"Could not read file {os.path.basename(filepath)}.") from read_e

        if df.empty: raise ValueError("No data could be parsed (potentially only comments/blank/bad lines).")
        original_rows = len(df); logging.debug(f"Read {original_rows} lines initially.")

        # --- Data Cleaning and Conversion ---
        df['Wavelength'] = pd.to_numeric(df['Wavelength'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        df['Intensity'] = pd.to_numeric(df['Intensity'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        initial_nan_count = df.isnull().any(axis=1).sum()
        if initial_nan_count > 0: logging.warning(f"Found {initial_nan_count} rows with non-numeric values. Dropping them."); df.dropna(subset=['Wavelength', 'Intensity'], inplace=True)
        rows_after_dropna = len(df)
        if df.empty: raise ValueError(f"No valid numeric Wavelength/Intensity pairs found after cleaning {original_rows} initial lines.")

        # --- Extract NumPy Arrays and Check Finite ---
        wavelengths = df['Wavelength'].to_numpy(dtype=float)
        raw_intensity = df['Intensity'].to_numpy(dtype=float)
        finite_mask = np.isfinite(wavelengths) & np.isfinite(raw_intensity)
        if not np.all(finite_mask):
            num_nonfinite = (~finite_mask).sum(); logging.warning(f"Found and removing {num_nonfinite} non-finite values (Inf/NaN).");
            wavelengths = wavelengths[finite_mask]; raw_intensity = raw_intensity[finite_mask]
        final_finite_rows = len(wavelengths)
        if final_finite_rows < 2: raise ValueError(f"Spectrum must contain >= 2 valid finite data points. Found {final_finite_rows}.")

        # --- Sorting and Deduplication ---
        logging.debug("Sorting data by wavelength..."); sort_indices = np.argsort(wavelengths)
        if not np.array_equal(sort_indices, np.arange(len(wavelengths))):
            logging.warning(f"Wavelength data in {os.path.basename(filepath)} was not sorted. Sorting now."); wavelengths = wavelengths[sort_indices]; raw_intensity = raw_intensity[sort_indices]
        logging.debug("Checking for duplicate wavelengths..."); unique_wavelengths, unique_indices = np.unique(wavelengths, return_index=True)
        final_unique_rows = len(unique_wavelengths)
        if final_unique_rows < len(wavelengths):
            num_dupes = len(wavelengths) - final_unique_rows; logging.warning(f"Found and removing {num_dupes} duplicate wavelength values (keeping first).");
            wavelengths = wavelengths[unique_indices]; raw_intensity = raw_intensity[unique_indices]
        if final_unique_rows < 2: raise ValueError(f"Spectrum has less than 2 unique valid points after cleaning.")

        # --- Create Spectrum Object using the IMPORTED class ---
        metadata = {
            "original_filename": os.path.basename(filepath),
            "load_timestamp": pd.Timestamp.now().isoformat(),
            "initial_rows_read": original_rows,
            "rows_after_conversion_drop": rows_after_dropna,
            "rows_after_finite_filter": final_finite_rows,
            "final_points": final_unique_rows,
            "guessed_delimiter": repr(guessed_delimiter),
            "used_comment_char": repr(used_comment)
        }
        # ***** USE THE IMPORTED Spectrum class from data_models *****
        spectrum = Spectrum(
            wavelengths=wavelengths,
            raw_intensity=raw_intensity,
            metadata=metadata,
            source_filepath=filepath # Store full path
        )
        logging.info(f"Successfully loaded: {spectrum}")
        return spectrum

    except FileNotFoundError: logging.error(f"File not found during load: {filepath}"); raise
    except ValueError as ve: logging.error(f"Value error loading {filepath}: {ve}"); raise ValueError(f"Invalid format or data in {os.path.basename(filepath)}: {ve}") from ve
    except ImportError: raise # Re-raise the import error if models weren't available
    except Exception as e: logging.error(f"Unexpected error loading {filepath}: {e}", exc_info=True); raise Exception(f"Failed to load {os.path.basename(filepath)}: {e}") from e

# --- Saving Functions ---

def save_spectrum_data(spectrum: Spectrum, filepath: str, include_processed: bool = True, delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT):
    """ Saves spectrum data (using core.data_models.Spectrum). """
    if not CORE_MODELS_IMPORTED: raise ImportError("Core Spectrum model unavailable. Cannot save spectrum.")
    if not isinstance(spectrum, Spectrum) or len(spectrum) == 0: logging.warning(f"Attempted to save empty or invalid spectrum object ({type(spectrum)}). Skipping save to {filepath}."); return
    logging.info(f"Saving spectrum data ({len(spectrum)} points) to: {filepath}")
    try:
        if not filepath: raise ValueError("Filepath cannot be empty for saving.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data_dict: Dict[str, np.ndarray] = {"Wavelength_nm": spectrum.wavelengths, "Raw_Intensity": spectrum.raw_intensity}
        columns = ["Wavelength_nm", "Raw_Intensity"]
        if include_processed and spectrum.processed_intensity is not None:
            if spectrum.processed_intensity.shape == spectrum.wavelengths.shape:
                data_dict["Processed_Intensity"] = spectrum.processed_intensity; columns.append("Processed_Intensity")
                if spectrum.baseline is not None and spectrum.baseline.shape == spectrum.wavelengths.shape:
                     data_dict["Baseline"] = spectrum.baseline; columns.append("Baseline"); logging.debug("Including baseline.")
                elif spectrum.baseline is not None: logging.warning(f"Baseline length mismatch, baseline NOT saved.")
            else: logging.warning(f"Processed intensity shape mismatch. Processed data NOT saved.")
        df_to_save = pd.DataFrame(data_dict)[columns]
        df_to_save.to_csv(filepath, sep=delimiter, header=True, index=False, float_format=float_format, encoding='utf-8', na_rep='NaN', lineterminator='\n')
        logging.info(f"Spectrum data saved successfully to {filepath}")
    except Exception as e: logging.error(f"Failed to save spectrum data to {filepath}: {e}", exc_info=True); raise

def save_peak_list(peaks: List[Peak], filepath: str, delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT):
    """ Saves the detected/fitted peak list (using core.data_models.Peak). """
    if not CORE_MODELS_IMPORTED: raise ImportError("Core Peak model unavailable. Cannot save peak list.")
    logging.info(f"Attempting to save peak list ({len(peaks)} peaks) to: {filepath}")
    output_columns = [ # Match columns from Peak.to_dataframe_row in data_models.py
        "Peak Index", "Detected Wavelength (nm)", "Raw Intensity", "Processed Intensity",
        "Fit Profile", "Fitted Center (nm)", "Fitted Amplitude", "Fitted Width (nm)",
        "Fitted FWHM (nm)", "Fit Mixing (eta)", "Fit R^2", "Fit AIC", "Fit BIC",
        "Fit Amp Error", "Fit Cen Error", "Fit Wid Error"#, "Correlated Matches" # Add back if needed
    ]
    if not filepath: raise ValueError("Filepath cannot be empty for saving peak list.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not peaks:
        logging.warning("Peak list is empty. Saving file with header only.")
        try: pd.DataFrame(columns=output_columns).to_csv(filepath, sep=delimiter, index=False, encoding='utf-8', lineterminator='\n')
        except Exception as e: logging.error(f"Failed to save empty peak list header: {e}", exc_info=True); raise
        return
    try:
        peak_data_list = []; successful_conversions = 0
        for i, peak in enumerate(peaks):
             if isinstance(peak, Peak) and hasattr(peak, 'to_dataframe_row') and callable(peak.to_dataframe_row):
                 try:
                     row_data = peak.to_dataframe_row();
                     if isinstance(row_data, dict): peak_data_list.append(row_data); successful_conversions += 1
                     else: logging.warning(f"Peak {i} 'to_dataframe_row' did not return dict.")
                 except Exception as row_e: logging.error(f"Error calling 'to_dataframe_row' for peak {i}: {row_e}.", exc_info=False)
             else: logging.error(f"Peak {i} invalid or missing 'to_dataframe_row'.")
        if not peak_data_list: logging.error("No valid peak data could be extracted. Cannot save."); return
        peak_df = pd.DataFrame(peak_data_list); logging.debug(f"Created DataFrame with {len(peak_df)} rows.")
        missing_cols = set(output_columns) - set(peak_df.columns)
        if missing_cols: logging.warning(f"Adding missing columns to save data: {missing_cols}"); peak_df[list(missing_cols)] = np.nan
        final_columns = [col for col in output_columns if col in peak_df.columns] # Use only available columns
        if len(final_columns) != len(output_columns): logging.warning(f"Some columns not generated: {set(output_columns) - set(final_columns)}")
        peak_df_to_save = peak_df[final_columns]
        peak_df_to_save.to_csv(filepath, sep=delimiter, header=True, index=False, float_format=float_format, encoding='utf-8', na_rep='NaN', lineterminator='\n')
        logging.info(f"Peak list ({len(peak_df_to_save)} peaks) saved successfully.")
    except Exception as e: logging.error(f"Failed to save peak list: {e}", exc_info=True); raise

def save_nist_matches(matches_df: Optional[pd.DataFrame], filepath: str, delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT):
    """ Saves the NIST match results DataFrame (if valid) to a CSV file. """
    if not isinstance(matches_df, pd.DataFrame) and matches_df is not None: logging.error("Invalid data type passed to save_nist_matches."); raise TypeError("Data must be pandas DataFrame or None.")
    save_dataframe(matches_df, filepath, delimiter, float_format, "NIST matches")

def save_dataframe(df: Optional[pd.DataFrame], filepath: str, delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT, data_description: str = "data"):
    """ Generic function to save any pandas DataFrame to CSV with error handling. """
    desc_cap = data_description.capitalize(); is_valid_df = isinstance(df, pd.DataFrame); shape_info = f"({df.shape[0]}x{df.shape[1]})" if is_valid_df else "(None or Empty)"
    logging.info(f"Saving {data_description} {shape_info} to: {filepath}")
    try:
        if not filepath: logging.error(f"Cannot save {data_description}: filepath is empty."); raise ValueError("Invalid filepath.")
        abs_filepath = os.path.abspath(filepath); dir_name = os.path.dirname(abs_filepath)
        if dir_name: os.makedirs(dir_name, exist_ok=True)
        if not is_valid_df or df.empty:
            log_msg = f"{desc_cap} DataFrame is None." if df is None else f"{desc_cap} DataFrame is empty."; logging.warning(f"{log_msg} Saving header only.");
            cols = df.columns if is_valid_df and not df.columns.empty else []
            pd.DataFrame(columns=cols).to_csv(filepath, sep=delimiter, index=False, encoding='utf-8', lineterminator='\n')
            logging.info(f"Saved empty {data_description} file to {filepath}.")
            return
        df.to_csv(filepath, sep=delimiter, header=True, index=False, float_format=float_format, encoding='utf-8', na_rep='NaN', lineterminator='\n')
        logging.info(f"{desc_cap} saved successfully ({df.shape[0]} rows) to {filepath}")
    except Exception as e: logging.error(f"Failed to save {data_description} to {filepath}: {e}", exc_info=True); raise

# Example Usage (Optional - can be kept for module testing)
if __name__ == '__main__':
    # ... (Example Usage code can remain as is, assuming core.data_models is importable) ...
    log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format)
    logging.getLogger().name = 'file_io_test'
    test_dir = "file_io_test_output"; os.makedirs(test_dir, exist_ok=True); logging.info(f"Test output directory: {os.path.abspath(test_dir)}")
    # ... rest of example usage megh ...