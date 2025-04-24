
"""
Handles interaction with NIST Atomic Spectra Database using Astroquery for online searching.
Local database searching is not implemented in this version.
"""

import logging
import time
import traceback
import numpy as np
from typing import List, Dict, Any, Optional

# --- Astroquery Imports ---
# Encapsulated to handle potential ImportError gracefully.
ASTROQUERY_AVAILABLE = False
Nist = None
u = None
Table = None
MaskedColumn = None
try:
    from astropy import units as u
    from astroquery.nist import Nist
    from astropy.table import Table, MaskedColumn # Import MaskedColumn for checking
    ASTROQUERY_AVAILABLE = True
    logging.info("Astroquery library found. Online NIST search enabled.")
except ImportError:
    logging.warning("Astroquery or Astropy not found. Online NIST search will be disabled. "
                    "Install with 'pip install astroquery'.")

# Import data models
from .data_models import NISTMatch

# --- Constants ---
# Default values used if not provided by calling function or config
DEFAULT_ONLINE_QUERY_DELAY_S = 1.0 # Reduced default delay slightly
DEFAULT_ONLINE_TOLERANCE_NM = 0.1
DEFAULT_ONLINE_TIMEOUT_S = 20 # Increased timeout slightly

# --- Online NIST Search Function ---

def search_online_nist(wavelength_nm: float,
                       tolerance_nm: float = DEFAULT_ONLINE_TOLERANCE_NM,
                       query_delay_s: float = DEFAULT_ONLINE_QUERY_DELAY_S,
                       timeout_s: float = DEFAULT_ONLINE_TIMEOUT_S
                       ) -> List[NISTMatch]:
    """
    Performs an online query to the NIST ASD for lines within a wavelength range.

    Args:
        wavelength_nm (float): The center wavelength (in nanometers) to search around.
        tolerance_nm (float): The search tolerance (+/- nm) around the center wavelength.
        query_delay_s (float): Delay *before* executing the query (seconds).
        timeout_s (float): Timeout duration for the query (seconds).

    Returns:
        List[NISTMatch]: A list of NISTMatch objects found, or an empty list if none found or error occurs.
    """
    if not ASTROQUERY_AVAILABLE:
        logging.error("Cannot perform online NIST search: Astroquery not available.")
        return []

    if not np.isfinite(wavelength_nm) or not np.isfinite(tolerance_nm) or tolerance_nm <= 0:
        logging.error(f"Invalid search parameters for online NIST search: wavelength={wavelength_nm}, tolerance={tolerance_nm}")
        return []

    # Apply delay before the query to be nice to the NIST server
    if query_delay_s > 0:
        logging.debug(f"Applying {query_delay_s:.1f}s delay before NIST query.")
        time.sleep(query_delay_s)

    # Define search range in Angstroms (Astroquery/NIST prefers AA)
    min_wave_aa = (wavelength_nm - tolerance_nm) * 10.0 * u.AA
    max_wave_aa = (wavelength_nm + tolerance_nm) * 10.0 * u.AA

    logging.info(f"Querying NIST online: {wavelength_nm:.3f} +/- {tolerance_nm:.3f} nm ({min_wave_aa:.2f} - {max_wave_aa:.2f})")

    matches: List[NISTMatch] = []
    try:
        # Execute the query - Vacuum wavelengths often preferred for consistency
        table = Nist.query(min_wave_aa, max_wave_aa,
                           energy_level_unit="eV", # Ensure energy levels are in eV
                           output_order="wavelength", # Sort results by wavelength
                           wavelength_type="vacuum", # Request vacuum wavelengths
                           timeout=timeout_s)

        if table is None or len(table) == 0:
            logging.info(f"No lines found online for range around {wavelength_nm:.3f} nm.")
            return []

        logging.info(f"NIST query returned {len(table)} potential lines near {wavelength_nm:.3f} nm.")

        # --- Define expected NIST column names ---
        # Prioritize Ritz Vacuum, then Observed Vacuum, etc.
        wav_col_priority = [ "Observed Wavelength Ritz VAC (nm)", "Observed Wavelength VAC (nm)", "Observed Wavelength Ritz AIR (nm)", "Observed Wavelength AIR (nm)", "Observed"]
        aki_col = "Aki (s^-1)"
        ek_col = "Ek (eV)" # Upper energy level (E_k) -> Our Ei for Boltzmann
        gk_col = "gk"      # Upper statistical weight (g_k) -> Our gi for Boltzmann
        ei_col = "Ei (eV)" # Lower energy level (E_i) -> Our Ek
        gi_col = "gi"      # Lower statistical weight (g_i) -> Our gk
        elem_col = "Element"
        spec_col = "Spectrum" # e.g., "Fe I", "Ca II"

        # Process each row in the returned table
        processed_count = 0
        for row in table:
            try:
                # --- Extract Wavelength ---
                obs_wav_nm = None
                for col in wav_col_priority:
                    if col in row.colnames and row[col] is not np.ma.masked:
                         try: val = float(row[col]); obs_wav_nm = val if np.isfinite(val) else None; break # Found valid wavelength
                         except (ValueError, TypeError): pass # Failed conversion, try next column
                if obs_wav_nm is None: continue # Skip row if no valid wavelength found

                # Ensure wavelength is within tolerance (NIST might return slightly outside)
                if abs(obs_wav_nm - wavelength_nm) > tolerance_nm * 1.01: continue

                # --- Extract Other Parameters ---
                def get_val(col_name, dtype=float, default=None):
                    """Helper to safely extract and convert potentially masked data."""
                    if col_name in row.colnames and row[col_name] is not np.ma.masked:
                        try: return dtype(row[col_name])
                        except (ValueError, TypeError): return default
                    return default

                element = get_val(elem_col, str)
                spectrum_str = get_val(spec_col, str)
                if not element or not spectrum_str: continue # Skip if essential ID is missing

                aki = get_val(aki_col, float); ei_upper = get_val(ek_col, float); gi_upper = get_val(gk_col, float); ei_lower = get_val(ei_col, float); gi_lower = get_val(gi_col, float);

                # --- Basic Validation of Atomic Data ---
                # Check essential data needed for Boltzmann/CF-LIBS (Aki, Ek, gk)
                if None in [aki, ei_upper, gi_upper] or not all(np.isfinite([aki, ei_upper, gi_upper])):
                    logging.debug(f"Skipping match {spectrum_str}@{obs_wav_nm:.3f} due to missing/invalid A, E_k, or g_k.")
                    continue
                if aki <= 0 or gi_upper <= 0: # Aki and g must be positive
                     logging.debug(f"Skipping match {spectrum_str}@{obs_wav_nm:.3f} due to non-positive A or g_k.")
                     continue

                # --- Parse Spectrum String ---
                parts = spectrum_str.split(); ion_state_str = "?"
                if len(parts)==2: ion_state_str = parts[1]
                elif len(parts)==1 and parts[0]==element: ion_state_str="I"; spectrum_str=f"{element} I" # Handle neutral like "H"

                # --- Create NISTMatch object ---
                # Note: Stores UPPER level E/g as ei/gi for CF-LIBS convenience
                match = NISTMatch(
                    element=element, ion_state_str=ion_state_str, wavelength_db=obs_wav_nm,
                    aki=aki, ei=ei_upper, gi=gi_upper, # Upper state E, g
                    ek=ei_lower, gk=gi_lower, # Lower state E, g
                    line_label=f"{spectrum_str} {obs_wav_nm:.3f}", source='NIST Online'
                )
                matches.append(match)
                processed_count += 1

            except Exception as row_err:
                 # Log error specific to row processing but continue with next row
                 logging.warning(f"Skipping online row due to error: {row_err}. Data: {dict(row)}", exc_info=False)
                 continue

    except ImportError: logging.error("Astroquery import failed during query."); return [] # Should be caught earlier
    except Exception as e: logging.error(f"Error during online NIST query: {e}", exc_info=True); return [] # Return empty list on error

    # Sort final matches by proximity to the query wavelength
    matches.sort(key=lambda m: abs(m.wavelength_db - wavelength_nm))
    logging.info(f"Processed {processed_count} valid matches from online query near {wavelength_nm:.3f} nm.")
    return matches

# --- Local Search Placeholder ---
# def search_local_nist(...) -> List[NISTMatch]:
#     logging.warning("Local NIST DB search is not implemented.")
#     return []

# --- Helper Function for Fetching Script (Keep commented or remove) ---
# def get_nist_element_ion_data(element: str, ion_stage: int, **query_kwargs) -> Optional[Table]:
#     """ Helper used by nist_data_fetcher.py """
#     # ... (Implementation from previous parts) ...
#     pass

# --- Column Mapping (Keep commented or remove, not needed for online search) ---
# NIST_COLUMN_MAP = { ... }
# def find_nist_column(available_columns: List[str], target_name: str) -> Optional[str]:
#     # ... (Implementation from previous parts) ...
#     pass
