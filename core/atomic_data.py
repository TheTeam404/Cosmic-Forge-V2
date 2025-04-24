
"""
Module for handling complex atomic data retrieval and calculation,
such as Partition Functions and Ionization Energies.

Loads data from CSV files if available, otherwise uses limited defaults.
Provides functions to retrieve data, handling interpolation for partition functions.

IMPORTANT: The accuracy of CF-LIBS calculations heavily depends on the quality
           and completeness of the data in partition_functions.csv and
           ionization_energies.csv. Users should generate these files from
           reliable sources (e.g., NIST ASD website levels data, literature)
           using the provided placeholder builder script (`database/atomic_data_builder.py`)
           as a template, or by obtaining pre-compiled files.
"""

import logging
import os
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any # Added Any

# Import helpers if needed later
from utils.helpers import get_project_root

# --- Constants ---
ATOMIC_DATA_DIR = os.path.join(get_project_root(), "database", "atomic_data")
PARTITION_FUNC_FILE = os.path.join(ATOMIC_DATA_DIR, "partition_functions.csv")
IONIZATION_ENERGY_FILE = os.path.join(ATOMIC_DATA_DIR, "ionization_energies.csv")

# --- Data Caches ---
_partition_function_cache: Dict[str, Any] = {"loaded": None} # Status: None, True, False
_ionization_energy_cache: Dict[str, Any] = {"loaded": None}

# --- Default/Fallback Data ---
DEFAULT_IONIZATION_ENERGIES = { # Used if file loading fails
    'H I': 13.59844, 'He I': 24.58739, 'Li I': 5.39172, 'Be I': 9.3227, 'B I': 8.29803,
    'C I': 11.2603, 'N I': 14.53414, 'O I': 13.61806, 'F I': 17.42282, 'Ne I': 21.56454,
    'Na I': 5.13908, 'Mg I': 7.64624, 'Al I': 5.98577, 'Si I': 8.15169, 'P I': 10.48669,
    'S I': 10.36001, 'Cl I': 12.96764, 'Ar I': 15.75962, 'K I': 4.34066, 'Ca I': 6.11316,
    'Fe I': 7.9024, 'Fe II': 16.1877, 'Ca II': 11.87172, 'Mg II': 15.03528, 'Al II': 18.82856,
    'Si II': 16.34585, 'Ti I': 6.8281, 'Ti II': 13.5755, 'Mn I': 7.43403, 'Mn II': 15.6399,
    'Ni I': 7.6398, 'Ni II': 18.16884, 'Cr I': 6.7665, 'Cr II': 16.4857,
    'Cu I': 7.72638, 'Cu II': 20.2924, 'Zn I': 9.3942, 'Zn II': 17.9644,
    'Sr I': 5.6949, 'Sr II': 11.03013, 'Ba I': 5.2117, 'Ba II': 10.00383,
    # Add more if needed for common fallback scenarios
}


# --- Partition Function Handling ---
def _find_csv_column(df_columns: List[str], target_options: List[str]) -> Optional[str]:
    """Finds the best matching column name case-insensitively."""
    df_cols_lower = {col.lower().strip(): col for col in df_columns}
    for option in target_options:
        option_lower = option.lower().strip()
        if option_lower in df_cols_lower:
            return df_cols_lower[option_lower] # Return original case
    return None

def _load_partition_functions():
    """ Loads partition function data from PARTITION_FUNC_FILE (e.g., CSV). """
    global _partition_function_cache
    if _partition_function_cache.get("loaded") is not None: return
    os.makedirs(ATOMIC_DATA_DIR, exist_ok=True)
    if not os.path.exists(PARTITION_FUNC_FILE): logging.warning(f"U(T) file not found: {PARTITION_FUNC_FILE}."); _partition_function_cache={"loaded":False}; return

    try:
        logging.info(f"Loading U(T) from {PARTITION_FUNC_FILE}...")
        df = pd.read_csv(PARTITION_FUNC_FILE, comment='#')
        # Find columns flexibly
        species_col = _find_csv_column(df.columns, ['Species', 'Ion', 'Element Spec'])
        temp_col = _find_csv_column(df.columns, ['Temperature_K', 'T(K)', 'Temperature', 'Temp'])
        u_col = _find_csv_column(df.columns, ['PartitionFunction_U', 'U(T)', 'Partition Function', 'U'])
        if not all([species_col, temp_col, u_col]): raise ValueError(f"File missing required columns (Species, Temp_K, U(T)). Found: {df.columns}")

        df[temp_col]=pd.to_numeric(df[temp_col],errors='coerce'); df[u_col]=pd.to_numeric(df[u_col],errors='coerce'); df.dropna(subset=[temp_col,u_col],inplace=True); df=df[df[u_col]>0]
        temp_cache: Dict[str, Dict[float, float]] = {}
        for _,row in df.iterrows(): species=str(row[species_col]).strip(); temp=float(row[temp_col]); u_val=float(row[u_col]);
        if species not in temp_cache: temp_cache[species]={}; temp_cache[species][temp]=u_val
        _partition_function_cache.clear()
        for species,data in temp_cache.items(): sorted_temps=sorted(data.keys());
        if len(sorted_temps)>=2: _partition_function_cache[species]={'temps':np.array(sorted_temps),'u_values':np.array([data[t] for t in sorted_temps])}
        elif len(sorted_temps)==1: _partition_function_cache[species]={'temps':np.array(sorted_temps),'u_values':np.array([data[sorted_temps[0]]])}
        _partition_function_cache["loaded"]=True; logging.info(f"Loaded U(T) data for {len(_partition_function_cache)-1} species.")
    except Exception as e: logging.error(f"Failed loading U(T): {e}",exc_info=True); _partition_function_cache={"loaded":False}

def get_partition_function(species: str, temperature_k: float) -> Optional[float]:
    """ Retrieves or interpolates the partition function U(T). """
    global _partition_function_cache
    if _partition_function_cache.get("loaded") is None: _load_partition_functions()
    if not _partition_function_cache.get("loaded", False): return None
    species_data = _partition_function_cache.get(species)
    if not species_data: logging.warning(f"U(T) data not found for {species}."); return None
    temps=species_data['temps']; u_values=species_data['u_values']
    if len(temps)<2:
        if len(temps)==1 and np.isclose(temperature_k,temps[0]): return u_values[0]
        logging.warning(f"Need >=2 pts for U(T) interp for {species}. Have {len(temps)}."); return None
    try: # Use linear interpolation
        interpolated_u = np.interp(temperature_k, temps, u_values)
        if temperature_k<temps[0] or temperature_k>temps[-1]: logging.warning(f"T={temperature_k:.0f}K outside U(T) range [{temps[0]:.0f}-{temps[-1]:.0f}]K for {species}. Extrapolated.")
        logging.debug(f"U(T={temperature_k:.0f}K) for {species}: {interpolated_u:.3f}"); return interpolated_u
    except Exception as e: logging.error(f"U(T) interp error for {species} @ {temperature_k}K: {e}"); return None

# --- Ionization Energy Handling ---
def _load_ionization_energies():
    """ Loads ionization energy data from IONIZATION_ENERGY_FILE. """
    global _ionization_energy_cache
    if _ionization_energy_cache.get("loaded") is not None: return
    os.makedirs(ATOMIC_DATA_DIR, exist_ok=True)
    if not os.path.exists(IONIZATION_ENERGY_FILE): logging.warning(f"V_ion file not found: {IONIZATION_ENERGY_FILE}. Using defaults."); _ionization_energy_cache=DEFAULT_IONIZATION_ENERGIES.copy(); _ionization_energy_cache["loaded"]=True; return
    try:
        logging.info(f"Loading V_ion from {IONIZATION_ENERGY_FILE}...")
        df = pd.read_csv(IONIZATION_ENERGY_FILE, comment='#'); df.columns=df.columns.str.strip()
        species_col = _find_csv_column(df.columns, ['Species', 'Ion', 'Element Spec Lower'])
        energy_col = _find_csv_column(df.columns, ['IonizationEnergy_eV', 'V_ion (eV)', 'Ionization Energy (eV)'])
        if not species_col or not energy_col: raise ValueError(f"File missing required columns (Species, IonizationEnergy_eV). Found: {df.columns}")

        df[energy_col]=pd.to_numeric(df[energy_col],errors='coerce'); df.dropna(subset=[energy_col],inplace=True); loaded_energies={}; loaded_count=0
        for _,row in df.iterrows(): species=str(row[species_col]).strip(); energy=float(row[energy_col]);
        if np.isfinite(energy) and energy>0: loaded_energies[species]=energy; loaded_count+=1
        # Start with defaults, update with file data
        _ionization_energy_cache=DEFAULT_IONIZATION_ENERGIES.copy(); _ionization_energy_cache.update(loaded_energies); _ionization_energy_cache["loaded"]=True; logging.info(f"Loaded/updated V_ion for {loaded_count} species from file.")
    except Exception as e: logging.error(f"Failed loading V_ion: {e}",exc_info=True);
    # Ensure cache is marked loaded even if file loading fails (use defaults)
    if not _ionization_energy_cache or _ionization_energy_cache.get("loaded") is None: _ionization_energy_cache=DEFAULT_IONIZATION_ENERGIES.copy(); _ionization_energy_cache["loaded"]=True


def get_ionization_energy(species_lower: str) -> Optional[float]:
    """ Retrieves the ionization energy (eV) for the specified lower ionization stage. """
    if _ionization_energy_cache.get("loaded") is None: _load_ionization_energies()
    # Perform case-insensitive lookup if direct match fails? Maybe not needed if file/defaults are consistent.
    energy = _ionization_energy_cache.get(species_lower)
    if energy is None: logging.warning(f"V_ion not found for: {species_lower}"); return None
    elif not np.isfinite(energy) or energy<=0: logging.warning(f"Invalid V_ion ({energy}) for {species_lower}"); return None
    logging.debug(f"V_ion for {species_lower}: {energy:.3f} eV"); return energy
