# --- START OF REFACTORED FILE database/atomic_data_builder.py ---
"""
Builds atomic data files used by the LIBS Forge application.

Reads atomic energy level data from source files, calculates partition functions
U(T) over a range of temperatures, extracts ionization energies V_ion, and
saves the results into CSV files ('partition_functions.csv', 'ionization_energies.csv').

*** IMPORTANT: USER ACTION REQUIRED ***
This script contains **PLACEHOLDER** logic for parsing source files.
You **MUST** modify the `parse_levels_file` and `extract_ionization_energy`
functions below to correctly handle the specific format of **YOUR** input data
files (e.g., downloaded NIST ASD level listings, external databases).
Failure to do so will result in incorrect or missing atomic data.
************************************

Potential Data Sources:
1. NIST Atomic Spectra Database (ASD): Requires parsing downloaded levels data (HTML or preferably TSV/CSV format if available).
2. External Compilations: Using pre-compiled tables from scientific literature or other atomic databases.

Workflow:
1. Obtain atomic level data files (energy levels, statistical weights 'g', and ideally ionization limits) for the elements/ions of interest.
2. Place these source files in the directory specified by `--input-dir` (default: 'database/source_atomic_levels/').
3. **Modify the `parse_levels_file` and `extract_ionization_energy` functions below to match your source file format.**
4. Run this script from the project root directory:
   `python database/atomic_data_builder.py [options]`
5. The script will generate/overwrite `partition_functions.csv` and `ionization_energies.csv`
   in the `database/atomic_data/` directory.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import traceback
from typing import List, Dict, Optional, Tuple, Set

# --- Setup Project Root Path ---
# Ensures the script can find core modules when run directly
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from core.atomic_data import ATOMIC_DATA_DIR, PARTITION_FUNC_FILE, IONIZATION_ENERGY_FILE
    from utils.helpers import setup_logging, get_project_root
except ImportError as e:
    print(f"[ERROR] Failed to import necessary modules: {e}")
    print("Please ensure the script is run from the project root directory or that")
    print("the project structure is correct and accessible in the Python path.")
    sys.exit(1)

# --- Constants ---
K_B_EV = 8.617333262e-5 # Boltzmann constant in eV/K
DEFAULT_TEMPERATURES_K = [
    3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 18000, 20000, 25000, 30000
]
# Placeholder column names expected after parsing
ENERGY_COL = 'Energy_eV'
G_WEIGHT_COL = 'g'

# --- Argument Parser ---
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build atomic data files (U(T), V_ion) from source level data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-dir",
        default=os.path.join(get_project_root(), "database", "source_atomic_levels"),
        help="Directory containing source atomic level data files (e.g., one file per species)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=ATOMIC_DATA_DIR,
        help="Directory to save the generated atomic data CSV files ('partition_functions.csv', 'ionization_energies.csv')."
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite existing output CSV files if they exist."
    )
    parser.add_argument(
        "--temperatures",
        nargs='+',
        type=float,
        default=DEFAULT_TEMPERATURES_K,
        help="List of temperatures (K) for which to calculate partition functions."
    )
    parser.add_argument(
        "--log-level",
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level for console output."
    )
    return parser.parse_args()

# --- Core Calculation Functions ---

def calculate_partition_function(levels_df: pd.DataFrame, temperature_k: float) -> Optional[float]:
    """
    Calculates the internal partition function U(T) for a given temperature.

    Assumes the input DataFrame contains columns named 'Energy_eV' and 'g'.

    Args:
        levels_df: DataFrame with energy levels and statistical weights.
        temperature_k: Temperature in Kelvin.

    Returns:
        The calculated partition function U(T), or None if calculation fails.
    """
    required_cols = {ENERGY_COL, G_WEIGHT_COL}
    if not required_cols.issubset(levels_df.columns):
        logging.error(f"Missing required columns for U(T) calculation. Need: {required_cols}")
        return None
    if not isinstance(temperature_k, (int, float)) or temperature_k <= 0:
        logging.error(f"Invalid temperature for U(T) calculation: {temperature_k}. Must be positive number.")
        return None

    try:
        # Ensure data is numeric, coercing errors to NaN
        energies = pd.to_numeric(levels_df[ENERGY_COL], errors='coerce')
        g_values = pd.to_numeric(levels_df[G_WEIGHT_COL], errors='coerce')

        # Filter out invalid entries (NaN energy/g, g <= 0)
        valid_mask = energies.notna() & g_values.notna() & (g_values > 0)
        if not valid_mask.any():
            logging.warning(f"No valid levels found for U(T) calculation at T={temperature_k}K after filtering.")
            return None # No valid levels to sum

        energies_valid = energies[valid_mask].to_numpy(dtype=float)
        g_valid = g_values[valid_mask].to_numpy(dtype=float)

        # Calculate Boltzmann factor, checking for numerical issues
        exp_arg = -energies_valid / (K_B_EV * temperature_k)
        # Avoid overflow in exp for very large negative arguments (high E/low T)
        # Although physically high E contributes little, exp(-large) -> 0
        # exp(large positive) -> inf (if negative E exists, which is unusual)
        # Set a practical limit on the exponent argument magnitude
        exp_arg = np.clip(exp_arg, -700, 700) # exp(-700) is near zero, exp(700) is huge but finite

        boltzmann_factor = np.exp(exp_arg)

        # Handle potential NaNs/Infs resulting from exp or previous steps
        # Treat non-finite contributions as zero
        boltzmann_factor = np.nan_to_num(boltzmann_factor, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate sum: U(T) = Sum[ g_i * exp(-E_i / (k_B * T)) ]
        partition_sum = np.sum(g_valid * boltzmann_factor)

        # Final check: Partition sum must be positive and finite
        if not np.isfinite(partition_sum) or partition_sum <= 1e-12: # Use a small threshold > 0
             logging.warning(f"Invalid partition sum calculated ({partition_sum:.3e}) at T={temperature_k}K. Check levels/g-values.")
             return None

        return partition_sum

    except Exception as e:
        logging.error(f"Error calculating U(T) at T={temperature_k}K: {e}", exc_info=True)
        return None

def extract_ionization_energy(levels_df: pd.DataFrame, filepath: str) -> Optional[float]:
    """
    *** PLACEHOLDER: User MUST implement this function ***

    Extracts the ionization energy (V_ion) for the species from the source data.
    This logic is highly dependent on the format of your input file(s).

    Args:
        levels_df: DataFrame containing the parsed level data (might be useful).
        filepath: Path to the original source file (may contain metadata).

    Returns:
        Ionization energy in eV, or None if not found or extraction fails.
    """
    # --- START OF USER MODIFICATION AREA ---
    logging.warning(f"Placeholder `extract_ionization_energy` called for {os.path.basename(filepath)}. "
                    "You MUST implement custom logic here to parse your specific file format.")

    # Example Strategy 1: Look for specific keywords in the original file
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
             for line in f:
                 line_lower = line.lower().strip()
                 # Modify keywords and parsing logic based on your file format
                 keywords = ["ionization energy", "ionization limit", "ionization potential"]
                 if any(kw in line_lower for kw in keywords):
                     # Example: Try to find a number after '=' or ':'
                     value_str = None
                     if '=' in line: value_str = line.split('=')[-1].strip()
                     elif ':' in line: value_str = line.split(':')[-1].strip()
                     # ... add more delimiters if needed ...

                     if value_str:
                         try:
                             # Extract the first number found
                             num_part = value_str.split()[0]
                             # Remove potential trailing characters like units (eV) if needed
                             num_part = num_part.replace('(','').replace(')','').rstrip('ev')
                             energy = float(num_part)
                             if energy > 0: # Basic sanity check
                                 logging.debug(f"Extracted V_ion = {energy} eV from line: '{line.strip()}'")
                                 return energy
                             else:
                                 logging.warning(f"Parsed non-positive potential V_ion ({energy}) from '{line.strip()}'")
                         except (ValueError, IndexError):
                             logging.debug(f"Could not parse float from value string '{value_str}' on line: '{line.strip()}'")
                             continue # Failed to parse, try next line
    except IOError as e:
        logging.error(f"Error reading file {filepath} to extract ionization energy: {e}")
        return None # File read error

    # Example Strategy 2: Check if ionization limit is the max energy in a specific column
    # if ENERGY_COL in levels_df.columns and not levels_df.empty:
    #     # This assumes the ionization limit is explicitly listed as the highest energy level
    #     try:
    #          max_energy = pd.to_numeric(levels_df[ENERGY_COL], errors='coerce').max()
    #          if pd.notna(max_energy) and max_energy > 0:
    #               # Check if the highest level has a specific marker indicating it's the limit?
    #               # if levels_df.loc[levels_df[ENERGY_COL] == max_energy, 'is_limit_marker_column'].iloc[0]: # Fictional column check
    #               #     return max_energy
    #               logging.debug(f"Found max energy {max_energy} eV, but unsure if it's V_ion without further context.")
    #     except Exception as e:
    #          logging.warning(f"Could not determine max energy for V_ion check: {e}")

    # If no ionization energy is found after trying implemented methods:
    logging.error(f"Failed to extract ionization energy for {os.path.basename(filepath)}. "
                  "Please implement parsing logic in `extract_ionization_energy`.")
    return None
    # --- END OF USER MODIFICATION AREA ---

def parse_levels_file(filepath: str) -> Optional[pd.DataFrame]:
    """
    *** PLACEHOLDER: User MUST implement this function ***

    Parses a single source atomic levels data file into a pandas DataFrame.
    The resulting DataFrame MUST contain columns named 'Energy_eV' and 'g'.

    Args:
        filepath (str): Path to the source levels data file.

    Returns:
        DataFrame with level data (minimally 'Energy_eV' and 'g' columns),
        or None if parsing fails or required columns are missing.
    """
    # --- START OF USER MODIFICATION AREA ---
    logging.warning(f"Placeholder `parse_levels_file` called for {os.path.basename(filepath)}. "
                    "You MUST implement custom logic here to parse your specific file format.")

    # Example Parsing Logic for a hypothetical Tab-Separated Value (TSV) file
    # Adjust parameters like `sep`, `header`, `comment`, `usecols`, `names` based on your file.
    try:
        # Example: Assuming TSV, skip first 3 rows, no header row in data, use specific columns
        # df = pd.read_csv(filepath, sep='\t', skiprows=3, header=None, comment='#',
        #                  usecols=[2, 4], # Hypothetical: column 2 is Energy, column 4 is g
        #                  names=['SourceEnergy', 'SourceG']) # Assign temporary names

        # Example: Assuming a more complex CSV with headers that need mapping
        # df = pd.read_csv(filepath, sep=',', comment='#')
        # # Define how source columns map to required columns
        # column_map = {
        #     'Level (eV)': ENERGY_COL,     # Replace 'Level (eV)' with actual column name in your file
        #     'Weight': G_WEIGHT_COL,       # Replace 'Weight' with actual column name in your file
        #     # Add other columns you might want to keep temporarily
        # }
        # # Keep only the columns we need and rename them
        # source_cols_needed = list(column_map.keys())
        # if not all(col in df.columns for col in source_cols_needed):
        #      missing = [col for col in source_cols_needed if col not in df.columns]
        #      logging.error(f"Missing required source columns in {filepath}: {missing}")
        #      return None
        # df = df[source_cols_needed].rename(columns=column_map)

        # --- If using the dummy data example: ---
        if "dummy" in os.path.basename(filepath).lower():
             logging.info(f"Using dummy data for {filepath}")
             dummy_levels = {'Energy_eV': [0.0, 2.1, 3.5, 4.8, 10.2], 'g': [1, 3, 5, 7, 2]}
             df = pd.DataFrame(dummy_levels)
             # --- End of dummy data ---
        else:
             # If not dummy, raise error as placeholder wasn't replaced
              raise NotImplementedError(f"Parsing logic required in `parse_levels_file` for {filepath}")

        # --- Post-Parsing Validation ---
        required_cols = {ENERGY_COL, G_WEIGHT_COL}
        if not required_cols.issubset(df.columns):
            logging.error(f"Parsing result for {filepath} is missing required columns. "
                          f"Need: {required_cols}, Got: {list(df.columns)}")
            return None

        # Ensure columns are numeric (or can be coerced), log issues
        for col in [ENERGY_COL, G_WEIGHT_COL]:
             original_non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
             if original_non_numeric > 0:
                  logging.warning(f"Column '{col}' in parsed {filepath} contained "
                                  f"{original_non_numeric} non-numeric entries that were coerced to NaN.")
        return df # Return the processed DataFrame

    except FileNotFoundError:
        logging.error(f"Source file not found: {filepath}")
        return None
    except NotImplementedError as nie:
         logging.error(nie) # Log the specific NotImplementedError
         return None
    except Exception as e:
        logging.error(f"Failed to parse levels file {filepath}: {e}", exc_info=True)
        return None
    # --- END OF USER MODIFICATION AREA ---


def process_species_file(
    filepath: str,
    temperatures_k: List[float]
) -> Tuple[Optional[float], Optional[Dict[float, float]], int, int]:
    """
    Processes a single source file: parses levels, calculates V_ion and U(T).

    Args:
        filepath: Path to the source file.
        temperatures_k: List of temperatures (K) for U(T) calculation.

    Returns:
        Tuple containing:
        - Ionization energy (eV) or None.
        - Dictionary of {Temperature: U(T)} or None.
        - Number of levels parsed (or 0).
        - Number of valid levels used for U(T) (or 0).
    """
    logging.info(f"Processing source file: {os.path.basename(filepath)}")
    levels_df = parse_levels_file(filepath)
    if levels_df is None:
        logging.error(f"Skipping file due to parsing error: {filepath}")
        return None, None, 0, 0

    num_parsed_levels = len(levels_df)
    logging.debug(f"Parsed {num_parsed_levels} potential levels from {os.path.basename(filepath)}.")

    # Calculate Partition Functions
    partition_functions = {}
    valid_levels_count_example = 0 # Get count from one temperature calculation
    temp_for_count = temperatures_k[len(temperatures_k)//2] # Pick a middle temp

    for temp in temperatures_k:
        u_t = calculate_partition_function(levels_df, temp)
        if u_t is not None:
            partition_functions[temp] = u_t
            if temp == temp_for_count: # Get valid count
                 energies = pd.to_numeric(levels_df[ENERGY_COL], errors='coerce')
                 g_values = pd.to_numeric(levels_df[G_WEIGHT_COL], errors='coerce')
                 valid_mask = energies.notna() & g_values.notna() & (g_values > 0)
                 valid_levels_count_example = valid_mask.sum()

    # Extract Ionization Energy
    ionization_energy = extract_ionization_energy(levels_df, filepath)

    # --- Logging Summary for the File ---
    if ionization_energy is None:
        logging.warning(f"Could not extract ionization energy from {os.path.basename(filepath)}.")
    else:
         logging.info(f"Extracted Ionization Energy = {ionization_energy:.5f} eV for {os.path.basename(filepath)}")

    if not partition_functions:
        logging.warning(f"Could not calculate any partition functions for {os.path.basename(filepath)}. Check levels and g-values.")
    else:
        num_temps = len(partition_functions)
        logging.info(f"Calculated {num_temps}/{len(temperatures_k)} partition functions for {os.path.basename(filepath)} (using ~{valid_levels_count_example}/{num_parsed_levels} valid levels).")

    # Return None for dict if empty, easier check later
    return ionization_energy, partition_functions if partition_functions else None, num_parsed_levels, valid_levels_count_example

def infer_species_from_filename(filename: str) -> Optional[str]:
    """
    Attempts to infer the species name (e.g., "Fe I", "O II") from the filename.
    NOTE: This logic is basic and might need adjustment based on your file naming convention.
    """
    base_name = os.path.splitext(filename)[0]
    # Common suffixes to remove
    suffixes_to_remove = ['_levels', '-levels', '_data', '-data', '_asd']
    for suffix in suffixes_to_remove:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]

    # Try splitting by common delimiters (e.g., 'Fe_I', 'O-II')
    delimiters = ['_', '-']
    for delim in delimiters:
        parts = base_name.split(delim)
        if len(parts) == 2:
            element, ion_stage = parts[0], parts[1]
            # Basic check: element looks alphabetic, ion stage might be Roman numeral or number
            # This is a weak check! Assumes Roman numerals are uppercase I, V, X etc.
            if element.isalpha() and ion_stage: # and ion_stage.isupper()?
                 # Format consistently: Capitalize element, keep ion stage as is (might be I, II, 1, 2 etc.)
                 species = f"{element.capitalize()} {ion_stage}"
                 logging.debug(f"Inferred species '{species}' from filename '{filename}'")
                 return species

    # Fallback if no delimiter found or pattern doesn't match
    logging.warning(f"Could not reliably infer species name from filename '{filename}'. "
                    "Filename should ideally be like 'Element_IonStage' (e.g., Fe_I, O_II).")
    return None


# --- Main Execution ---
def main():
    """Main script execution function."""
    args = parse_arguments()

    # Setup logging
    log_file = os.path.join(get_project_root(), "logs", "atomic_data_builder.log")
    setup_logging(log_level_str=args.log_level, log_file=log_file, logger_name=None) # Use root logger

    logging.warning("--- Atomic Data Builder ---")
    logging.warning("*** IMPORTANT: Ensure `parse_levels_file` and `extract_ionization_energy` functions are correctly implemented for your data format! ***")
    logging.info(f"Input Directory (Source Levels): {args.input_dir}")
    logging.info(f"Output Directory (Generated CSVs): {args.output_dir}")
    logging.info(f"Temperatures for U(T) (K): {args.temperatures}")
    logging.info(f"Overwrite Existing Output: {args.overwrite}")

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logging.critical(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Ensure output directory exists
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        logging.critical(f"Could not create output directory {args.output_dir}: {e}")
        sys.exit(1)

    # Check if output files exist and handle overwrite logic
    ion_path = os.path.join(args.output_dir, os.path.basename(IONIZATION_ENERGY_FILE))
    part_path = os.path.join(args.output_dir, os.path.basename(PARTITION_FUNC_FILE))

    if not args.overwrite:
        if os.path.exists(ion_path):
             logging.warning(f"Output file '{ion_path}' already exists. Use --overwrite to replace it. Skipping V_ion generation.")
             # Set flag to skip saving later
             skip_ion_save = True
        else: skip_ion_save = False
        if os.path.exists(part_path):
            logging.warning(f"Output file '{part_path}' already exists. Use --overwrite to replace it. Skipping U(T) generation.")
            # Set flag to skip saving later
            skip_part_save = True
        else: skip_part_save = False
        if skip_ion_save and skip_part_save:
             logging.info("Both output files exist and --overwrite not specified. Nothing to do.")
             sys.exit(0)
    else:
         skip_ion_save = False
         skip_part_save = False


    # --- Process Files ---
    all_ionization_data: List[Dict] = []
    all_partition_data: List[Dict] = []
    processed_files: Set[str] = set() # Keep track of processed species
    file_count = 0
    success_count = 0 # Files from which *any* data was extracted

    source_files = sorted([f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))])

    if not source_files:
         logging.warning(f"No files found in input directory: {args.input_dir}")
         sys.exit(0)

    logging.info(f"Found {len(source_files)} files in input directory. Processing...")

    for filename in source_files:
        filepath = os.path.join(args.input_dir, filename)
        file_count += 1

        species = infer_species_from_filename(filename)
        if not species:
            logging.warning(f"Could not determine species for '{filename}', skipping.")
            continue
        if species in processed_files:
            logging.warning(f"Species '{species}' already processed (likely duplicate filename pattern), skipping '{filename}'.")
            continue

        try:
            V_ion, U_T_dict, _, _ = process_species_file(filepath, args.temperatures)

            data_extracted = False
            if V_ion is not None:
                all_ionization_data.append({'Species': species, 'IonizationEnergy_eV': V_ion})
                data_extracted = True
            if U_T_dict:
                for temp, u_val in U_T_dict.items():
                    all_partition_data.append({
                        'Species': species,
                        'Temperature_K': temp,
                        'PartitionFunction_U': u_val
                    })
                data_extracted = True

            if data_extracted:
                success_count += 1
                processed_files.add(species) # Mark species as processed

        except Exception as e:
            logging.error(f"Unhandled error processing file {filepath}: {e}", exc_info=True)
            # Continue to next file

    logging.info(f"Finished processing. Successfully extracted data for {success_count}/{file_count} files.")

    # --- Save Results ---
    # Save Ionization Energies
    if all_ionization_data and not skip_ion_save:
        try:
            ion_df = pd.DataFrame(all_ionization_data).sort_values('Species').reset_index(drop=True)
            ion_df.to_csv(ion_path, index=False, float_format='%.5f')
            logging.info(f"Saved Ionization Energies ({len(ion_df)} entries) to: {ion_path}")
        except Exception as e:
            logging.error(f"Failed to save Ionization Energy data to {ion_path}: {e}", exc_info=True)
    elif not all_ionization_data:
        logging.warning("No Ionization Energy data was generated.")
    elif skip_ion_save:
        logging.info("Skipped saving Ionization Energy file (already exists).")


    # Save Partition Functions
    if all_partition_data and not skip_part_save:
        try:
            part_df = pd.DataFrame(all_partition_data).sort_values(['Species', 'Temperature_K']).reset_index(drop=True)
            part_df.to_csv(part_path, index=False, float_format='%.5e') # Use scientific notation for U(T)
            logging.info(f"Saved Partition Functions ({len(part_df)} entries) to: {part_path}")
        except Exception as e:
            logging.error(f"Failed to save Partition Function data to {part_path}: {e}", exc_info=True)
    elif not all_partition_data:
        logging.warning("No Partition Function data was generated.")
    elif skip_part_save:
        logging.info("Skipped saving Partition Function file (already exists).")


    logging.info("--- Atomic Data Builder Finished ---")
    if success_count == 0 and file_count > 0:
         logging.error("!!! No data was successfully extracted from any source file. !!!")
         logging.error("!!! Please check your input files and IMPLEMENT the required parsing logic in `parse_levels_file` and `extract_ionization_energy` functions within this script. !!!")

if __name__ == "__main__":
    main()
# --- END OF REFACTORED FILE database/atomic_data_builder.py ---