
"""
Script to fetch NIST Atomic Spectra Database data for specified elements and
ionization stages using astroquery and save it locally as CSV files.

This script iterates through elements and ionization states, queries NIST,
and saves the results in a structured directory.

WARNING: Running this for all elements and many ion stages can take a VERY
         long time (hours or even days) and generate a large amount of data.
         It also places significant load on the NIST servers. Please use responsibly
         and check NIST's usage policies. Consider fetching only the elements
         you need, perhaps with a lower --max-ion setting initially.
"""

import os
import sys
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

# --- Setup Project Root Path ---
# Ensure the script can find core modules when run directly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Assumes script is in database/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local Imports ---
# Check for core dependencies first
_CORE_AVAILABLE = False
try:
    from utils.helpers import get_project_root, setup_logging # Import setup_logging if desired for file output
    _CORE_AVAILABLE = True
except ImportError:
    # Fallback basic logging if utils not found (e.g., running script standalone)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-7s - %(message)s', datefmt='%H:%M:%S')
    logging.warning("Could not import from utils. Using basic console logging.")
    # Define get_project_root locally if needed
    if 'get_project_root' not in globals():
        get_project_root = lambda: project_root # Simple lambda fallback

# Now import nist_manager, checking its dependency
try:
    from core.nist_manager import get_nist_element_ion_data, ASTROQUERY_AVAILABLE
    if not ASTROQUERY_AVAILABLE:
        logging.critical("Astroquery library is required but not installed or failed to import. Please run:")
        logging.critical("pip install astroquery")
        sys.exit(1)
except ImportError as e:
    logging.critical(f"Error importing core modules (nist_manager): {e}")
    logging.critical("Please ensure the script is run from within the project structure "
                     "or the project root is in the Python path.")
    sys.exit(1)

# --- Constants ---
DEFAULT_MAX_EMPTY_CONSECUTIVE = 3
DEFAULT_QUERY_DELAY_S = 2.0
DEFAULT_MAX_WORKERS = 4 # Be mindful of NIST server load

# --- Argument Parser ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fetch NIST ASD data locally.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e","--elements", nargs='+', default=None, help="List of specific element symbols (e.g., Fe Ca Si). Default: Common LIBS elements.")
    parser.add_argument("-o","--output-dir", default=os.path.join(get_project_root(),"database","nist_data_cache"), help="Base directory to save CSV files.")
    parser.add_argument("--max-ion", type=int, default=7, help="Maximum ionization stage (e.g., 7 for VII).")
    parser.add_argument("--max-empty", type=int, default=DEFAULT_MAX_EMPTY_CONSECUTIVE, help="Stop after N consecutive empty ion stages.")
    parser.add_argument("--delay", type=float, default=DEFAULT_QUERY_DELAY_S, help="Delay (s) between queries per worker.")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="Max parallel download threads.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing CSV files.")
    parser.add_argument("--log-level", default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help="Console logging level.")
    return parser.parse_args()

# --- List of Common Elements (Fallback) ---
COMMON_LIBS_ELEMENTS = ["H","Li","Be","B","C","N","O","F","Na","Mg","Al","Si","P","S","Cl","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Rb","Sr","Y","Zr","Nb","Mo","Ag","Cd","In","Sn","Sb","Te","I","Cs","Ba","La","Ce","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","U"]

# --- Roman Numeral Map (Extend if needed for higher stages) ---
ROMAN_MAP_REV = {1:'I', 2:'II', 3:'III', 4:'IV', 5:'V', 6:'VI', 7:'VII', 8:'VIII', 9:'IX', 10:'X', 11:'XI', 12:'XII'}

# --- Worker Function ---
def fetch_and_save_worker(element: str, ion_stage: int, element_dir: str, delay: float, overwrite: bool) -> Tuple[str, int, bool]:
    """ Worker task: fetches data for one ion, saves, returns status. """
    roman = ROMAN_MAP_REV.get(ion_stage)
    if not roman: logging.warning(f"Cannot convert ion stage {ion_stage} to Roman. Skipping."); return element, ion_stage, False
    species_name = f"{element} {roman}"
    filename = f"{element}_{roman}.csv"; output_path = os.path.join(element_dir, filename)
    if os.path.exists(output_path) and not overwrite: logging.info(f"Skipping {species_name}: Exists."); return element, ion_stage, True # Treat existing as success
    time.sleep(delay) # Apply delay before query
    try:
        table = get_nist_element_ion_data(element, ion_stage) # Uses helper
        if table is not None and len(table) > 0:
            os.makedirs(element_dir, exist_ok=True)
            table.write(output_path, format="csv", overwrite=True)
            logging.info(f"Saved: {species_name} ({len(table)} lines) -> {filename}")
            return element, ion_stage, True # Data found and saved
        else: logging.info(f"No data found for {species_name}."); return element, ion_stage, False # No data
    except Exception as e: logging.error(f"Error processing {species_name}: {e}"); return element, ion_stage, False # Error

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_arguments()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s-%(levelname)-7s-%(message)s', datefmt='%H:%M:%S')
    # Optional: Use setup_logging from utils if more complex config needed later
    # if _CORE_AVAILABLE: setup_logging({'logging':{'log_level_console': args.log_level}})

    elements_to_fetch = args.elements or COMMON_LIBS_ELEMENTS
    logging.info(f"--- NIST Data Fetcher ---")
    logging.info(f"Elements: {', '.join(elements_to_fetch)}")
    logging.info(f"Max Ion Stage: {args.max_ion}")
    logging.info(f"Output Dir: {args.output_dir}")
    logging.info(f"Overwrite: {args.overwrite}, Delay: {args.delay}s, Workers: {args.workers}")
    print("-" * 25, flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    total_files_processed = 0; total_success = 0; total_errors = 0
    element_empty_counts = {el: 0 for el in elements_to_fetch}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        tasks_to_submit = []
        # Generate all potential tasks first
        for element in elements_to_fetch:
            element_dir = os.path.join(args.output_dir, element)
            for ion_stage in range(1, args.max_ion + 1):
                 tasks_to_submit.append((element, ion_stage, element_dir))

        logging.info(f"Submitting {len(tasks_to_submit)} potential tasks...")
        # Submit tasks
        for element, ion_stage, element_dir in tasks_to_submit:
             future = executor.submit(fetch_and_save_worker, element, ion_stage, element_dir, args.delay, args.overwrite)
             futures[future] = (element, ion_stage)

        total_tasks = len(futures)
        logging.info(f"Waiting for {total_tasks} tasks to complete...")
        # Process results as they complete
        for future in as_completed(futures):
            element, ion_stage = futures[future]
            total_files_processed += 1
            progress_percent = (total_files_processed / total_tasks) * 100
            print(f"Progress: {total_files_processed}/{total_tasks} ({progress_percent:.1f}%) - Last: {element} {ROMAN_MAP_REV.get(ion_stage, '?')}", end='\r', flush=True)

            try:
                elem_res, ion_res, success = future.result()
                if success:
                    total_success += 1
                    element_empty_counts[elem_res] = 0 # Reset counter
                else:
                    element_empty_counts[elem_res] += 1 # Increment empty/error count
            except Exception as e:
                 logging.error(f"Error getting result for {element} {ion_stage}: {e}")
                 total_errors += 1
                 element_empty_counts[element] += 1

            # Note: Stopping logic based on empty counts is less effective in parallel
            # as all tasks are submitted upfront. It mainly prevents over-logging success.
            if element_empty_counts[element] >= args.max_empty:
                 logging.debug(f"Max empty reached for {element}. Further results ignored if empty.")

    print("\n" + "="*30) # Newline after progress indicator
    logging.info("NIST data fetching process finished.")
    logging.info(f"Total Tasks Processed: {total_files_processed}")
    logging.info(f"Successful Fetches/Skips: {total_success}")
    logging.info(f"Errors during processing: {total_errors}")
    print("="*30)