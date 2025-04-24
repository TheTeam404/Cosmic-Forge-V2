
"""
Peak detection algorithms for LIBS spectra. Primarily uses SciPy's find_peaks.
Includes options for basic filtering based on height, distance, width, prominence.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# --- SciPy Imports ---
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.error("SciPy library not found. Peak detection functionality unavailable. Install with 'pip install scipy'.")
    # Dummy function to avoid NameErrors elsewhere, raises error if called
    def find_peaks(*args, **kwargs): raise ImportError("SciPy not installed.")

# Import data models
from .data_models import Spectrum, Peak

def detect_peaks_scipy(spectrum: Spectrum,
                       rel_height_percent: float = 5.0,
                       min_distance_points: int = 5,
                       min_width_points: Optional[float] = None, # Allow float input from UI
                       prominence: Optional[float] = None
                       ) -> List[Peak]:
    """
    Detects peaks in the processed spectrum using scipy.signal.find_peaks.

    Args:
        spectrum (Spectrum): The Spectrum object containing processed data.
                             Must have `processed_intensity` attribute set.
        rel_height_percent (float): Minimum peak height relative to the intensity range
                                   (max-min) of the processed data (0-100).
        min_distance_points (int): Minimum horizontal distance (in data points)
                                  between neighbouring peaks.
        min_width_points (Optional[float]): Minimum width of peaks in data points. If float,
                                           will be rounded to nearest integer >= 0.
                                           Set to 0 or None to disable.
        prominence (Optional[float]): Minimum vertical distance (prominence) required for a peak.
                                      Set to 0 or None to disable.

    Returns:
        List[Peak]: A list of detected Peak objects, sorted by wavelength.
                    Returns empty list if no processed data, SciPy unavailable, or no peaks found.
    """
    if not SCIPY_AVAILABLE:
        logging.error("Cannot detect peaks: SciPy unavailable.")
        return []

    if spectrum.processed_intensity is None:
        logging.error("Cannot detect peaks: Processed intensity data is missing. Run processing first.")
        return []
    if len(spectrum.processed_intensity) < 3: # Need at least 3 points for peak detection
        logging.warning("Cannot detect peaks: Spectrum has less than 3 data points.")
        return []

    x_processed = spectrum.wavelengths
    y_processed = spectrum.processed_intensity

    # --- Input Validation and Preparation ---
    # Handle potential NaNs/Infs in processed data robustly
    finite_mask = np.isfinite(y_processed)
    if not np.all(finite_mask):
        logging.warning("NaNs/Infs detected in processed data for peak detection. Results may be affected.")
        # Option 1: Let find_peaks handle it (might ignore NaNs or error)
        y_search = y_processed
        # Option 2: Interpolate or fill NaNs (can introduce artifacts)
        # y_search = pd.Series(y_processed).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').to_numpy()
        # Option 3: Search only on finite segments (complex)
        # For now, proceed with original data, letting find_peaks handle non-finite values if it can.
    else:
        y_search = y_processed

    # Ensure y_search is 1D
    if y_search.ndim != 1:
         logging.error(f"Peak detection input 'y_search' must be 1D, got shape {y_search.shape}.")
         return []

    # Calculate absolute height threshold based on relative percentage
    # Use only finite values for range calculation
    y_finite = y_search[np.isfinite(y_search)]
    if len(y_finite) == 0: logging.warning("No finite data points for peak detection range."); return []
    min_intensity = np.min(y_finite)
    max_intensity = np.max(y_finite)
    data_range = max_intensity - min_intensity

    height_threshold: Optional[float] = None
    if data_range > 1e-9: # Avoid division by zero or issues with flat lines
        if 0 < rel_height_percent <= 100:
            height_threshold = min_intensity + (rel_height_percent / 100.0) * data_range
        else:
            logging.debug(f"Relative height ({rel_height_percent}%) outside (0, 100]. Disabling height threshold.")
    else:
        logging.warning("Data range is too small to calculate relative height threshold.")

    # Validate distance parameter
    if min_distance_points < 1:
        logging.warning(f"Minimum peak distance ({min_distance_points}) must be >= 1. Using 1.")
        min_distance_points = 1

    # Validate and prepare optional parameters for find_peaks
    find_peaks_kwargs = {'distance': int(min_distance_points)} # Ensure integer
    if height_threshold is not None:
        find_peaks_kwargs['height'] = height_threshold
    if min_width_points is not None and min_width_points > 0:
        # find_peaks expects width as number of samples, must be non-negative integer
        find_peaks_kwargs['width'] = max(0, int(round(min_width_points)))
    if prominence is not None and prominence > 0:
        find_peaks_kwargs['prominence'] = prominence

    logging.info(f"Running find_peaks with kwargs: {find_peaks_kwargs}")

    # --- Execute find_peaks ---
    try:
        peak_indices, properties = find_peaks(y_search, **find_peaks_kwargs)
        num_found = len(peak_indices)
        logging.info(f"SciPy find_peaks identified {num_found} raw peaks.")
        if num_found == 0: return []

        # --- Create Peak Objects ---
        detected_peaks: List[Peak] = []
        for i, idx in enumerate(peak_indices):
            # Ensure index is valid within original arrays
            if not (0 <= idx < len(x_processed)):
                logging.warning(f"Detected peak index {idx} is out of bounds (spectrum length {len(x_processed)}). Skipping.")
                continue

            # Get values at the exact detected index
            detected_wavelength = x_processed[idx]
            detected_intensity = y_processed[idx] # Use processed intensity at peak index

            # Check if detected intensity is valid (could be NaN if input had NaNs)
            if not np.isfinite(detected_intensity):
                 logging.warning(f"Peak at index {idx} has non-finite processed intensity ({detected_intensity}). Skipping.")
                 continue

            # Find corresponding raw intensity (careful about index validity)
            raw_intensity_at_peak = np.nan
            if spectrum.raw_intensity is not None:
                 if idx < len(spectrum.raw_intensity):
                      raw_intensity_at_peak = spectrum.raw_intensity[idx]
                 else: # Should not happen if lengths match, but safety check
                      logging.warning(f"Raw intensity array length mismatch for peak index {idx}.")

            # Create Peak object
            try:
                peak_obj = Peak(
                    detected_index=int(idx),
                    detected_wavelength=float(detected_wavelength),
                    detected_intensity=float(detected_intensity),
                    raw_intensity_at_peak=float(raw_intensity_at_peak) # Will store NaN if raw not found/valid
                )
                detected_peaks.append(peak_obj)
            except ValueError as ve: # Catch errors during Peak creation (e.g., non-finite values)
                logging.warning(f"Could not create Peak object for index {idx}: {ve}")

        # Sort final list of valid peaks by wavelength
        detected_peaks.sort(key=lambda p: p.wavelength_detected)
        logging.info(f"Created {len(detected_peaks)} valid Peak objects.")
        return detected_peaks

    except ImportError: logging.error("SciPy find_peaks failed: Library not found."); return []
    except Exception as e: logging.error(f"Error during SciPy peak detection: {e}", exc_info=True); return []


# --- Placeholder for NIST-Guided Detection ---
# def detect_peaks_nist_guided(...) -> List[Peak]:
#     logging.warning("NIST-guided peak detection not implemented.")
#     return []
