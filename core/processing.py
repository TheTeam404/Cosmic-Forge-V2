# -*- coding: utf-8 -*-
"""
Core signal processing functions for LIBS spectra.

Includes baseline correction, smoothing, wavelet denoising, peak profile functions,
and noise analysis capabilities.
"""

import logging
import numpy as np
from typing import Tuple, Optional, List, Union, Any # Added Any

# --- SciPy Import Handling ---
# Check for SciPy availability, needed for Savitzky-Golay smoothing.
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning(
        "SciPy not found. Savitzky-Golay smoothing will be unavailable. "
        "Install with 'pip install scipy'."
    )
    # Define a placeholder function to raise an error if called
    def savgol_filter(*args, **kwargs):
        raise ImportError(
            "SciPy is required for Savitzky-Golay smoothing but is not installed."
        )

# --- PyWavelets Import Handling ---
# Check for PyWavelets availability, needed for wavelet denoising.
try:
    import pywt # Default import name for PyWavelets
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    logging.warning(
        "PyWavelets not found. Wavelet denoising will be unavailable. "
        "Install with 'pip install PyWavelets'."
    )
    # Define dummy functions from pywt if needed elsewhere, or just rely on the check
    class pywt: # Dummy class to avoid NameErrors if pywt is used directly elsewhere
        @staticmethod
        def wavedec(*args, **kwargs):
             raise ImportError("PyWavelets required but not installed.")
        @staticmethod
        def threshold(*args, **kwargs):
             raise ImportError("PyWavelets required but not installed.")
        @staticmethod
        def waverec(*args, **kwargs):
             raise ImportError("PyWavelets required but not installed.")
        @staticmethod
        def scale2level(*args, **kwargs): # For MAD calculation
             raise ImportError("PyWavelets required but not installed.")

# --- Utility Import ---
# Assuming 'utils.helpers.ensure_odd' exists and ensures an integer is odd.
# If not available, replace with: def ensure_odd(n): return n if n % 2 != 0 else n + 1
try:
    from utils.helpers import ensure_odd
except ImportError:
    logging.warning("utils.helpers.ensure_odd not found. Using basic implementation.")
    def ensure_odd(n: int) -> int:
        """Ensures an integer is odd."""
        n = int(n)
        return n if n % 2 != 0 else n + 1

# --- Constants ---
# Factor to convert Gaussian sigma to FWHM
FWHM_GAUSS_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0)) # Approx 2.35482
# Small epsilon to avoid division by zero or log(0) issues
EPSILON = 1e-9
# Constant for MAD threshold calculation
MAD_NORMALIZATION_CONST = 0.6745 # Approximate normalization for Gaussian noise

# --- Profile Functions ---
# (gaussian, lorentzian, pseudo_voigt functions remain unchanged)
def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    if sigma <= EPSILON:
        logging.warning(f"Gaussian sigma ({sigma:.2e}) too small or non-positive. Returning Inf.")
        return np.full_like(x, np.inf)
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))

def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    if gamma <= EPSILON:
        logging.warning(f"Lorentzian gamma ({gamma:.2e}) too small or non-positive. Returning Inf.")
        return np.full_like(x, np.inf)
    return amplitude * (gamma**2 / ((x - center)**2 + gamma**2))

def pseudo_voigt(
    x: np.ndarray, amplitude: float, center: float, sigma: float, eta: float
) -> np.ndarray:
    if sigma <= EPSILON:
        logging.warning(f"Pseudo-Voigt sigma ({sigma:.2e}) too small or non-positive. Returning Inf.")
        return np.full_like(x, np.inf)
    eta_bounded = np.clip(eta, 0.0, 1.0) # Ensure eta is within [0, 1]
    gamma = max(EPSILON, (sigma * FWHM_GAUSS_FACTOR) / 2.0)
    gauss_part = (1.0 - eta_bounded) * np.exp(-((x - center)**2) / (2 * sigma**2))
    loren_part = eta_bounded * (gamma**2 / ((x - center)**2 + gamma**2))
    return amplitude * (gauss_part + loren_part)


# --- Baseline Correction Algorithms ---
# (baseline_poly, baseline_snip, _interpolate_finite remain unchanged)

def _interpolate_finite(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Internal helper to interpolate NaN/Inf values using linear interpolation."""
    y_interp = y.copy()
    finite_mask = np.isfinite(y_interp)
    all_finite = np.all(finite_mask)

    if not all_finite:
        nan_mask = ~finite_mask
        try:
            finite_indices = np.flatnonzero(finite_mask)
            nan_indices = np.flatnonzero(nan_mask)

            if len(finite_indices) < 2:
                logging.error("Cannot interpolate NaNs: Fewer than 2 finite data points.")
                y_interp[nan_mask] = 0.0
                finite_mask = np.isfinite(y_interp)
                if not np.any(finite_mask):
                   logging.error("Interpolation fallback failed, all data seems non-finite.")
                   return y, np.zeros_like(y, dtype=bool), False
            else:
                y_interp[nan_mask] = np.interp(nan_indices, finite_indices, y_interp[finite_indices])
                logging.warning(f"Interpolated {np.sum(nan_mask)} NaN/Inf values using linear interpolation.")
        except Exception as e:
            logging.error(f"Error during NaN interpolation: {e}. Returning data with NaNs potentially zeroed.", exc_info=True)
            y_interp[~np.isfinite(y_interp)] = 0.0
            finite_mask = np.isfinite(y_interp) # Recompute mask
            if not np.any(finite_mask):
                return y, np.zeros_like(y, dtype=bool), False
        if not np.all(np.isfinite(y_interp)):
            logging.error("Failed to remove all NaNs/Infs through interpolation and fallback. Errors may occur.")
            finite_mask = np.isfinite(y_interp) # Update mask based on current state

    return y_interp, finite_mask, True # Return interpolated, original finite mask, success flag

def baseline_poly(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    order: int = 3,
    percentile: float = 10.0,
    max_iterations: int = 10, # Added for iterative refinement (optional)
    tolerance: float = 0.001, # Added for iterative refinement (optional)
    **kwargs # Accept unused kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    # Function body unchanged...
    if order < 0:
        logging.debug("Polynomial baseline skipped (order < 0).")
        return intensity, np.zeros_like(intensity)
    if not isinstance(wavelengths, np.ndarray) or not isinstance(intensity, np.ndarray):
        raise TypeError("Wavelengths and intensity must be NumPy arrays.")
    if wavelengths.shape != intensity.shape:
        raise ValueError(f"Wavelength ({wavelengths.shape}) and Intensity ({intensity.shape}) arrays must have the same shape.")
    n_points = len(wavelengths)
    min_points_required = order + 1
    if n_points < min_points_required:
        logging.warning(f"Polynomial baseline skipped: Need at least {min_points_required} points for order {order}, but got {n_points}.")
        return intensity, np.zeros_like(intensity)
    if not (0 < percentile <= 100):
        original_percentile = percentile
        percentile = 10.0
        logging.warning(f"Invalid percentile ({original_percentile}). Using default: {percentile}%.")
    intensity_processed, original_finite_mask, interp_ok = _interpolate_finite(intensity)
    if not interp_ok:
        logging.error("Failed to handle NaNs/Infs. Skipping polynomial baseline.")
        return intensity, np.zeros_like(intensity)
    current_intensity = intensity_processed
    baseline = np.zeros_like(intensity_processed)
    for iteration in range(max_iterations if max_iterations > 0 else 1):
        try:
            valid_for_percentile = intensity_processed[original_finite_mask] if iteration == 0 else current_intensity[original_finite_mask]
            if len(valid_for_percentile) == 0:
                 logging.error("No finite points available to calculate percentile. Skipping baseline.")
                 return intensity, np.zeros_like(intensity)
            threshold = np.percentile(valid_for_percentile, percentile)
            mask = (current_intensity <= threshold) & original_finite_mask # Always respect original NaNs
            n_masked_pts = np.sum(mask)
            if n_masked_pts < min_points_required:
                logging.warning( f"Iteration {iteration+1}: Only {n_masked_pts} points below {percentile:.1f}% " f"percentile (threshold={threshold:.2f}). Need {min_points_required}. " "Using lowest intensity points as fallback.")
                finite_indices = np.where(original_finite_mask)[0]
                if len(finite_indices) < min_points_required:
                    logging.error(f"Fallback failed: Only {len(finite_indices)} finite points available. " "Cannot fit polynomial. Skipping baseline.")
                    return intensity, np.zeros_like(intensity)
                sorted_finite_indices = finite_indices[np.argsort(intensity_processed[finite_indices])]
                fallback_indices = sorted_finite_indices[:min_points_required]
                mask = np.zeros_like(intensity_processed, dtype=bool)
                mask[fallback_indices] = True
                n_masked_pts = np.sum(mask) # Should be min_points_required
                logging.debug(f"Fallback selected {n_masked_pts} lowest finite points.")
            x_masked = wavelengths[mask]
            y_masked = intensity_processed[mask] # Use the imputed intensity for fitting
            if not (np.all(np.isfinite(x_masked)) and np.all(np.isfinite(y_masked))):
                 logging.error("Non-finite values detected in data selected for polyfit after masking/fallback. Skipping.")
                 return intensity, np.zeros_like(intensity)
            if len(x_masked) < min_points_required: # Should be caught above, but double check
                logging.error("Insufficient points for polyfit after final checks. Skipping.")
                return intensity, np.zeros_like(intensity)
            coeffs = np.polyfit(x_masked, y_masked, order)
            new_baseline = np.polyval(coeffs, wavelengths)
            if max_iterations > 1:
                diff = np.abs(baseline - new_baseline)
                if np.all(diff[original_finite_mask] < tolerance):
                    logging.info(f"Polynomial baseline converged after {iteration + 1} iterations.")
                    baseline = new_baseline
                    break
            baseline = new_baseline
            current_intensity = intensity_processed - baseline
        except (np.linalg.LinAlgError, ValueError) as e:
            logging.error(f"Polynomial baseline fitting failed: {e}", exc_info=True)
            return intensity, np.zeros_like(intensity)
        except Exception as e:
            logging.error(f"Unexpected error during polynomial baseline: {e}", exc_info=True)
            return intensity, np.zeros_like(intensity)
    corrected_intensity = intensity_processed - baseline
    corrected_intensity[~original_finite_mask] = intensity[~original_finite_mask]
    baseline[~original_finite_mask] = np.nan # Baseline is undefined where original was NaN
    logging.info(f"Polynomial baseline applied (order={order}, percentile={percentile:.1f}%, iterations={iteration+1 if max_iterations > 0 else 1}).")
    return corrected_intensity, baseline

def baseline_snip(
    wavelengths: np.ndarray, # Kept for API consistency, but not used by SNIP
    intensity: np.ndarray,
    max_iterations: int = 100,
    increasing_window: bool = True,
    **kwargs # Accept unused kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    # Function body unchanged...
    if not isinstance(intensity, np.ndarray): raise TypeError("Intensity must be a NumPy array.")
    n_points = len(intensity)
    if n_points < 3: logging.warning(f"SNIP requires >= 3 points, got {n_points}. Skipping."); return intensity, np.zeros_like(intensity)
    if max_iterations < 1: logging.warning(f"SNIP max_iterations ({max_iterations}) must be >= 1. Setting to 1."); max_iterations = 1
    intensity_processed, original_finite_mask, interp_ok = _interpolate_finite(intensity)
    if not interp_ok: logging.error("Failed to handle NaNs/Infs. Skipping SNIP baseline."); return intensity, np.zeros_like(intensity)
    baseline = intensity_processed.copy()
    logging.info(f"Starting SNIP baseline estimation (max_iter={max_iterations}, increasing_window={increasing_window}).")
    try:
        for k in range(1, max_iterations + 1):
            window_size = k if increasing_window else max_iterations
            if window_size >= n_points // 2: logging.debug(f"SNIP iteration {k}: Window size ({window_size}) reached limit. Stopping."); break
            prev_baseline = baseline.copy() # Store previous iteration's baseline
            avg_neighbors = (prev_baseline[:-2 * window_size] + prev_baseline[2 * window_size:]) / 2.0
            points_to_clip = prev_baseline[window_size:-window_size] > avg_neighbors
            baseline[window_size:-window_size][points_to_clip] = avg_neighbors[points_to_clip]
        baseline = np.minimum(baseline, intensity_processed)
        corrected_intensity = intensity_processed - baseline
        corrected_intensity[~original_finite_mask] = intensity[~original_finite_mask]
        baseline[~original_finite_mask] = np.nan # Baseline undefined where original was NaN
        logging.info("SNIP baseline estimation complete.")
        return corrected_intensity, baseline
    except Exception as e:
        logging.error(f"Error during SNIP baseline correction: {e}", exc_info=True)
        return intensity, np.zeros_like(intensity)


# --- Smoothing Algorithms ---
# (smooth_savitzky_golay remains unchanged)
def smooth_savitzky_golay(
    intensity: np.ndarray,
    smoothing_window: int = 11,
    smoothing_polyorder: int = 3,
    **kwargs # Accept unused kwargs
) -> np.ndarray:
    # Function body unchanged...
    if not SCIPY_AVAILABLE: logging.error("Cannot smooth: SciPy library is unavailable."); return intensity
    if not isinstance(intensity, np.ndarray): raise TypeError("Intensity must be a NumPy array.")
    n_points = len(intensity)
    if n_points < 3: logging.warning(f"Data too short ({n_points} points) for Sav-Gol smoothing. Returning original."); return intensity
    try:
        wl = ensure_odd(int(smoothing_window));
        if wl < 3: logging.warning(f"Sav-Gol window ({smoothing_window}) too small. Adjusting to 3."); wl = 3
        po = int(smoothing_polyorder);
        if po < 0: logging.warning(f"Sav-Gol polyorder ({smoothing_polyorder}) cannot be negative. Adjusting to 0 (moving average)."); po = 0
        if wl > n_points: old_wl = wl; wl = n_points if n_points % 2 != 0 else n_points - 1; wl = max(3, wl); logging.warning(f"Sav-Gol window ({old_wl}) > data length ({n_points}). Adjusting window to {wl}.")
        if po >= wl: old_po = po; po = max(0, wl - 2); logging.warning(f"Sav-Gol polyorder ({old_po}) >= adjusted window ({wl}). Adjusting polyorder to {po}.")
    except (ValueError, TypeError) as e: logging.error(f"Invalid Sav-Gol parameters (window={smoothing_window}, order={smoothing_polyorder}): {e}. Returning original."); return intensity
    intensity_processed, original_finite_mask, interp_ok = _interpolate_finite(intensity)
    if not interp_ok: logging.error("Failed to handle NaNs/Infs prior to smoothing. Returning original."); return intensity
    elif not np.all(original_finite_mask): logging.debug("Applied linear interpolation to handle NaNs/Infs before smoothing.")
    try:
        if wl > len(intensity_processed) or po >= wl: logging.error(f"Internal Error: Adjusted Sav-Gol parameters (wl={wl}, po={po}) still invalid after processing. Skipping."); return intensity
        smoothed_intensity = savgol_filter(intensity_processed, window_length=wl, polyorder=po)
        smoothed_intensity[~original_finite_mask] = intensity[~original_finite_mask]
        logging.info(f"Applied Savitzky-Golay smoothing (window={wl}, order={po}).")
        return smoothed_intensity
    except ValueError as e: logging.error(f"Error applying Savitzky-Golay filter: {e}. Returning original.", exc_info=True); return intensity
    except Exception as e: logging.error(f"Unexpected error during Savitzky-Golay smoothing: {e}", exc_info=True); return intensity

# --- Denoising Algorithms ---

def denoise_wavelet(
    intensity: np.ndarray,
    wavelet_type: str = 'db8',
    level: Optional[int] = None,
    mode: str = 'soft',
    threshold_sigma_factor: float = 3.0,
    **kwargs # Catch unused args
) -> np.ndarray:
    """
    Applies wavelet denoising to the intensity data.

    Uses Median Absolute Deviation (MAD) to estimate noise level and applies
    soft or hard thresholding to detail coefficients.

    Args:
        intensity: Intensity array to denoise. Handles NaNs/Infs by
                   replacing them with zeros (may affect results).
        wavelet_type: Type of wavelet to use (e.g., 'db4', 'sym8', 'coif5').
        level: Decomposition level. If None, automatically estimated by pywt.
        mode: Thresholding mode ('soft' or 'hard').
        threshold_sigma_factor: Factor to multiply MAD by for threshold value.
        kwargs: Accepts extra arguments for compatibility.

    Returns:
        The denoised intensity array. Returns the original array if denoising
        cannot be performed (e.g., PyWavelets unavailable, errors).
    """
    if not PYWAVELETS_AVAILABLE:
        logging.error("Cannot denoise: PyWavelets library is unavailable.")
        return intensity # Return original data

    if not isinstance(intensity, np.ndarray):
        raise TypeError("Intensity must be a NumPy array.")

    n_points = len(intensity)
    if n_points < 2:
        logging.warning(f"Data too short ({n_points} points) for wavelet denoising. Returning original.")
        return intensity

    # --- Handle NaNs/Infs (Simple Approach: Replace with zero) ---
    # More sophisticated handling (interpolation) might be better but adds complexity.
    intensity_processed = intensity.copy()
    original_finite_mask = np.isfinite(intensity_processed)
    if not np.all(original_finite_mask):
         logging.warning("Wavelet denoising input contains NaNs/Infs. Replacing with 0.0. Results may be affected.")
         intensity_processed[~original_finite_mask] = 0.0
         # Check if replacement caused issues
         if not np.all(np.isfinite(intensity_processed)):
              logging.error("Failed to handle non-finite values even after replacement. Skipping wavelet denoising.")
              return intensity


    try:
        # --- Parameter Validation ---
        # Check if wavelet exists
        try:
            wavelet = pywt.Wavelet(wavelet_type)
        except ValueError:
            logging.error(f"Invalid wavelet type '{wavelet_type}'. Available families: {pywt.families()}. Skipping denoising.")
            return intensity

        # Determine decomposition level
        if level is None:
            # Default level calculation (can be adjusted)
            level = pywt.dwt_max_level(n_points, wavelet.dec_len)
            logging.debug(f"Wavelet level automatically determined: {level}")
        else:
            level = int(level) # Ensure integer
            max_level = pywt.dwt_max_level(n_points, wavelet.dec_len)
            if level < 1:
                logging.warning(f"Wavelet level ({level}) cannot be less than 1. Using 1.")
                level = 1
            elif level > max_level:
                 logging.warning(f"Wavelet level ({level}) exceeds maximum possible level ({max_level}) for data length {n_points} and wavelet '{wavelet_type}'. Using max level.")
                 level = max_level

        # Validate mode
        if mode not in ['soft', 'hard']:
            logging.warning(f"Invalid thresholding mode '{mode}'. Using default 'soft'.")
            mode = 'soft'

        # Validate sigma factor
        if threshold_sigma_factor <= 0:
            logging.warning(f"Threshold sigma factor ({threshold_sigma_factor}) must be positive. Using default 3.0.")
            threshold_sigma_factor = 3.0


        # --- Wavelet Decomposition ---
        # Use padding mode 'symmetric' or 'reflect' to handle boundaries gracefully
        coeffs = pywt.wavedec(intensity_processed, wavelet, level=level, mode='symmetric')
        # Example coeffs structure: [cA_n, cD_n, cD_n-1, ..., cD_1]

        # --- Noise Estimation (MAD from finest detail coefficients cD1) ---
        # Extract detail coefficients at the finest level (cD_1)
        coeffs_cd1 = coeffs[-1]
        if len(coeffs_cd1) == 0:
             logging.warning("No detail coefficients found at level 1. Cannot estimate noise. Skipping denoising.")
             return intensity

        # Calculate Median Absolute Deviation (MAD)
        # Ensure calculation is robust against edge cases or all-zero coeffs
        median_cd1 = np.median(coeffs_cd1)
        mad = np.median(np.abs(coeffs_cd1 - median_cd1))
        if mad < EPSILON:
            # If MAD is effectively zero (e.g., constant signal or very sparse coeffs),
            # noise estimation is unreliable. Skip thresholding or use alternative.
            logging.warning(f"MAD of detail coefficients is near zero ({mad:.2e}). Noise estimation may be unreliable. Thresholding might be skipped or ineffective.")
            # Option 1: Skip thresholding (effectively returns smoothed signal) -> Set threshold very high
            # Option 2: Use a fallback small noise estimate?
            sigma = EPSILON # Use a tiny sigma, threshold will be small
        else:
            # Estimate noise standard deviation (sigma) using MAD
            sigma = mad / MAD_NORMALIZATION_CONST


        # --- Threshold Calculation ---
        # Universal Threshold: sigma * sqrt(2 * log(N)), where N is signal length
        # Using a simpler factor-based threshold here as specified in config
        threshold_value = threshold_sigma_factor * sigma
        logging.info(f"Applying Wavelet Denoising: Wavelet='{wavelet_type}', Level={level}, Mode='{mode}', "
                     f"Est. Noise (Sigma)≈{sigma:.3f}, Threshold={threshold_value:.3f}")


        # --- Thresholding Detail Coefficients ---
        # Apply thresholding to detail coefficients (cD_n to cD_1)
        # Do not threshold the approximation coefficients (cA_n)
        coeffs_thresholded = [coeffs[0]] # Keep approximation coeffs unchanged
        for i in range(1, len(coeffs)): # Iterate through detail coefficient arrays (cD_n to cD_1)
            coeffs_thresholded.append(pywt.threshold(coeffs[i], value=threshold_value, mode=mode))


        # --- Wavelet Reconstruction ---
        denoised_intensity = pywt.waverec(coeffs_thresholded, wavelet, mode='symmetric')

        # Ensure output length matches input length (padding might cause mismatch)
        if len(denoised_intensity) != n_points:
             logging.warning(f"Wavelet reconstruction length ({len(denoised_intensity)}) differs from input ({n_points}). Truncating/padding.")
             # Simple truncation/padding (might not be ideal for all boundary modes)
             if len(denoised_intensity) > n_points:
                  denoised_intensity = denoised_intensity[:n_points]
             else: # Pad with zeros (or edge values?) if shorter
                  pad_width = n_points - len(denoised_intensity)
                  # Pad symmetrically if possible, default to padding end
                  denoised_intensity = np.pad(denoised_intensity, (0, pad_width), mode='constant')

        # Restore original NaNs/Infs in the output
        denoised_intensity[~original_finite_mask] = intensity[~original_finite_mask]

        logging.info("Wavelet denoising applied successfully.")
        return denoised_intensity

    except ImportError: # Should be caught by PYWAVELETS_AVAILABLE check, but safety net
        logging.error("PyWavelets not available during denoise execution.")
        return intensity
    except Exception as e:
        logging.error(f"Error during wavelet denoising: {e}", exc_info=True)
        return intensity # Return original data on failure

# --- Noise Analysis ---
# (analyze_noise function remains unchanged)
def analyze_noise(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    signal_free_regions: Optional[List[Tuple[float, float]]] = None
) -> Tuple[Optional[float], Optional[List[Tuple[float, float]]]]:
    # Function body unchanged...
    if not isinstance(wavelengths, np.ndarray) or not isinstance(intensity, np.ndarray): raise TypeError("Wavelengths and intensity must be NumPy arrays.")
    if wavelengths.shape != intensity.shape: raise ValueError("Wavelength and Intensity arrays must have the same shape.")
    if signal_free_regions is None or not signal_free_regions: logging.warning("No signal-free regions provided for noise analysis. Cannot estimate noise."); return None, signal_free_regions
    noise_segments = []; valid_regions_used = []
    for region in signal_free_regions:
        try:
            start_wl, end_wl = map(float, region)
            if start_wl >= end_wl: logging.warning(f"Skipping invalid noise region {region}: start >= end."); continue
            region_mask = (wavelengths >= start_wl) & (wavelengths <= end_wl); intensity_in_region = intensity[region_mask]
            if intensity_in_region.size == 0: logging.warning(f"Skipping noise region {region}: No data points found in this range."); continue
            finite_intensity_in_region = intensity_in_region[np.isfinite(intensity_in_region)]
            if finite_intensity_in_region.size < 2: logging.warning(f"Skipping noise region {region}: Fewer than 2 finite data points for std dev calculation."); continue
            noise_segments.append(finite_intensity_in_region); valid_regions_used.append(region)
        except (TypeError, ValueError) as e: logging.warning(f"Skipping invalid noise region {region}: Error parsing limits ({e})."); continue
        except Exception as e: logging.error(f"Unexpected error processing noise region {region}: {e}", exc_info=True); continue
    if not noise_segments: logging.error("Could not find any valid data points in the specified signal-free regions."); return None, valid_regions_used
    try:
        all_noise_points = np.concatenate(noise_segments)
        if all_noise_points.size < 2: logging.error("Insufficient total finite data points (< 2) across all valid noise regions."); return None, valid_regions_used
        noise_std_dev = np.std(all_noise_points)
        logging.info(f"Estimated noise standard deviation: {noise_std_dev:.4f} (from {len(valid_regions_used)} regions, {all_noise_points.size} points).")
        return noise_std_dev, valid_regions_used
    except Exception as e: logging.error(f"Failed to calculate final noise standard deviation: {e}", exc_info=True); return None, valid_regions_used


# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    # Generate sample data (Sine wave + Gaussian peak + Noise + Baseline)
    np.random.seed(42)
    wavelengths = np.linspace(200, 800, 1024) # Use power of 2 length for easier wavelet levels
    baseline_true = 50 + 0.1 * (wavelengths - 200) # Linear baseline
    signal = (
        gaussian(wavelengths, amplitude=200, center=350, sigma=5) +
        gaussian(wavelengths, amplitude=150, center=600, sigma=10) +
        lorentzian(wavelengths, amplitude=80, center=700, gamma=8)
    )
    noise = np.random.normal(0, 10, size=wavelengths.shape) # Noise std dev = 10
    intensity_raw = signal + baseline_true + noise

    # --- Test Baseline Correction ---
    print("\n--- Testing Polynomial Baseline ---")
    intensity_poly_corrected, baseline_poly_calc = baseline_poly(
        wavelengths, intensity_raw, order=2, percentile=5.0, max_iterations=1
    )

    # --- Test Denoising ---
    print("\n--- Testing Wavelet Denoising ---")
    if PYWAVELETS_AVAILABLE:
        # Apply denoising AFTER baseline correction (common practice)
        denoised_intensity = denoise_wavelet(
            intensity_poly_corrected, # Denoise the baseline-corrected signal
            wavelet_type='db8',
            level=4,
            mode='soft',
            threshold_sigma_factor=3.0
        )
        print(f"Original mean (post-baseline): {np.nanmean(intensity_poly_corrected):.2f}, Denoised mean: {np.nanmean(denoised_intensity):.2f}")
    else:
        print("Skipping wavelet denoising test: PyWavelets not installed.")
        denoised_intensity = intensity_poly_corrected # Use non-denoised if lib missing

    # --- Test Smoothing ---
    print("\n--- Testing Savitzky-Golay Smoothing ---")
    # Smooth the denoised signal
    smoothed_intensity = smooth_savitzky_golay(
        denoised_intensity, smoothing_window=15, smoothing_polyorder=3
    )
    print(f"Denoised mean: {np.nanmean(denoised_intensity):.2f}, Smoothed mean: {np.nanmean(smoothed_intensity):.2f}")


    # --- Optional: Plotting results ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 10))

        plt.subplot(3, 1, 1)
        plt.plot(wavelengths, intensity_raw, label='Raw Intensity', alpha=0.7, color='gray')
        plt.plot(wavelengths, signal + baseline_true, label='Signal + True Baseline', alpha=0.8, color='blue')
        plt.title('Original Data vs Signal')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(wavelengths, intensity_poly_corrected, 'r-', label='Poly Corrected', alpha=0.8)
        if PYWAVELETS_AVAILABLE:
            plt.plot(wavelengths, denoised_intensity, 'g-', label='Wavelet Denoised (db8 L4)', alpha=0.9)
        plt.plot(wavelengths, signal, 'k--', label='True Signal') # Original signal without baseline/noise
        plt.title('Baseline Corrected & Denoised Data')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        if PYWAVELETS_AVAILABLE:
             plt.plot(wavelengths, denoised_intensity, label=f'Denoised (Wavelet)', color='g', alpha=0.7)
        else:
             plt.plot(wavelengths, intensity_poly_corrected, label=f'Baseline Corrected', color='r', alpha=0.7)
        plt.plot(wavelengths, smoothed_intensity, label=f'Smoothed (SavGol W=15, P=3)', color='purple', linewidth=1.5)
        plt.title('Denoised vs Smoothed Data')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping plots.")
    except Exception as e:
        print(f"\nError during plotting: {e}")