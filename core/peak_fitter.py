"""
Peak fitting functions for individual peaks in LIBS spectra.
Includes profile definitions, fitting logic using scipy.optimize.curve_fit,
and model selection based on AIC/BIC.
"""

import logging
import numpy as np
import pandas as pd  # Kept import as it might be used elsewhere, but not in this snippet
import traceback
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

# --- SciPy Imports ---
try:
    from scipy.optimize import curve_fit
    from scipy.special import voigt_profile # Keep if true Voigt might be added later
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Provide placeholder functions that raise ImportError if called
    def curve_fit(*args, **kwargs): raise ImportError("SciPy library not found. Peak fitting functionality unavailable. Install with 'pip install scipy'.")
    def voigt_profile(*args, **kwargs): raise ImportError("SciPy library not found. Peak fitting functionality unavailable. Install with 'pip install scipy'.")
    logging.error("SciPy library not found. Peak fitting functionality unavailable. Install with 'pip install scipy'.")


# --- Local Imports (adjust path as needed) ---
# Assume data_models contains Spectrum and FitResult (defined below if needed)
# Assume utils.helpers contains get_project_root (not used here, but kept for context)
# from .data_models import Spectrum, FitResult # Using local definitions for now
# from utils.helpers import get_project_root


# --- Constants and Configuration ---
FWHM_GAUSS_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.35482
MIN_ROI_POINTS = 5  # Minimum number of data points required in ROI for fitting
MIN_WIDTH_NM = 1e-6 # Minimum allowed width parameter during fitting to avoid numerical issues
DEFAULT_MAX_ITER = 2000 # Default max iterations for curve_fit

# --- Enums for Configuration ---
class ProfileType(Enum):
    GAUSSIAN = "Gaussian"
    LORENTZIAN = "Lorentzian"
    PSEUDO_VOIGT = "PseudoVoigt"
    # VOIGT = "Voigt" # Add if implementing true Voigt

class BaselineMode(Enum):
    NONE = "None"
    LOCAL_LINEAR = "LocalLinear" # Fit linear baseline to edges of ROI
    SLOPE = "Slope"          # Simple linear baseline between ROI endpoints

class ModelSelectionCriterion(Enum):
    AIC = "AIC"
    BIC = "BIC"


# --- Data Models (Simplified - replace with your actual data_models if different) ---
@dataclass
class Spectrum:
    wavelengths: np.ndarray
    intensity: np.ndarray # Assuming this is the *raw* intensity
    # Add other relevant spectrum metadata if needed

@dataclass
class FitResult:
    # Fields WITHOUT default values MUST come first
    profile_type: ProfileType
    amplitude: float
    center: float
    width: float # Sigma for Gaussian/PV, Gamma (HWHM) for Lorentzian
    params: np.ndarray = field(repr=False) # Raw fitted parameters (No explicit default)

    # Fields WITH default values follow
    mixing_param_eta: Optional[float] = None # Only for PseudoVoigt
    params_covariance: Optional[np.ndarray] = field(default=None, repr=False)
    r_squared: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    success: bool = False
    message: str = "" # Optional message (e.g., reason for failure)
    roi_wavelengths: Optional[np.ndarray] = field(default=None, repr=False)
    roi_intensity_corrected: Optional[np.ndarray] = field(default=None, repr=False)
    fitted_curve: Optional[np.ndarray] = field(default=None, repr=False)


# --- Profile Functions ---

def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Gaussian profile function (amplitude, center, standard deviation sigma)."""
    # Bounds in curve_fit should prevent sigma <= 0
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))

def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    """Lorentzian profile function (amplitude, center, HWHM gamma)."""
    # Bounds in curve_fit should prevent gamma <= 0
    return amplitude * (gamma**2 / ((x - center)**2 + gamma**2))

def pseudo_voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, eta: float) -> np.ndarray:
    """
    Pseudo-Voigt profile using Gaussian sigma and a mixing parameter eta (0=Gauss, 1=Lorentz).
    FWHM is derived from sigma and eta.
    """
    # Bounds will handle sigma > 0 and 0 <= eta <= 1
    # Derive Lorentzian HWHM gamma based on Gaussian sigma's FWHM for consistent width scaling
    # This specific definition assumes FWHM(PV) approx FWHM(Gauss) used for sigma
    gamma = (sigma * FWHM_GAUSS_FACTOR) / 2.0
    gauss_part = (1.0 - eta) * np.exp(-((x - center)**2) / (2 * sigma**2))
    loren_part = eta * (gamma**2 / ((x - center)**2 + gamma**2))
    return amplitude * (gauss_part + loren_part)

# Dictionary mapping ProfileType Enum to function and parameter count
PROFILE_FUNCTIONS: Dict[ProfileType, Tuple[Callable, int]] = {
    ProfileType.GAUSSIAN: (gaussian, 3),
    ProfileType.LORENTZIAN: (lorentzian, 3),
    ProfileType.PSEUDO_VOIGT: (pseudo_voigt, 4),
}

# --- Goodness-of-Fit Calculations ---

def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    if y_true is None or y_pred is None or len(y_true) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot < 1e-12: # Handle constant data case
        return 1.0 if ss_res < 1e-12 else 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    return max(0.0, r2) # Ensure R2 is not negative for very poor fits

def calculate_aic_bic(n: int, k: int, rss: float) -> Tuple[float, float]:
    """
    Calculates AIC and BIC using the residual sum of squares (RSS).
    Assumes errors are normally distributed.
    """
    if n <= k or rss < 0 or not np.isfinite(rss) or n == 0:
        return np.inf, np.inf # Not computable
    if rss < 1e-12: # Very good fit, avoid log(0)
        rss = 1e-12 # Use a tiny floor value

    try:
        # Formula based on likelihood for normally distributed errors with estimated variance
        log_likelihood_term = n * np.log(rss / n)
        aic = 2 * k + log_likelihood_term
        bic = k * np.log(n) + log_likelihood_term
        # Add correction for small sample sizes (AICc) if desired:
        # aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if (n - k - 1) > 0 else np.inf
        return aic, bic
    except (ValueError, FloatingPointError): # Catch potential math errors
        logging.warning(f"AIC/BIC calculation failed (n={n}, k={k}, rss={rss})", exc_info=True)
        return np.inf, np.inf


# --- Helper Functions for Fitting ---

def _determine_roi(wavelengths: np.ndarray,
                   intensity: np.ndarray,
                   peak_index: int,
                   roi_factor: float,
                   min_roi_width_nm: float,
                   explicit_roi_wl: Optional[Tuple[float, float]] = None
                   ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Determines the Region of Interest (ROI) indices and wavelengths.

    Returns:
        Tuple containing (roi_indices, wl_roi, int_roi, center_wl_guess) or None if ROI is invalid.
    """
    center_wl_guess = wavelengths[peak_index]
    n_points = len(wavelengths)

    if explicit_roi_wl and len(explicit_roi_wl) == 2 and explicit_roi_wl[0] < explicit_roi_wl[1]:
        # Use explicitly provided ROI wavelengths
        roi_start_wl, roi_end_wl = explicit_roi_wl
        roi_indices = np.where((wavelengths >= roi_start_wl) & (wavelengths <= roi_end_wl))[0]
        logging.debug(f"Peak @ {center_wl_guess:.3f}nm: Using provided ROI [{roi_start_wl:.3f}-{roi_end_wl:.3f}]nm")
    else:
        # Automatic ROI calculation based on estimated FWHM
        try:
            # Estimate amplitude and baseline near the peak for FWHM calc
            local_min_idx = max(0, peak_index - 10)
            local_max_idx = min(n_points, peak_index + 11)
            local_intensity_range = intensity[local_min_idx:local_max_idx]

            # Check for NaNs in local range
            if np.all(np.isnan(local_intensity_range)):
                 logging.warning(f"Peak @ {center_wl_guess:.3f}nm: Cannot estimate amplitude/baseline near peak due to NaNs.")
                 # Fallback: use a fixed number of points around the peak index? Or fail?
                 # Let's try a fixed point range as a fallback for ROI determination
                 roi_half_points = 10 # Arbitrary fallback width in points
                 roi_start_idx = max(0, peak_index - roi_half_points)
                 roi_end_idx = min(n_points - 1, peak_index + roi_half_points)
                 roi_indices = np.arange(roi_start_idx, roi_end_idx + 1)

            else:
                amplitude_guess = intensity[peak_index] - np.nanmin(local_intensity_range)
                amplitude_guess = max(amplitude_guess, 1e-9) # Ensure positive amplitude guess
                half_max_val = np.nanmin(local_intensity_range) + amplitude_guess / 2.0

                # Find indices where intensity crosses half max
                left_indices = np.where(intensity[:peak_index+1] < half_max_val)[0]
                idx_left = left_indices[-1] if len(left_indices) > 0 else max(0, peak_index - 5) # Fallback: 5 points left

                right_indices = np.where(intensity[peak_index:] < half_max_val)[0]
                # Need to offset right_indices because it starts from peak_index
                idx_right = peak_index + right_indices[0] if len(right_indices) > 0 else min(n_points - 1, peak_index + 5) # Fallback: 5 points right

                fwhm_guess_nm = abs(wavelengths[idx_right] - wavelengths[idx_left])
                # Ensure FWHM guess is reasonable (at least pixel width)
                min_pixel_width = np.min(np.diff(wavelengths)) if n_points > 1 else 1e-6
                fwhm_guess_nm = max(fwhm_guess_nm, min_pixel_width)

                roi_half_width_nm = max((roi_factor / 2.0) * fwhm_guess_nm, min_roi_width_nm / 2.0)
                roi_start_wl = center_wl_guess - roi_half_width_nm
                roi_end_wl = center_wl_guess + roi_half_width_nm
                roi_indices = np.where((wavelengths >= roi_start_wl) & (wavelengths <= roi_end_wl))[0]

        except Exception as e:
            logging.error(f"Error calculating auto ROI for peak @ {center_wl_guess:.3f}nm: {e}", exc_info=True)
            return None

    # Validate and potentially expand ROI size
    if len(roi_indices) < MIN_ROI_POINTS:
        logging.warning(f"Peak @ {center_wl_guess:.3f}nm: Initial ROI too narrow ({len(roi_indices)} pts). Attempting expansion.")
        needed = MIN_ROI_POINTS - len(roi_indices)
        expand_left = needed // 2
        expand_right = needed - expand_left

        start_idx = roi_indices[0] if len(roi_indices) > 0 else peak_index
        end_idx = roi_indices[-1] if len(roi_indices) > 0 else peak_index

        final_start_idx = max(0, start_idx - expand_left)
        final_end_idx = min(n_points - 1, end_idx + expand_right)
        roi_indices = np.arange(final_start_idx, final_end_idx + 1)

        if len(roi_indices) < MIN_ROI_POINTS:
            logging.error(f"Peak @ {center_wl_guess:.3f}nm: ROI still too narrow ({len(roi_indices)} pts) after expansion. Cannot fit.")
            return None

    wl_roi = wavelengths[roi_indices]
    int_roi = intensity[roi_indices] # Use original intensity for baseline correction

    # Final check for sufficient valid data points in ROI
    if np.sum(np.isfinite(int_roi)) < MIN_ROI_POINTS:
        logging.error(f"Peak @ {center_wl_guess:.3f}nm: Not enough finite data points ({np.sum(np.isfinite(int_roi))}) in the final ROI. Cannot fit.")
        return None

    logging.debug(f"Peak @ {center_wl_guess:.3f}nm: Final ROI [{wl_roi[0]:.3f}-{wl_roi[-1]:.3f}]nm ({len(wl_roi)} pts)")
    return roi_indices, wl_roi, int_roi, center_wl_guess


def _apply_local_baseline(wl_roi: np.ndarray,
                          int_roi: np.ndarray,
                          baseline_mode: BaselineMode
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Applies local baseline correction within the ROI."""
    local_baseline = np.zeros_like(int_roi)
    int_roi_corrected = int_roi.copy() # Start with original intensity

    if baseline_mode == BaselineMode.LOCAL_LINEAR and len(wl_roi) >= 4:
        try:
            # Use ~10% of points at each edge, minimum 2 points
            num_edge = max(2, int(len(wl_roi) * 0.1))
            if len(wl_roi) >= 2 * num_edge: # Ensure enough points for edges
                edge_idx = np.concatenate((np.arange(num_edge), np.arange(len(wl_roi) - num_edge, len(wl_roi))))
                edge_wl, edge_int = wl_roi[edge_idx], int_roi[edge_idx]
                # Filter out NaNs before polyfit
                finite_mask = np.isfinite(edge_wl) & np.isfinite(edge_int)
                if np.sum(finite_mask) >= 2: # Need at least 2 points for linear fit
                    coeffs = np.polyfit(edge_wl[finite_mask], edge_int[finite_mask], 1)
                    local_baseline = np.polyval(coeffs, wl_roi)
                    int_roi_corrected = int_roi - local_baseline
                else:
                     logging.warning(f"LocalLinear baseline failed: Not enough finite edge points in ROI [{wl_roi[0]:.3f}-{wl_roi[-1]:.3f}]nm. Using None.")
                     baseline_mode = BaselineMode.NONE # Fallback
            else:
                 logging.warning(f"LocalLinear baseline failed: ROI too small for edge calculation in ROI [{wl_roi[0]:.3f}-{wl_roi[-1]:.3f}]nm. Using None.")
                 baseline_mode = BaselineMode.NONE # Fallback

        except (np.linalg.LinAlgError, ValueError) as e:
            logging.warning(f"LocalLinear baseline polyfit failed for ROI [{wl_roi[0]:.3f}-{wl_roi[-1]:.3f}]nm: {e}. Using None.")
            baseline_mode = BaselineMode.NONE # Fallback

    elif baseline_mode == BaselineMode.SLOPE and len(wl_roi) >= 2:
        wl_diff = wl_roi[-1] - wl_roi[0]
        if abs(wl_diff) > 1e-9 and np.isfinite(int_roi[0]) and np.isfinite(int_roi[-1]):
            slope = (int_roi[-1] - int_roi[0]) / wl_diff
            local_baseline = int_roi[0] + slope * (wl_roi - wl_roi[0])
            int_roi_corrected = int_roi - local_baseline
        else:
             logging.warning(f"Slope baseline failed (zero width or NaN endpoints) for ROI [{wl_roi[0]:.3f}-{wl_roi[-1]:.3f}]nm. Using None.")
             baseline_mode = BaselineMode.NONE # Fallback

    # Ensure baseline is not applied if mode ended up as NONE
    if baseline_mode == BaselineMode.NONE:
        local_baseline = np.zeros_like(int_roi)
        int_roi_corrected = int_roi # Ensure we use original if baseline failed

    return int_roi_corrected, local_baseline


def _get_initial_guesses_and_bounds(profile_type: ProfileType,
                                    wl_roi: np.ndarray,
                                    int_roi_corrected: np.ndarray,
                                    center_wl_guess: float
                                   ) -> Tuple[List[float], Tuple[List[float], List[float]]]:
    """Estimates initial parameters and bounds for a given profile type."""

    # Robust amplitude guess after baseline correction
    try:
        amp_guess = np.nanmax(int_roi_corrected)
        if not np.isfinite(amp_guess) or amp_guess <= 0:
             # Fallback: estimate from range if max is non-positive or NaN
             min_val = np.nanmin(int_roi_corrected)
             max_val = np.nanmax(int_roi_corrected)
             if np.isfinite(min_val) and np.isfinite(max_val):
                 amp_guess = max(max_val - min_val, 1e-9) # Use range
             else: # If still invalid, use a small default
                 amp_guess = 1e-6
        amp_guess = max(amp_guess, 1e-9) # Ensure positive
    except ValueError: # Handle case where int_roi_corrected might be all NaN
        amp_guess = 1e-6 # Small default amplitude

    # Robust width guess (sigma/gamma)
    try:
        half_max_val = np.nanmin(int_roi_corrected) + amp_guess / 2.0
        above_half_max = np.where(int_roi_corrected >= half_max_val)[0]
        if len(above_half_max) > 1:
            fwhm_guess_nm = abs(wl_roi[above_half_max[-1]] - wl_roi[above_half_max[0]])
        else: # Fallback if FWHM cannot be estimated
            fwhm_guess_nm = abs(wl_roi[-1] - wl_roi[0]) / 3.0 # Guess ~1/3 of ROI width

        # Ensure minimum pixel width considered
        min_pixel_width = np.min(np.diff(wl_roi)) if len(wl_roi) > 1 else MIN_WIDTH_NM
        fwhm_guess_nm = max(fwhm_guess_nm, min_pixel_width)

    except Exception: # Broad catch if index/value errors occur with NaNs
        fwhm_guess_nm = abs(wl_roi[-1] - wl_roi[0]) / 3.0 # Fallback width
        fwhm_guess_nm = max(fwhm_guess_nm, MIN_WIDTH_NM)


    # Sigma guess (for Gaussian/PV) from FWHM
    sigma_guess = fwhm_guess_nm / FWHM_GAUSS_FACTOR
    sigma_guess = max(sigma_guess, MIN_WIDTH_NM) # Ensure positive floor

    # Gamma guess (for Lorentzian) from FWHM
    gamma_guess = fwhm_guess_nm / 2.0
    gamma_guess = max(gamma_guess, MIN_WIDTH_NM) # Ensure positive floor

    # Eta guess (for PV)
    eta_guess = 0.5

    # Bounds
    min_wl, max_wl = wl_roi[0], wl_roi[-1]
    min_width_bound = MIN_WIDTH_NM
    # Max width can be larger than ROI, e.g., full ROI width
    max_width_bound = abs(max_wl - min_wl) if abs(max_wl - min_wl) > MIN_WIDTH_NM else 1.0
    max_amp_bound = amp_guess * 5.0 # Allow amplitude to increase significantly
    min_amp_bound = 0.0 # Amplitude cannot be negative

    # Center bounds slightly inside ROI to avoid edge fitting issues
    center_buffer = abs(wl_roi[1] - wl_roi[0]) if len(wl_roi) > 1 else 0.01 * abs(max_wl-min_wl)
    center_min_bound = min_wl + center_buffer
    center_max_bound = max_wl - center_buffer
    if center_min_bound >= center_max_bound: # Handle very narrow ROI
        center_min_bound = min_wl
        center_max_bound = max_wl


    # Clamp initial guesses within bounds
    center_wl_guess = np.clip(center_wl_guess, center_min_bound, center_max_bound)
    sigma_guess = np.clip(sigma_guess, min_width_bound, max_width_bound)
    gamma_guess = np.clip(gamma_guess, min_width_bound, max_width_bound)
    amp_guess = np.clip(amp_guess, min_amp_bound, max_amp_bound)

    # Define parameters and bounds based on profile type
    if profile_type == ProfileType.GAUSSIAN:
        p0 = [amp_guess, center_wl_guess, sigma_guess]
        bounds = ([min_amp_bound, center_min_bound, min_width_bound],
                  [max_amp_bound, center_max_bound, max_width_bound])
    elif profile_type == ProfileType.LORENTZIAN:
        p0 = [amp_guess, center_wl_guess, gamma_guess]
        bounds = ([min_amp_bound, center_min_bound, min_width_bound],
                  [max_amp_bound, center_max_bound, max_width_bound])
    elif profile_type == ProfileType.PSEUDO_VOIGT:
        p0 = [amp_guess, center_wl_guess, sigma_guess, eta_guess]
        bounds = ([min_amp_bound, center_min_bound, min_width_bound, 0.0], # Eta bounds [0, 1]
                  [max_amp_bound, center_max_bound, max_width_bound, 1.0])
    else:
        raise ValueError(f"Unsupported profile type for guessing: {profile_type}")

    return p0, bounds


def _perform_single_fit(profile_type: ProfileType,
                        wl_roi: np.ndarray,
                        int_roi_corrected: np.ndarray,
                        max_iterations: int
                       ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Performs curve_fit for a single profile type."""

    # Filter out NaN values before fitting - curve_fit cannot handle them
    finite_mask = np.isfinite(wl_roi) & np.isfinite(int_roi_corrected)
    if np.sum(finite_mask) < PROFILE_FUNCTIONS[profile_type][1]: # Need at least as many points as parameters
         return None, None, f"Not enough finite data points ({np.sum(finite_mask)}) in ROI for {profile_type.value} fit."

    wl_fit = wl_roi[finite_mask]
    int_fit = int_roi_corrected[finite_mask]

    try:
        fit_func, n_params = PROFILE_FUNCTIONS[profile_type]
        p0, bounds = _get_initial_guesses_and_bounds(profile_type, wl_fit, int_fit, wl_fit[np.argmax(int_fit)]) # Recalculate guesses on filtered data?

        params_opt, params_cov = curve_fit(
            fit_func,
            wl_fit,
            int_fit,
            p0=p0,
            bounds=bounds,
            maxfev=max_iterations,
            method='trf', # Trust Region Reflective is good for bounds
            # Add sigma=weights if error estimates are available for weighted LS
        )

        # Basic check for valid covariance (positive diagonal)
        if params_cov is None or not np.all(np.isfinite(params_cov)) or np.any(np.diag(params_cov) < 0):
             # Sometimes curve_fit returns inf variance when fit is poor or singular
             logging.warning(f"Fit converged for {profile_type.value}, but covariance matrix is invalid. Treating as failed fit.")
             return None, None, f"{profile_type.value} fit covariance invalid."

        return params_opt, params_cov, f"{profile_type.value} fit successful."

    except (RuntimeError, ValueError) as e:
        # RuntimeError often means optimal parameters not found (max iterations, bad jacobian)
        # ValueError can arise from input shape mismatches or issues within the model function
        return None, None, f"{profile_type.value} fit failed: {e}"
    except Exception as e:
        # Catch any other unexpected errors during fitting
        logging.error(f"Unexpected error during {profile_type.value} fit:", exc_info=True)
        return None, None, f"Unexpected error during {profile_type.value} fit: {e}"


def _evaluate_fit(profile_type: ProfileType,
                  params: np.ndarray,
                  cov: np.ndarray,
                  wl_roi: np.ndarray, # Use original ROI wl for evaluation metrics
                  int_roi_corrected: np.ndarray # Use original ROI corrected intensity
                 ) -> FitResult:
    """Calculates metrics and creates a FitResult object for a successful fit."""

    fit_func, n_params = PROFILE_FUNCTIONS[profile_type]

    # Calculate fitted curve on the original ROI wavelengths (including any NaNs)
    y_fitted = np.full_like(wl_roi, np.nan)
    finite_mask = np.isfinite(wl_roi) & np.isfinite(int_roi_corrected)
    if np.any(finite_mask):
        y_fitted[finite_mask] = fit_func(wl_roi[finite_mask], *params)

    # Calculate metrics only on the finite points used for fitting
    wl_fit = wl_roi[finite_mask]
    int_fit = int_roi_corrected[finite_mask]
    y_fit_points = y_fitted[finite_mask]

    rss = np.sum((int_fit - y_fit_points)**2)
    r2 = calculate_r_squared(int_fit, y_fit_points)
    n_fit_points = len(int_fit)
    aic, bic = calculate_aic_bic(n_fit_points, n_params, rss)

    # Extract common parameters
    amplitude = params[0]
    center = params[1]
    width = params[2] # Sigma for Gauss/PV, Gamma for Lorentz
    eta = params[3] if profile_type == ProfileType.PSEUDO_VOIGT else None

    # Create FitResult ensuring correct field order from definition
    return FitResult(
        profile_type=profile_type,
        amplitude=amplitude,
        center=center,
        width=width,
        params=params,                 # Non-default field moved up
        mixing_param_eta=eta,          # Default fields follow
        params_covariance=cov,
        r_squared=r2,
        aic=aic,
        bic=bic,
        success=True,
        message=f"{profile_type.value} fit evaluation complete.",
        roi_wavelengths=wl_roi,
        roi_intensity_corrected=int_roi_corrected,
        fitted_curve=y_fitted,
    )


# --- Main Fitting Function ---

def fit_peak(spectrum: Spectrum,
             peak_index: int,
             processed_intensity: np.ndarray,
             roi_factor: float = 7.0,
             min_roi_width_nm: float = 0.1,
             profiles_to_fit: Optional[List[ProfileType]] = None,
             model_selection: ModelSelectionCriterion = ModelSelectionCriterion.AIC,
             baseline_mode: BaselineMode = BaselineMode.LOCAL_LINEAR,
             max_fit_iterations: int = DEFAULT_MAX_ITER,
             roi_wavelengths: Optional[Tuple[float, float]] = None
            ) -> Tuple[Optional[FitResult], Dict[ProfileType, FitResult]]:
    """
    Fits specified profiles to a single peak within a defined ROI of the processed intensity.

    Performs local baseline correction within the ROI before fitting.
    Selects the best model based on the specified criterion (AIC or BIC).

    Args:
        spectrum (Spectrum): Spectrum object containing wavelengths.
        peak_index (int): Index of the estimated peak center in the spectrum arrays.
        processed_intensity (np.ndarray): Intensity array (e.g., baseline-corrected) to fit.
        roi_factor (float): Multiplier for estimated FWHM to determine auto ROI width. Ignored if roi_wavelengths is set.
        min_roi_width_nm (float): Minimum width of the auto ROI in nm. Ignored if roi_wavelengths is set.
        profiles_to_fit (Optional[List[ProfileType]]): Profiles to attempt fitting. Defaults to all implemented types.
        model_selection (ModelSelectionCriterion): Criterion (AIC or BIC) for best model selection.
        baseline_mode (BaselineMode): Method for local baseline subtraction within the ROI before fitting.
        max_fit_iterations (int): Max iterations for curve_fit.
        roi_wavelengths (Optional[Tuple[float, float]]): Explicit ROI [min_wl, max_wl]. Overrides auto ROI if provided.

    Returns:
        Tuple[Optional[FitResult], Dict[ProfileType, FitResult]]:
            - Best FitResult object based on the selection criterion (or None if all fits fail).
            - Dictionary {ProfileType: FitResult} for all attempted fits (including failures).
    """
    if not SCIPY_AVAILABLE:
        logging.error("Cannot fit peak: SciPy is unavailable.")
        return None, {}
    if not isinstance(spectrum, Spectrum) or spectrum.wavelengths is None or processed_intensity is None:
         logging.error("Invalid input: Spectrum or processed_intensity is missing.")
         return None, {}
    if peak_index < 0 or peak_index >= len(spectrum.wavelengths) or peak_index >= len(processed_intensity):
        logging.error(f"Invalid peak index: {peak_index} for data length {len(spectrum.wavelengths)}")
        return None, {}
    if len(spectrum.wavelengths) != len(processed_intensity):
        logging.error(f"Shape mismatch: Wavelengths ({len(spectrum.wavelengths)}) vs Intensity ({len(processed_intensity)})")
        return None, {}

    # --- 1. Determine Region of Interest (ROI) ---
    try:
        roi_result = _determine_roi(
            spectrum.wavelengths,
            processed_intensity, # Use the provided processed intensity for ROI determination
            peak_index,
            roi_factor,
            min_roi_width_nm,
            roi_wavelengths
        )
        if roi_result is None:
            # Error already logged in _determine_roi
            return None, {}
        roi_indices, wl_roi, int_roi_raw, center_wl_guess = roi_result

    except Exception as e:
        logging.error(f"Unexpected error defining ROI for peak near index {peak_index}: {e}", exc_info=True)
        return None, {}

    # --- 2. Local Baseline Correction ---
    # Note: This baseline is applied *locally* within the ROI to the *processed_intensity*
    # It's intended to remove residual local slope/offset *before* peak profile fitting.
    try:
        int_roi_corrected, _ = _apply_local_baseline(wl_roi, int_roi_raw, baseline_mode)
    except Exception as e:
         logging.error(f"Unexpected error applying local baseline for peak near index {peak_index}: {e}", exc_info=True)
         # Proceed without local correction? Or fail? Let's proceed but log clearly.
         logging.warning("Proceeding without local baseline correction due to error.")
         int_roi_corrected = int_roi_raw # Use the uncorrected ROI intensity


    # --- 3. Fit Profiles ---
    if profiles_to_fit is None:
        profiles_to_fit = list(PROFILE_FUNCTIONS.keys()) # Default to all defined profiles

    all_results: Dict[ProfileType, FitResult] = {}

    for profile_type in profiles_to_fit:
        if profile_type not in PROFILE_FUNCTIONS:
            logging.warning(f"Skipping unsupported profile type: {profile_type}")
            # Create a placeholder failure result
            all_results[profile_type] = FitResult(profile_type=profile_type, amplitude=np.nan, center=np.nan, width=np.nan, params=np.array([]), success=False, message="Unsupported profile type")
            continue

        logging.debug(f"Attempting {profile_type.value} fit for peak near {center_wl_guess:.3f}nm...")
        params_opt, params_cov, fit_message = _perform_single_fit(
            profile_type,
            wl_roi,
            int_roi_corrected,
            max_fit_iterations
        )

        if params_opt is not None and params_cov is not None:
            try:
                result = _evaluate_fit(
                    profile_type,
                    params_opt,
                    params_cov,
                    wl_roi, # Pass original ROI wl/intensity for context
                    int_roi_corrected
                )
                all_results[profile_type] = result
                logging.debug(f"Successfully fitted and evaluated {profile_type.value}.")
            except Exception as e:
                logging.error(f"Error evaluating successful {profile_type.value} fit: {e}", exc_info=True)
                # Create a failure result, but store params if available
                all_results[profile_type] = FitResult(profile_type=profile_type, amplitude=np.nan, center=np.nan, width=np.nan, params=params_opt if params_opt is not None else np.array([]), success=False, message=f"Evaluation error: {e}")
        else:
            # Fit failed, record failure
            logging.warning(f"Fit failed for {profile_type.value} near {center_wl_guess:.3f}nm: {fit_message}")
            # Create a failure result
            all_results[profile_type] = FitResult(profile_type=profile_type, amplitude=np.nan, center=np.nan, width=np.nan, params=np.array([]), success=False, message=fit_message)


    # --- 4. Select Best Fit ---
    best_fit_result: Optional[FitResult] = None
    best_score = np.inf

    successful_fits = [res for res in all_results.values() if res.success]

    if not successful_fits:
        logging.warning(f"No successful fits found for peak near {center_wl_guess:.3f}nm.")
        return None, all_results # Return None for best, but all attempted results

    for result in successful_fits:
        score = np.inf
        if model_selection == ModelSelectionCriterion.AIC and result.aic is not None:
            score = result.aic
        elif model_selection == ModelSelectionCriterion.BIC and result.bic is not None:
            score = result.bic
        else: # Fallback or handle case where criterion value is None
            logging.warning(f"Could not get {model_selection.value} score for {result.profile_type.value}, skipping for best fit selection.")
            continue

        if np.isfinite(score) and score < best_score:
            best_score = score
            best_fit_result = result

    if best_fit_result:
        logging.info(f"Best fit for peak near {center_wl_guess:.3f}nm: {best_fit_result.profile_type.value} "
                     f"({model_selection.value}={best_score:.2f}, R2={best_fit_result.r_squared:.3f})")
    else:
        # This might happen if all successful fits had non-finite AIC/BIC scores
        logging.warning(f"Could not determine best fit among successful fits for peak near {center_wl_guess:.3f}nm.")
        # Optionally, return the first successful fit as a fallback? Or None? Let's return None.

    return best_fit_result, all_results

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':

    # Configure logging for detailed output during testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    if not SCIPY_AVAILABLE:
        logging.error("Example usage requires SciPy. Exiting.")
    else:
        # 1. Create Sample Data (e.g., a Gaussian peak with noise)
        wavelengths = np.linspace(500, 510, 201) # 201 points from 500 to 510 nm
        center_true = 505.0
        amplitude_true = 100.0
        sigma_true = 0.2
        noise_level = 5.0

        # Baseline (optional, could be added)
        baseline = 10 + 0.5 * (wavelengths - 500)

        # Ideal peak + baseline + noise
        intensity_signal = gaussian(wavelengths, amplitude_true, center_true, sigma_true)
        intensity_noisy = intensity_signal + baseline + np.random.normal(0, noise_level, size=wavelengths.shape)

        # Assume some processing removed the coarse baseline, but maybe not perfectly
        processed_intensity = intensity_noisy - (10 + 0.5 * (wavelengths - 500)) # Imperfect baseline removal
        # processed_intensity = intensity_noisy # Or fit with baseline removal enabled

        # Create Spectrum object
        spectrum_data = Spectrum(wavelengths=wavelengths, intensity=intensity_noisy) # Raw intensity in spectrum obj

        # 2. Find Peak Index (simplified - use your actual peak finding logic)
        # Here, we know the approximate center, find the index near it
        peak_idx_est = np.argmin(np.abs(wavelengths - center_true))
        # Verify the index corresponds to a local maximum in processed_intensity
        peak_idx = peak_idx_est # Assume peak finding correctly identified this index

        logging.info(f"Simulated peak center: {center_true:.3f}nm, Estimated index: {peak_idx}")

        # 3. Call fit_peak
        best_fit, all_fits = fit_peak(
            spectrum=spectrum_data,
            peak_index=peak_idx,
            processed_intensity=processed_intensity, # Pass the data to be fitted
            roi_factor=6.0,             # How many FWHM for auto ROI
            min_roi_width_nm=0.3,       # Minimum auto ROI width
            profiles_to_fit=[ProfileType.GAUSSIAN, ProfileType.LORENTZIAN, ProfileType.PSEUDO_VOIGT],
            model_selection=ModelSelectionCriterion.BIC, # Choose AIC or BIC
            baseline_mode=BaselineMode.LOCAL_LINEAR,    # Try removing local tilt in ROI
            # roi_wavelengths=(504.0, 506.0) # Optional: Override auto ROI
        )

        # 4. Analyze Results
        print("\n--- Fitting Summary ---")
        if best_fit:
            print(f"Best Fit Profile: {best_fit.profile_type.value}")
            print(f"  Center: {best_fit.center:.4f} nm")
            print(f"  Amplitude: {best_fit.amplitude:.2f}")
            print(f"  Width (sigma/gamma): {best_fit.width:.4f}")
            if best_fit.mixing_param_eta is not None:
                print(f"  Eta (PV): {best_fit.mixing_param_eta:.3f}")
            print(f"  R-squared: {best_fit.r_squared:.4f}")
            print(f"  AIC: {best_fit.aic:.2f}")
            print(f"  BIC: {best_fit.bic:.2f}")

            # Optional: Plotting
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                # Plot data points within the ROI used for the best fit
                if best_fit.roi_wavelengths is not None and best_fit.roi_intensity_corrected is not None:
                     plt.plot(best_fit.roi_wavelengths, best_fit.roi_intensity_corrected, 'bo', label='Data in ROI (Corrected)', markersize=4)
                     # Plot the fitted curve
                     if best_fit.fitted_curve is not None:
                           plt.plot(best_fit.roi_wavelengths, best_fit.fitted_curve, 'r-', label=f'Best Fit ({best_fit.profile_type.value})')
                else: # Fallback if ROI data not stored in result
                     plt.plot(wavelengths, processed_intensity, 'b.', label='Processed Data')

                plt.title(f"Peak Fit Example (Best: {best_fit.profile_type.value})")
                plt.xlabel("Wavelength (nm)")
                plt.ylabel("Intensity (Processed)")
                plt.legend()
                plt.grid(True)
                plt.show()
            except ImportError:
                print("\nInstall matplotlib (pip install matplotlib) to see the plot.")

        else:
            print("No successful fit was found.")

        print("\n--- All Attempted Fits ---")
        for profile, result in all_fits.items():
            status = "Success" if result.success else f"Failure ({result.message})"
            print(f"- {profile.value}: {status}")
            if result.success:
                 print(f"    Center={result.center:.4f}, Amp={result.amplitude:.2f}, Width={result.width:.4f}, R2={result.r_squared:.3f}, AIC={result.aic:.2f}, BIC={result.bic:.2f}")