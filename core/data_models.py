
"""
Core data structures (models) used throughout the application.
These classes help standardize how spectral data, peaks, fits, etc., are represented.
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

# --- Constants ---
# Factor to convert sigma (Gaussian std dev) to FWHM
FWHM_GAUSS_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0)) # ~2.35482

class Spectrum:
    """Represents a single LIBS spectrum."""
    def __init__(self,
                 wavelengths: np.ndarray,
                 raw_intensity: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None,
                 source_filepath: Optional[str] = None):
        """
        Initializes a Spectrum object.

        Args:
            wavelengths (np.ndarray): Array of wavelength values (nm). Should be sorted.
            raw_intensity (np.ndarray): Array of corresponding raw intensity values.
            metadata (Optional[Dict[str, Any]]): Dictionary for metadata.
            source_filepath (Optional[str]): Original file path of the spectrum.

        Raises:
            ValueError: If wavelengths and intensity arrays have mismatched shapes or are not 1D.
        """
        if wavelengths.shape != raw_intensity.shape:
            raise ValueError("Wavelengths and Intensity arrays must have the same shape.")
        if wavelengths.ndim != 1:
            raise ValueError("Wavelengths and Intensity must be 1-dimensional arrays.")
        if len(wavelengths) == 0:
             raise ValueError("Wavelengths and Intensity arrays cannot be empty.")

        self.wavelengths: np.ndarray = wavelengths
        self.raw_intensity: np.ndarray = raw_intensity
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.source_filepath: Optional[str] = source_filepath

        # --- Processed Data (initialized later by processing steps) ---
        self.processed_intensity: Optional[np.ndarray] = None
        self.baseline: Optional[np.ndarray] = None
        # --- Noise Analysis Data (Placeholders - to be populated by noise analysis) ---
        self.noise_regions: Optional[List[Tuple[float, float]]] = None # List of (start_wl, end_wl) tuples
        self.noise_std_dev: Optional[float] = None # Estimated noise level in signal-free regions

    @property
    def filename(self) -> Optional[str]:
        """Returns the base filename from the source filepath, if available."""
        if self.source_filepath:
            try:
                return os.path.basename(self.source_filepath)
            except Exception:
                return self.source_filepath # Fallback if path manipulation fails
        return None

    def update_processed(self,
                         processed_intensity: np.ndarray,
                         baseline: Optional[np.ndarray] = None):
        """
        Updates the processed intensity and optionally the baseline.

        Args:
            processed_intensity (np.ndarray): The processed intensity array.
            baseline (Optional[np.ndarray]): The calculated baseline array.

        Raises:
            ValueError: If array shapes do not match wavelengths.
        """
        if processed_intensity.shape != self.wavelengths.shape:
            raise ValueError(f"Processed intensity shape {processed_intensity.shape} mismatch with wavelengths {self.wavelengths.shape}.")
        self.processed_intensity = processed_intensity

        if baseline is not None:
            if baseline.shape != self.wavelengths.shape:
                raise ValueError(f"Baseline shape {baseline.shape} mismatch with wavelengths {self.wavelengths.shape}.")
            self.baseline = baseline
        else:
             self.baseline = None # Ensure baseline is cleared if not provided

    def get_data_range(self, wl_min: float, wl_max: float, processed: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns wavelength and intensity data within a specified wavelength range.

        Args:
            wl_min (float): Minimum wavelength.
            wl_max (float): Maximum wavelength.
            processed (bool): If True, returns processed intensity (if available),
                              otherwise returns raw intensity.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Wavelengths and corresponding intensities in the range.
                                          Returns empty arrays if no data in range.
        """
        intensity_source = self.processed_intensity if processed and self.processed_intensity is not None else self.raw_intensity
        if intensity_source is None: # Should not happen if raw_intensity is always present
             return np.array([]), np.array([])
        mask = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
        return self.wavelengths[mask], intensity_source[mask]

    def __len__(self) -> int:
        """Returns the number of data points in the spectrum."""
        return len(self.wavelengths)

    def __repr__(self) -> str:
        wl_range = f"[{self.wavelengths[0]:.2f}-{self.wavelengths[-1]:.2f}] nm" if len(self) > 0 else "[Empty]"
        file_info = f", file='{self.filename}'" if self.filename else ""
        proc_info = ", Processed" if self.processed_intensity is not None else ""
        return f"Spectrum(points={len(self)}, range={wl_range}{proc_info}{file_info})"


class FitResult:
    """Represents the result of fitting a specific profile to a peak."""
    def __init__(self,
                 profile_type: str, # e.g., 'Gaussian', 'Lorentzian', 'Voigt', 'PseudoVoigt'
                 amplitude: float,
                 center: float,
                 width: float, # Primary width parameter (e.g., sigma, HWHM gamma)
                 fwhm: Optional[float] = None, # Calculated or fitted FWHM
                 mixing_param_eta: Optional[float] = None, # For PseudoVoigt
                 params_covariance: Optional[np.ndarray] = None, # Covariance matrix from fit
                 r_squared: Optional[float] = None, # Goodness-of-fit metric
                 aic: Optional[float] = None, # Model selection criterion
                 bic: Optional[float] = None, # Model selection criterion
                 success: bool = True, # Did the fit converge successfully?
                 message: str = "" # Optional message from the fitter
                 ):
        self.profile_type = profile_type; self.amplitude = amplitude; self.center = center; self.width = width; self.mixing_param_eta = mixing_param_eta; self.r_squared = r_squared; self.aic = aic; self.bic = bic; self.success = success; self.message = message
        # Calculate FWHM if not provided explicitly by the fitter
        self.fwhm = fwhm if fwhm is not None and np.isfinite(fwhm) else self._calculate_fwhm()
        # Calculate parameter errors from covariance matrix
        self.param_errors = self._calculate_errors(params_covariance) # List: [amp_err, cen_err, wid_err, (eta_err)]

    def _calculate_fwhm(self) -> Optional[float]:
        """Calculates FWHM based on profile type and width parameter if possible."""
        if self.width is None or not np.isfinite(self.width) or self.width <= 0: return np.nan
        try:
            if self.profile_type == 'Gaussian': return self.width * FWHM_GAUSS_FACTOR # width = sigma
            elif self.profile_type == 'Lorentzian': return self.width * 2.0 # width = gamma (HWHM)
            elif self.profile_type in ['Voigt', 'PseudoVoigt']: # FWHM is complex, rely on fitter or approximation
                 # Approximation for PseudoVoigt based on its Gaussian sigma if width is sigma
                 if self.mixing_param_eta is not None: return self.width * FWHM_GAUSS_FACTOR # Approx using Gaussian component
                 else: return np.nan # Cannot calculate without eta or direct value
            else: return np.nan
        except Exception: return np.nan # Catch potential math errors

    def _calculate_errors(self, cov: Optional[np.ndarray]) -> List[Optional[float]]:
        """Estimates parameter errors (std dev) from the diagonal of the covariance matrix."""
        num_expected_params = 3
        if self.profile_type == 'PseudoVoigt': num_expected_params = 4 # Amp, Cen, Sigma, Eta
        default_errors: List[Optional[float]] = [np.nan] * num_expected_params

        if cov is None or not isinstance(cov, np.ndarray) or cov.shape != (num_expected_params, num_expected_params):
            return default_errors
        try:
            diag_variances = np.diag(cov)
            # Check for negative variances (can happen with poor fits)
            valid_mask = (diag_variances >= 0) & np.isfinite(diag_variances)
            errors = np.full_like(diag_variances, np.nan)
            errors[valid_mask] = np.sqrt(diag_variances[valid_mask])
            return errors.tolist()
        except Exception as e:
            logging.warning(f"Error calculating parameter errors from covariance: {e}")
            return default_errors

    def get_param_dict(self, include_errors=False) -> Dict[str, Any]:
         """Returns fit parameters as a dictionary."""
         params = {
             "Fit Profile": self.profile_type,
             "Fitted Amplitude": self.amplitude,
             "Fitted Center (nm)": self.center,
             "Fitted Width (nm)": self.width, # Primary width (sigma/gamma)
             "Fitted FWHM (nm)": self.fwhm,
             "Fit R^2": self.r_squared,
             "Fit AIC": self.aic,
             "Fit BIC": self.bic,
         }
         if self.profile_type == 'PseudoVoigt':
              params["Fit Mixing (eta)"] = self.mixing_param_eta
         if include_errors:
              errs = self.param_errors
              params["Fit Amp Error"] = errs[0] if len(errs)>0 else np.nan
              params["Fit Cen Error"] = errs[1] if len(errs)>1 else np.nan
              params["Fit Wid Error"] = errs[2] if len(errs)>2 else np.nan
              if self.profile_type == 'PseudoVoigt' and len(errs)>3:
                   params["Fit Eta Error"] = errs[3] # Add eta error if present
         return params


    def __repr__(self) -> str:
        return f"Fit({self.profile_type}, cen={self.center:.4f}, amp={self.amplitude:.2f}, wid={self.width:.4f})"

    def __eq__(self, other): # Check for approximate equality
        if not isinstance(other, FitResult): return NotImplemented
        if self.profile_type != other.profile_type: return False
        params_self = [self.amplitude, self.center, self.width, self.mixing_param_eta or 0]
        params_other = [other.amplitude, other.center, other.width, other.mixing_param_eta or 0]
        return np.allclose(params_self, params_other, atol=1e-6, equal_nan=True)

    def __hash__(self): # Hash based on rounded core parameters
        return hash((self.profile_type, round(self.amplitude,6), round(self.center,6), round(self.width,6)))


class Peak:
    """Represents a detected peak in a spectrum, potentially with fit results and identification."""
    def __init__(self, detected_index:int, detected_wavelength:float, detected_intensity:float, raw_intensity_at_peak:float):
        if not all(np.isfinite([detected_wavelength, detected_intensity, raw_intensity_at_peak])):
             logging.warning(f"Peak created with non-finite values (idx={detected_index}): wl={detected_wavelength}, int_proc={detected_intensity}, int_raw={raw_intensity_at_peak}")
        self.index: int = int(detected_index); self.wavelength_detected: float = float(detected_wavelength); self.intensity_processed: float = float(detected_intensity); self.intensity_raw: float = float(raw_intensity_at_peak)
        self.best_fit: Optional[FitResult] = None; self.alternative_fits: Dict[str, FitResult] = {}; self.potential_matches: List['NISTMatch'] = []

    def add_fit_result(self, fit_result: FitResult, is_best: bool = False):
        """Adds a fitting result, updates best_fit if applicable."""
        if not isinstance(fit_result, FitResult): return
        if is_best: self.best_fit = fit_result
        # Always store the fit result in alternatives, even if it's the best one
        self.alternative_fits[fit_result.profile_type] = fit_result

    def add_nist_match(self, match: 'NISTMatch'):
        if not isinstance(match, NISTMatch): return
        # Avoid adding duplicate matches (based on element, ion, db_wavelength)
        for existing_match in self.potential_matches:
            if (match.element == existing_match.element and
                match.ion_state_str == existing_match.ion_state_str and
                np.isclose(match.wavelength_db, existing_match.wavelength_db, atol=1e-4)):
                return # Already have this match
        self.potential_matches.append(match); self.potential_matches.sort(key=lambda m: abs(m.wavelength_db - self.wavelength_fitted_or_detected))

    @property
    def wavelength_fitted_or_detected(self) -> float:
        """Returns the fitted center wavelength if available and valid, otherwise the detected."""
        if self.best_fit and self.best_fit.center is not None and np.isfinite(self.best_fit.center): return self.best_fit.center
        return self.wavelength_detected

    def to_dataframe_row(self) -> Dict[str, Any]:
        """Converts peak data (including best fit) to a dictionary for DataFrames/Tables."""
        row = {"Peak Index":self.index, "Detected Wavelength (nm)":self.wavelength_detected, "Raw Intensity":self.intensity_raw, "Processed Intensity":self.intensity_processed}
        # Add columns for best fit parameters using FitResult's dict method
        fit_params = self.best_fit.get_param_dict(include_errors=True) if self.best_fit else {}
        row.update(fit_params)
        # Ensure all expected columns exist, even if fit failed (populate with NaN)
        expected_fit_cols = ["Fit Profile","Fitted Amplitude","Fitted Center (nm)","Fitted Width (nm)","Fitted FWHM (nm)","Fit Mixing (eta)","Fit R^2","Fit AIC","Fit BIC","Fit Amp Error","Fit Cen Error","Fit Wid Error"] # Adjusted width name
        for col in expected_fit_cols:
            row.setdefault(col, np.nan) # Use NaN for missing fit data
        return row

    def __repr__(self) -> str: fit=f",fit='{self.best_fit.profile_type}'" if self.best_fit else ""; match=f",matches={len(self.potential_matches)}" if self.potential_matches else ""; return f"Peak(idx={self.index}, wl={self.wavelength_detected:.3f}{fit}{match})"


class NISTMatch:
    """Represents a potential match from the NIST database."""
    # Note: ei/gi stored here refer to UPPER levels (NIST Ek/gk) for direct use in Boltzmann plots
    def __init__(self, element:str, ion_state_str:str, wavelength_db:float, aki:Optional[float]=None, ei:Optional[float]=None, # Upper E (Ek)
                 ek:Optional[float]=None, # Lower E (Ei)
                 gi:Optional[float]=None, # Upper g (gk)
                 gk:Optional[float]=None, # Lower g (gi)
                 line_label:Optional[str]=None, source:str='Unknown'):
        self.element=str(element); self.ion_state_str=str(ion_state_str); self.wavelength_db=float(wavelength_db); self.aki=float(aki) if aki is not None else None; self.ei=float(ei) if ei is not None else None; self.ek=float(ek) if ek is not None else None; self.gi=float(gi) if gi is not None else None; self.gk=float(gk) if gk is not None else None; self.line_label=str(line_label) if line_label else f"{element} {ion_state_str} {wavelength_db:.3f}"; self.source=str(source)
        self.query_peak_index: Optional[int] = None; self.query_peak_wavelength: Optional[float] = None # Store info about the peak that generated this match query

    @property
    def ion_state_int(self) -> Optional[int]: roman_map={'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7}; return roman_map.get(self.ion_state_str.upper())

    def to_dataframe_row(self, peak_wavelength: float, peak_intensity: float) -> Dict[str, Any]:
        """Converts match data to a dictionary for the NIST results table."""
        delta = peak_wavelength - self.wavelength_db if np.isfinite(peak_wavelength) else np.nan
        # Match columns expected by NistSearchView table
        return {"Peak λ (nm)":peak_wavelength, "Intensity":peak_intensity, "Source":self.source, "Elem":self.element, "Ion":self.ion_state_str, "DB λ (nm)":self.wavelength_db, "Δλ (nm)":delta, "Aki (s⁻¹)":self.aki, "Ei (eV)":self.ei, # Upper E
                "gi":self.gi, # Upper g
                "Line Label":self.line_label,
               }
    def __repr__(self) -> str: return f"NIST({self.element}{self.ion_state_str}@{self.wavelength_db:.3f},src={self.source})"

