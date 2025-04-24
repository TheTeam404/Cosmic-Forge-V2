
# This file makes 'core' a Python package.
# It can also be used to expose certain core functionalities directly.

# Expose core data models for convenience
from .data_models import Spectrum, Peak, FitResult, NISTMatch

# Expose primary functions from each submodule (optional, depends on usage pattern)
from .file_io import load_spectrum_from_file, save_spectrum_data, save_peak_list, save_nist_matches, save_dataframe
from .processing import baseline_poly, baseline_snip, smooth_savitzky_golay
from .peak_detector import detect_peaks_scipy
from .peak_fitter import fit_peak
from .nist_manager import search_online_nist, ASTROQUERY_AVAILABLE
from .cflibs import calculate_boltzmann_temp, calculate_electron_density_saha, calculate_cf_libs_conc
from .analysis_backends import (
    check_sklearn_availability, scale_data, run_pca, run_pls_regression,
    run_classification # Add others like run_clustering if implemented
)
from .atomic_data import get_partition_function, get_ionization_energy
from .session_manager import SessionManager

# Define __all__ for explicit public API if desired (good practice)
__all__ = [
    # Data Models
    "Spectrum", "Peak", "FitResult", "NISTMatch",
    # File IO
    "load_spectrum_from_file", "save_spectrum_data", "save_peak_list",
    "save_nist_matches", "save_dataframe",
    # Processing
    "baseline_poly", "baseline_snip", "smooth_savitzky_golay",
    # Peak Detection
    "detect_peaks_scipy",
    # Peak Fitting
    "fit_peak",
    # NIST Manager
    "search_online_nist", "ASTROQUERY_AVAILABLE",
    # CF-LIBS
    "calculate_boltzmann_temp", "calculate_electron_density_saha", "calculate_cf_libs_conc",
    # Analysis Backends (ML)
    "check_sklearn_availability", "scale_data", "run_pca", "run_pls_regression",
    "run_classification",
    # Atomic Data
    "get_partition_function", "get_ionization_energy",
    # Session Management
    "SessionManager",
]
