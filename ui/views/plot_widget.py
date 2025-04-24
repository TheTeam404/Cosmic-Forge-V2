# -*- coding: utf-8 -*-
"""
Custom Matplotlib plot widget integrated with PyQt6 for displaying spectra,
peaks, fits, NIST lines, and handling interactive elements like highlighting
and annotations on hover/click.
"""
import logging
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt # Required for colormap access (plt.get_cmap)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text as MplText # Use explicit type hint alias
from matplotlib.artist import Artist
from matplotlib.legend import Legend # For type hinting legend
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, pyqtBoundSignal
# Use string literals for data model type hints (e.g., 'Spectrum') within THIS file.
# This is the standard way to resolve type checker errors caused by complex import
# situations or circular dependencies without affecting runtime behavior.
from typing import List, Optional, Dict, Any, Tuple, Union

from core.peak_fitter import ProfileType

# Ensure using QtAgg backend for compatibility with PyQt6
matplotlib.use('QtAgg')

# --- Project Root Calculation (Attempt) ---
# Removed for brevity, assume project structure handles imports or it's added elsewhere

# --- Import Core Components with Fallbacks ---
try:
    # Import the actual classes needed for runtime checks and operations
    from core.data_models import Spectrum, Peak, FitResult, NISTMatch
    # Import the fitting functions (used internally for generating fit line Y values)
    from core.processing import gaussian, lorentzian, pseudo_voigt
    CORE_MODULES_AVAILABLE = True
    logging.debug("PlotWidget: Core data models and processing functions imported successfully.")
except ImportError as e:
    logging.error(f"PlotWidget: CRITICAL - Failed to import core modules: {e}. Plotting will be severely limited. Using dummy placeholders.")
    class Spectrum: pass # type: ignore
    class Peak: pass # type: ignore
    class FitResult: pass # type: ignore
    class NISTMatch: pass # type: ignore
    def gaussian(*args, **kwargs): return np.zeros_like(args[0]) if len(args) > 0 and isinstance(args[0], np.ndarray) else None # type: ignore
    def lorentzian(*args, **kwargs): return np.zeros_like(args[0]) if len(args) > 0 and isinstance(args[0], np.ndarray) else None # type: ignore
    def pseudo_voigt(*args, **kwargs): return np.zeros_like(args[0]) if len(args) > 0 and isinstance(args[0], np.ndarray) else None # type: ignore
    CORE_MODULES_AVAILABLE = False
except Exception as e:
    # Catch other potential exceptions during import
     logging.critical(f"PlotWidget: An unexpected error occurred during core module import: {e}", exc_info=True)
     Spectrum = Peak = FitResult = NISTMatch = object # type: ignore
     gaussian = lorentzian = pseudo_voigt = lambda *args, **kwargs: None # type: ignore
     CORE_MODULES_AVAILABLE = False

# --- Type Hint Aliases ---
PlotElement = Optional[Artist] # Type alias for optional Matplotlib artists
FitLineDict = Dict[Tuple[int, str], Line2D] # Type alias for the dictionary storing fit lines


class SpectrumPlotWidget(QWidget):
    """
    A Matplotlib plotting widget embedded in PyQt6 for visualizing spectral data.
    """

    peak_clicked: pyqtBoundSignal = pyqtSignal(int)

    # --- Initialization ---
    def __init__(self, parent: Optional[QWidget] = None, config: Optional[Dict] = None):
        """ Initializes the SpectrumPlotWidget. """
        super().__init__(parent)
        self.config = config if config is not None else {}
        logging.info("Initializing SpectrumPlotWidget.")
        if not CORE_MODULES_AVAILABLE:
             logging.warning("PlotWidget initialized, but core modules failed to import. Functionality will be limited.")

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self._plot_elements: Dict[str, Union[PlotElement, FitLineDict]] = {
            'raw': None, 'proc': None, 'base': None, 'det': None, 'fit': None,
            'fits': {}, 'nist_lines': None, 'highlight': None, 'legend': None
        }
        self._nist_annotations: List[MplText] = []
        self._peaks_ref: List['Peak'] = []
        self._matches_ref: List['NISTMatch'] = []
        self._highlighted_peak_list_index: Optional[int] = None
        self._highlighted_fit_result: Optional['FitResult'] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.updateGeometry()

        self._setup_initial_plot_state()
        self._connect_matplotlib_events()
        self.apply_theme_colors()

    def _setup_initial_plot_state(self):
        """Configures the initial appearance of the plot axes and annotation."""
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Spectrum Plot")
        self.ax.grid(True, linestyle=':', alpha=0.6, zorder=-10)
        self.annot = self.ax.annotate(
            "", xy=(0,0), xytext=(20,20), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.85),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            visible=False, clip_on=True
        )

    def _connect_matplotlib_events(self):
        """Connects Matplotlib canvas events to handler methods."""
        if hasattr(self, 'hover_connection_id') and self.hover_connection_id:
            try: self.canvas.mpl_disconnect(self.hover_connection_id)
            except Exception: pass
        if hasattr(self, 'click_connection_id') and self.click_connection_id:
            try: self.canvas.mpl_disconnect(self.click_connection_id)
            except Exception: pass
        self.hover_connection_id = self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.click_connection_id = self.canvas.mpl_connect("pick_event", self._on_pick)
        logging.debug("Matplotlib event handlers connected (hover, pick).")

    # --- Artist Management ---
    def _remove_artist(self, key: str):
        """ Safely removes a plot element (or group) referenced by `key`. """
        element = self._plot_elements.get(key)
        removed = False
        try:
            if key == 'fits' and isinstance(element, dict):
                fits_dict: FitLineDict = element
                for line_artist in list(fits_dict.values()):
                    if line_artist: line_artist.remove()
                self._plot_elements['fits'] = {}; removed = True
            elif key == 'nist_lines' and isinstance(element, LineCollection):
                element.remove()
                self._plot_elements['nist_lines'] = None
                for ann in self._nist_annotations: ann.remove()
                self._nist_annotations = []; removed = True
            elif isinstance(element, Artist):
                element.remove()
                self._plot_elements[key] = None; removed = True
        except (AttributeError, ValueError, TypeError) as e:
            logging.debug(f"Issue removing plot element '{key}' (may already be gone): {e}")
        except Exception as e:
            logging.error(f"Unexpected error removing plot element '{key}': {e}", exc_info=True)
        # Reset key if removal was attempted or successful
        if removed or element is not None:
            if key == 'fits': self._plot_elements['fits'] = {}
            elif key == 'nist_lines': self._plot_elements['nist_lines'] = None; self._nist_annotations = []
            elif key in self._plot_elements: self._plot_elements[key] = None
        # if removed: logging.debug(f"Removed plot element: {key}")


    def clear_plot(self, redraw: bool = True):
        """ Clears all plotted data, annotations, and resets axes. """
        logging.info("Clearing plot widget.")
        try:
            xlim, ylim = None, None
            if self.ax and (self.ax.has_data() or self.ax.get_xlim() != (0.0, 1.0) or self.ax.get_ylim() != (0.0, 1.0)):
                try:
                     xlim = self.ax.get_xlim()
                     ylim = self.ax.get_ylim()
                     if not (all(np.isfinite(v) for v in xlim + ylim) and xlim[0] < xlim[1] and ylim[0] < ylim[1]):
                          logging.warning(f"Invalid limits detected before clear: X={xlim}, Y={ylim}. Will autoscale.")
                          xlim, ylim = None, None
                except Exception as e:
                     logging.warning(f"Could not get axes limits before clearing: {e}. Will autoscale.")
                     xlim, ylim = None, None

            for key in list(self._plot_elements.keys()):
                 self._remove_artist(key)

            self.ax.cla()
            self._setup_initial_plot_state()

            self._peaks_ref = []
            self._matches_ref = []
            self._highlighted_peak_list_index = None
            self._highlighted_fit_result = None

            if xlim is not None and ylim is not None:
                try:
                    self.ax.set_xlim(xlim)
                    self.ax.set_ylim(ylim)
                    logging.debug(f"Restored previous plot limits: X={xlim}, Y={ylim}")
                except ValueError as ve:
                    logging.warning(f"Could not restore previous plot limits after clear: {ve}. Autoscaling.")
                    self.ax.relim()
                    self.ax.autoscale_view(True, True, True)
            else:
                 self.ax.relim()
                 self.ax.autoscale_view(True, True, True)

            if redraw:
                self._redraw_canvas()
        except Exception as e:
            logging.error(f"Error during clear_plot: {e}", exc_info=True)

    # --- Plotting Methods ---

    # ***** NEW METHOD ADDED *****
        # Use STRING LITERAL 'Spectrum' for type hint
    def plot_spectrum(self, spectrum: Optional[Spectrum],
                      show_raw: bool = True, show_processed: bool = True, show_baseline: bool = True):
        """
        Plots the main spectral data lines (raw, processed, baseline).
        Clears previous spectrum lines before plotting.

        Args:
            spectrum (Optional[Spectrum]): The Spectrum data object to plot.
            show_raw (bool): Whether to plot the raw intensity line.
            show_processed (bool): Whether to plot the processed intensity line.
            show_baseline (bool): Whether to plot the baseline intensity line.
        """
        logging.debug(f"Received request to plot spectrum: {spectrum}")
        if not CORE_MODULES_AVAILABLE:
            logging.error("Cannot plot spectrum: Core modules not loaded.")
            return

        # ***** REFINED CHECK - More Detailed Logging *****
        valid_plot_data = True
        # 1. Check if spectrum object itself is valid and is the expected type
        if not isinstance(spectrum, Spectrum):
             logging.warning(f"Cannot plot spectrum: Invalid object type received ({type(spectrum)}). Expected core.data_models.Spectrum.")
             valid_plot_data = False
        # 2. Check if the 'wavelengths' attribute exists and is not None (only if object was valid type)
        elif not hasattr(spectrum, 'wavelengths') or spectrum.wavelengths is None:
            logging.warning("Cannot plot spectrum: Spectrum object missing 'wavelengths' attribute or it is None.")
            valid_plot_data = False
        # 3. Check if wavelengths array is actually populated (only if attribute exists and is not None)
        elif not hasattr(spectrum.wavelengths, '__len__') or len(spectrum.wavelengths) == 0:
             logging.warning(f"Cannot plot spectrum: Wavelengths array is empty or does not support len() (type: {type(spectrum.wavelengths)}).")
             valid_plot_data = False

        if not valid_plot_data:
             # Clear existing spectrum lines if invalid data is received or checks fail
             self._remove_artist('raw')
             self._remove_artist('proc')
             self._remove_artist('base')
             self._update_legend()
             self._redraw_canvas()
             return
        # ***** END REFINED CHECK *****


        # --- Clear Previous Spectrum Lines Only ---
        # Do not clear peaks/fits/etc., just the spectrum lines themselves
        self._remove_artist('raw')
        self._remove_artist('proc')
        self._remove_artist('base')

        legend_handles = [] # Start fresh list for legend

        # --- Get Data ---
        wl = spectrum.wavelengths
        # Use getattr safely, defaulting to None if attribute is missing
        raw = getattr(spectrum, 'raw_intensity', None)
        proc = getattr(spectrum, 'processed_intensity', None)
        base = getattr(spectrum, 'baseline', None)

        # --- Get Colors ---
        plot_colors = self.config.get('plotting', {})
        raw_c = plot_colors.get('raw_data_color', '#888888')
        proc_c = plot_colors.get('processed_data_color', '#00FFFF')
        base_c = plot_colors.get('baseline_color', '#FF8C00')

        # --- Plot Lines ---
        something_plotted = False
        try:
            # Plot Raw Data
            # Add explicit check for valid array shape and content
            if show_raw and isinstance(raw, np.ndarray) and raw.shape == wl.shape and np.any(np.isfinite(raw)):
                line_raw, = self.ax.plot(wl, raw, label="Raw Data", color=raw_c, lw=0.8, alpha=0.7, zorder=1)
                self._plot_elements['raw'] = line_raw
                legend_handles.append(line_raw)
                something_plotted = True
            else:
                 self._plot_elements['raw'] = None
                 if show_raw: logging.debug("Raw data not plotted (missing, wrong shape, or all NaN).")


            # Plot Processed Data
            if show_processed and isinstance(proc, np.ndarray) and proc.shape == wl.shape and np.any(np.isfinite(proc)):
                line_proc, = self.ax.plot(wl, proc, label="Processed Data", color=proc_c, lw=1.0, zorder=2)
                self._plot_elements['proc'] = line_proc
                legend_handles.append(line_proc)
                something_plotted = True
            else:
                 self._plot_elements['proc'] = None
                 if show_processed: logging.debug("Processed data not plotted (missing, wrong shape, or all NaN).")

            # Plot Baseline
            if show_baseline and isinstance(base, np.ndarray) and base.shape == wl.shape and np.any(np.isfinite(base)):
                line_base, = self.ax.plot(wl, base, label="Baseline", color=base_c, lw=0.8, ls='--', alpha=0.8, zorder=1.5)
                self._plot_elements['base'] = line_base
                legend_handles.append(line_base)
                something_plotted = True
            else:
                 self._plot_elements['base'] = None
                 if show_baseline: logging.debug("Baseline data not plotted (missing, wrong shape, or all NaN).")

            # If only one line was plotted, ensure limits are recalculated
            if something_plotted: # Use flag instead of counting keys
                self.ax.relim()
                self.ax.autoscale_view(True, True, True)
                logging.debug("Autoscaled axes after plotting spectrum lines.")
            else:
                 logging.warning("No valid spectrum data lines (raw, processed, or baseline) were plotted.")


            # --- Update Legend and Redraw ---
            # Add any existing non-spectrum handles (like peaks) back to the legend handles list
            for key, element in self._plot_elements.items():
                 if key not in ['raw', 'proc', 'base', 'legend', 'fits', 'nist_lines', 'highlight'] and isinstance(element, Artist) and element.get_label() and not element.get_label().startswith('_'):
                      if element not in legend_handles:
                           legend_handles.append(element)

            self._update_legend(legend_handles)
            self._redraw_canvas()
            logging.debug(f"Finished plotting spectrum lines. Plotted Raw: {self._plot_elements['raw'] is not None}, Processed: {self._plot_elements['proc'] is not None}, Baseline: {self._plot_elements['base'] is not None}")

        except Exception as e:
            logging.error(f"Error occurred during spectrum line plotting: {e}", exc_info=True)
            # Attempt to clean up partially plotted elements?
            self._remove_artist('raw')
            self._remove_artist('proc')
            self._remove_artist('base')
            self._update_legend()
            self._redraw_canvas()

    # Use STRING LITERAL 'Peak' for type hint
    def plot_peaks(self, peaks: List['Peak']):
        """ Plots markers for detected and fitted peaks based on a list of Peak objects. """
        if not CORE_MODULES_AVAILABLE: logging.error("Cannot plot peaks: Core modules not loaded."); return
        if not isinstance(peaks, list): logging.error(f"plot_peaks expects a list, but got {type(peaks)}. Ignoring."); return

        self._peaks_ref = peaks

        self._remove_artist('det')
        self._remove_artist('fit')
        self._remove_artist('highlight')
        self._highlighted_peak_list_index = None

        current_legend: Optional[Legend] = self._plot_elements.get('legend') # type: ignore
        handles = []
        if current_legend and hasattr(current_legend, 'legendHandles'):
            handles = [h for h in current_legend.legendHandles if isinstance(h, Line2D)] # Keep only non-scatter handles

        if not peaks:
            logging.debug("plot_peaks called with empty list. Peak markers cleared.")
            self._update_legend(handles)
            self._redraw_canvas()
            return

        logging.info(f"Plotting {len(peaks)} peak markers.")

        line_for_y = self._plot_elements.get('proc') or self._plot_elements.get('raw')
        if not isinstance(line_for_y, Line2D):
            logging.warning("Cannot plot peak markers: No 'processed' or 'raw' spectrum line plotted.")
            self._update_legend(handles)
            self._redraw_canvas()
            return

        try:
            plot_wl, plot_int = line_for_y.get_xdata(), line_for_y.get_ydata()
            if len(plot_wl) == 0: logging.warning("Cannot plot peak markers: Base spectrum line is empty."); self._update_legend(handles); self._redraw_canvas(); return
        except Exception as e:
            logging.error(f"Error getting data from base spectrum line for peak plotting: {e}", exc_info=True); return

        det_x, det_y, det_indices = [], [], []
        fit_x, fit_y, fit_indices = [], [], []

        for i, peak in enumerate(peaks):
            if not isinstance(peak, Peak) or not hasattr(peak, 'wavelength_fitted_or_detected'): logging.warning(f"Skipping invalid peak data at index {i}."); continue
            marker_wl = peak.wavelength_fitted_or_detected
            marker_y = np.interp(marker_wl, plot_wl, plot_int, left=np.nan, right=np.nan)
            if not (np.isfinite(marker_wl) and np.isfinite(marker_y)): logging.warning(f"Skipping peak {i} at λ={marker_wl:.4f}: Invalid coordinates (Y={marker_y:.2f})."); continue

            has_fit = hasattr(peak, 'best_fit') and peak.best_fit and getattr(peak.best_fit, 'success', False)
            if has_fit: fit_x.append(marker_wl); fit_y.append(marker_y); fit_indices.append(i)
            else: det_x.append(marker_wl); det_y.append(marker_y); det_indices.append(i)

        plot_colors = self.config.get('plotting', {})
        det_color = plot_colors.get('peak_detected_color', 'red')
        fit_color = plot_colors.get('peak_fitted_color', 'lime')
        picker_radius = 5

        try:
            if det_x:
                scatter_det = self.ax.scatter(det_x, det_y, marker="x", color=det_color, s=35, label=f"Detected ({len(det_x)})", zorder=5, picker=picker_radius)
                scatter_det.peak_list_indices = det_indices
                self._plot_elements['det'] = scatter_det
                handles.append(scatter_det)
            if fit_x:
                scatter_fit = self.ax.scatter(fit_x, fit_y, marker="o", facecolors='none', edgecolors=fit_color, s=50, label=f"Fitted ({len(fit_x)})", zorder=6, picker=picker_radius)
                scatter_fit.peak_list_indices = fit_indices
                self._plot_elements['fit'] = scatter_fit
                handles.append(scatter_fit)
        except Exception as e:
            logging.error(f"Error creating peak scatter plots: {e}", exc_info=True)

        self._update_legend(handles)
        self._redraw_canvas()

    # ... (rest of the methods: plot_fit_lines, highlight_fit_line, plot_nist_matches, etc.) ...
    # ... (They remain largely the same as provided previously) ...

    # Use STRING LITERALS 'Peak' and 'FitResult' for type hints
    def plot_fit_lines(self, peaks: List['Peak'], highlight_fit: Optional['FitResult'] = None):
        """ Plots individual fit profile lines for successful fits within the peaks list. """
        if not CORE_MODULES_AVAILABLE: logging.error("Cannot plot fit lines: Core modules not loaded."); return
        if not isinstance(peaks, list): logging.error(f"plot_fit_lines expects a list, but got {type(peaks)}. Ignoring."); return

        self._remove_artist('fits')
        self._highlighted_fit_result = highlight_fit

        if not peaks: logging.debug("plot_fit_lines called with empty peak list."); self._redraw_canvas(); return

        base_line = self._plot_elements.get('proc') or self._plot_elements.get('raw')
        if not isinstance(base_line, Line2D): logging.warning("Cannot plot fit lines: No base spectrum ('proc' or 'raw') found."); return
        try:
            plot_wl = base_line.get_xdata();
            if len(plot_wl) == 0: logging.warning("Cannot plot fit lines: Base spectrum has no wavelength data."); return
        except Exception as e: logging.error(f"Error getting wavelength data for fit lines: {e}", exc_info=True); return

        baseline_y: Optional[np.ndarray] = None
        baseline_element = self._plot_elements.get('base')
        if isinstance(baseline_element, Line2D):
            try:
                baseline_y_data = baseline_element.get_ydata()
                if len(baseline_y_data) == len(plot_wl): baseline_y = baseline_y_data
                else: logging.warning(f"Baseline length ({len(baseline_y_data)}) differs from wavelength length ({len(plot_wl)}). Fit lines will not include baseline offset.")
            except Exception as e: logging.error(f"Error getting baseline Y-data: {e}")

        new_fits_dict: FitLineDict = {}
        plot_colors = self.config.get('plotting', {})
        default_fit_color = plot_colors.get('fit_line_color', 'magenta')
        highlight_color = plot_colors.get('highlight_fit_color', 'yellow')

        num_plotted = 0
        for peak_index, peak in enumerate(peaks):
            if not isinstance(peak, Peak): continue
            fits_to_plot: Dict[str, 'FitResult'] = {}
            if hasattr(peak, 'alternative_fits') and isinstance(peak.alternative_fits, dict):
                for prof, fit in peak.alternative_fits.items():
                    if isinstance(fit, FitResult) and getattr(fit, 'success', False): fits_to_plot[prof] = fit
            if hasattr(peak, 'best_fit') and isinstance(peak.best_fit, FitResult) and getattr(peak.best_fit, 'success', False):
                profile_type = getattr(peak.best_fit, 'profile_type', 'unknown_best_fit')
                fits_to_plot[str(profile_type)] = peak.best_fit # Use string profile type as key

            if not fits_to_plot: continue

            for profile_name, fit in fits_to_plot.items():
                 if not all(hasattr(fit, attr) for attr in ['center', 'amplitude', 'width']): logging.warning(f"Skipping fit {profile_name} for peak {peak_index}: Missing essential attributes."); continue
                 if not all(np.isfinite(getattr(fit, attr, np.nan)) for attr in ['center', 'amplitude', 'width']): logging.warning(f"Skipping fit {profile_name} for peak {peak_index}: Non-finite parameters."); continue

                 center = fit.center
                 fwhm = getattr(fit, 'fwhm', None); width = fit.width
                 plot_range_multiplier = 3.0; half_plot_width = 0.5

                 if fwhm is not None and np.isfinite(fwhm) and fwhm > 1e-6: half_plot_width = fwhm * plot_range_multiplier / 2.0
                 elif width is not None and np.isfinite(width) and width > 1e-9: half_plot_width = width * 1.5 * plot_range_multiplier / 2.0
                 else: logging.debug(f"Using default plot width for fit P{peak_index}/{profile_name}")

                 min_wl, max_wl = center - half_plot_width, center + half_plot_width
                 mask = (plot_wl >= min_wl) & (plot_wl <= max_wl)
                 x_fit = plot_wl[mask]
                 if len(x_fit) < 5: x_fit = np.linspace(min_wl, max_wl, 100)
                 if len(x_fit) == 0: continue

                 y_fit_relative = self._generate_fit_y(fit, x_fit)
                 if y_fit_relative is None: continue

                 if baseline_y is not None:
                      try: y_baseline_interp = np.interp(x_fit, plot_wl, baseline_y); y_fit_absolute = y_fit_relative + y_baseline_interp
                      except Exception as interp_e: logging.warning(f"Error interpolating baseline for fit P{peak_index}/{profile_name}: {interp_e}. Plotting relative to zero."); y_fit_absolute = y_fit_relative
                 else: y_fit_absolute = y_fit_relative

                 is_highlighted = (highlight_fit is not None and fit == highlight_fit)
                 color = highlight_color if is_highlighted else default_fit_color
                 ls = '-' if is_highlighted else '-.'; lw = 1.5 if is_highlighted else 0.8
                 alpha = 1.0 if is_highlighted else 0.7; zorder = 7 if is_highlighted else 4

                 try:
                      line_key = (peak_index, profile_name)
                      line, = self.ax.plot(x_fit, y_fit_absolute, color=color, ls=ls, lw=lw, alpha=alpha, zorder=zorder, label='_nolegend_')
                      line.fit_result_ref = fit # Store reference
                      if not hasattr(fit, 'peak_index'): fit.peak_index = peak_index
                      new_fits_dict[line_key] = line
                      num_plotted += 1
                 except Exception as plot_e: logging.error(f"Error plotting fit line P{peak_index}/{profile_name}: {plot_e}", exc_info=True)

        self._plot_elements['fits'] = new_fits_dict
        if num_plotted > 0: logging.info(f"Plotted {num_plotted} individual fit lines.")
        if highlight_fit: logging.info(f"Highlighted fit: Peak {getattr(highlight_fit, 'peak_index', 'N/A')} / {getattr(highlight_fit, 'profile_type', 'N/A')}")
        self._redraw_canvas()

    # Use STRING LITERAL 'FitResult' for type hint
    def highlight_fit_line(self, fit_result: Optional['FitResult']):
        """ Highlights a specific fit line by replotting all fits with new styling. """
        if not CORE_MODULES_AVAILABLE: logging.error("Cannot highlight fit line: Core modules not loaded."); return
        if fit_result is not None and not isinstance(fit_result, FitResult): logging.warning(f"highlight_fit_line called with invalid type: {type(fit_result)}. Clearing highlight."); fit_result = None

        current_highlight = self._highlighted_fit_result
        if fit_result != current_highlight:
            logging.info(f"Changing fit highlight to: {fit_result if fit_result else 'None'}")
            self.plot_fit_lines(self._peaks_ref, highlight_fit=fit_result)
        else:
            logging.debug(f"Fit highlight requested ({fit_result}) is same as current. No change.")

    # Use STRING LITERAL 'NISTMatch' for type hint
    def plot_nist_matches(self, matches: List['NISTMatch'], clear_previous: bool = True):
        """ Plots vertical lines and labels for NIST database matches. """
        if not CORE_MODULES_AVAILABLE: logging.error("Cannot plot NIST matches: Core modules not loaded."); return
        if not isinstance(matches, list): logging.error(f"plot_nist_matches expects a list, got {type(matches)}. Ignoring."); return

        if clear_previous: self.clear_nist_matches()
        if not matches: logging.debug("plot_nist_matches called with empty list."); self._redraw_canvas(); return

        self._matches_ref = matches
        logging.info(f"Plotting {len(matches)} NIST matches.")

        lines_segments = []; lines_colors = []; new_nist_annotations: List[MplText] = []

        try:
            current_ylim = self.ax.get_ylim()
            if current_ylim == (0.0, 1.0) and not self.ax.has_data():
                y_max_guess = 1.0
                for key in ['proc', 'raw']:
                    line = self._plot_elements.get(key)
                    if isinstance(line, Line2D) and len(line.get_ydata()) > 0:
                        try: y_max_guess = max(y_max_guess, np.nanmax(line.get_ydata()))
                        except ValueError: pass
                current_ylim = (self.ax.dataLim.y0 if self.ax.dataLim.y0 < y_max_guess else 0, y_max_guess * 1.1)
            yrange = current_ylim[1] - current_ylim[0]; yrange = 1.0 if yrange <= 0 else yrange
            line_y_start = current_ylim[0]; line_y_end = current_ylim[0] + yrange * 0.85; label_y_pos = current_ylim[0] + yrange * 0.90
        except Exception as e: logging.error(f"Could not determine Y limits for NIST lines: {e}. Using relative defaults."); line_y_start, line_y_end, label_y_pos = 0.0, 0.85, 0.90

        unique_elements = sorted(list(set(getattr(m, 'element', None) for m in matches if hasattr(m, 'element') and m.element)))
        color_map = {}; default_color = 'grey'
        if unique_elements:
             try: cmap = plt.get_cmap('tab10'); num_colors = min(cmap.N, 10); color_map = {elem: cmap(i % num_colors) for i, elem in enumerate(unique_elements)}
             except Exception as e: logging.warning(f"Could not get colormap for NIST elements: {e}. Using default color.")

        num_plotted = 0
        for match in matches:
            if not isinstance(match, NISTMatch) or not hasattr(match, 'wavelength_db'): continue
            wl_db = match.wavelength_db;
            if not np.isfinite(wl_db): continue

            lines_segments.append([(wl_db, line_y_start), (wl_db, line_y_end)])
            element = getattr(match, 'element', None)
            color = color_map.get(element, default_color) if element else default_color
            lines_colors.append(color)
            ion_state_str = getattr(match, 'ion_state_str', '')
            label = f"{element} {ion_state_str}".strip() if element else f"? {ion_state_str}".strip()

            try: txt = self.ax.text(wl_db, label_y_pos, label, rotation=90, ha='center', va='bottom', fontsize=7, color=color, clip_on=True); new_nist_annotations.append(txt); num_plotted += 1
            except Exception as text_e: logging.error(f"Failed to create NIST text annotation for '{label}' at {wl_db:.2f}: {text_e}")

        if lines_segments:
            try:
                line_collection = LineCollection(lines_segments, colors=lines_colors, linewidths=0.7, alpha=0.8, label='_nolegend_', zorder=0)
                self.ax.add_collection(line_collection)
                self._plot_elements['nist_lines'] = line_collection; self._nist_annotations = new_nist_annotations
                logging.debug(f"Added {num_plotted} NIST lines/annotations via LineCollection.")
            except Exception as e:
                logging.error(f"Failed to add NIST LineCollection to plot: {e}", exc_info=True);
                for ann in new_nist_annotations: ann.remove(); self._nist_annotations = []
        else: logging.debug("No valid NIST match data resulted in plottable lines.")
        self._redraw_canvas()


    def clear_nist_matches(self):
        """Removes NIST match lines and their associated text labels."""
        logging.debug("Clearing NIST matches from plot.")
        self._remove_artist('nist_lines')
        self._matches_ref = []


    def highlight_peak(self, peak_list_index: Optional[int]):
        """ Highlights a specific peak marker using its index in the original list. """
        self._remove_artist('highlight')
        self._highlighted_peak_list_index = peak_list_index

        if peak_list_index is None: logging.debug("Clearing peak highlight."); self._redraw_canvas(); return
        if not isinstance(self._peaks_ref, list) or not (0 <= peak_list_index < len(self._peaks_ref)): logging.warning(f"Cannot highlight peak: Index {peak_list_index} invalid (size {len(self._peaks_ref)})."); self._redraw_canvas(); return
        peak = self._peaks_ref[peak_list_index];
        if not isinstance(peak, Peak) or not hasattr(peak, 'wavelength_fitted_or_detected'): logging.warning(f"Cannot highlight peak index {peak_list_index}: Invalid Peak object."); self._redraw_canvas(); return

        highlight_x = peak.wavelength_fitted_or_detected
        line_for_y = self._plot_elements.get('proc') or self._plot_elements.get('raw')
        if not isinstance(line_for_y, Line2D): logging.warning(f"Cannot determine highlight Y-coordinate for peak index {peak_list_index}: Base spectrum line not found."); self._redraw_canvas(); return

        highlight_y = np.nan
        try:
            wl_data, int_data = line_for_y.get_xdata(), line_for_y.get_ydata()
            if len(wl_data) > 0: highlight_y = np.interp(highlight_x, wl_data, int_data, left=np.nan, right=np.nan)
        except Exception as e: logging.error(f"Error interpolating highlight Y-coordinate: {e}", exc_info=True)

        if np.isfinite(highlight_x) and np.isfinite(highlight_y):
            plot_colors = self.config.get('plotting', {}); highlight_color = plot_colors.get('highlight_peak_color', 'yellow')
            try:
                highlight_scatter = self.ax.scatter([highlight_x], [highlight_y], marker='o', s=150, facecolors='none', edgecolors=highlight_color, lw=1.5, zorder=10, label='_nolegend_')
                self._plot_elements['highlight'] = highlight_scatter
                logging.info(f"Highlighted peak list index {peak_list_index} at ({highlight_x:.4f}, {highlight_y:.2f})")
            except Exception as e: logging.error(f"Error plotting highlight marker: {e}", exc_info=True)
        else: logging.warning(f"Cannot highlight peak index {peak_list_index}: Invalid/NaN coordinates (X={highlight_x}, Y={highlight_y}).")
        self._redraw_canvas()


    # --- Helper Methods ---

    # Use STRING LITERAL 'FitResult' for type hint
    def _generate_fit_y(self, fit: 'FitResult', x: np.ndarray) -> Optional[np.ndarray]:
        """ Calculates the Y values for a given FitResult object over an array of X values. """
        if not CORE_MODULES_AVAILABLE:
             if not hasattr(self, '_logged_missing_core_func_error'): logging.error("Cannot generate fit Y: Core functions unavailable."); self._logged_missing_core_func_error = True
             return None
        if not isinstance(fit, FitResult): logging.warning("Cannot generate fit Y: Invalid FitResult."); return None

        try:
            amp = getattr(fit, 'amplitude', np.nan); cen = getattr(fit, 'center', np.nan); wid = getattr(fit, 'width', np.nan)
            prof = getattr(fit, 'profile_type', None); prof_str = str(prof) # Use string representation for checks

            if not prof: logging.warning(f"Cannot generate fit Y: Missing profile type."); return None
            if not (np.isfinite(amp) and np.isfinite(cen)): logging.warning(f"Cannot generate fit Y for '{prof_str}': Invalid amp ({amp}) or cen ({cen})."); return None

            # Determine the actual function based on the profile type (might be Enum or string)
            if prof == ProfileType.GAUSSIAN or prof_str == "Gaussian":
                if not np.isfinite(wid) or wid <= 0: logging.warning(f"Invalid width (sigma={wid}) for Gaussian."); return None
                return gaussian(x, amp, cen, wid)
            elif prof == ProfileType.LORENTZIAN or prof_str == "Lorentzian":
                if not np.isfinite(wid) or wid <= 0: logging.warning(f"Invalid width (gamma={wid}) for Lorentzian."); return None
                return lorentzian(x, amp, cen, wid)
            elif prof == ProfileType.PSEUDO_VOIGT or prof_str == "PseudoVoigt":
                 # PseudoVoigt function uses sigma and eta
                 eta = getattr(fit, 'mixing_param_eta', np.nan)
                 sigma = wid # Assuming 'width' stored in FitResult for PV is sigma
                 if not (np.isfinite(sigma) and np.isfinite(eta)) or sigma <= 0: logging.warning(f"Invalid parameters for PseudoVoigt (sigma={sigma}, eta={eta})."); return None
                 eta_clipped = np.clip(eta, 0.0, 1.0); # Ensure eta is in [0, 1]
                 if eta != eta_clipped: logging.debug(f"Clipped PseudoVoigt eta from {eta} to {eta_clipped}.")
                 return pseudo_voigt(x, amp, cen, sigma, eta_clipped)
            else: logging.warning(f"Cannot generate fit Y: Unknown profile type '{prof_str}'."); return None

        except Exception as e: logging.error(f"Error generating Y-values for fit '{getattr(fit, 'profile_type', 'N/A')}': {e}", exc_info=True); return None


    def _update_legend(self, specific_handles: Optional[List[Artist]] = None):
        """ Refreshes the plot legend. """
        try:
            handles_to_consider: List[Artist] = []; labels_to_consider: List[str] = []
            if specific_handles is not None:
                 handles_to_consider = specific_handles
                 labels_to_consider = [getattr(h, 'get_label', lambda: '')() for h in handles_to_consider]
            else: handles_to_consider, labels_to_consider = self.ax.get_legend_handles_labels()

            valid_legend_items: Dict[str, Artist] = {}
            for handle, label in zip(handles_to_consider, labels_to_consider):
                if handle and label and not label.startswith('_'): valid_legend_items[label] = handle

            old_legend = self._plot_elements.get('legend');
            if isinstance(old_legend, Legend):
                try: old_legend.remove()
                except Exception as e: logging.debug(f"Issue removing previous legend: {e}")
            self._plot_elements['legend'] = None

            if valid_legend_items:
                new_legend = self.ax.legend(list(valid_legend_items.values()), list(valid_legend_items.keys()), fontsize='small', loc='best')
                self._plot_elements['legend'] = new_legend
                self._apply_legend_theme()
                logging.debug(f"Legend updated with labels: {list(valid_legend_items.keys())}")
            else: logging.debug("No valid items for legend.")
        except Exception as e: logging.error(f"Failed to update plot legend: {e}", exc_info=True)


    def _apply_legend_theme(self):
        """Applies theme colors (background, text) to the current legend frame and text."""
        legend = self._plot_elements.get('legend')
        if not isinstance(legend, Legend): return
        try:
            # Basic theme detection (improve if needed)
            bg = plt.rcParams.get('figure.facecolor', '#FFFFFF')
            is_dark = matplotlib.colors.to_rgb(bg)[0] < 0.5
            text_color = 'white' if is_dark else 'black'
            bg_color_str = '#333333' if is_dark else '#FFFFFF'
            edge_color_str = '#555555' if is_dark else '#CCCCCC'

            # Use config overrides if available
            plotting_cfg = self.config.get('plotting', {})
            text_color = plotting_cfg.get('legend_text_color', text_color)
            bg_color_str = plotting_cfg.get('legend_background_color', bg_color_str)
            edge_color_str = plotting_cfg.get('legend_edge_color', edge_color_str)

            frame = legend.get_frame()
            bg_color_rgba = matplotlib.colors.to_rgba(bg_color_str, alpha=0.85)
            frame.set(facecolor=bg_color_rgba, edgecolor=edge_color_str, linewidth=0.5)
            for text in legend.get_texts(): text.set_color(text_color)
        except Exception as e: logging.warning(f"Could not apply theme settings to legend: {e}", exc_info=True)


    def _update_annotation(self, target_element: Artist, data_info: Dict):
        """ Updates the text and position of the hover annotation box. """
        try:
            pos = (data_info.get('x', 0), data_info.get('y', 0)); self.annot.xy = pos; text = ""
            if isinstance(target_element, Line2D):
                label = getattr(target_element, 'get_label', lambda: '')().lower()
                prefix = "Data"
                if label.startswith("raw"): prefix = "Raw"
                elif label.startswith("proc"): prefix = "Processed"
                elif label.startswith("base"): prefix = "Baseline"
                elif hasattr(target_element, 'fit_result_ref'):
                     fit_ref = getattr(target_element, 'fit_result_ref'); profile = getattr(fit_ref,'profile_type', 'Unknown') if fit_ref else 'Unknown'; prefix = f"Fit ({profile})"
                elif label and not label.startswith('_'): prefix = label.capitalize()
                text = f"{prefix}\nλ: {pos[0]:.4f}\nI: {pos[1]:.2f}"
            elif isinstance(target_element, PathCollection):
                label = getattr(target_element, 'get_label', lambda: '')().lower(); prefix = "Peak"
                if label.startswith("det"): prefix = "Detected Peak"
                elif label.startswith("fit"): prefix = "Fitted Peak"
                elif label and '(' in label: prefix = label.split('(')[0].strip().capitalize()

                peak_info = data_info.get('peak_info')
                if isinstance(peak_info, dict):
                    det_wav = peak_info.get('Detected Wavelength (nm)', np.nan); fit_cen = peak_info.get('Fitted Center (nm)', np.nan)
                    fit_fwhm = peak_info.get('Fitted FWHM (nm)', np.nan); proc_int = peak_info.get('Processed Intensity', np.nan)
                    peak_height = peak_info.get('Fitted Amplitude', np.nan); prof = peak_info.get('Fit Profile', '')
                    text = f"{prefix}"; text += f"\nλ Detect: {det_wav:.4f}" if np.isfinite(det_wav) else f"\nλ Hover: {pos[0]:.4f}"; text += f"\nI Detect: {proc_int:.2f}" if np.isfinite(proc_int) else f"\nI Hover: {pos[1]:.2f}"
                    if prof and isinstance(prof, str) and prof != '': text += f"\nFit: {prof}";
                    if np.isfinite(fit_cen): text += f"\n Fit λ: {fit_cen:.4f}";
                    if np.isfinite(peak_height): text += f"\n Fit Amp: {peak_height:.2f}";
                    if np.isfinite(fit_fwhm): text += f"\n Fit FWHM: {fit_fwhm:.4f}"
                else: text = f"{prefix}\nλ: {pos[0]:.4f}\nI: {pos[1]:.2f}"

            if text: self.annot.set_text(text); self.annot.set_visible(True)
            else: self.annot.set_visible(False)
        except Exception as e: logging.error(f"Error updating annotation: {e}", exc_info=True); self.annot.set_visible(False)

    # --- Event Handlers ---

    def _on_hover(self, event):
        """ Handles mouse motion events over the canvas for hover effects. """
        if not event.inaxes == self.ax:
            if self.annot.get_visible(): self.annot.set_visible(False); self._redraw_canvas()
            return

        target_element: Optional[Artist] = None; info: Dict[str, Any] = {}; min_dist_sq = float('inf')
        peak_scatters = [s for s in [self._plot_elements.get('det'), self._plot_elements.get('fit')] if isinstance(s, PathCollection)]
        for scatter in peak_scatters:
            contains, ind_dict = scatter.contains(event)
            if contains:
                scatter_indices = ind_dict['ind']
                if len(scatter_indices) > 0:
                    offsets = scatter.get_offsets(); points_in_radius = offsets[scatter_indices]
                    distances = np.sum((points_in_radius - [event.xdata, event.ydata])**2, axis=1); closest_idx_in_subset = np.argmin(distances)
                    scatter_index = scatter_indices[closest_idx_in_subset]
                    if scatter_index < len(offsets):
                        target_element = scatter; pos = offsets[scatter_index]; info = {'x': pos[0], 'y': pos[1]}
                        peak_list_indices = getattr(scatter, 'peak_list_indices', None)
                        if peak_list_indices and isinstance(peak_list_indices, list) and scatter_index < len(peak_list_indices):
                            original_list_index = peak_list_indices[scatter_index]
                            if 0 <= original_list_index < len(self._peaks_ref):
                                peak_obj = self._peaks_ref[original_list_index]
                                if peak_obj and hasattr(peak_obj, 'to_dataframe_row'): info['peak_info'] = peak_obj.to_dataframe_row()
                                else: info['peak_info'] = {'Index': original_list_index}
                        break # Found scatter point
        if target_element is None:
            lines_to_check: List[Line2D] = []
            for key in ['proc', 'raw', 'base']:
                 line = self._plot_elements.get(key);
                 if isinstance(line, Line2D): lines_to_check.append(line)
            fits_dict = self._plot_elements.get('fits')
            if isinstance(fits_dict, dict): lines_to_check.extend(l for l in fits_dict.values() if isinstance(l, Line2D))
            for line in lines_to_check:
                 contains, ind_dict = line.contains(event, radius=5)
                 if contains:
                     x_data, y_data = line.get_data(); indices = ind_dict['ind']
                     if len(indices) > 0:
                          distances_sq = (x_data[indices] - event.xdata)**2 + (y_data[indices] - event.ydata)**2
                          closest_local_idx = np.argmin(distances_sq); closest_dist_sq = distances_sq[closest_local_idx]
                          if closest_dist_sq < min_dist_sq:
                               min_dist_sq = closest_dist_sq; target_element = line
                               global_index = indices[closest_local_idx]
                               info = {'x': x_data[global_index], 'y': y_data[global_index]}

        if target_element: self._update_annotation(target_element, info); self._redraw_canvas()
        elif self.annot.get_visible(): self.annot.set_visible(False); self._redraw_canvas()

    def _on_pick(self, event):
        """ Handles pick events (typically clicks) on artists with 'picker' enabled. """
        artist = event.artist; indices = event.ind
        if isinstance(artist, PathCollection) and hasattr(artist, 'peak_list_indices') and indices:
            scatter_index = indices[0]; peak_list_indices = artist.peak_list_indices
            if peak_list_indices and isinstance(peak_list_indices, list) and scatter_index < len(peak_list_indices):
                original_list_index = peak_list_indices[scatter_index]
                logging.info(f"Peak clicked: Scatter Index={scatter_index}, Original Peak List Index={original_list_index}")
                self.peak_clicked.emit(original_list_index)
            else: logging.warning(f"Picked scatter point (index {scatter_index}), but could not map back. Artist: {artist.get_label()}")

    # --- Theming and Redrawing ---

    def apply_theme_colors(self, config: Optional[Dict] = None):
        """ Applies color theme settings from the configuration to plot elements. """
        if config is None: config = self.config
        if not config: logging.debug("apply_theme_colors: No config provided, using rcParams."); pass # Allow using rcParams only
        else: logging.debug("Applying theme colors from configuration.")

        try:
            style_cfg = config.get('style', {}); plotting_cfg = config.get('plotting', {})
            if not plotting_cfg and 'plotting' in style_cfg: plotting_cfg = style_cfg.get('plotting', {})
            theme_name = style_cfg.get('default_theme', None); is_dark = False
            if theme_name: is_dark = 'dark' in theme_name.lower()
            else: default_bg = plt.rcParams.get('figure.facecolor', '#FFFFFF'); is_dark = matplotlib.colors.to_rgb(default_bg)[0] < 0.5
            logging.debug(f"Theme detected as {'dark' if is_dark else 'light'}.")

            def get_color(key: str, default_light: str, default_dark: str) -> str:
                 config_val = plotting_cfg.get(key); rc_val = plt.rcParams.get(f'cosmic.{key}', plt.rcParams.get(key.replace('_','.')))
                 return config_val or rc_val or (default_dark if is_dark else default_light) # type: ignore

            bg_color = get_color('background_color', '#FFFFFF', '#2E2E2E'); text_color = get_color('text_color', '#000000', '#FFFFFF')
            grid_color = get_color('grid_color', '#CCCCCC', '#444444'); raw_c = get_color('raw_data_color', '#AAAAAA', '#888888')
            proc_c = get_color('processed_data_color', '#0000FF', '#00FFFF'); base_c = get_color('baseline_color', '#FFA500', '#FF8C00')
            det_c = get_color('peak_detected_color', '#FF0000', '#FF4500'); fit_c = get_color('peak_fitted_color', '#00FF00', '#ADFF2F')
            fit_line_c = get_color('fit_line_color', '#FF00FF', '#DA70D6'); hl_peak_c = get_color('highlight_peak_color', '#FFFF00', '#FFFF00')
            hl_fit_c = get_color('highlight_fit_color', '#FFFF00', '#FFFFE0'); annot_bg_c = get_color('annotation_background_color', '#FFFFE0', '#3C3F41')
            annot_text_c = get_color('annotation_text_color', '#000000', '#FFFFFF')

            self.figure.set_facecolor(bg_color); self.ax.set_facecolor(bg_color)
            for spine in self.ax.spines.values(): spine.set_color(text_color)
            self.ax.xaxis.label.set_color(text_color); self.ax.yaxis.label.set_color(text_color)
            self.ax.tick_params(axis='x', colors=text_color); self.ax.tick_params(axis='y', colors=text_color)
            self.ax.title.set_color(text_color); self.ax.grid(True, color=grid_color, linestyle=':', alpha=0.6)

            elements = self._plot_elements
            if isinstance(elements.get('raw'), Line2D): elements['raw'].set_color(raw_c) # type: ignore
            if isinstance(elements.get('proc'), Line2D): elements['proc'].set_color(proc_c) # type: ignore
            if isinstance(elements.get('base'), Line2D): elements['base'].set_color(base_c) # type: ignore
            if isinstance(elements.get('det'), PathCollection): elements['det'].set_color(det_c) # type: ignore
            if isinstance(elements.get('fit'), PathCollection): elements['fit'].set_edgecolor(fit_c); elements['fit'].set_facecolor('none') # type: ignore
            if isinstance(elements.get('highlight'), PathCollection): elements['highlight'].set_edgecolor(hl_peak_c) # type: ignore

            fits_dict = elements.get('fits');
            if isinstance(fits_dict, dict):
                 highlight_ref = self._highlighted_fit_result
                 for line in fits_dict.values():
                     is_highlighted = (highlight_ref is not None and hasattr(line, 'fit_result_ref') and line.fit_result_ref == highlight_ref)
                     line.set_color(hl_fit_c if is_highlighted else fit_line_c)

            self._apply_legend_theme()
            if self.annot:
                 self.annot.get_bbox_patch().set(facecolor=annot_bg_c, edgecolor=annot_text_c, alpha=0.85); self.annot.set_color(annot_text_c)
                 if hasattr(self.annot, 'arrow_patch') and self.annot.arrow_patch: self.annot.arrow_patch.set_color(annot_text_c)

            self._redraw_canvas(); logging.debug("Theme colors applied.")
        except Exception as e: logging.error(f"Error applying theme colors to plot: {e}", exc_info=True)

    def _redraw_canvas(self):
        """ Requests an idle redraw of the Matplotlib canvas. """
        try: self.canvas.draw_idle()
        except Exception as e: logging.error(f"Error occurred during canvas redraw: {e}", exc_info=True)