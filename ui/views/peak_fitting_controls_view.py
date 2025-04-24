# --- START OF REFACTORED FILE libs_cosmic_forge/ui/views/peak_fitting_controls_view.py ---
"""
Control Panel View for Peak Fitting settings and actions using CollapsibleBox.

Provides controls for global fitting parameters, triggers fitting for all peaks,
displays detailed fit results (multiple profiles) for a selected peak,
highlights the best fit, allows adjusting the ROI for the selected peak,
and triggers refitting for only the selected peak.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QPushButton, QLabel, QHBoxLayout, QComboBox,
                             QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QMessageBox, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt, QVariant, pyqtSlot # Added QVariant
from PyQt6.QtGui import QFont, QColor, QBrush

from ui.widgets.info_button import InfoButton
from ui.widgets.collapsible_box import CollapsibleBox
from core.data_models import Peak, FitResult

class PeakFittingControlPanel(QWidget):
    """Control panel for peak fitting using collapsible sections."""

    # Signal emitted to fit all detected peaks with current global settings
    fit_peaks_triggered = pyqtSignal(dict)
    # Signal emitted to refit a single peak
    # Sends: peak_list_index (int), settings (dict, potentially with specific ROI)
    refit_single_peak_requested = pyqtSignal(int, dict)
    # Signal emitted when a user selects a specific fit profile row in the details table
    # Sends: fit_result (FitResult object)
    show_specific_fit = pyqtSignal(object) # Use object for FitResult type

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config.get('peak_fitting', {}) # Get sub-config for fitting
        self.current_peak_data: Optional[Peak] = None
        self._current_peak_list_index: Optional[int] = None # Store index for refitting
        self.current_model_selection: str = self.config.get('model_selection', 'AIC')
        self._dark_mode: bool = False # Track theme for background colors

        self._init_ui()
        self._load_defaults()
        self.results_box.setVisible(False) # Initially hide results until a peak is selected

    def _init_ui(self):
        """Initializes the UI components and layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8) # Increased spacing slightly

        # --- Global Fitting Parameters ---
        self.fit_params_box = CollapsibleBox("Global Fitting Parameters", self)
        fit_params_content = QWidget()
        fit_params_layout = QFormLayout(fit_params_content)
        fit_params_layout.setContentsMargins(8, 8, 8, 8) # Padding inside box
        fit_params_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        fit_params_layout.setHorizontalSpacing(10)
        fit_params_layout.setVerticalSpacing(8)

        # ROI Factor
        self.roi_factor_dspin = QDoubleSpinBox()
        self.roi_factor_dspin.setRange(2.0, 20.0); self.roi_factor_dspin.setDecimals(1)
        self.roi_factor_dspin.setSingleStep(0.5); self.roi_factor_dspin.setToolTip("Default fitting Region of Interest (ROI) width = Factor × Estimated FWHM.")
        fit_params_layout.addRow("Default ROI Factor:", self.roi_factor_dspin)

        # Min ROI Width
        self.min_roi_width_dspin = QDoubleSpinBox()
        self.min_roi_width_dspin.setRange(0.01, 5.0); self.min_roi_width_dspin.setDecimals(2)
        self.min_roi_width_dspin.setSingleStep(0.05); self.min_roi_width_dspin.setSuffix(" nm")
        self.min_roi_width_dspin.setToolTip("Minimum width for the default fitting ROI.")
        fit_params_layout.addRow("Default Min ROI Width:", self.min_roi_width_dspin)

        # Model Selection (AIC/BIC)
        ms_hbox = QHBoxLayout()
        self.model_select_combo = QComboBox()
        self.model_select_combo.addItems(["AIC", "BIC"]) # Akaike / Bayesian Information Criterion
        self.model_select_combo.setToolTip("Criterion used to automatically select the 'best' fit profile among alternatives (lower score is better).")
        self.model_select_combo.currentTextChanged.connect(self._update_model_selection_criterion)
        ms_hbox.addWidget(self.model_select_combo)
        ms_hbox.addWidget(InfoButton(self._show_model_select_info, "Model Selection Help", self))
        fit_params_layout.addRow("Best Fit Criterion:", ms_hbox)

        # Max Iterations
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(100, 10000); self.max_iter_spin.setSingleStep(100)
        self.max_iter_spin.setToolTip("Maximum number of iterations allowed for the fitting algorithm (scipy.optimize.curve_fit).")
        fit_params_layout.addRow("Max Iterations:", self.max_iter_spin)

        self.fit_params_box.setContentLayout(fit_params_layout) # Use setContent for QFormLayout
        main_layout.addWidget(self.fit_params_box)

        # --- Fit All Button ---
        self.fit_button = QPushButton("Fit All Detected Peaks")
        self.fit_button.setToolTip("Apply fitting using the above global parameters to all currently detected peaks.")
        self.fit_button.clicked.connect(self._emit_fit_all_signal)
        main_layout.addWidget(self.fit_button)

        # --- Selected Peak Details & Refit Box ---
        self.results_box = CollapsibleBox("Selected Peak Details & Refit", self)
        results_content = QWidget()
        results_layout = QVBoxLayout(results_content)
        results_layout.setContentsMargins(8, 8, 8, 8)
        results_layout.setSpacing(8)

        # Label for selected peak info
        self.selected_peak_label = QLabel("Select a peak from the list or plot to view details.")
        self.selected_peak_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_peak_label.setStyleSheet("font-style: italic; padding: 5px;")
        results_layout.addWidget(self.selected_peak_label)

        # Table for fit results of the selected peak
        self.fit_details_table = QTableWidget()
        # Define columns clearly - ensure order matches population logic
        self.fit_details_columns = ["Profile", "Amplitude", "Amp Err", "Center", "Cen Err", "Width", "Wid Err", "FWHM/Mix", "R²", "Score"]
        self.fit_details_table.setColumnCount(len(self.fit_details_columns))
        self.fit_details_table.setHorizontalHeaderLabels(self.fit_details_columns)
        self.fit_details_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) # Read-only
        self.fit_details_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows) # Select whole row
        self.fit_details_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection) # Only one row selectable
        self.fit_details_table.verticalHeader().setVisible(False) # Hide row numbers
        self.fit_details_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents) # Auto-resize cols initially
        # self.fit_details_table.horizontalHeader().setStretchLastSection(True) # ResizeToContents often better with many cols
        self.fit_details_table.setMinimumHeight(110) # Adjust height
        self.fit_details_table.setMaximumHeight(120)
        self.fit_details_table.itemSelectionChanged.connect(self._handle_fit_profile_selection) # Signal for row selection
        results_layout.addWidget(self.fit_details_table)

        # GroupBox for ROI adjustment and Refit button
        self.roi_adjust_group = QGroupBox("Adjust ROI & Refit Selected Peak")
        self.roi_adjust_group.setToolTip("Manually define the wavelength range (ROI) used for fitting only this specific peak.")
        roi_layout = QFormLayout(self.roi_adjust_group)
        roi_layout.setSpacing(6)
        roi_layout.setContentsMargins(8, 10, 8, 8)

        self.roi_min_wl_dspin = QDoubleSpinBox()
        self.roi_min_wl_dspin.setDecimals(4); self.roi_min_wl_dspin.setRange(0, 3000) # Wider range
        self.roi_min_wl_dspin.setSuffix(" nm"); self.roi_min_wl_dspin.setToolTip("Manually set the start wavelength for the fitting ROI.")
        self.roi_min_wl_dspin.setKeyboardTracking(False) # Emit valueChanged only when focus lost or Enter pressed
        self.roi_min_wl_dspin.setEnabled(False) # Disabled until ROI data is loaded

        self.roi_max_wl_dspin = QDoubleSpinBox()
        self.roi_max_wl_dspin.setDecimals(4); self.roi_max_wl_dspin.setRange(0, 3000) # Wider range
        self.roi_max_wl_dspin.setSuffix(" nm"); self.roi_max_wl_dspin.setToolTip("Manually set the end wavelength for the fitting ROI.")
        self.roi_max_wl_dspin.setKeyboardTracking(False)
        self.roi_max_wl_dspin.setEnabled(False) # Disabled until ROI data is loaded

        self.refit_button = QPushButton("Refit Selected Peak")
        self.refit_button.setToolTip("Refit only this peak using the global fitting parameters but with the specific ROI defined above.")
        self.refit_button.clicked.connect(self._emit_refit_signal)
        self.refit_button.setEnabled(False) # Disabled until ROI data is loaded

        roi_layout.addRow("ROI Min:", self.roi_min_wl_dspin)
        roi_layout.addRow("ROI Max:", self.roi_max_wl_dspin)
        roi_layout.addRow(self.refit_button)
        results_layout.addWidget(self.roi_adjust_group)
        self.roi_adjust_group.setVisible(False) # Hide initially

        self.results_box.setContentLayout(results_layout) # Use setContent
        main_layout.addWidget(self.results_box)

        main_layout.addStretch() # Push content to the top
        self.setLayout(main_layout)

    def _load_defaults(self):
        """Loads default values from config into UI widgets."""
        logging.debug("Loading peak fitting default parameters.")
        self.roi_factor_dspin.setValue(self.config.get('roi_factor', 7.0))
        self.min_roi_width_dspin.setValue(self.config.get('min_roi_width_nm', 0.1))
        self.current_model_selection = self.config.get('model_selection', 'AIC')
        self.model_select_combo.setCurrentText(self.current_model_selection)
        self.max_iter_spin.setValue(self.config.get('max_iterations', 2000))
        self._update_score_column_header() # Set initial header

    def _update_model_selection_criterion(self, criterion: str):
        """Updates the internal state when the model selection combo changes."""
        if self.current_model_selection == criterion: return # No change
        self.current_model_selection = criterion
        logging.debug(f"Model selection criterion changed to: {criterion}")
        self._update_score_column_header()
        # Re-display details to update scores and potentially highlighting
        # Check if peak data exists before attempting to redisplay
        if self.current_peak_data is not None:
            self.display_peak_fit_details(self.current_peak_data, self._current_peak_list_index)

    def _update_score_column_header(self):
        """Updates the header of the 'Score' column in the details table."""
        try:
            # Find column index by name
            score_col_idx = self.fit_details_columns.index("Score")
            header_item = QTableWidgetItem(f"{self.current_model_selection}")
            # Optional: Add tooltip to header
            header_item.setToolTip(f"{self.current_model_selection} Score (lower is better)")
            self.fit_details_table.setHorizontalHeaderItem(score_col_idx, header_item)
        except ValueError:
            logging.error("Could not find 'Score' column in fit_details_columns to update header.")
        except Exception as e:
             logging.error(f"Error updating score column header: {e}", exc_info=True)

    def get_fitting_settings(self) -> dict:
        """Collects current global fitting settings from the UI."""
        settings = {
            # Profiles to attempt fitting with. Could be made configurable via UI/config later.
            'profiles_to_fit': ['Gaussian', 'Lorentzian', 'PseudoVoigt'],
            'roi_factor': self.roi_factor_dspin.value(),
            'min_roi_width_nm': self.min_roi_width_dspin.value(),
            'model_selection': self.model_select_combo.currentText(),
            'max_fit_iterations': self.max_iter_spin.value(),
            # 'baseline_mode' is determined within fit_peak based on context or options
        }
        logging.debug(f"Gathered fitting settings: {settings}")
        return settings

    @pyqtSlot()
    def _emit_fit_all_signal(self):
        """Emits the signal to fit all peaks."""
        settings = self.get_fitting_settings()
        logging.info("Fit All Peaks button clicked. Emitting signal.")
        self.fit_peaks_triggered.emit(settings)

    @pyqtSlot()
    def _emit_refit_signal(self):
        """Emits the signal to refit the currently selected peak with current settings and specified ROI."""
        if self.current_peak_data is None or self._current_peak_list_index is None:
            QMessageBox.warning(self, "No Peak Selected", "Please select a peak from the list or plot before refitting.")
            return

        settings = self.get_fitting_settings()
        roi_min = self.roi_min_wl_dspin.value()
        roi_max = self.roi_max_wl_dspin.value()

        # Validate the manual ROI
        if roi_min < roi_max:
            settings['roi_wavelengths'] = [roi_min, roi_max]
            # Remove default ROI params if providing explicit wavelengths
            settings.pop('roi_factor', None)
            settings.pop('min_roi_width_nm', None)
            logging.info(f"Requesting refit for Peak List Index {self._current_peak_list_index} "
                         f"(Peak Index: {self.current_peak_data.index}) "
                         f"with manual ROI [{roi_min:.4f}, {roi_max:.4f}] nm.")
            self.refit_single_peak_requested.emit(self._current_peak_list_index, settings)
        else:
            # User entered invalid range
            QMessageBox.warning(self, "Invalid ROI",
                                "The specified ROI Min wavelength must be less than the ROI Max wavelength.")
            # Do not emit the signal if ROI is invalid

    def _create_table_item(self, text: str, is_best_fit: bool = False,
                           alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                           tooltip: Optional[str] = None,
                           user_data: Any = None) -> QTableWidgetItem:
        """Helper function to create and format a QTableWidgetItem."""
        item = QTableWidgetItem(text)
        item.setTextAlignment(alignment)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable) # Make read-only

        if is_best_fit:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            # Use theme-aware background color
            # Access parent safely for theme manager
            parent_widget = self.parent()
            is_dark = False
            if parent_widget and hasattr(parent_widget, 'theme_manager'):
                 is_dark = 'dark' in parent_widget.theme_manager.current_theme_name
            best_bg_color = QColor("#3a5f4a") if is_dark else QColor("#d0e0d0") # Dark Green / Light Green
            item.setBackground(QBrush(best_bg_color))
        # else: # Optional: Explicitly set default background if needed, usually inherits
        #    default_bg_color = self.palette().base().color()
        #    item.setBackground(QBrush(default_bg_color))

        if tooltip:
            item.setToolTip(tooltip)

        if user_data is not None:
            # Store the associated FitResult object (or other data)
            # Use QVariant for PyQt compatibility if needed, though direct storage often works
            item.setData(Qt.ItemDataRole.UserRole, QVariant(user_data))

        return item

    def display_peak_fit_details(self, peak: Optional[Peak], peak_list_index: Optional[int] = None):
        """
        Displays the fitting results for the selected peak in the table.

        Args:
            peak: The Peak object containing fit results.
            peak_list_index: The index of this peak in the main peak list (for refitting).
        """
        self.current_peak_data = peak
        self._current_peak_list_index = peak_list_index # Store index for refit signal

        # Clear table and selection first
        self.fit_details_table.setSortingEnabled(False) # Disable sorting during population
        self.fit_details_table.clearSelection()
        self.fit_details_table.setRowCount(0)

        has_peak_data = peak is not None
        self.results_box.setVisible(has_peak_data)

        if not has_peak_data:
            self.selected_peak_label.setText("Select a peak to view details.")
            self.roi_adjust_group.setVisible(False) # Hide ROI group if no peak
            return

        # Update label with selected peak info
        list_idx_str = f"List Idx: {peak_list_index}" if peak_list_index is not None else "List Idx: N/A"
        self.selected_peak_label.setText(f"Fit Details: Peak @ {peak.wavelength_fitted_or_detected:.4f} nm "
                                         f"({list_idx_str}, Spectrum Idx: {peak.index})")

        # --- Populate Fit Details Table ---
        # Combine best fit and alternative fits into one dictionary for iteration
        all_fits: Dict[str, FitResult] = {}
        if peak.alternative_fits:
             all_fits.update(peak.alternative_fits)
        if peak.best_fit:
             # Ensure best_fit is included, potentially overwriting if it was also in alternatives
             all_fits[peak.best_fit.profile_type] = peak.best_fit

        if not all_fits:
            self.selected_peak_label.setText(f"No successful fits found for Peak @ {peak.wavelength_fitted_or_detected:.4f} nm")
            self.roi_adjust_group.setVisible(False) # Hide if no fits
            return
        else:
             self.roi_adjust_group.setVisible(True) # Show if fits exist


        # --- Handle ROI Display and Enablement ---
        actual_roi = None
        # Try to get ROI from best_fit first, then any fit
        if peak.best_fit and hasattr(peak.best_fit, 'roi_wavelengths') and peak.best_fit.roi_wavelengths:
            # TODO: Ensure core.peak_fitter.fit_peak stores roi_wavelengths in FitResult
            actual_roi = peak.best_fit.roi_wavelengths
            logging.debug(f"ROI from best fit: {actual_roi}")
        else:
             first_fit = next(iter(all_fits.values()), None)
             if first_fit and hasattr(first_fit, 'roi_wavelengths') and first_fit.roi_wavelengths:
                 actual_roi = first_fit.roi_wavelengths
                 logging.debug(f"ROI from first available fit: {actual_roi}")

        if actual_roi and len(actual_roi) == 2:
             self.roi_min_wl_dspin.setValue(actual_roi[0])
             self.roi_max_wl_dspin.setValue(actual_roi[1])
             self.roi_adjust_group.setEnabled(True)
             self.roi_min_wl_dspin.setEnabled(True)
             self.roi_max_wl_dspin.setEnabled(True)
             self.refit_button.setEnabled(True)
             self.roi_adjust_group.setTitle("Adjust ROI & Refit Selected Peak")
        else:
             # If ROI info isn't available from fit results, disable manual adjustment
             logging.warning(f"ROI info not found in FitResult(s) for peak {peak.index}. Disabling manual ROI adjustment.")
             # Set placeholder values but keep disabled
             center_guess = peak.wavelength_fitted_or_detected
             self.roi_min_wl_dspin.setValue(center_guess - 0.1)
             self.roi_max_wl_dspin.setValue(center_guess + 0.1)
             self.roi_adjust_group.setEnabled(False)
             self.roi_min_wl_dspin.setEnabled(False)
             self.roi_max_wl_dspin.setEnabled(False)
             self.refit_button.setEnabled(False)
             self.roi_adjust_group.setTitle("Adjust ROI & Refit Selected Peak (ROI Info Unavailable)")

        # --- Populate Table Rows ---
        # Map column names to their indices for easier access
        col_map = {name: i for i, name in enumerate(self.fit_details_columns)}
        score_label = self.current_model_selection # AIC or BIC

        row = 0
        # Iterate through expected profiles to maintain order
        for profile_name in ['Gaussian', 'Lorentzian', 'PseudoVoigt']:
            fit_result = all_fits.get(profile_name)
            if fit_result is None:
                continue # Skip if this profile wasn't successfully fitted

            self.fit_details_table.insertRow(row)
            is_best = (peak.best_fit is not None and peak.best_fit.profile_type == profile_name)

            # Helper for safe formatting
            def format_val(val, prec=4): return f"{val:.{prec}f}" if val is not None and np.isfinite(val) else "N/A"
            def format_err(err, prec=1): return f"{err:.{prec}e}" if err is not None and np.isfinite(err) and err != 0 else ""

            # Get parameters and errors safely
            amp = fit_result.amplitude
            cen = fit_result.center
            wid = fit_result.width
            eta = fit_result.mixing_param_eta # Only for PseudoVoigt
            amp_err = cen_err = wid_err = eta_err = None
            if fit_result.param_errors is not None:
                 try: amp_err = fit_result.param_errors[0]
                 except IndexError: pass
                 try: cen_err = fit_result.param_errors[1]
                 except IndexError: pass
                 try: wid_err = fit_result.param_errors[2]
                 except IndexError: pass
                 # TODO: Add eta error if fit_peak provides it (e.g., param_errors[3])
                 # try: eta_err = fit_result.param_errors[3]
                 # except IndexError: pass


            # Profile Name (with best fit marker *) - Store FitResult here
            profile_text = f"{profile_name}{' *' if is_best else ''}"
            item_prof = self._create_table_item(profile_text, is_best, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, user_data=fit_result)
            self.fit_details_table.setItem(row, col_map["Profile"], item_prof)

            # Amplitude
            item_amp = self._create_table_item(format_val(amp, 2), is_best, tooltip=f"Amplitude")
            self.fit_details_table.setItem(row, col_map["Amplitude"], item_amp)
            # Amp Error
            item_amp_err = self._create_table_item(format_err(amp_err), is_best, tooltip=f"Amplitude Std. Error")
            self.fit_details_table.setItem(row, col_map["Amp Err"], item_amp_err)

            # Center
            item_cen = self._create_table_item(format_val(cen, 4), is_best, tooltip=f"Center Wavelength (nm)")
            self.fit_details_table.setItem(row, col_map["Center"], item_cen)
            # Cen Error
            item_cen_err = self._create_table_item(format_err(cen_err), is_best, tooltip=f"Center Std. Error (nm)")
            self.fit_details_table.setItem(row, col_map["Cen Err"], item_cen_err)

            # Width (Sigma or Gamma)
            item_wid = self._create_table_item(format_val(wid, 4), is_best, tooltip=f"Width (σ/γ) (nm)")
            self.fit_details_table.setItem(row, col_map["Width"], item_wid)
            # Wid Error
            item_wid_err = self._create_table_item(format_err(wid_err), is_best, tooltip=f"Width Std. Error (nm)")
            self.fit_details_table.setItem(row, col_map["Wid Err"], item_wid_err)

            # FWHM / Mixing Param (Eta)
            fwhm_mix_val = ""
            fwhm_mix_tip = ""
            if profile_name == 'PseudoVoigt' and eta is not None and np.isfinite(eta):
                fwhm_mix_val = f"η={format_val(eta, 3)}" # Display Eta value
                fwhm_mix_tip = "Mixing Param (η)" # Tooltip explains Eta
                # TODO: Add eta_err to tooltip if available
            elif fit_result.fwhm is not None and np.isfinite(fit_result.fwhm):
                fwhm_mix_val = format_val(fit_result.fwhm, 4) # Display FWHM
                fwhm_mix_tip = "FWHM (nm)" # Tooltip explains FWHM
            item_fm = self._create_table_item(fwhm_mix_val, is_best, alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter, tooltip=fwhm_mix_tip)
            self.fit_details_table.setItem(row, col_map["FWHM/Mix"], item_fm)

            # R²
            item_r2 = self._create_table_item(format_val(fit_result.r_squared, 3), is_best, alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter, tooltip="R² Fit Metric")
            self.fit_details_table.setItem(row, col_map["R²"], item_r2)

            # Score (AIC or BIC)
            score_val = fit_result.aic if score_label == 'AIC' else fit_result.bic
            item_score = self._create_table_item(format_val(score_val, 2), is_best, alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter, tooltip=f"{score_label} Score")
            self.fit_details_table.setItem(row, col_map["Score"], item_score)

            row += 1

        self.fit_details_table.resizeColumnsToContents()
        self.fit_details_table.setSortingEnabled(True)
        # Optionally select the best fit row after populating
        for r in range(self.fit_details_table.rowCount()):
             profile_item = self.fit_details_table.item(r, col_map["Profile"])
             # Retrieve the FitResult to check its type directly
             fit_data_variant = profile_item.data(Qt.ItemDataRole.UserRole) if profile_item else None
             fit_data = fit_data_variant.value() if isinstance(fit_data_variant, QVariant) else fit_data_variant
             if peak.best_fit and fit_data and isinstance(fit_data, FitResult) and fit_data.profile_type == peak.best_fit.profile_type:
                  self.fit_details_table.selectRow(r)
                  break # Select first match and exit


    @pyqtSlot()
    def _handle_fit_profile_selection(self):
        """Emits signal when a row (fit profile) is selected in the details table."""
        selected_items = self.fit_details_table.selectedItems()
        if not selected_items:
            return # No selection or selection cleared

        # Get the item from the first column (Profile) of the selected row
        selected_row = selected_items[0].row()
        profile_item = self.fit_details_table.item(selected_row, 0) # Column 0 is "Profile"

        if profile_item:
            # Retrieve the FitResult object stored in UserRole
            fit_data_variant = profile_item.data(Qt.ItemDataRole.UserRole)
            if fit_data_variant is not None: # Check if data exists
                # Handle potential QVariant wrapping if used, otherwise access directly
                fit_data = fit_data_variant.value() if isinstance(fit_data_variant, QVariant) else fit_data_variant

                if isinstance(fit_data, FitResult):
                    if self.current_peak_data: # Ensure current peak context is valid
                        logging.debug(f"User selected fit profile: {fit_data.profile_type} for peak index {self.current_peak_data.index}")
                        self.show_specific_fit.emit(fit_data) # Emit the selected FitResult
                    else:
                         logging.warning("Profile selected but current_peak_data is None.")
                else:
                    logging.warning(f"UserRole data in profile selection is not a FitResult object, type is {type(fit_data)}.")
            else:
                logging.warning("No UserRole data found for selected profile item.")
        else:
            logging.warning("Could not get profile item for selected row.")


    def setEnabled(self, enabled: bool):
        """Overrides setEnabled to also control the 'Fit All' button and clear display if disabled."""
        super().setEnabled(enabled)
        # Also enable/disable the main action button
        self.fit_button.setEnabled(enabled)
        # If the whole panel is disabled, clear the details view
        if not enabled:
            # Pass None to clear the display
            self.display_peak_fit_details(None, None)


    def _show_model_select_info(self):
        """Shows help information about AIC/BIC."""
        QMessageBox.information(self, "Model Selection Criterion Help",
            "This setting determines how the 'best' fit profile is automatically chosen when multiple profiles (Gaussian, Lorentzian, PseudoVoigt) are attempted:\n\n"
            "- **AIC (Akaike Information Criterion):** Balances goodness of fit with model complexity (number of parameters). Generally preferred when prediction is the goal.\n\n"
            "- **BIC (Bayesian Information Criterion):** Similar to AIC, but penalizes model complexity more strongly. Tends to favor simpler models.\n\n"
            "The profile with the *lowest* score (AIC or BIC) is typically selected as the best fit (*).")

    def apply_theme_colors(self, config: Dict):
         """Updates UI elements based on theme change (e.g., table highlights)."""
         try:
            # Determine dark mode based on theme name from parent config if possible
            parent_widget = self.parent()
            is_dark = False
            if parent_widget and hasattr(parent_widget, 'theme_manager'):
                 is_dark = 'dark' in parent_widget.theme_manager.current_theme_name
            self._dark_mode = is_dark
            # Force redraw/repopulation of the details table if a peak is selected
            if self.current_peak_data:
                self.display_peak_fit_details(self.current_peak_data, self._current_peak_list_index)
         except Exception as e:
              logging.warning(f"Could not apply theme update to PeakFitting panel: {e}")


# --- END OF REFACTORED FILE libs_cosmic_forge/ui/views/peak_fitting_controls_view.py ---