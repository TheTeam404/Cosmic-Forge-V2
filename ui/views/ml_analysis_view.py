# --- START OF REFACTORED FILE libs_cosmic_forge/ui/views/ml_analysis_view.py ---
"""
View widget for performing Machine Learning based analysis (PCA, PLS, Classification)
on multiple spectra. Includes preprocessing controls, results visualization,
and results saving capabilities.
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, QComboBox,
                             QPushButton, QLabel, QHBoxLayout, QMessageBox, QSplitter,
                             QSpinBox, QFileDialog, QListWidget, QListWidgetItem,
                             QAbstractItemView, QCheckBox, QDoubleSpinBox, QApplication) # Added QApplication
from PyQt6.QtCore import pyqtSignal, Qt, QThread, QObject, pyqtSlot
from PyQt6.QtGui import QIcon

from ui.widgets.info_button import InfoButton
from ui.widgets.collapsible_box import CollapsibleBox
from ui.views.plot_widget import SpectrumPlotWidget
from core.data_models import Spectrum
from core.analysis_backends import (check_sklearn_availability, scale_data, run_pca,
                                    run_pls_regression, run_classification)
from core.file_io import load_spectrum_from_file, save_dataframe
from core.processing import baseline_poly # Assuming simple baseline is sufficient here

# --- Preprocessing Worker ---
class PreprocessingWorker(QObject):
    """
    Worker object to preprocess multiple spectra in a background thread.

    Handles wavelength alignment, optional baseline correction, and optional scaling.
    """
    # Signals:
    # Emits processed matrix (np.ndarray), common wavelengths (np.ndarray), labels (list)
    preprocessing_complete = pyqtSignal(object, object, list)
    # Emits current index, total count, current filename string
    progress_update = pyqtSignal(int, int, str)
    # Emits error message string
    error_occurred = pyqtSignal(str)
    # Emitted when the run method finishes, regardless of success/error/stop
    finished = pyqtSignal()

    def __init__(self, spectra: List[Spectrum], settings: dict):
        super().__init__()
        self.spectra = spectra
        self.settings = settings
        self._is_running = True # Flag to allow stopping the worker

    def stop(self):
        """Requests the worker to stop processing."""
        logging.debug("Preprocessing worker stop requested.")
        self._is_running = False

    def run(self):
        """Performs the preprocessing steps."""
        logging.info(f"Preprocessing worker started for {len(self.spectra)} spectra.")
        processed_intensities = []
        labels = []
        common_wl = None
        n_total = len(self.spectra)
        successful_count = 0

        try:
            # --- 1. Wavelength Alignment ---
            valid_spectra = [s for s in self.spectra if s and s.wavelengths is not None and len(s.wavelengths) > 1]
            if len(valid_spectra) < 2:
                raise ValueError("Need at least 2 valid spectra with wavelength data for ML analysis.")

            all_wls = [s.wavelengths for s in valid_spectra]
            # Determine common range, handling potential non-finite values
            try:
                min_wl = np.nanmax([np.nanmin(wl) for wl in all_wls])
                max_wl = np.nanmin([np.nanmax(wl) for wl in all_wls])
            except ValueError: # Handles cases where all wavelengths might be NaN in a spectrum
                 raise ValueError("Could not determine wavelength bounds. Check input spectra for valid wavelength data.")


            if min_wl >= max_wl:
                raise ValueError(f"No overlapping wavelength range found across spectra (Min={min_wl:.2f}, Max={max_wl:.2f}).")

            # Determine number of points for common axis (e.g., median length)
            # Ensure a minimum number of points for meaningful analysis
            median_len = int(np.median([len(wl) for wl in all_wls]))
            n_points = max(median_len, self.settings.get('min_common_wl_points', 500))
            common_wl = np.linspace(min_wl, max_wl, n_points)
            logging.debug(f"ML common wavelength axis created: {n_points} points [{min_wl:.2f} - {max_wl:.2f}] nm.")

            # --- 2. Process Each Spectrum ---
            do_baseline = self.settings.get('do_baseline', True)
            baseline_method = self.settings.get('baseline_method', 'Polynomial (Linear)')
            # TODO: Make baseline parameters configurable if needed
            baseline_order = 1
            baseline_percentile = 10

            for i, spec in enumerate(self.spectra):
                if not self._is_running:
                    logging.info("Preprocessing worker stopped.")
                    self.finished.emit()
                    return

                # Use original filename if available, otherwise generate one
                fname = os.path.basename(spec.filename) if spec.filename else f"Spectrum {i+1}"
                self.progress_update.emit(i + 1, n_total, fname)

                try:
                    if spec is None or spec.wavelengths is None or spec.raw_intensity is None or len(spec.wavelengths) < 2:
                        raise ValueError("Spectrum object or its data is invalid.")

                    wl = spec.wavelengths
                    intensity = spec.raw_intensity.copy() # Work on a copy

                    # Interpolate onto common wavelength axis
                    # Handle potential NaNs in input intensity robustly if needed
                    interp_intensity = np.interp(common_wl, wl, intensity, left=np.nan, right=np.nan)
                    if np.isnan(interp_intensity).any():
                         logging.warning(f"NaNs produced during interpolation for {fname}. Attempting to continue.")
                         interp_intensity = np.nan_to_num(interp_intensity) # Replace NaN with 0

                    proc_intensity = interp_intensity

                    # Apply baseline correction if requested
                    if do_baseline and "Polynomial" in baseline_method:
                        # Note: Using a simple linear baseline here. More complex methods might be needed.
                        proc_intensity, _ = baseline_poly(
                            common_wl, proc_intensity,
                            order=baseline_order, percentile=baseline_percentile
                        )

                    # Check for non-finite values after processing
                    if not np.all(np.isfinite(proc_intensity)):
                        raise ValueError("Non-finite values detected after preprocessing steps.")

                    processed_intensities.append(proc_intensity)
                    labels.append(fname) # Use filename as label
                    successful_count += 1

                except Exception as e:
                    logging.warning(f"Skipping spectrum '{fname}' during preprocessing: {e}")
                    # Continue to the next spectrum

            if not self._is_running: # Check again after loop
                logging.info("Preprocessing worker stopped.")
                self.finished.emit()
                return

            if successful_count < 2:
                raise ValueError(f"Preprocessing failed: Fewer than 2 spectra were processed successfully ({successful_count}/{n_total}).")

            # --- 3. Create Matrix and Scale ---
            matrix = np.vstack(processed_intensities)
            logging.info(f"Preprocessing matrix created with shape: {matrix.shape}")

            do_scale = self.settings.get('do_scale', True)
            if do_scale:
                scaled_matrix = scale_data(matrix) # Uses sklearn StandardScaler
                if scaled_matrix is not None:
                    matrix = scaled_matrix
                    logging.info("Preprocessing data scaling applied (mean=0, std=1).")
                else:
                    logging.warning("Data scaling failed. Using unscaled data.")

            # --- 4. Emit Results ---
            if self._is_running: # Final check before emitting
                 self.preprocessing_complete.emit(matrix, common_wl, labels)
                 logging.info("Preprocessing complete. Emitting results.")

        except Exception as e:
            logging.error(f"Error during preprocessing worker run: {e}", exc_info=True)
            if self._is_running: # Don't emit error if intentionally stopped
                 self.error_occurred.emit(f"Preprocessing failed: {e}")
        finally:
            logging.debug("Preprocessing worker finished.")
            self.finished.emit() # Ensure finished signal is always emitted

# --- Main View Widget ---
class MLAnalysisView(QWidget):
    """View for Multivariate/Machine Learning Analysis on multiple spectra."""

    # Signal for updating the main window's status bar
    # Args: message (str), timeout (int, milliseconds)
    status_update = pyqtSignal(str, int)

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config.get('machine_learning', {}) # Get ML sub-config
        self._spectra_list: List[Spectrum] = [] # Original loaded spectra
        self._processed_matrix: Optional[np.ndarray] = None # Result of preprocessing
        self._common_wavelengths: Optional[np.ndarray] = None # Wavelength axis for matrix
        self._labels: List[str] = [] # Labels corresponding to matrix rows (spectra)
        self._analysis_results: Optional[pd.DataFrame] = None # Results table from analysis

        self.sklearn_ok: bool = check_sklearn_availability()
        self._preprocess_thread: Optional[QThread] = None
        self._preprocess_worker: Optional[PreprocessingWorker] = None
        self._is_preprocessing: bool = False
        self._is_analyzing: bool = False
        self._last_save_dir: str = os.path.expanduser("~") # Default save location

        self._init_ui()
        self._connect_signals()
        self._update_button_states()

        if not self.sklearn_ok:
            logging.warning("Scikit-learn library not found. ML features will be disabled.")
            QMessageBox.warning(
                self,
                "Dependency Missing",
                "Scikit-learn library not found.\nMachine Learning features will be disabled.\n\n"
                "Please install it using pip:\n  `pip install scikit-learn`"
            )

    def _init_ui(self):
        """Initializes the UI components and layout."""
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(5)

        # --- Left Panel: Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)
        left_layout.setSpacing(8) # Increased spacing

        # --- Spectra List Box ---
        file_box = CollapsibleBox("1. Loaded Spectra", self)
        file_content = QWidget()
        file_layout = QVBoxLayout(file_content)
        file_layout.setContentsMargins(8, 8, 8, 8)
        file_layout.setSpacing(8)
        # Removed Load button - use File Menu action
        file_layout.addWidget(QLabel("Use 'File -> Load Multiple Spectra...' to load data."))
        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection) # No selection needed
        self.file_list_widget.setFixedHeight(120)
        self.file_list_widget.setToolTip("List of spectra currently loaded for ML analysis.")
        file_layout.addWidget(QLabel("Spectra Loaded:"))
        file_layout.addWidget(self.file_list_widget)
        file_box.setContentLayout(file_layout)
        left_layout.addWidget(file_box)

        # --- Preprocessing Box ---
        self.preprocess_box = CollapsibleBox("2. Preprocessing Options", self)
        preprocess_content = QWidget()
        preprocess_layout = QFormLayout(preprocess_content)
        preprocess_layout.setContentsMargins(8, 8, 8, 8)
        preprocess_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        preprocess_layout.setSpacing(8)
        self.preprocess_baseline_combo = QComboBox()
        # Keep options simple for ML preprocessing
        self.preprocess_baseline_combo.addItems(["Polynomial (Linear)", "None"])
        self.preprocess_baseline_combo.setToolTip("Apply simple linear baseline correction before analysis.")
        preprocess_layout.addRow("Baseline Correction:", self.preprocess_baseline_combo)

        self.preprocess_scale_checkbox = QCheckBox("Standard Scale Data")
        self.preprocess_scale_checkbox.setChecked(self.config.get('preprocess_scale_default', True))
        self.preprocess_scale_checkbox.setToolTip("Scale each feature (wavelength) across spectra to have zero mean and unit variance.")
        preprocess_layout.addRow(self.preprocess_scale_checkbox)

        # Add a note about when preprocessing runs
        preprocess_note = QLabel("Preprocessing (incl. wavelength alignment) runs automatically before analysis if needed, or if this section is checked.")
        preprocess_note.setWordWrap(True)
        preprocess_note.setStyleSheet("font-style: italic; color: gray;")
        preprocess_layout.addRow(preprocess_note)

        # Use setWidget to allow collapsing correctly with QFormLayout
        self.preprocess_box.setContentLayout(preprocess_layout) # Use setContent for forms
        left_layout.addWidget(self.preprocess_box)


        # --- Analysis Method Box ---
        self.analysis_box = CollapsibleBox("3. Analysis Method & Parameters", self)
        analysis_content = QWidget()
        analysis_layout = QFormLayout(analysis_content)
        analysis_layout.setContentsMargins(8, 8, 8, 8)
        analysis_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        analysis_layout.setSpacing(8)

        self.analysis_method_combo = QComboBox()
        available_methods = ["PCA", "PLS (Example)", "RF Classify (Example)", "GBT Classify (Example)"] if self.sklearn_ok else ["No Methods Available"]
        self.analysis_method_combo.addItems(available_methods)
        self.analysis_method_combo.currentTextChanged.connect(self._update_parameter_widgets)
        self.analysis_method_combo.setEnabled(self.sklearn_ok) # Disable if sklearn missing
        analysis_layout.addRow("Method:", self.analysis_method_combo)

        # Container for parameter widgets (dynamically shown/hidden)
        self.params_container = QWidget()
        params_layout = QVBoxLayout(self.params_container)
        params_layout.setContentsMargins(0, 5, 0, 0) # Add top margin
        params_layout.setSpacing(5)

        # PCA Parameters
        self.pca_params_widget = QWidget()
        pca_f = QFormLayout(self.pca_params_widget)
        pca_f.setContentsMargins(0, 0, 0, 0)
        self.pca_n_components_spin = QSpinBox()
        self.pca_n_components_spin.setRange(1, 20); self.pca_n_components_spin.setValue(self.config.get('pca', {}).get('default_n_components', 3))
        self.pca_n_components_spin.setToolTip("Number of Principal Components to calculate and display.")
        pca_f.addRow("Num Components:", self.pca_n_components_spin)
        self.pca_params_widget.setVisible(False) # Initially hidden
        params_layout.addWidget(self.pca_params_widget)

        # PLS Parameters (Example)
        self.pls_params_widget = QWidget()
        pls_f = QFormLayout(self.pls_params_widget)
        pls_f.setContentsMargins(0, 0, 0, 0)
        self.pls_n_components_spin = QSpinBox()
        self.pls_n_components_spin.setRange(1, 50); self.pls_n_components_spin.setValue(self.config.get('pls', {}).get('default_n_components', 5))
        self.pls_n_components_spin.setToolTip("Number of PLS components.")
        self.pls_target_wl_dspin = QDoubleSpinBox()
        self.pls_target_wl_dspin.setRange(100, 1200); self.pls_target_wl_dspin.setDecimals(2)
        self.pls_target_wl_dspin.setValue(self.config.get('pls', {}).get('default_target_wl', 404.58))
        self.pls_target_wl_dspin.setSuffix(" nm")
        self.pls_target_wl_dspin.setToolTip("Example: Wavelength intensity to predict (Y variable).")
        pls_f.addRow("Num Components:", self.pls_n_components_spin)
        pls_f.addRow("Target WL:", self.pls_target_wl_dspin)
        self.pls_params_widget.setVisible(False)
        params_layout.addWidget(self.pls_params_widget)

        # Random Forest Parameters (Example)
        self.rf_params_widget = QWidget()
        rf_f = QFormLayout(self.rf_params_widget)
        rf_f.setContentsMargins(0, 0, 0, 0)
        self.rf_n_estimators_spin = QSpinBox()
        self.rf_n_estimators_spin.setRange(10, 1000); self.rf_n_estimators_spin.setValue(self.config.get('RandomForest', {}).get('n_estimators', 100))
        self.rf_n_estimators_spin.setToolTip("Number of trees in the forest.")
        rf_f.addRow("Num Estimators:", self.rf_n_estimators_spin)
        self.rf_params_widget.setVisible(False)
        params_layout.addWidget(self.rf_params_widget)

        # Gradient Boosting Parameters (Example)
        self.gbt_params_widget = QWidget()
        gbt_f = QFormLayout(self.gbt_params_widget)
        gbt_f.setContentsMargins(0, 0, 0, 0)
        self.gbt_n_estimators_spin = QSpinBox()
        self.gbt_n_estimators_spin.setRange(10, 1000); self.gbt_n_estimators_spin.setValue(self.config.get('GBT', {}).get('n_estimators', 100))
        self.gbt_n_estimators_spin.setToolTip("Number of boosting stages (trees).")
        self.gbt_learning_rate_dspin = QDoubleSpinBox()
        self.gbt_learning_rate_dspin.setRange(0.001, 1.0); self.gbt_learning_rate_dspin.setDecimals(3)
        self.gbt_learning_rate_dspin.setValue(self.config.get('GBT', {}).get('learning_rate', 0.1))
        self.gbt_learning_rate_dspin.setToolTip("Learning rate shrinks the contribution of each tree.")
        gbt_f.addRow("Num Estimators:", self.gbt_n_estimators_spin)
        gbt_f.addRow("Learning Rate:", self.gbt_learning_rate_dspin)
        self.gbt_params_widget.setVisible(False)
        params_layout.addWidget(self.gbt_params_widget)

        # Add container to main analysis layout
        analysis_layout.addRow(self.params_container)
        self.analysis_box.setContentLayout(analysis_layout) # Use setContent
        left_layout.addWidget(self.analysis_box)

        # --- Action Buttons ---
        action_hbox = QHBoxLayout()
        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.setIcon(QIcon.fromTheme("system-run", QIcon.fromTheme("media-playback-start"))) # Fallback
        # Tooltip set dynamically by _update_button_states
        action_hbox.addWidget(self.run_analysis_button)

        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.setIcon(QIcon.fromTheme("document-save"))
        self.save_results_button.setToolTip("Save analysis results table to a CSV file.")
        self.save_results_button.setEnabled(False) # Enabled when results are available
        action_hbox.addWidget(self.save_results_button)

        left_layout.addLayout(action_hbox)
        left_layout.addStretch() # Push controls up

        # --- Right Panel: Results Plot ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0) # Add spacing to left of right panel
        plot_group = QGroupBox("Analysis Results Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.results_plot_widget = SpectrumPlotWidget(config=self.config, parent=self) # Pass config
        self.results_plot_widget.ax.set_title("Multivariate Analysis Results")
        # Axes labels will be set by the specific analysis method
        plot_layout.addWidget(self.results_plot_widget)
        right_layout.addWidget(plot_group)

        # --- Assemble Splitter ---
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 1) # Give left panel less space initially
        main_splitter.setStretchFactor(1, 2) # Give plot more space
        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(main_splitter)
        self.setLayout(outer_layout)

        # Set initial parameter widget visibility
        self._update_parameter_widgets(self.analysis_method_combo.currentText())


    def _connect_signals(self):
        """Connects internal signals and slots."""
        # self.load_files_button.clicked.connect(self._load_spectra_files) # Removed button
        self.run_analysis_button.clicked.connect(self._trigger_analysis)
        self.save_results_button.clicked.connect(self._save_results)


    def _update_button_states(self):
        """Updates the enabled state and tooltips of action buttons based on current state."""
        spectra_loaded = len(self._spectra_list) >= 2
        can_preprocess = self.sklearn_ok and spectra_loaded and not self._is_preprocessing and not self._is_analyzing
        # Can run analysis if sklearn ok, spectra loaded, not busy, AND data is preprocessed
        can_run_analysis = self.sklearn_ok and spectra_loaded and not self._is_preprocessing and not self._is_analyzing and self._processed_matrix is not None

        # Determine state for Run button
        run_enabled = self.sklearn_ok and spectra_loaded and not self._is_preprocessing and not self._is_analyzing
        run_tooltip = "Run Analysis"
        if not self.sklearn_ok: run_tooltip = "Requires scikit-learn installation."
        elif not spectra_loaded: run_tooltip = "Load at least 2 spectra first."
        elif self._is_preprocessing: run_tooltip = "Preprocessing in progress..."
        elif self._is_analyzing: run_tooltip = "Analysis in progress..."
        elif self._processed_matrix is None: run_tooltip = "Run Analysis (will preprocess first)"

        self.run_analysis_button.setEnabled(run_enabled)
        self.run_analysis_button.setToolTip(run_tooltip)

        # Determine state for Save button
        can_save = self._analysis_results is not None and not self._analysis_results.empty and not self._is_analyzing
        self.save_results_button.setEnabled(can_save)
        self.save_results_button.setToolTip("Save results table to CSV" if can_save else "No analysis results available to save")

        # Enable/disable controls based on busy state
        is_busy = self._is_preprocessing or self._is_analyzing
        self.preprocess_box.setEnabled(not is_busy)
        self.analysis_box.setEnabled(not is_busy and self.sklearn_ok)
        self.file_list_widget.setEnabled(not is_busy)


    def _update_parameter_widgets(self, method_text: str):
        """Shows/hides parameter widgets based on selected analysis method."""
        method_key = method_text.split(" ")[0].lower() # e.g., "pca", "pls"
        logging.debug(f"Updating parameter widgets for method: {method_key}")
        self.pca_params_widget.setVisible(method_key == "pca")
        self.pls_params_widget.setVisible(method_key == "pls")
        self.rf_params_widget.setVisible(method_key == "rf")
        self.gbt_params_widget.setVisible(method_key == "gbt")


    def set_spectra_list(self, spectra: List[Spectrum]):
        """Receives the list of spectra loaded by the main window."""
        self._spectra_list = spectra if spectra else []
        logging.info(f"ML View received {len(self._spectra_list)} spectra.")

        # Update the list widget display
        self.file_list_widget.clear()
        for i, s in enumerate(self._spectra_list):
            display_name = os.path.basename(s.filename) if s.filename else f"Spectrum {i+1}"
            self.file_list_widget.addItem(QListWidgetItem(display_name))

        # Reset analysis state when new spectra are loaded
        self._processed_matrix = None
        self._common_wavelengths = None
        self._labels = []
        self._analysis_results = None
        self.results_plot_widget.clear_plot()
        self._update_button_states()


    def _stop_preprocessing(self):
        """Stops the background preprocessing worker and thread."""
        if self._preprocess_thread and self._preprocess_thread.isRunning():
            logging.info("Attempting to stop preprocessing worker...")
            if self._preprocess_worker:
                self._preprocess_worker.stop() # Signal worker to stop

            # Give thread time to finish gracefully
            if not self._preprocess_thread.quit() and not self._preprocess_thread.wait(1500): # Wait 1.5 sec
                 logging.warning("Preprocessing thread did not quit gracefully, terminating.")
                 self._preprocess_thread.terminate() # Force terminate
                 self._preprocess_thread.wait() # Wait after termination

            logging.info("Preprocessing stopped.")
        else:
            logging.debug("Stop preprocessing called, but no active thread found.")

        # Clean up references
        self._preprocess_thread = None
        self._preprocess_worker = None
        self._is_preprocessing = False
        self._update_button_states()


    def _preprocess_spectra_threaded(self):
        """Starts the preprocessing worker in a background thread."""
        if self._is_preprocessing:
            logging.warning("Preprocessing already in progress.")
            return
        if len(self._spectra_list) < 2:
             logging.warning("Cannot start preprocessing: Less than 2 spectra loaded.")
             return # Should be caught by button state, but double check

        # Ensure any previous thread is stopped
        self._stop_preprocessing()

        # Gather settings from UI
        settings = {
            'do_baseline': self.preprocess_baseline_combo.currentText() != "None",
            'baseline_method': self.preprocess_baseline_combo.currentText(),
            'do_scale': self.preprocess_scale_checkbox.isChecked(),
            'min_common_wl_points': self.config.get('min_common_wl_points', 500) # Pass config value
        }

        self._is_preprocessing = True
        self._update_button_states()
        self.status_update.emit("Preprocessing spectra...", 0)

        # Create worker and thread
        self._preprocess_thread = QThread()
        self._preprocess_worker = PreprocessingWorker(self._spectra_list, settings)
        self._preprocess_worker.moveToThread(self._preprocess_thread)

        # Connect signals from worker to slots in this view
        self._preprocess_worker.preprocessing_complete.connect(self._handle_preprocessing_complete)
        self._preprocess_worker.progress_update.connect(self._handle_preprocessing_progress)
        self._preprocess_worker.error_occurred.connect(self._handle_preprocessing_error)
        # Connect finished signal to ensure cleanup happens
        self._preprocess_worker.finished.connect(self._preprocessing_finished)

        # Connect thread signals
        self._preprocess_thread.started.connect(self._preprocess_worker.run)
        # Ensure thread quits after worker finishes
        self._preprocess_worker.finished.connect(self._preprocess_thread.quit)
        # Optional: Clean worker object when thread finishes
        # self._preprocess_thread.finished.connect(self._preprocess_worker.deleteLater)

        # Start the thread
        self._preprocess_thread.start()
        logging.info("Preprocessing thread started.")


    @pyqtSlot(object, object, list)
    def _handle_preprocessing_complete(self, matrix: np.ndarray, wavelengths: np.ndarray, labels: list):
        """Handles successful completion of preprocessing."""
        logging.info(f"Preprocessing successful. Received matrix shape: {matrix.shape}, "
                     f"Wavelengths: {len(wavelengths)}, Labels: {len(labels)}")
        self._processed_matrix = matrix
        self._common_wavelengths = wavelengths
        self._labels = labels
        # Note: finished signal will handle state update


    @pyqtSlot(int, int, str)
    def _handle_preprocessing_progress(self, current: int, total: int, filename: str):
        """Updates status bar with preprocessing progress."""
        self.status_update.emit(f"Preprocessing {current}/{total}: {os.path.basename(filename)}", 0) # Show only filename


    @pyqtSlot(str)
    def _handle_preprocessing_error(self, error_msg: str):
        """Handles errors reported by the preprocessing worker."""
        logging.error(f"Preprocessing Error: {error_msg}")
        QMessageBox.critical(self, "Preprocessing Error", f"An error occurred during preprocessing:\n{error_msg}")
        self._processed_matrix = None # Ensure matrix is cleared on error
        # Note: finished signal will handle state update


    @pyqtSlot()
    def _preprocessing_finished(self):
        """Cleans up after the preprocessing thread finishes."""
        logging.debug("Preprocessing finished signal received.")
        was_successful = self._processed_matrix is not None
        # State updates must happen in the main thread
        self._is_preprocessing = False
        self._update_button_states()

        # Clean up thread/worker objects *after* state is updated
        # These checks are needed as finished might be called after manual stop
        if self._preprocess_thread:
             self._preprocess_thread = None
        if self._preprocess_worker:
             self._preprocess_worker = None

        # Final status message
        msg = "Preprocessing complete." if was_successful else "Preprocessing failed or stopped."
        self.status_update.emit(msg, 5000)


    @pyqtSlot()
    def _trigger_analysis(self):
        """Initiates the analysis process, preprocessing if necessary."""
        if self._is_analyzing or self._is_preprocessing:
            logging.warning("Analysis trigger ignored: Already busy.")
            return
        if len(self._spectra_list) < 2:
            QMessageBox.warning(self, "Not Enough Data", "Please load at least 2 spectra files for analysis.")
            return
        if not self.sklearn_ok:
            QMessageBox.critical(self, "Dependency Missing", "Scikit-learn is required for analysis. Please install it.")
            return

        # Decide if preprocessing needs to run
        # Run if matrix is None OR if the preprocess box is checked (user wants to re-run with current settings)
        needs_preprocessing = self._processed_matrix is None or self.preprocess_box.isChecked()

        if needs_preprocessing:
            logging.info("Preprocessing required before analysis. Starting preprocessing thread.")
            self._processed_matrix = None # Clear any old matrix if re-preprocessing
            self._analysis_results = None # Clear old results
            self.results_plot_widget.clear_plot()
            self._preprocess_spectra_threaded()
            # Inform user that preprocessing started, analysis needs separate trigger
            QMessageBox.information(self, "Preprocessing Started",
                                    "Preprocessing the spectra now.\n"
                                    "Click 'Run Analysis' again after the 'Preprocessing complete' status message appears.")
        elif self._processed_matrix is not None:
            logging.info("Preprocessed data available. Running selected analysis.")
            self._run_selected_analysis()
        else:
             # Should not happen if logic is correct, but catch anyway
             logging.error("Analysis trigger state error: No processed matrix, but preprocessing not requested.")
             QMessageBox.warning(self, "State Error", "Cannot run analysis. Try running preprocessing first.")


    def _run_selected_analysis(self):
        """Executes the selected analysis method using the preprocessed data."""
        if self._processed_matrix is None:
            QMessageBox.warning(self, "Data Error", "Preprocessed data is not available. Please preprocess first.")
            return
        if self._is_analyzing:
            logging.warning("Analysis already in progress.")
            return

        method_txt = self.analysis_method_combo.currentText()
        method_key = method_txt.split(" ")[0].lower() # e.g., "pca"
        self.status_update.emit(f"Running {method_txt} analysis...", 0)
        self._is_analyzing = True
        self._analysis_results = None # Clear previous results
        self._update_button_states()
        QApplication.processEvents() # Update UI

        plot_widget = self.results_plot_widget
        plot_widget.clear_plot(redraw=False) # Clear plot but don't redraw yet
        success = False
        results_data = None
        X = self._processed_matrix
        y_labels = self._labels # For indexing results DataFrame

        try:
            if method_key == "pca":
                n_components = self.pca_n_components_spin.value()
                logging.info(f"Running PCA with n_components={n_components}")
                result_tuple = run_pca(X, n_components)
                if result_tuple:
                    scores, explained_variance_ratio = result_tuple
                    pc1 = scores[:, 0]
                    pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(pc1) # Handle n=1 case
                    var1 = explained_variance_ratio[0] * 100
                    var2 = explained_variance_ratio[1] * 100 if len(explained_variance_ratio) > 1 else 0
                    # Plotting
                    plot_widget.ax.scatter(pc1, pc2, c=np.arange(len(pc1)), cmap='viridis', alpha=0.8)
                    plot_widget.ax.set_xlabel(f"PC 1 ({var1:.1f}%)")
                    plot_widget.ax.set_ylabel(f"PC 2 ({var2:.1f}%)")
                    plot_widget.ax.set_title(f"PCA Scores Plot ({n_components} Components)")
                    # Add labels if not too many points
                    if len(y_labels) <= 30:
                         for i, txt in enumerate(y_labels):
                             plot_widget.ax.text(pc1[i]*1.01, pc2[i]*1.01, os.path.basename(txt), fontsize=8, clip_on=True) # Use basename
                    success = True
                    results_data = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(scores.shape[1])], index=y_labels)
                else: raise RuntimeError("PCA backend function returned None.")

            elif method_key == "pls":
                # Example PLS for predicting intensity at a specific wavelength
                n_components = self.pls_n_components_spin.value()
                target_wl = self.pls_target_wl_dspin.value()
                if self._common_wavelengths is None: raise ValueError("Common wavelengths not available for PLS target.")
                # Find index of wavelength closest to target
                target_idx = np.argmin(np.abs(self._common_wavelengths - target_wl))
                actual_target_wl = self._common_wavelengths[target_idx]
                logging.info(f"Running PLS Regression (n={n_components}) to predict intensity at {actual_target_wl:.2f} nm.")
                y = X[:, target_idx] # Intensity at target wavelength is the dependent variable
                X_pls = np.delete(X, target_idx, axis=1) # Use remaining wavelengths as predictors
                result_tuple = run_pls_regression(X_pls, y, n_components)
                if result_tuple:
                    y_pred, r_squared = result_tuple
                    # Plotting
                    plot_widget.ax.scatter(y, y_pred, c=np.arange(len(y)), cmap='viridis', alpha=0.8)
                    lims = [min(np.min(y), np.min(y_pred)), max(np.max(y), np.max(y_pred))]
                    plot_widget.ax.plot(lims, lims, 'r--', alpha=0.7, label=f'R²={r_squared:.3f}')
                    plot_widget.ax.set_xlabel(f"Actual Intensity @ {actual_target_wl:.1f} nm")
                    plot_widget.ax.set_ylabel("Predicted Intensity")
                    plot_widget.ax.set_title(f"PLS Regression ({n_components} Comp)")
                    plot_widget._update_legend()
                    success = True
                    results_data = pd.DataFrame({'Label': y_labels, 'Actual_Intensity': y, 'Predicted_Intensity': y_pred.flatten()})
                else: raise RuntimeError("PLS backend function returned None.")

            elif method_key == "rf" or method_key == "gbt":
                # Example Classification: Assume first half are class 0, second half class 1
                n_samples = X.shape[0]
                if n_samples < 2: raise ValueError("Need at least 2 samples for classification.")
                y_true = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
                classifier_params = {}
                if method_key == "rf":
                    classifier_params['n_estimators'] = self.rf_n_estimators_spin.value()
                    model_name = "RandomForest"
                    title_prefix = "Random Forest"
                else: # GBT
                    classifier_params['n_estimators'] = self.gbt_n_estimators_spin.value()
                    classifier_params['learning_rate'] = self.gbt_learning_rate_dspin.value()
                    model_name = "GBT"
                    title_prefix = "Gradient Boosting"
                logging.info(f"Running {title_prefix} Classification with params: {classifier_params}")

                result_tuple = run_classification(X, y_true, model_name, **classifier_params)
                if result_tuple:
                    y_pred, accuracy = result_tuple
                    # Plotting: Sample index vs predicted class, colored by true class
                    plot_widget.ax.scatter(np.arange(n_samples), y_pred, c=y_true, cmap='coolwarm', alpha=0.8, label="Predicted Class")
                    # Plot true classes as small dots for reference
                    plot_widget.ax.plot(np.arange(n_samples), y_true, 'k.', ms=3, alpha=0.5, label="True Class")
                    plot_widget.ax.set_xlabel("Sample Index")
                    plot_widget.ax.set_ylabel("Class Label")
                    plot_widget.ax.set_yticks([0, 1]) # Assuming binary classification
                    plot_widget.ax.set_yticklabels(['Class 0', 'Class 1'])
                    plot_widget.ax.set_title(f"{title_prefix} Classification (Accuracy = {accuracy:.3f})")
                    plot_widget._update_legend()
                    success = True
                    results_data = pd.DataFrame({'Label': y_labels, 'True_Class': y_true, 'Predicted_Class': y_pred})
                else: raise RuntimeError(f"{title_prefix} backend function returned None.")

            else:
                QMessageBox.warning(self, "Not Implemented", f"Analysis method '{method_txt}' is not yet implemented.")
                # No results, success remains False

            # --- Post-Analysis ---
            if success and results_data is not None:
                self._analysis_results = results_data
                plot_widget.apply_theme_colors(self.parent().config if self.parent() else {}) # Apply theme to plot
                plot_widget._redraw_canvas()
                self.status_update.emit(f"{method_txt} analysis complete.", 5000)
            elif success and results_data is None:
                 # Should not happen if logic is correct
                 raise RuntimeError(f"{method_txt} reported success but returned no results data.")
            else:
                # If backend didn't raise error but returned None/False
                 raise RuntimeError(f"{method_txt} analysis failed or was aborted by backend.")

        except Exception as e:
            logging.error(f"Error during {method_txt} analysis: {e}", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during {method_txt} analysis:\n{e}")
            self.status_update.emit(f"{method_txt} analysis failed.", 5000)
            plot_widget.clear_plot()
            self._analysis_results = None # Clear results on error
            success = False # Ensure success is false
        finally:
            self._is_analyzing = False
            self._update_button_states() # Update button states after analysis finishes/fails


    def _save_results(self):
        """Saves the current analysis results DataFrame to a CSV file."""
        if self._analysis_results is None or self._analysis_results.empty:
            QMessageBox.warning(self, "No Results", "There are no analysis results available to save.")
            return

        method = self.analysis_method_combo.currentText().split(" ")[0].lower()
        default_filename = f"ml_results_{method}.csv"
        file_filter = "CSV Files (*.csv);;All Files (*)"

        # Use QFileDialog.getSaveFileName for the save dialog
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            f"Save {method.upper()} Analysis Results",
            os.path.join(self._last_save_dir, default_filename),
            file_filter
        )

        if filepath:
            self._last_save_dir = os.path.dirname(filepath) # Remember directory
            self.status_update.emit(f"Saving {method.upper()} results to {os.path.basename(filepath)}...", 0)
            QApplication.processEvents() # Allow status update to show

            try:
                # Ensure the results DataFrame has the index (labels) included if desired
                df_to_save = self._analysis_results.copy()
                # Check if index has a name, if not, maybe set it from 'Label' column if it exists
                if df_to_save.index.name is None and 'Label' in df_to_save.columns:
                     df_to_save = df_to_save.set_index('Label') # Set 'Label' as index before saving

                save_dataframe(df_to_save, filepath) # Use core function
                self.status_update.emit(f"Results saved: {os.path.basename(filepath)}", 5000)
                logging.info(f"ML analysis results saved to: {filepath}")

            except Exception as e:
                logging.error(f"Failed to save ML results to {filepath}: {e}", exc_info=True)
                QMessageBox.critical(self, "Save Error", f"Failed to save results file:\n{e}")
                self.status_update.emit("Save failed.", 5000)
        else:
            # User cancelled the save dialog
            self.status_update.emit("Save cancelled.", 3000)


    def clear_all(self):
        """Resets the entire ML analysis view."""
        logging.debug("Clearing ML Analysis view.")
        self._stop_preprocessing() # Ensure background task is stopped
        self._spectra_list = []
        self.file_list_widget.clear()
        self._processed_matrix = None
        self._common_wavelengths = None
        self._labels = []
        self._analysis_results = None
        self.results_plot_widget.clear_plot()
        self.status_update.emit("ML view cleared.", 1000)
        self._update_button_states()


    def setEnabled(self, enabled: bool):
        """Overrides setEnabled to update button states accordingly."""
        super().setEnabled(enabled)
        # Update buttons, considering the external enabled state
        self._update_button_states()
        if not enabled: # If view is disabled externally, ensure buttons reflect this
             self.run_analysis_button.setEnabled(False)
             self.save_results_button.setEnabled(False)


    def closeEvent(self, event):
        """Ensures background thread is stopped when the widget is closed."""
        logging.debug("MLAnalysisView close event triggered.")
        self._stop_preprocessing()
        super().closeEvent(event)

    def apply_theme_colors(self, config: Dict):
         """Applies color settings from the theme to the plot."""
         if hasattr(self, 'results_plot_widget') and self.results_plot_widget:
             self.results_plot_widget.apply_theme_colors(config) # Delegate

# --- END OF REFACTORED FILE libs_cosmic_forge/ui/views/ml_analysis_view.py ---