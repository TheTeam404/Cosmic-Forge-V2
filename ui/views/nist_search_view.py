# --- START OF REFACTORED FILE libs_cosmic_forge/ui/views/nist_search_view.py ---
"""
View widget for searching the NIST Atomic Spectra Database (ASD) online
and displaying the potential line matches found for detected peaks.
"""
import logging
import time # Added missing import
import traceback
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QTableWidget, QHeaderView,
                             QTableWidgetItem, QAbstractItemView, QLabel, QHBoxLayout,
                             QPushButton, QLineEdit, QDoubleSpinBox, QMenu, QProgressDialog,
                             QMessageBox , QFormLayout)
from PyQt6.QtCore import pyqtSignal, Qt, QThread, QObject, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QColor, QBrush # Added QColor, QBrush

# Import core components
from core.data_models import Peak, NISTMatch
from core.nist_manager import search_online_nist, ASTROQUERY_AVAILABLE
from ui.widgets.info_button import InfoButton # Assuming InfoButton is used, otherwise remove

# --- Worker Thread for Online Search ---
class OnlineSearchWorker(QObject):
    """Worker object to perform online NIST searches in a background thread."""
    # Signals
    results_ready = pyqtSignal(list) # Emits List[NISTMatch]
    search_progress = pyqtSignal(int, int) # current processed peak index, total peaks
    search_finished = pyqtSignal(str) # Final status message (success, failure, cancelled)
    error_occurred = pyqtSignal(str) # Error message for specific query failures

    def __init__(self, peaks_to_search: List[Peak], tolerance_nm: float, config: dict):
        super().__init__()
        # Filter out any None peaks or peaks without valid search wavelengths immediately
        self.peaks_to_search = [
            p for p in peaks_to_search if p is not None and np.isfinite(p.wavelength_fitted_or_detected)
        ]
        self.tolerance_nm = tolerance_nm
        self.db_config = config.get('database', {})
        self._is_running = True # Flag to control execution loop
        logging.debug(f"OnlineSearchWorker created for {len(self.peaks_to_search)} valid peaks.")

    def stop(self):
        """Requests the worker to stop processing."""
        logging.debug("OnlineSearchWorker stop requested.")
        self._is_running = False

    def run(self):
        """Executes the online search loop."""
        if not ASTROQUERY_AVAILABLE:
            logging.error("Astroquery library not found, cannot perform online search.")
            self.error_occurred.emit("Astroquery library not installed. Cannot search online.")
            self.search_finished.emit("Search failed (Astroquery missing).")
            return

        all_matches: List[NISTMatch] = []
        total_peaks = len(self.peaks_to_search)

        if total_peaks == 0:
            logging.warning("No valid peaks provided to OnlineSearchWorker.")
            self.search_finished.emit("No valid peaks provided for search.")
            return

        # Get config parameters with defaults
        query_delay_s = self.db_config.get('online_query_delay_s', 1.0) # Default 1 sec delay
        timeout_s = self.db_config.get('online_search_timeout_s', 20) # Default 20 sec timeout
        query_errors = 0

        logging.info(f"Online search worker started: {total_peaks} peaks, "
                     f"tolerance={self.tolerance_nm:.3f} nm, delay={query_delay_s}s, timeout={timeout_s}s")

        for i, peak in enumerate(self.peaks_to_search):
            if not self._is_running:
                logging.info("Online search worker stopping mid-execution.")
                break # Exit loop if stop requested

            current_progress = i + 1
            self.search_progress.emit(current_progress, total_peaks)

            # Wavelength is already validated in __init__
            wl_search = peak.wavelength_fitted_or_detected
            logging.debug(f"Worker searching online ({current_progress}/{total_peaks}) for peak index {peak.index} @ {wl_search:.4f} nm...")

            try:
                # --- Apply Delay (if not the first query and delay > 0) ---
                if i > 0 and query_delay_s > 0:
                    time_slept = 0
                    sleep_interval = 0.1 # Check stop flag frequently during sleep
                    while time_slept < query_delay_s and self._is_running:
                        time.sleep(sleep_interval)
                        time_slept += sleep_interval
                    if not self._is_running: break # Check again immediately after sleep loop finishes

                if not self._is_running: break # Break before making the potentially long query

                # --- Perform NIST Query ---
                matches_for_peak: Optional[List[NISTMatch]] = search_online_nist(
                    wavelength_nm=wl_search,
                    tolerance_nm=self.tolerance_nm,
                    query_delay_s=0, # Delay is handled manually above
                    timeout_s=timeout_s
                )

                # Process results if query didn't timeout or fail internally
                if matches_for_peak is not None:
                    logging.debug(f"Found {len(matches_for_peak)} potential matches for {wl_search:.4f} nm.")
                    # Add reference back to the original peak (index and wavelength used)
                    for match in matches_for_peak:
                         match.query_peak_index = peak.index
                         match.query_peak_wavelength = wl_search
                    all_matches.extend(matches_for_peak)
                else:
                     # search_online_nist returns None on timeout or other errors handled internally
                     logging.warning(f"No results returned from search_online_nist for {wl_search:.4f} nm (likely timeout or internal error).")
                     # Don't count this as a query_error unless search_online_nist raises Exception
                     # self.error_occurred.emit(f"Query timed out or failed for {wl_search:.4f} nm.")
                     # query_errors += 1

            except Exception as e:
                # Catch unexpected errors during the query call itself
                logging.error(f"Unhandled exception during online query for {wl_search:.4f} nm: {e}", exc_info=True)
                self.error_occurred.emit(f"Query failed for {wl_search:.4f} nm (Exception: {e}). Check logs.")
                query_errors += 1
                # Continue to the next peak despite the error

        # --- Final Emission ---
        if self._is_running: # If the loop completed without being stopped
            self.results_ready.emit(all_matches) # Emit collected results
            status = f"Online search complete. Found {len(all_matches)} potential matches."
            if query_errors > 0:
                 status += f" ({query_errors} query errors occurred)"
            self.search_finished.emit(status)
            logging.info(status)
        else:
            # If stopped, do not emit results_ready, just finish
            self.search_finished.emit("Online search cancelled by user.")
            logging.info("Online search worker finished after cancellation request.")


# --- Main View Widget ---
class NistSearchView(QWidget):
    """Displays NIST search controls and results table."""

    # Signal emitted after online search completes and results are processed.
    # Emits the raw list of NISTMatch objects found.
    online_results_obtained = pyqtSignal(list) # List[NISTMatch]

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config # Store full config for access to 'database' sub-config etc.
        self._peaks_ref: List[Peak] = [] # Reference to the current list of detected peaks
        self._current_matches_df: Optional[pd.DataFrame] = None # DataFrame for table display
        self._online_search_thread: Optional[QThread] = None
        self._online_search_worker: Optional[OnlineSearchWorker] = None
        self._progress_dialog: Optional[QProgressDialog] = None
        self._is_searching: bool = False # Flag to track if search is active

        self._init_ui()
        self._connect_signals()
        self._update_button_states() # Set initial button state

    def _init_ui(self):
        """Initializes the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5) # Reduced margins
        main_layout.setSpacing(8) # Increased spacing

        # --- Controls Group ---
        controls_box = QGroupBox("NIST Online Search")
        ctrl_layout = QFormLayout(controls_box)
        ctrl_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        ctrl_layout.setSpacing(8)

        # Tolerance SpinBox
        self.tolerance_dspin = QDoubleSpinBox()
        self.tolerance_dspin.setRange(0.001, 10.0) # Adjusted range slightly
        self.tolerance_dspin.setDecimals(3)
        self.tolerance_dspin.setSingleStep(0.01)
        default_tolerance = self.config.get('database', {}).get('default_search_tolerance_nm', 0.1)
        self.tolerance_dspin.setValue(default_tolerance)
        self.tolerance_dspin.setSuffix(" nm")
        self.tolerance_dspin.setToolTip("Wavelength match tolerance (± nm) for NIST queries.")
        self.tolerance_dspin.setKeyboardTracking(False) # Update value on focus loss/enter
        ctrl_layout.addRow("Tolerance (±):", self.tolerance_dspin)

        # Action Button(s)
        btn_hbox = QHBoxLayout()
        self.search_online_btn = QPushButton("Search Online (NIST ASD)")
        self.search_online_btn.setToolTip("Search the online NIST ASD database for potential line matches for all detected peaks.")
        self.search_online_btn.setIcon(QIcon.fromTheme("network-transmit-receive", QIcon.fromTheme("internet-web-browser"))) # Fallback icon
        # Removed local search button - focus on online search
        btn_hbox.addWidget(self.search_online_btn)
        btn_hbox.addStretch() # Push button left
        ctrl_layout.addRow(btn_hbox)

        main_layout.addWidget(controls_box)

        # --- Results Table ---
        results_group = QGroupBox("Search Results") # Added groupbox for table
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(5, 8, 5, 5)
        self.results_table = QTableWidget()
        # Define columns (ensure consistency with _populate_results_table and NISTMatch.to_dataframe_row)
        self.columns = ["Peak λ (nm)", "Source", "Elem", "Ion", "DB λ (nm)", "Δλ (nm)", "Aki (s⁻¹)", "Eᵢ (eV)", "gᵢ", "Line Label"]
        self.results_table.setColumnCount(len(self.columns))
        self.results_table.setHorizontalHeaderLabels(self.columns)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) # Read-only
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection) # Allow multi-select (e.g., for copying)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSortingEnabled(True)
        self.results_table.verticalHeader().setVisible(False) # Hide row numbers
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive) # Allow user resize
        self.results_table.horizontalHeader().setStretchLastSection(True) # Stretch last column initially
        self.results_table.setWordWrap(False) # Prevent wrapping in cells
        # self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu) # Context menu might be useful later (e.g., copy)

        results_layout.addWidget(self.results_table)
        main_layout.addWidget(results_group, 1) # Give table vertical stretch

        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect signals from UI elements."""
        self.search_online_btn.clicked.connect(self._trigger_online_search)
        # No action needed on table selection currently
        # self.results_table.itemSelectionChanged.connect(self._handle_selection_change)

    def set_peaks_reference(self, peak_list: List[Peak]):
        """Updates the internal reference to the list of detected peaks."""
        self._peaks_ref = peak_list if peak_list else []
        logging.debug(f"NIST search view received {len(self._peaks_ref)} peaks as reference.")
        # Clear previous results when peaks change
        self.clear_results()
        # Update button states based on new peak list
        self._update_button_states()

    def _update_button_states(self):
        """Updates the enabled state and tooltip of the search button."""
        has_peaks = bool(self._peaks_ref)
        can_run = has_peaks and ASTROQUERY_AVAILABLE and not self._is_searching
        self.search_online_btn.setEnabled(can_run)

        tooltip = "Search the online NIST ASD database."
        if self._is_searching:
            tooltip = "Online search currently in progress..."
        elif not ASTROQUERY_AVAILABLE:
            tooltip = "Cannot search: Astroquery library is not installed."
        elif not has_peaks:
            tooltip = "Cannot search: No peaks detected or loaded."

        self.search_online_btn.setToolTip(tooltip)

    def clear_results(self):
        """Clears the results table and internal DataFrame."""
        # Stop any running search first
        self._stop_running_search()

        logging.debug("Clearing NIST search results table and data.")
        self.results_table.setSortingEnabled(False) # Disable sorting for clearing
        # self.results_table.clearContents() # Keeps headers
        self.results_table.setRowCount(0) # Faster than clearContents for large tables
        self.results_table.setSortingEnabled(True)
        self._current_matches_df = None


    def _trigger_online_search(self):
        """Starts the online search process in a background thread."""
        # --- Pre-checks ---
        if self._is_searching:
             logging.warning("Online search trigger ignored: Already searching.")
             return
        if not self._peaks_ref:
            QMessageBox.warning(self, "No Peaks Detected", "Cannot start search: Please detect peaks first.")
            return
        if not ASTROQUERY_AVAILABLE:
            QMessageBox.critical(self, "Dependency Missing", "Cannot start search: Astroquery library is required.\nInstall using: `pip install astroquery`")
            return

        # --- Prepare Search ---
        self._stop_running_search() # Ensure any previous search is stopped
        self.clear_results() # Clear previous results display
        tolerance = self.tolerance_dspin.value()
        peaks_for_worker = self._peaks_ref # Pass the current reference list

        logging.info(f"Starting online NIST search for {len(peaks_for_worker)} peaks with tolerance ±{tolerance:.3f} nm.")
        self._is_searching = True
        self._update_button_states()
        self.search_online_btn.setText("Searching...") # Update button text

        # --- Setup Progress Dialog ---
        self._progress_dialog = QProgressDialog("Searching NIST Online...", "Cancel", 0, len(peaks_for_worker), self)
        self._progress_dialog.setWindowTitle("Online Search Progress")
        self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal) # Block interaction with main window
        self._progress_dialog.setAutoClose(False) # Keep open until explicitly closed
        self._progress_dialog.setAutoReset(False) # Keep settings until explicitly reset
        self._progress_dialog.canceled.connect(self._stop_running_search) # Connect cancel button

        # --- Setup Worker & Thread ---
        self._online_search_thread = QThread()
        self._online_search_worker = OnlineSearchWorker(peaks_for_worker, tolerance, self.config)
        self._online_search_worker.moveToThread(self._online_search_thread)

        # --- Connect Worker Signals ---
        self._online_search_worker.results_ready.connect(self._handle_online_results)
        self._online_search_worker.search_progress.connect(self._update_progress)
        self._online_search_worker.search_finished.connect(self._online_search_finished)
        self._online_search_worker.error_occurred.connect(self._handle_search_error)

        # --- Connect Thread Signals ---
        self._online_search_thread.started.connect(self._online_search_worker.run)
        # Clean up worker/thread when thread finishes
        self._online_search_thread.finished.connect(self._cleanup_thread_worker)
        # Also ensure worker is deleted later to avoid memory leaks if movedToThread
        self._online_search_thread.finished.connect(self._online_search_worker.deleteLater)

        # --- Start Search ---
        self._online_search_thread.start()
        self._progress_dialog.setValue(0) # Initialize progress
        self._progress_dialog.show()


    @pyqtSlot(list) # List[NISTMatch]
    def _handle_online_results(self, raw_matches: List[NISTMatch]):
        """Processes the raw match list from the worker and populates the table."""
        logging.info(f"NIST view received {len(raw_matches)} raw matches from worker.")

        if not raw_matches:
            self._current_matches_df = pd.DataFrame(columns=self.columns) # Empty DF with correct columns
            self._populate_results_table(self._current_matches_df)
            self.online_results_obtained.emit([]) # Emit empty list
            return

        # Convert NISTMatch objects to dictionary rows suitable for DataFrame
        match_data_rows = []
        # Create a quick lookup map for peak data (intensity, wavelength) based on index
        peak_map = {p.index: p for p in self._peaks_ref if p is not None}

        for match in raw_matches:
            query_peak_idx = getattr(match, 'query_peak_index', None)
            peak_ref = peak_map.get(query_peak_idx) if query_peak_idx is not None else None

            # Get peak intensity and wavelength used for query
            peak_intensity = peak_ref.intensity_processed if peak_ref else np.nan
            peak_wavelength = peak_ref.wavelength_fitted_or_detected if peak_ref else getattr(match, 'query_peak_wavelength', np.nan)

            # Use the NISTMatch method to create the row dictionary
            # This method should handle calculating delta_lambda internally
            try:
                row_dict = match.to_dataframe_row(peak_wavelength, peak_intensity)
                if row_dict: # Ensure method returns a dict
                     match_data_rows.append(row_dict)
            except Exception as e:
                 logging.error(f"Error converting NISTMatch to DataFrame row: {e} for match {match!r}", exc_info=True)


        if not match_data_rows:
            logging.warning("No valid DataFrame rows created from received matches.")
            self._current_matches_df = pd.DataFrame(columns=self.columns)
            self._populate_results_table(self._current_matches_df)
            self.online_results_obtained.emit([])
            return

        # Create DataFrame and sort
        # Ensure columns match self.columns definition
        self._current_matches_df = pd.DataFrame(match_data_rows)
        try:
             # Sort by absolute difference first, then by peak wavelength
             self._current_matches_df.sort_values(
                 by=["Δλ (nm)", "Peak λ (nm)"],
                 key=lambda col: col.abs() if col.name == "Δλ (nm)" else col, # Apply abs only to delta
                 inplace=True,
                 na_position='last' # Put NaNs at the end
             )
        except KeyError as e:
             logging.error(f"Sorting failed, column missing: {e}. Displaying unsorted.")
        except Exception as e:
             logging.error(f"Error sorting NIST results: {e}", exc_info=True)

        # Populate the UI table with the sorted DataFrame
        self._populate_results_table(self._current_matches_df)

        # Emit the original raw list of NISTMatch objects for MainWindow correlation
        self.online_results_obtained.emit(raw_matches)


    @pyqtSlot(int, int)
    def _update_progress(self, current: int, total: int):
        """Updates the progress dialog."""
        if self._progress_dialog and self._progress_dialog.isVisible():
            self._progress_dialog.setMaximum(total)
            self._progress_dialog.setValue(current)


    @pyqtSlot(str)
    def _online_search_finished(self, status_msg: str):
        """Handles the completion of the online search (success, failure, cancel)."""
        logging.info(f"Online search finished signal received: {status_msg}")

        # Update state immediately
        self._is_searching = False
        self._update_button_states()
        self.search_online_btn.setText("Search Online (NIST ASD)") # Reset button text

        # Close and clean up progress dialog
        if self._progress_dialog:
            logging.debug("Closing progress dialog.")
            self._progress_dialog.close() # Close it cleanly
            self._progress_dialog = None

        # Optional: Update main window status bar
        parent = self.parent()
        if parent and hasattr(parent, 'update_status') and callable(parent.update_status):
            parent.update_status(status_msg, 5000)


    @pyqtSlot(str)
    def _handle_search_error(self, error_msg: str):
        """Logs errors reported by the worker (doesn't stop the search)."""
        # Errors reported here are typically non-fatal for the whole search
        logging.error(f"Online search query error reported: {error_msg}")
        # Optionally show a non-modal notification or log to a dedicated area?
        # For now, just log it. The final status message will indicate errors occurred.
        # QMessageBox.warning(self,"Online Search Query Error",f"A query failed:\n{error_msg}\nSearch will continue.")


    @pyqtSlot()
    def _cleanup_thread_worker(self):
        """Cleans up thread and worker references after thread finishes."""
        logging.debug("Cleaning up NIST search worker/thread objects.")
        # Worker might already be deleted by deleteLater, thread should be cleaned up by Qt
        self._online_search_worker = None
        self._online_search_thread = None
        # Ensure state reflects not searching, although _online_search_finished should handle this
        if self._is_searching:
             logging.warning("Cleanup called but _is_searching was still True.")
             self._is_searching = False
             self._update_button_states()


    def _stop_running_search(self):
        """Requests the running search thread/worker to stop."""
        if self._online_search_thread and self._online_search_thread.isRunning():
            logging.info("Requesting online search stop...")
            self._is_searching = False # Update state flag immediately
            if self._online_search_worker:
                self._online_search_worker.stop() # Tell worker to stop processing loop

            # Don't force quit here; let the worker finish its current step
            # and emit finished signal naturally for proper cleanup.
            # self._online_search_thread.quit()
            # self._online_search_thread.wait(500) # Brief wait optional

            # Cancel/close the progress dialog immediately
            if self._progress_dialog:
                logging.debug("Cancelling progress dialog due to stop request.")
                self._progress_dialog.cancel() # This closes the dialog

            self._update_button_states() # Reflect stop request in UI
            logging.info("Stop request sent to online search worker.")


    def _populate_results_table(self, df: pd.DataFrame):
        """Populates the results QTableWidget with data from the DataFrame."""
        self.results_table.setSortingEnabled(False) # Essential for performance
        self.results_table.setRowCount(0) # Clear existing rows efficiently

        if df is None or df.empty:
            logging.debug("Populating NIST table: DataFrame is empty.")
            self.results_table.setSortingEnabled(True)
            return

        logging.debug(f"Populating NIST table with {len(df)} rows.")
        self.results_table.setRowCount(len(df))
        col_indices = {name: i for i, name in enumerate(self.columns)} # Cache indices

        # Iterate through DataFrame rows
        for r_idx, row_data in df.iterrows(): # Use iterrows() for named access
            # Populate cells for the current row
            for col_name in self.columns:
                col_idx = col_indices.get(col_name)
                if col_idx is not None:
                    value = row_data.get(col_name) # Get value safely
                    source = row_data.get('Source', '?') # Get source for formatting
                    # Create and format the item using the helper
                    table_item = self._format_table_item(value, col_name, source)
                    self.results_table.setItem(r_idx, col_idx, table_item)

        # Restore sorting and resize columns after population
        self.results_table.resizeColumnsToContents()
        self.results_table.setSortingEnabled(True)


    def _format_table_item(self, value: Any, column_name: str, source: str) -> QTableWidgetItem:
        """Creates and formats a QTableWidgetItem based on value, column, and source."""
        item = QTableWidgetItem()
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable) # Read-only

        text = ""
        alignment = Qt.AlignmentFlag.AlignVCenter # Default vertical alignment

        # --- Formatting based on Type and Column ---
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            text = "N/A"
            alignment |= Qt.AlignmentFlag.AlignCenter
            item.setForeground(QBrush(QColor('gray')))
        elif isinstance(value, (int, np.integer)):
            text = str(value)
            alignment |= Qt.AlignmentFlag.AlignRight
        elif isinstance(value, (float, np.floating)):
            # Specific formatting based on column name
            if "Aki" in column_name: text = f"{value:.2e}"; alignment |= Qt.AlignmentFlag.AlignRight
            elif "λ" in column_name: text = f"{value:.4f}"; alignment |= Qt.AlignmentFlag.AlignRight # Wavelengths
            elif "Δλ" in column_name: text = f"{value:+.4f}"; alignment |= Qt.AlignmentFlag.AlignRight # Delta lambda show sign
            elif "E" in column_name or "g" in column_name: text = f"{value:.3f}"; alignment |= Qt.AlignmentFlag.AlignRight # Energy/Stat Weight
            else: text = f"{value:.3f}"; alignment |= Qt.AlignmentFlag.AlignRight # Default float
            item.setToolTip(str(value)) # Show full value in tooltip
        else: # Assume string
            text = str(value)
            # Specific alignment/coloring for certain string columns
            if column_name == "Source":
                alignment |= Qt.AlignmentFlag.AlignCenter
                # Example colors (adjust as needed)
                if "Online" in value or "NIST" in value: item.setForeground(QBrush(QColor("#008000"))) # Green for NIST
                elif "Local" in value: item.setForeground(QBrush(QColor("#0000FF"))) # Blue for Local
                else: alignment |= Qt.AlignmentFlag.AlignLeft # Default left-align other strings
            elif column_name == "Ion" or column_name == "Elem":
                alignment |= Qt.AlignmentFlag.AlignCenter
            else:
                alignment |= Qt.AlignmentFlag.AlignLeft

        item.setText(text)
        item.setTextAlignment(alignment)

        # Set data for numerical sorting (using DisplayRole which QTableWidget uses by default for sorting)
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
             item.setData(Qt.ItemDataRole.DisplayRole, float(value)) # Store as float for consistent sorting

        return item


    def _handle_selection_change(self):
        """Placeholder for handling selection changes in the results table."""
        # Currently not used, but could be used to highlight corresponding peaks on the main plot, etc.
        pass


    def closeEvent(self, event):
        """Ensures the search thread is stopped when the widget is closed."""
        logging.debug("NISTSearchView close event triggered.")
        self._stop_running_search()
        super().closeEvent(event)

# --- END OF REFACTORED FILE libs_cosmic_forge/ui/views/nist_search_view.py ---