
"""
View widget to display the list of detected peaks in a table.
Allows selection of peaks to view details or perform actions.
Includes highlighting of the selected row.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QTableWidget, QHeaderView,
                             QTableWidgetItem, QAbstractItemView, QLabel, QHBoxLayout,
                             QPushButton)
from PyQt6.QtCore import pyqtSignal, Qt

# Import data models
from core.data_models import Peak

class PeakListView(QWidget):
    """Displays detected peaks in a table and allows selection."""
    # Signal emitted when a peak row is selected in the table
    # Sends the original index of the selected Peak object within the MainWindow's list
    # Emits -1 if selection is cleared.
    peak_selected = pyqtSignal(int) # Emits list index of the selected peak
    # Signal to request clearing selection (e.g., when data is reprocessed)
    clear_selection_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._peak_list_ref: List[Peak] = [] # Reference to the main window's peak list
        # Use a dictionary to map the original peak list index to the table row index
        # This is more robust if the peak list itself doesn't change order but the table does (sorting)
        self._list_index_to_table_row: Dict[int, int] = {}
        self._selected_list_index: Optional[int] = None # Track selected original index

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0) # No margins for the dock content
        main_layout.setSpacing(5)

        # --- Table Widget ---
        self.peak_table = QTableWidget()
        # Define columns - Adjust based on what's most useful to see directly
        self.columns = [
            "Index",            # Original index in spectrum array
            "WL (Detected)",    # Detected Wavelength
            "WL (Fitted)",      # Fitted Wavelength
            "Intensity (Proc)", # Processed Intensity at peak
            "Amplitude (Fit)",  # Fitted Amplitude
            "FWHM (Fit)",       # Fitted FWHM
            "Fit Profile",      # Best fit profile
            "Fit R²",           # Best fit R²
        ]
        self.peak_table.setColumnCount(len(self.columns))
        self.peak_table.setHorizontalHeaderLabels(self.columns)

        # --- Table Settings ---
        self.peak_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) # Read-only
        self.peak_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.peak_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.peak_table.setAlternatingRowColors(True)
        self.peak_table.setSortingEnabled(True)
        self.peak_table.verticalHeader().setVisible(False)
        # Resize columns initially, allow interactive resize
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.peak_table.horizontalHeader().setStretchLastSection(False) # Don't stretch last initially
        # Set initial reasonable column widths
        self.peak_table.setColumnWidth(self.columns.index("Index"), 50)
        self.peak_table.setColumnWidth(self.columns.index("WL (Detected)"), 100)
        self.peak_table.setColumnWidth(self.columns.index("WL (Fitted)"), 100)
        self.peak_table.setColumnWidth(self.columns.index("Intensity (Proc)"), 100)
        self.peak_table.setColumnWidth(self.columns.index("Amplitude (Fit)"), 100)
        self.peak_table.setColumnWidth(self.columns.index("FWHM (Fit)"), 80)
        self.peak_table.setColumnWidth(self.columns.index("Fit Profile"), 90)
        self.peak_table.setColumnWidth(self.columns.index("Fit R²"), 60)


        main_layout.addWidget(self.peak_table)

        # --- Summary Label ---
        self.summary_label = QLabel("Peaks: 0")
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.summary_label.setStyleSheet("font-size: 8pt; color: gray;") # Smaller summary text
        main_layout.addWidget(self.summary_label)

        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect signals for table interaction."""
        self.peak_table.itemSelectionChanged.connect(self._handle_selection_change)
        self.clear_selection_requested.connect(self.clear_selection) # Connect external clear request

    def update_peak_list(self, peak_list: List[Peak]):
        """
        Populates the table with data from the provided list of Peak objects.

        Args:
            peak_list (List[Peak]): The list of detected/fitted Peak objects from MainWindow.
        """
        if peak_list is None: # Handle potential None input
            peak_list = []

        self._peak_list_ref = peak_list # Store reference to the list itself
        self.peak_table.setSortingEnabled(False)
        self.peak_table.clearContents()
        self.peak_table.setRowCount(len(peak_list))
        self._list_index_to_table_row.clear() # Clear the index map
        self._selected_list_index = None # Clear selection tracking

        logging.debug(f"Updating peak list view with {len(peak_list)} peaks.")

        for table_row_idx, peak_obj in enumerate(peak_list):
            # Find the actual index in the referenced list (should usually match table_row_idx initially)
            try:
                original_list_index = self._peak_list_ref.index(peak_obj)
                self._list_index_to_table_row[original_list_index] = table_row_idx # Map list_idx -> table_row
                self._populate_row(table_row_idx, peak_obj, original_list_index)
            except ValueError:
                 logging.error(f"Peak object not found in reference list during update. Skipping row {table_row_idx}.")
            except Exception as e:
                 logging.error(f"Error populating row {table_row_idx}: {e}", exc_info=True)


        self.peak_table.resizeColumnsToContents() # Adjust column widths after populating
        # Re-apply stretch if needed after resize
        # self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.peak_table.setSortingEnabled(True)
        self.summary_label.setText(f"Peaks Found: {len(peak_list)}")


    def _populate_row(self, table_row_idx: int, peak: Peak, original_list_index: int):
        """Populates a single row in the table with data from a Peak object."""

        # Helper to create items, handle None/NaN, and store original index
        def create_item(value, precision: Optional[int] = None, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, store_index=False):
            item = QTableWidgetItem()
            data_for_sorting = None # Store numeric data for sorting

            if value is None or (isinstance(value, float) and np.isnan(value)):
                text = "" # Display empty for None/NaN
            elif isinstance(value, (int, np.integer)):
                text = str(value)
                data_for_sorting = int(value)
            elif isinstance(value, (float, np.floating)):
                fmt = f".{precision}f" if precision is not None else ".4f" # Default 4 decimals for floats
                text = f"{value:{fmt}}"
                data_for_sorting = float(value)
            else:
                text = str(value) # String values
                alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                data_for_sorting = text # Sort strings alphabetically

            item.setText(text)
            item.setTextAlignment(alignment)
            if data_for_sorting is not None:
                # Store numeric/string data in DisplayRole or UserRole+1 for sorting
                # Using DisplayRole seems simplest if text representation is acceptable
                # Or use a dedicated sort role: Qt.ItemDataRole.UserRole + 1 ? Let's try DisplayRole.
                item.setData(Qt.ItemDataRole.DisplayRole, data_for_sorting)

            # Store original list index in UserRole of the first column's item ('Index')
            if store_index:
                item.setData(Qt.ItemDataRole.UserRole, original_list_index)

            return item

        # Populate columns based on self.columns list order
        col_map = {name: idx for idx, name in enumerate(self.columns)}

        # Use helper for each column
        self.peak_table.setItem(table_row_idx, col_map["Index"], create_item(peak.index, precision=0, alignment=Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignVCenter, store_index=True))
        self.peak_table.setItem(table_row_idx, col_map["WL (Detected)"], create_item(peak.wavelength_detected, 4))
        self.peak_table.setItem(table_row_idx, col_map["Intensity (Proc)"], create_item(peak.intensity_processed, 1))
        self.peak_table.setItem(table_row_idx, col_map["Intensity (Raw)"], create_item(peak.intensity_raw, 1))

        # Fit results (check if best_fit exists)
        fit = peak.best_fit
        fit_profile = fit.profile_type if fit and fit.success else None
        fit_center = fit.center if fit and fit.success else None
        fit_fwhm = fit.fwhm if fit and fit.success else None
        fit_r2 = fit.r_squared if fit and fit.success else None
        fit_amp = fit.amplitude if fit and fit.success else None # Get amplitude

        self.peak_table.setItem(table_row_idx, col_map["WL (Fitted)"], create_item(fit_center, 4))
        self.peak_table.setItem(table_row_idx, col_map["Amplitude (Fit)"], create_item(fit_amp, 1))
        self.peak_table.setItem(table_row_idx, col_map["FWHM (Fit)"], create_item(fit_fwhm, 4))
        self.peak_table.setItem(table_row_idx, col_map["Fit Profile"], create_item(fit_profile, alignment=Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignVCenter))
        self.peak_table.setItem(table_row_idx, col_map["Fit R²"], create_item(fit_r2, 3))


    def _handle_selection_change(self):
        """Emits the peak_selected signal with the ORIGINAL list index."""
        selected_items = self.peak_table.selectedItems()
        newly_selected_list_index: Optional[int] = None # Default to None (deselected)

        if selected_items:
            selected_row = selected_items[0].row()
            # Retrieve the original list index using UserRole data stored in the first column's item
            first_col_item = self.peak_table.item(selected_row, 0) # Column 0 is "Index"
            if first_col_item:
                 original_list_index = first_col_item.data(Qt.ItemDataRole.UserRole)
                 # Validate the retrieved index
                 if original_list_index is not None and isinstance(original_list_index, int) and 0 <= original_list_index < len(self._peak_list_ref):
                     newly_selected_list_index = original_list_index
                 else:
                      logging.warning(f"Invalid UserRole data or index out of range in table row {selected_row}. Data: {original_list_index}")
            else:
                 logging.warning(f"Could not get item for selected row {selected_row}, column 0 to retrieve original index.")

        # Emit signal only if the selection actually changed the effective list index
        if newly_selected_list_index != self._selected_list_index:
            self._selected_list_index = newly_selected_list_index # Update internal tracking
            if self._selected_list_index is not None:
                logging.debug(f"Peak list selection changed. Emitting original list index: {self._selected_list_index}")
                self.peak_selected.emit(self._selected_list_index)
            else:
                 # Selection was cleared or mapped index was invalid
                 logging.debug("Peak list selection cleared or invalid.")
                 self.peak_selected.emit(-1) # Emit -1 for deselection


    def clear_selection(self):
        """Clears the current selection in the table."""
        self.peak_table.clearSelection()
        # Selection change signal will handle updating internal state and emitting -1

    def select_peak_by_index(self, peak_list_index: int):
         """Selects the table row corresponding to the original peak list index."""
         target_table_row = None
         # Find the table row by checking UserRole data in the first column
         # This is robust to table sorting
         for r in range(self.peak_table.rowCount()):
             item = self.peak_table.item(r, 0) # Column 0 = "Index"
             if item and item.data(Qt.ItemDataRole.UserRole) == peak_list_index:
                 target_table_row = r
                 break

         if target_table_row is not None:
              # Prevent emission cascade: block signals, select, scroll, unblock
              self.peak_table.blockSignals(True)
              try:
                  self.peak_table.selectRow(target_table_row)
                  # Ensure the selected row is visible
                  self.peak_table.scrollToItem(self.peak_table.item(target_table_row, 0),
                                                 QAbstractItemView.ScrollHint.EnsureVisible)
                  # Manually update internal state IF NEEDED (usually signal handles it)
                  # if self._selected_list_index != peak_list_index:
                  #     self._selected_list_index = peak_list_index
                  logging.debug(f"Programmatically selected row {target_table_row} for list index {peak_list_index}")
              finally:
                    self.peak_table.blockSignals(False)
         else:
              logging.warning(f"Could not find table row for peak list index {peak_list_index} to select.")