
"""
View widget for CF-LIBS calculations (Electron Density via Saha-Boltzmann, Concentrations).
This view provides controls and displays results for these advanced analyses.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, QTableWidget,
                             QHeaderView, QTableWidgetItem, QAbstractItemView, QLabel,
                             QHBoxLayout, QPushButton, QLineEdit, QDoubleSpinBox, QMessageBox)
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot
from PyQt6.QtGui import QIcon # Added QIcon

# Import UI Elements
from ui.widgets.info_button import InfoButton
# Import Core Elements
from core.data_models import Peak # Not directly used, but conceptually related

class CfLibsView(QWidget):
    """Displays controls and results for CF-LIBS analysis."""
    # Signal to request calculation of Electron Density (Ne)
    calculate_ne_requested = pyqtSignal(str, str, float) # species1, species2, temperature_k
    # Signal to request calculation of Concentrations
    calculate_conc_requested = pyqtSignal(float,object) # temperature_k, ne_cm3 (optional)
    # Signal to indicate calculation is complete (optional, for status updates)
    # calculation_complete = pyqtSignal(str, bool) # calculation_type, success (Maybe add later if needed)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config.get('cflibs', {}) # Get CF-LIBS sub-config
        # Store last calculated values if needed
        self.last_temp_k: Optional[float] = None
        self.last_ne_cm3: Optional[float] = None
        self.last_conc_df: Optional[pd.DataFrame] = None

        self._init_ui()
        self._connect_signals()
        self._update_button_states() # Initial state

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(15)

        # --- Electron Density (Ne) Group ---
        ne_group = QGroupBox("Electron Density (N\u2091) - Saha (Placeholder)")
        ne_layout = QFormLayout(ne_group)
        ne_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        ne_layout.setHorizontalSpacing(10)
        ne_layout.setVerticalSpacing(8)

        # Species Input
        self.ne_species1_input = QLineEdit()
        self.ne_species1_input.setPlaceholderText("e.g., Fe I (Lower Ion State)")
        self.ne_species2_input = QLineEdit()
        self.ne_species2_input.setPlaceholderText("e.g., Fe II (Higher Ion State)")
        species_info_btn = InfoButton(self._show_ne_info, tooltip_text="Help on Nₑ calculation", parent=self)
        species_hbox = QHBoxLayout()
        species_hbox.addWidget(self.ne_species1_input)
        species_hbox.addWidget(QLabel(" & "))
        species_hbox.addWidget(self.ne_species2_input)
        species_hbox.addWidget(species_info_btn)

        # Temperature Input (Required for Saha)
        self.ne_temp_input = QDoubleSpinBox()
        self.ne_temp_input.setRange(3000, 50000) # Reasonable plasma temp range
        self.ne_temp_input.setDecimals(0)
        self.ne_temp_input.setSuffix(" K")
        self.ne_temp_input.setToolTip("Plasma temperature (Tₑ) in Kelvin, needed for Saha equation.\nObtain from Boltzmann plot or enter manually.")
        # Connect valueChanged here or later? Later in _connect_signals

        self.calculate_ne_button = QPushButton("Calculate N\u2091")
        self.calculate_ne_button.setToolTip("Estimate electron density using selected species and temperature (Placeholder).")
        self.calculate_ne_button.setIcon(QIcon.fromTheme("view-statistics")) # Example icon
        self.ne_result_label = QLabel("N\u2091 Result: N/A")
        self.ne_result_label.setStyleSheet("font-weight: bold; padding: 3px;")

        ne_layout.addRow("Species Pair:", species_hbox)
        ne_layout.addRow("Plasma Temp (T\u2091):", self.ne_temp_input)
        ne_layout.addRow(self.calculate_ne_button)
        ne_layout.addRow(self.ne_result_label)
        main_layout.addWidget(ne_group)

        # --- Concentration Group ---
        conc_group = QGroupBox("Concentration (CF-LIBS) - (Placeholder)")
        conc_layout = QFormLayout(conc_group)
        conc_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        conc_layout.setHorizontalSpacing(10)
        conc_layout.setVerticalSpacing(8)

        # Inputs (Temperature is primary, Ne optional but recommended)
        conc_inputs_hbox = QHBoxLayout()
        self.conc_temp_label = QLabel("Using T\u2091:")
        self.conc_temp_value = QLabel("N/A") # Display temp used
        self.conc_temp_value.setStyleSheet("font-weight: bold;")
        self.conc_ne_label = QLabel("and N\u2091:")
        self.conc_ne_value = QLabel("N/A") # Display Ne used
        self.conc_ne_value.setStyleSheet("font-weight: bold;")
        conc_inputs_hbox.addWidget(self.conc_temp_label)
        conc_inputs_hbox.addWidget(self.conc_temp_value)
        conc_inputs_hbox.addSpacing(20)
        conc_inputs_hbox.addWidget(self.conc_ne_label)
        conc_inputs_hbox.addWidget(self.conc_ne_value)
        conc_inputs_hbox.addStretch()

        self.calculate_conc_button = QPushButton("Calculate Concentrations")
        self.calculate_conc_button.setToolTip("Estimate relative element concentrations (CF-LIBS Placeholder). Requires Tₑ and ideally Nₑ.")
        self.calculate_conc_button.setIcon(QIcon.fromTheme("view-statistics")) # Example icon
        conc_info_btn = InfoButton(self._show_conc_info, tooltip_text="Help on CF-LIBS calculation", parent=self)

        # Results Table
        self.conc_results_table = QTableWidget()
        self.conc_columns = ["Element", "Concentration (%)"]
        self.conc_results_table.setColumnCount(len(self.conc_columns))
        self.conc_results_table.setHorizontalHeaderLabels(self.conc_columns)
        self.conc_results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.conc_results_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.conc_results_table.verticalHeader().setVisible(False)
        self.conc_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch) # Stretch columns
        self.conc_results_table.setMinimumHeight(150)

        conc_button_hbox = QHBoxLayout()
        conc_button_hbox.addWidget(self.calculate_conc_button)
        conc_button_hbox.addStretch()
        conc_button_hbox.addWidget(conc_info_btn)
        conc_layout.addRow("Input Params:", conc_inputs_hbox)
        conc_layout.addRow(conc_button_hbox)
        conc_layout.addRow(self.conc_results_table)
        main_layout.addWidget(conc_group)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect button clicks and input changes."""
        self.calculate_ne_button.clicked.connect(self._request_ne_calculation)
        self.calculate_conc_button.clicked.connect(self._request_conc_calculation)
        self.ne_temp_input.valueChanged.connect(self._temp_input_changed)

    def _show_ne_info(self):
        QMessageBox.information(self, "Electron Density (Saha-Boltzmann)",
            "Estimates electron density (N\u2091) using the ratio of intensities "
            "from lines of two *consecutive ionization stages* of the *same element*.\n\n"
            "Requires:\n"
            "- Correct Species Pair (e.g., 'Fe I', 'Fe II').\n"
            "- Accurate atomic data (A\u2096\u1d62, g\u2096, E\u2096).\n"
            "- Plasma Temperature (T\u2091) in Kelvin.\n"
            "- Ionization energy and Partition functions (loaded from atomic_data files or defaults).\n\n"
            "**Note:** The current implementation uses a simplified formula and average line properties. Requires external data files (`partition_functions.csv`, `ionization_energies.csv`) in `database/atomic_data/` for accuracy.")

    def _show_conc_info(self):
        QMessageBox.information(self, "Concentration Estimation (CF-LIBS)",
            "Calibration-Free LIBS (CF-LIBS) determines relative elemental "
            "composition assuming LTE and Optically Thin plasma.\n\n"
            "Requires:\n"
            "- Identified lines with accurate atomic data (A\u2096\u1d62, g\u2096, E\u2096).\n"
            "- Plasma Temperature (T\u2091).\n"
            "- Electron Density (N\u2091) (currently optional in this placeholder).\n"
            "- Partition functions (loaded from `database/atomic_data/partition_functions.csv`).\n\n"
            "**Note:** The current implementation uses a simplified formula based on average line properties and assumes optical thinness. Requires external data files for accuracy.")


    def _update_button_states(self):
        """Enable/disable calculation buttons based on required inputs."""
        temp_valid = self.last_temp_k is not None and np.isfinite(self.last_temp_k) and self.last_temp_k > 0
        sp1_str = self.ne_species1_input.text().strip()
        sp2_str = self.ne_species2_input.text().strip()
        # Basic check for "Elem Ion" format
        sp1_valid = len(sp1_str.split()) == 2 and sp1_str.split()[1].isupper()
        sp2_valid = len(sp2_str.split()) == 2 and sp2_str.split()[1].isupper()

        self.calculate_ne_button.setEnabled(temp_valid and sp1_valid and sp2_valid)
        self.calculate_conc_button.setEnabled(temp_valid) # Ne is optional for conc calc placeholder


    @pyqtSlot(float)
    def _temp_input_changed(self, value: float):
        """Update internal temperature if user manually changes spinbox."""
        # Only update if different to avoid potential loops
        if self.last_temp_k is None or not np.isclose(self.last_temp_k, value):
            self.last_temp_k = value if value > 0 else None
            self.update_temperature(self.last_temp_k) # Update display labels too


    def update_temperature(self, temp_k: Optional[float]):
        """Receives the calculated temperature (e.g., from Boltzmann view) and updates UI."""
        self.last_temp_k = temp_k
        temp_str = f"{temp_k:.0f} K" if temp_k is not None and np.isfinite(temp_k) else "N/A"
        # Update spinbox only if value is different (and valid)
        current_spin_val = self.ne_temp_input.value()
        new_spin_val = temp_k if temp_k is not None and np.isfinite(temp_k) else 0
        if abs(current_spin_val - new_spin_val) > 1e-3: # Allow for float comparison issues
             self.ne_temp_input.blockSignals(True) # Prevent triggering _temp_input_changed
             self.ne_temp_input.setValue(new_spin_val)
             self.ne_temp_input.blockSignals(False)
        self.conc_temp_value.setText(temp_str) # Update display label
        self._update_button_states()


    def update_electron_density(self, ne_cm3: Optional[float]):
        """Updates the electron density display after calculation."""
        self.last_ne_cm3 = ne_cm3
        ne_str = f"{ne_cm3:.2e} cm⁻³" if ne_cm3 is not None and np.isfinite(ne_cm3) else "N/A"
        self.ne_result_label.setText(f"N\u2091 Result: {ne_str}")
        self.conc_ne_value.setText(ne_str) # Update display label for concentration input
        self._update_button_states()


    def display_concentrations(self, conc_df: Optional[pd.DataFrame]):
        """Displays the calculated concentrations in the table."""
        self.last_conc_df = conc_df; self.conc_results_table.setSortingEnabled(False); self.conc_results_table.setRowCount(0)
        if conc_df is None or conc_df.empty: logging.info("No concentration results."); item=QTableWidgetItem("(No results or failed)"); item.setTextAlignment(Qt.AlignmentFlag.AlignCenter); self.conc_results_table.setRowCount(1); self.conc_results_table.setItem(0,0,item); self.conc_results_table.setSpan(0,0,1,2); return

        logging.info(f"Displaying {len(conc_df)} concentration results."); self.conc_results_table.setRowCount(len(conc_df))
        for r_idx, (_, row) in enumerate(conc_df.iterrows()):
            elem=row.get("Element","?"); conc=row.get("Concentration"); conc_str=f"{conc*100:.2f}" if conc is not None and np.isfinite(conc) else "N/A";
            item_elem=QTableWidgetItem(str(elem)); item_elem.setTextAlignment(Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignVCenter); item_conc=QTableWidgetItem(conc_str); item_conc.setTextAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
            self.conc_results_table.setItem(r_idx,0,item_elem); self.conc_results_table.setItem(r_idx,1,item_conc)
        self.conc_results_table.resizeColumnsToContents(); self.conc_results_table.setSortingEnabled(True)


    @pyqtSlot()
    def _request_ne_calculation(self):
        """Gathers inputs and emits signal to calculate Ne."""
        species1 = self.ne_species1_input.text().strip(); species2 = self.ne_species2_input.text().strip(); temp_k = self.ne_temp_input.value()
        if not species1 or len(species1.split())!=2 or not species1.split()[1].isupper(): QMessageBox.warning(self,"Input Error","Enter valid Species 1 (e.g., 'Fe I')."); return
        if not species2 or len(species2.split())!=2 or not species2.split()[1].isupper(): QMessageBox.warning(self,"Input Error","Enter valid Species 2 (e.g., 'Fe II')."); return
        if temp_k <= 0 or not np.isfinite(temp_k): QMessageBox.warning(self,"Input Error","Enter valid Tₑ > 0 K."); return
        logging.info(f"Requesting Ne calc: {species1}/{species2}, T={temp_k:.0f}K"); self.calculate_ne_requested.emit(species1, species2, temp_k)


    @pyqtSlot()
    def _request_conc_calculation(self):
        """Gathers inputs and emits signal to calculate concentrations."""
        if self.last_temp_k is None or not np.isfinite(self.last_temp_k): QMessageBox.warning(self,"Input Error","Valid Tₑ required (from Boltzmann or manual input)."); return
        ne_val = self.last_ne_cm3 if self.last_ne_cm3 is not None and np.isfinite(self.last_ne_cm3) else None # Pass None if invalid/missing
        logging.info(f"Requesting Conc calc: T={self.last_temp_k:.0f}K, Ne={ne_val or 'N/A'}"); self.calculate_conc_requested.emit(self.last_temp_k, ne_val)


    def clear_all(self):
        """Clears inputs and results in this view."""
        self.ne_species1_input.clear(); self.ne_species2_input.clear(); self.ne_temp_input.setValue(0); self.ne_result_label.setText("N\u2091 Result: N/A"); self.conc_temp_value.setText("N/A"); self.conc_ne_value.setText("N/A"); self.conc_results_table.setRowCount(0); self.last_temp_k=None; self.last_ne_cm3=None; self.last_conc_df=None; self._update_button_states()

    def setEnabled(self, enabled: bool):
         """Enable/disable controls, but keep view itself generally enabled."""
         # super().setEnabled(enabled) # Don't disable the whole view
         self.ne_species1_input.setEnabled(enabled)
         self.ne_species2_input.setEnabled(enabled)
         self.ne_temp_input.setEnabled(enabled)
         self.calculate_ne_button.setEnabled(enabled and self.calculate_ne_button.isEnabled()) # Keep internal logic
         self.calculate_conc_button.setEnabled(enabled and self.calculate_conc_button.isEnabled())
         if not enabled: # Clear results if view is conceptually disabled
             self.clear_all()