# --- START OF REFACTORED FILE ui/views/boltzmann_plot_view.py ---
"""
View widget for performing Boltzmann plot analysis and displaying results.

Allows users to:
1. Specify a target species (e.g., "Fe I").
2. Populate a table with candidate spectral lines matching that species,
   using previously identified peaks and NIST correlations.
3. Select which lines to include in the Boltzmann plot.
4. Calculate the electron temperature (Te) based on a linear fit to the plot.
5. Visualize the Boltzmann plot and the calculated results.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, QTableWidget,
                             QHeaderView, QTableWidgetItem, QAbstractItemView, QLabel,
                             QHBoxLayout, QPushButton, QLineEdit, QMessageBox, QSplitter,
                             QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot
from PyQt6.QtGui import QIcon, QBrush, QColor

# Import UI Elements
from ui.widgets.info_button import InfoButton
# Use a dedicated plot widget instance for the Boltzmann plot
from ui.views.plot_widget import SpectrumPlotWidget

# Import Core Elements
# Data models are needed indirectly via the input DataFrame structure
# from core.data_models import Peak, NISTMatch
from core.cflibs import calculate_boltzmann_temp, K_B_EV # Import calculation & constant

class BoltzmannPlotView(QWidget):
    """Displays controls and results for Boltzmann plot temperature calculation."""

    # Signal emitted when the user clicks "Populate Lines"
    # Carries the target species string (e.g., "Fe I").
    populate_lines_requested = pyqtSignal(str)

    # Signal emitted after a calculation attempt.
    # Carries:
    #   success (bool): Whether a valid temperature was calculated.
    #   temperature_k (Optional[float]): Calculated temperature in Kelvin, or None.
    #   r_squared (Optional[float]): R-squared value of the fit, or None.
    #   plot_data (Optional[pd.DataFrame]): Data used for plotting, or None.
    calculation_complete = pyqtSignal(bool, object, object, object)

    # Minimum number of lines required for a valid fit
    MIN_LINES_FOR_FIT = 3

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config.get('cflibs', {}) # Get CF-LIBS sub-config (might contain related settings)
        self._candidate_lines_df: Optional[pd.DataFrame] = None # Stores DataFrame passed to display_candidate_lines
        self._plot_data: Optional[pd.DataFrame] = None # Stores data generated by calculate_boltzmann_temp

        self._init_ui()
        self._connect_signals()
        # Disable calculation initially until lines are populated and selected
        self.calculate_button.setEnabled(False)
        self.apply_theme_colors(self.config) # Apply initial theme

    def _init_ui(self):
        """Initializes the UI components."""
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(5)

        # --- Left Panel: Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0) # Add spacing to the right of left panel
        left_layout.setSpacing(10)

        # Input Group (Species Selection)
        input_group = QGroupBox("1. Setup & Input")
        input_layout = QFormLayout(input_group)
        input_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        input_layout.setSpacing(8)
        species_hbox = QHBoxLayout()
        self.species_input = QLineEdit()
        self.species_input.setPlaceholderText("e.g., Fe I, Ca II")
        self.species_input.setToolTip("Target species (Element + Ion State like 'Fe I', 'Ca II'). Case-insensitive.")
        species_info_btn = InfoButton(self._show_species_info, "Help on Target Species", self)
        species_hbox.addWidget(self.species_input, 1) # Give line edit more stretch
        species_hbox.addWidget(species_info_btn)
        input_layout.addRow("Target Species:", species_hbox)
        self.populate_button = QPushButton("Populate Lines")
        self.populate_button.setToolTip("Find identified lines matching the target species with necessary atomic data.")
        self.populate_button.setIcon(QIcon.fromTheme("edit-find-replace", QIcon.fromTheme("go-down"))) # Fallback icon
        input_layout.addRow(self.populate_button)
        left_layout.addWidget(input_group)

        # Lines Table Group (Candidate Lines)
        lines_group = QGroupBox("2. Select Lines for Plot")
        lines_layout = QVBoxLayout(lines_group)
        self.lines_table = QTableWidget()
        self.lines_columns = ["Use", "Peak λ (nm)", "Intensity", "Elem", "Ion", "DB λ (nm)", "E_k (eV)", "g_k", "A_ki (s⁻¹)"]
        self.lines_table.setColumnCount(len(self.lines_columns))
        self.lines_table.setHorizontalHeaderLabels(self.lines_columns)
        self.lines_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection) # Don't highlight rows
        self.lines_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) # Read-only table
        self.lines_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.lines_table.horizontalHeader().setStretchLastSection(False) # Prevent last column stretching initially
        self.lines_table.verticalHeader().setVisible(False)
        self.lines_table.setMinimumHeight(180)
        # Allow table to expand vertically and horizontally
        self.lines_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lines_layout.addWidget(self.lines_table)
        left_layout.addWidget(lines_group, 1) # Give table group more stretch factor

        # Calculation Group (Trigger and Result)
        calc_group = QGroupBox("3. Calculate Temperature")
        calc_layout = QVBoxLayout(calc_group)
        self.calculate_button = QPushButton("Calculate Tₑ")
        self.calculate_button.setIcon(QIcon.fromTheme("view-statistics", QIcon.fromTheme("applications-mathematics"))) # Fallback icon
        self.calculate_button.setToolTip(f"Perform Boltzmann plot fit using selected lines (min {self.MIN_LINES_FOR_FIT}).")
        self.result_label = QLabel("Result: Tₑ = N/A")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-weight: bold; font-size: 10pt; padding: 3px; border: 1px solid gray; border-radius: 3px;")
        calc_layout.addWidget(self.calculate_button)
        calc_layout.addWidget(self.result_label)
        left_layout.addWidget(calc_group)

        # --- Right Panel: Boltzmann Plot ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0) # Add spacing to the left of right panel
        plot_group = QGroupBox("Boltzmann Plot")
        plot_layout = QVBoxLayout(plot_group)
        # Use a SpectrumPlotWidget instance for consistency (axes, toolbar, theming)
        self.boltzmann_plot_widget = SpectrumPlotWidget(config=self.config, parent=self)
        # Set specific labels for Boltzmann plot
        self.boltzmann_plot_widget.ax.set_xlabel("Upper Energy Level E$_k$ (eV)")
        self.boltzmann_plot_widget.ax.set_ylabel(r"ln( I $\lambda$ / (A$_{ki}$ g$_k$) )")
        self.boltzmann_plot_widget.ax.set_title("Boltzmann Plot (Select Lines & Calculate)")
        plot_layout.addWidget(self.boltzmann_plot_widget)
        right_layout.addWidget(plot_group)

        # --- Assemble Splitter ---
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        # Give the plot panel more initial horizontal space (ratio 1:2)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)

        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(main_splitter)
        self.setLayout(outer_layout)

    def _connect_signals(self):
        """Connects internal signals and slots."""
        self.populate_button.clicked.connect(self._request_populate_lines)
        self.calculate_button.clicked.connect(self._trigger_calculation)
        # Connect itemChanged signal AFTER populating to avoid triggers during setup
        # self.lines_table.itemChanged.connect(self._handle_line_selection_change) -> Moved connection

    def _show_species_info(self):
        """Displays help message for the target species input."""
        QMessageBox.information(self, "Target Species Help",
            "Enter the element symbol and ionization state for the Boltzmann plot analysis.\n"
            "Format: 'Element IonState' (e.g., 'Fe I', 'Ca II', 'O 1', 'Si 2').\n\n"
            "The calculation requires multiple identified emission lines belonging to this *exact* species. "
            "These lines must also have known atomic data (Upper energy E\u2096, statistical weight g\u2096, transition probability A\u2096\u1d62) "
            "retrieved from the NIST search or other sources.\n\n"
            "Click 'Populate Lines' after entering the species to find suitable lines among the detected peaks.")

    def clear_all(self):
        """Resets the view to its initial state."""
        logging.debug("Clearing Boltzmann plot view.")
        self.species_input.clear()
        # Disconnect signal temporarily to avoid triggers during clear
        try:
            self.lines_table.itemChanged.disconnect(self._handle_line_selection_change)
        except TypeError: # Raised if not connected
             pass
        self.lines_table.setRowCount(0)
        self.lines_table.setSortingEnabled(False)
        self._candidate_lines_df = None
        self._plot_data = None
        self.result_label.setText("Result: Tₑ = N/A")
        self.calculate_button.setEnabled(False)

        self.boltzmann_plot_widget.clear_plot(redraw=False) # Clear plot without redrawing yet
        self.boltzmann_plot_widget.ax.set_xlabel("Upper Energy Level E$_k$ (eV)")
        self.boltzmann_plot_widget.ax.set_ylabel(r"ln( I $\lambda$ / (A$_{ki}$ g$_k$) )")
        self.boltzmann_plot_widget.ax.set_title("Boltzmann Plot")
        self.boltzmann_plot_widget._redraw_canvas() # Redraw after resetting labels/title

    @pyqtSlot()
    def _request_populate_lines(self):
        """Validates species input and emits the populate_lines_requested signal."""
        species = self.species_input.text().strip()
        if not species:
            QMessageBox.warning(self, "Missing Input", "Please enter the target species (e.g., 'Fe I').")
            return

        # Basic validation: should have two parts (Element, Ion State)
        parts = species.split()
        if len(parts) != 2:
            QMessageBox.warning(self, "Invalid Format",
                                "Species format seems incorrect. Please use 'Element IonState' (e.g., 'Fe I', 'Ca II').")
            return

        logging.info(f"Requesting population of candidate lines for species: {species}")
        self.populate_lines_requested.emit(species) # Emit signal for MainWindow


    def display_candidate_lines(self, lines_df: Optional[pd.DataFrame]):
        """
        Populates the lines table with candidate lines found for the target species.

        Args:
            lines_df: DataFrame containing candidate lines. Expected columns are
                      defined in `self.lines_columns` (excluding "Use").
        """
        # Disconnect signal while populating
        try:
            self.lines_table.itemChanged.disconnect(self._handle_line_selection_change)
        except TypeError:
            pass

        self.lines_table.setSortingEnabled(False) # Disable sorting during population
        self.lines_table.setRowCount(0) # Clear previous content
        self._candidate_lines_df = lines_df # Store the DataFrame
        self._plot_data = None # Clear previous plot data
        self.boltzmann_plot_widget.clear_plot() # Clear plot
        self.result_label.setText("Result: Tₑ = N/A") # Reset result label

        if lines_df is None or lines_df.empty:
            logging.info("No candidate lines provided to display for Boltzmann plot.")
            # Optionally show info message if lines were expected
            # QMessageBox.information(self,"No Lines Found", "No suitable lines with required atomic data were found for the specified species.")
            self.calculate_button.setEnabled(False)
            self.lines_table.setSortingEnabled(True) # Re-enable sorting
            return # Nothing more to do

        required_source_cols = {'Peak λ (nm)', 'Intensity', 'Elem', 'Ion', 'DB λ (nm)', 'E_k (eV)', 'g_k', 'A_ki (s⁻¹)'}
        missing_cols = required_source_cols - set(lines_df.columns)
        if missing_cols:
             logging.error(f"Input DataFrame for Boltzmann lines is missing required columns: {missing_cols}")
             QMessageBox.critical(self, "Data Error", f"The provided line data is missing essential columns:\n{', '.join(missing_cols)}")
             self.calculate_button.setEnabled(False)
             self.lines_table.setSortingEnabled(True) # Re-enable sorting
             return

        logging.info(f"Displaying {len(lines_df)} candidate lines for Boltzmann plot.")
        self.lines_table.setRowCount(len(lines_df))
        col_map = {name: idx for idx, name in enumerate(self.lines_columns)} # Map column name to index

        # Helper function to create table items with appropriate formatting and alignment
        def create_formatted_item(value: Any, precision: Optional[int] = None, scientific: bool = False, alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) -> QTableWidgetItem:
            item = QTableWidgetItem()
            text = ""
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                text = "N/A" # Or ""
                item.setForeground(QBrush(QColor('gray'))) # Gray out invalid data
            elif isinstance(value, (int, np.integer)):
                text = str(value)
            elif isinstance(value, (float, np.floating)):
                if scientific:
                    text = f"{value:.2e}" # Scientific notation
                elif precision is not None:
                    text = f"{value:.{precision}f}" # Fixed precision
                else:
                    text = f"{value:.4g}" # General format, up to 4 significant digits
            else:
                text = str(value)
                # Default left align for strings, center for Elem/Ion
                is_elem_ion = self.lines_columns[col_map["Elem"]] == 'Elem' or self.lines_columns[col_map["Ion"]] == 'Ion'
                alignment = Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter if is_elem_ion else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

            item.setText(text)
            item.setTextAlignment(alignment)
            # Tooltip can show full precision if formatted text truncates it
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                 item.setToolTip(str(value))

            # Set item flags (read-only)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable) # Ensure not editable
            return item

        # Populate the table row by row
        for row_idx, data_row in lines_df.iterrows():
            # Checkbox column ("Use")
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk_item.setCheckState(Qt.CheckState.Checked) # Default to checked
            chk_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.lines_table.setItem(row_idx, col_map["Use"], chk_item)

            # Data columns using the helper
            self.lines_table.setItem(row_idx, col_map["Peak λ (nm)"], create_formatted_item(data_row.get('Peak λ (nm)'), 4))
            self.lines_table.setItem(row_idx, col_map["Intensity"], create_formatted_item(data_row.get('Intensity'), 1)) # Low precision often okay
            self.lines_table.setItem(row_idx, col_map["Elem"], create_formatted_item(data_row.get('Elem')))
            self.lines_table.setItem(row_idx, col_map["Ion"], create_formatted_item(data_row.get('Ion')))
            self.lines_table.setItem(row_idx, col_map["DB λ (nm)"], create_formatted_item(data_row.get('DB λ (nm)'), 4))
            self.lines_table.setItem(row_idx, col_map["E_k (eV)"], create_formatted_item(data_row.get('E_k (eV)'), 4))
            self.lines_table.setItem(row_idx, col_map["g_k"], create_formatted_item(data_row.get('g_k'), 0)) # Usually integer
            self.lines_table.setItem(row_idx, col_map["A_ki (s⁻¹)"], create_formatted_item(data_row.get('A_ki (s⁻¹)'), scientific=True))

        self.lines_table.resizeColumnsToContents() # Adjust column widths
        self.lines_table.horizontalHeader().setStretchLastSection(True) # Allow last column (Aki) to stretch
        self.lines_table.setSortingEnabled(True) # Re-enable sorting

        # Connect signal AFTER population
        self.lines_table.itemChanged.connect(self._handle_line_selection_change)
        # Trigger initial check for button state
        self._handle_line_selection_change()


    def _handle_line_selection_change(self):
        """Updates the 'Calculate' button state based on the number of selected lines."""
        selected_count = 0
        use_col_idx = self.lines_columns.index("Use") # Get index of "Use" column

        for row in range(self.lines_table.rowCount()):
            item = self.lines_table.item(row, use_col_idx)
            if item and item.checkState() == Qt.CheckState.Checked:
                selected_count += 1

        min_lines = self.config.get('min_lines_for_boltzmann', self.MIN_LINES_FOR_FIT)
        can_calculate = selected_count >= min_lines
        self.calculate_button.setEnabled(can_calculate)
        logging.debug(f"{selected_count} lines selected for Boltzmann plot (min required: {min_lines}). Calculate button enabled: {can_calculate}")


    @pyqtSlot()
    def _trigger_calculation(self):
        """Gathers selected line data, triggers calculation, and displays results."""
        if self._candidate_lines_df is None or self._candidate_lines_df.empty:
            logging.warning("Calculation triggered, but no candidate line data is available.")
            return

        use_col_idx = self.lines_columns.index("Use")
        selected_indices = [
            r for r in range(self.lines_table.rowCount())
            if (item := self.lines_table.item(r, use_col_idx)) and item.checkState() == Qt.CheckState.Checked
        ]

        min_lines = self.config.get('min_lines_for_boltzmann', self.MIN_LINES_FOR_FIT)
        if len(selected_indices) < min_lines:
            QMessageBox.warning(self, "Not Enough Lines",
                                f"Please select at least {min_lines} lines from the table to perform the Boltzmann plot calculation.")
            return

        # Get the subset of the original DataFrame corresponding to selected rows
        selected_df = self._candidate_lines_df.iloc[selected_indices].copy()

        # --- Prepare DataFrame for calculation function ---
        # Define the mapping from display names to calculation function argument names
        rename_map = {
            'Intensity': 'intensity',
            'Peak λ (nm)': 'wavelength_nm', # Use peak wavelength for intensity term
            'E_k (eV)': 'ei_upper',
            'g_k': 'gi_upper',
            'A_ki (s⁻¹)': 'aki'
        }
        # Also need a label for plotting, use DB wavelength if available
        label_col_source = 'DB λ (nm)'

        required_calc_cols = set(rename_map.keys())
        missing_cols = required_calc_cols - set(selected_df.columns)
        if missing_cols:
            logging.error(f"Cannot calculate Boltzmann T: DataFrame is missing required columns: {missing_cols}")
            QMessageBox.critical(self, "Data Error", f"Cannot perform calculation. The line data is missing columns:\n{', '.join(missing_cols)}")
            return

        # Select and rename columns for the calculation function
        calc_df = selected_df[list(rename_map.keys())].rename(columns=rename_map)

        # Add label column if source exists
        if label_col_source in selected_df.columns:
             # Format label nicely, handle potential NaNs
             calc_df['label'] = selected_df[label_col_source].apply(lambda x: f"{x:.2f} nm" if pd.notna(x) else "?")
        else:
             calc_df['label'] = None # No labels available

        logging.info(f"Calculating Boltzmann temperature using {len(calc_df)} selected lines.")

        # --- Perform Calculation ---
        try:
            temp_k, temp_err, r_squared, plot_data_df = calculate_boltzmann_temp(calc_df)
            self._plot_data = plot_data_df # Store data used for plotting (for saving, etc.)
            success = temp_k is not None and np.isfinite(temp_k) # Define success condition
        except Exception as e:
            logging.error(f"Error during Boltzmann calculation: {e}", exc_info=True)
            QMessageBox.critical(self, "Calculation Error", f"An error occurred during the Boltzmann calculation:\n{e}")
            success = False
            temp_k, temp_err, r_squared, plot_data_df = None, None, None, None
            self._plot_data = None

        # --- Display Results ---
        if success:
            t_str = f"{temp_k:.0f}"
            err_str = f"{temp_err:.0f}" if temp_err is not None and np.isfinite(temp_err) else "N/A"
            r2_str = f"{r_squared:.3f}" if r_squared is not None and np.isfinite(r_squared) else "N/A"
            self.result_label.setText(f"Result: Tₑ ≈ {t_str} ± {err_str} K (R² = {r2_str})")
            logging.info(f"Boltzmann Calculation Result: T={t_str} ± {err_str} K, R²={r2_str}")
        else:
            r2_str = f"{r_squared:.3f}" if r_squared is not None and np.isfinite(r_squared) else "N/A"
            msg = "Calculation Failed"
            if r_squared is not None: msg += f" (R²={r2_str})" # Show R² even if temp is invalid
            self.result_label.setText(f"Result: {msg}")
            logging.warning(f"Boltzmann calculation failed or produced invalid temperature. R²={r2_str}")

        # Emit signal regardless of success, sending all results
        self.calculation_complete.emit(success, temp_k, r_squared, self._plot_data)

        # Update the plot visualization
        self._update_boltzmann_plot(self._plot_data, temp_k, r_squared)


    def _update_boltzmann_plot(self, plot_data: Optional[pd.DataFrame], temp_k: Optional[float], r_squared: Optional[float]):
        """Updates the Boltzmann plot with the calculated data and fit line."""
        plot_widget = self.boltzmann_plot_widget
        ax = plot_widget.ax
        plot_widget.clear_plot(redraw=False) # Clear previous plot elements

        # Set labels and default title
        ax.set_xlabel("Upper Energy Level E$_k$ (eV)")
        ax.set_ylabel(r"ln( I $\lambda$ / (A$_{ki}$ g$_k$) )")
        title = "Boltzmann Plot"

        if plot_data is None or plot_data.empty:
            ax.set_title(title + " (No data)")
            logging.debug("Updating Boltzmann plot: No data to plot.")
            plot_widget._redraw_canvas()
            return

        # Basic check for required plot columns
        if 'x_energy_ev' not in plot_data.columns or 'y_boltzmann_term' not in plot_data.columns:
             logging.error("Plot data DataFrame is missing required columns 'x_energy_ev' or 'y_boltzmann_term'.")
             ax.set_title(title + " (Data Error)")
             plot_widget._redraw_canvas()
             return

        x = plot_data['x_energy_ev'].values
        y = plot_data['y_boltzmann_term'].values

        # Filter out non-finite values to prevent plotting/fitting errors
        finite_mask = np.isfinite(x) & np.isfinite(y)
        x_plot = x[finite_mask]
        y_plot = y[finite_mask]

        if len(x_plot) == 0:
            ax.set_title(title + " (No valid data points)")
            logging.debug("Updating Boltzmann plot: No finite data points to plot.")
            plot_widget._redraw_canvas()
            return

        # --- Plotting ---
        # Get colors from theme manager if possible
        scatter_color = self.config.get('plot_colors', {}).get('scatter_point', 'blue')
        fit_color = self.config.get('plot_colors', {}).get('fit_line', 'red')

        # Plot scatter points
        scatter = ax.scatter(x_plot, y_plot, marker='o', color=scatter_color, label=f"Data ({len(x_plot)} points)")

        # Add text labels to points (optional, can be crowded)
        if 'label' in plot_data.columns and len(x_plot) < 25: # Only label if few points
            labels = plot_data['label'].values[finite_mask]
            try:
                for i, txt in enumerate(labels):
                     if txt: # Check if label is not None or empty
                         ax.text(x_plot[i], y_plot[i], f' {txt}', fontsize=7, va='bottom', ha='left', clip_on=True)
            except Exception as e:
                logging.warning(f"Could not add text labels to Boltzmann plot: {e}")


        # Plot fit line if calculation was successful and R² is reasonable
        fit_line = None
        plot_fit = (temp_k is not None and np.isfinite(temp_k) and
                    r_squared is not None and np.isfinite(r_squared) and
                    len(x_plot) >= 2) # Need at least 2 points for a line

        if plot_fit:
            try:
                # Calculate slope and intercept from temperature
                slope = -1.0 / (temp_k * K_B_EV)
                # Calculate intercept based on mean (more robust than using endpoint)
                intercept = np.mean(y_plot) - slope * np.mean(x_plot)

                # Create line points for plotting
                x_line = np.array([np.min(x_plot), np.max(x_plot)])
                y_line = intercept + slope * x_line

                # Plot the line
                fit_line, = ax.plot(x_line, y_line, '--', color=fit_color, lw=1.5, label=f'Fit (R²={r_squared:.3f})')
                # Update title with temperature
                title += f": T$_e$ ≈ {temp_k:.0f} K"
            except Exception as e:
                logging.warning(f"Could not calculate or plot Boltzmann fit line: {e}")
                title += " (Fit Error)"
        elif temp_k is None and r_squared is not None: # Indicate failed fit if R2 exists
            title += " (Invalid Fit)"

        ax.set_title(title)

        # Update legend
        handles = [scatter]
        if fit_line: handles.append(fit_line)
        plot_widget._update_legend(handles=handles, loc='best') # Use helper for consistency

        # Apply theme colors (important after plotting)
        self.apply_theme_colors(self.config)

        # Redraw the canvas
        plot_widget._redraw_canvas()
        logging.debug("Boltzmann plot updated.")

    def set_restored_data(self, temperature_k: Optional[float], plot_data: Optional[pd.DataFrame]):
         """
         Applies state loaded from a session file.

         Args:
             temperature_k: The loaded plasma temperature.
             plot_data: The loaded DataFrame used for the last plot calculation.
         """
         logging.info("Restoring state to Boltzmann view.")
         self._plot_data = plot_data # Store the loaded data
         # We don't have the original candidate lines or selected species easily,
         # so we can only restore the result label and the plot.

         # Update result label
         if temperature_k is not None and np.isfinite(temperature_k):
             # R-squared isn't typically saved directly with just temp, so show simple result
             t_str = f"{temperature_k:.0f}"
             self.result_label.setText(f"Result: Tₑ ≈ {t_str} K (Restored)")
             # Can't reliably determine R2 or success without re-calculating or saving more state
             success = True
             r_squared = None # Mark as unknown
         else:
             self.result_label.setText("Result: Tₑ = N/A (Restored)")
             success = False
             r_squared = None

         # Re-plot the loaded data
         self._update_boltzmann_plot(self._plot_data, temperature_k, r_squared)

         # Emit signal to notify MainWindow about the restored state
         self.calculation_complete.emit(success, temperature_k, r_squared, self._plot_data)
         # Calculation button remains disabled as we don't have the selection state
         self.calculate_button.setEnabled(False)


    def apply_theme_colors(self, config: Dict):
        """Applies color settings from the theme to the plot."""
        if hasattr(self, 'boltzmann_plot_widget') and self.boltzmann_plot_widget:
            self.boltzmann_plot_widget.apply_theme_colors(config) # Delegate to plot widget


# --- END OF REFACTORED FILE ui/views/boltzmann_plot_view.py ---