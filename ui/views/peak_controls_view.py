"""
Control Panel View for Peak Detection settings using CollapsibleBox.
"""
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QPushButton, QLabel, QHBoxLayout, QComboBox , QMessageBox)
from PyQt6.QtCore import pyqtSignal, Qt

# Import helper functions/classes
from ui.widgets.info_button import InfoButton
from ui.widgets.collapsible_box import CollapsibleBox

class PeakDetectionControlPanel(QWidget):
    """Control panel for peak detection settings using collapsible sections."""
    detect_peaks_triggered = pyqtSignal(dict)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        # Get the peak_detection section, or an empty dict if it doesn't exist
        self.peak_detection_config = config.get('peak_detection', {})
        # Ensure self.config always refers to the peak_detection section or empty dict
        self.config = self.peak_detection_config

        self._init_ui()
        self._load_defaults()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5) # Reduced spacing

        # --- Detection Method Selection ---
        method_box = QGroupBox("Detection Method") # Simple group box for method selection
        method_layout = QFormLayout(method_box)
        method_layout.setContentsMargins(8, 8, 8, 8)
        method_layout.setHorizontalSpacing(10)
        method_layout.setVerticalSpacing(8)

        m_hbox = QHBoxLayout()
        self.peak_method_combo = QComboBox()
        self.peak_method_combo.addItems(["ScipyFindPeaks", "NISTGuided"]) # NISTGuided is placeholder
        self.peak_method_combo.setToolTip("Select peak detection algorithm.\nNISTGuided requires further implementation and NIST line data.")
        m_hbox.addWidget(self.peak_method_combo)
        m_hbox.addWidget(InfoButton(self._show_detect_method_info, tooltip_text="Help on Detection Methods"))
        method_layout.addRow("Method:", m_hbox)
        main_layout.addWidget(method_box)

        # --- Scipy Find Peaks Parameters Box ---
        self.scipy_box = CollapsibleBox("Scipy Parameters", self)
        scipy_content = QWidget()
        scipy_layout = QFormLayout(scipy_content)
        scipy_layout.setContentsMargins(5,5,5,5); scipy_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); scipy_layout.setHorizontalSpacing(10); scipy_layout.setVerticalSpacing(8)

        self.peak_rel_height_dspin = QDoubleSpinBox(); self.peak_rel_height_dspin.setRange(0.1, 100.0); self.peak_rel_height_dspin.setDecimals(1); self.peak_rel_height_dspin.setValue(5.0); self.peak_rel_height_dspin.setSuffix(" %"); self.peak_rel_height_dspin.setToolTip("Minimum peak height relative to spectrum intensity range (max-min).")
        self.peak_min_dist_spin = QSpinBox(); self.peak_min_dist_spin.setRange(1, 1000); self.peak_min_dist_spin.setValue(5); self.peak_min_dist_spin.setToolTip("Minimum required horizontal distance between neighboring peaks (in data points).")
        self.peak_min_width_spin = QSpinBox(); self.peak_min_width_spin.setRange(0, 100); self.peak_min_width_spin.setValue(0); self.peak_min_width_spin.setToolTip("Minimum required peak width (in data points). Set to 0 to disable.")
        self.peak_prominence_dspin = QDoubleSpinBox(); self.peak_prominence_dspin.setRange(0.0, 1e9); self.peak_prominence_dspin.setDecimals(2); self.peak_prominence_dspin.setValue(0.0); self.peak_prominence_dspin.setToolTip("Minimum required peak prominence (vertical distance to neighbors). Set to 0 to disable.")

        scipy_layout.addRow("Min Rel Height:", self.peak_rel_height_dspin)
        scipy_layout.addRow("Min Distance (pts):", self.peak_min_dist_spin)
        scipy_layout.addRow("Min Width (pts):", self.peak_min_width_spin)
        scipy_layout.addRow("Min Prominence:", self.peak_prominence_dspin)
        self.scipy_box.setContentLayout(scipy_layout)
        main_layout.addWidget(self.scipy_box)

        # --- NIST Guided Parameters Box ---
        self.nist_box = CollapsibleBox("NIST-Guided Parameters", self)
        nist_content = QWidget()
        nist_layout = QFormLayout(nist_content)
        nist_layout.setContentsMargins(5,5,5,5); nist_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); nist_layout.setHorizontalSpacing(10); nist_layout.setVerticalSpacing(8)

        self.nist_tolerance_dspin = QDoubleSpinBox(); self.nist_tolerance_dspin.setRange(0.01, 5.0); self.nist_tolerance_dspin.setDecimals(3); self.nist_tolerance_dspin.setValue(0.5); self.nist_tolerance_dspin.setSuffix(" nm"); self.nist_tolerance_dspin.setToolTip("Search window (+/- nm) around known NIST lines.")
        self.nist_prom_factor_dspin = QDoubleSpinBox(); self.nist_prom_factor_dspin.setRange(0.1, 1.0); self.nist_prom_factor_dspin.setDecimals(2); self.nist_prom_factor_dspin.setValue(0.5); self.nist_prom_factor_dspin.setToolTip("Minimum prominence factor relative to local range (placeholder).")
        # Add ComboBox to select which NIST elements/ions to guide search? Needs more thought.
        # self.nist_species_select = ...

        nist_layout.addRow("Tolerance (+/-):", self.nist_tolerance_dspin)
        nist_layout.addRow("Prominence Factor:", self.nist_prom_factor_dspin)
        self.nist_box.setContentLayout(nist_layout)
        main_layout.addWidget(self.nist_box)

        # --- Detect Button ---
        self.detect_button = QPushButton("Detect Peaks"); self.detect_button.setToolTip("Run peak detection using the current method and parameters."); self.detect_button.clicked.connect(self._emit_detect_signal); main_layout.addWidget(self.detect_button)
        main_layout.addStretch(); self.setLayout(main_layout)

        # Connect signals & update visibility
        self.peak_method_combo.currentTextChanged.connect(self._update_parameter_boxes)
        self._update_parameter_boxes(self.peak_method_combo.currentText())

    def _load_defaults(self):
        logging.debug("Loading peak detection defaults.")
        # Use self.config which is already the peak_detection section or {}
        method = self.config.get('default_method', 'ScipyFindPeaks')
        self.peak_method_combo.setCurrentText(method)

        # --- SciPy Defaults with None Check ---
        scipy_config = self.config.get('scipy_find_peaks', {})

        # Relative Height (DoubleSpinBox expects float)
        rel_height_val = scipy_config.get('rel_height_percent', 5.0)
        if rel_height_val is None:
            logging.warning("Config value for 'rel_height_percent' is None, using default 5.0.")
            rel_height_val = 5.0
        try:
            self.peak_rel_height_dspin.setValue(float(rel_height_val))
        except (TypeError, ValueError):
             logging.warning(f"Invalid config value '{rel_height_val}' for 'rel_height_percent', using default 5.0.")
             self.peak_rel_height_dspin.setValue(5.0)

        # Min Distance (SpinBox expects int)
        min_dist_val = scipy_config.get('min_distance_points', 5)
        if min_dist_val is None:
            logging.warning("Config value for 'min_distance_points' is None, using default 5.")
            min_dist_val = 5
        try:
            self.peak_min_dist_spin.setValue(int(min_dist_val))
        except (TypeError, ValueError):
             logging.warning(f"Invalid config value '{min_dist_val}' for 'min_distance_points', using default 5.")
             self.peak_min_dist_spin.setValue(5)

        # Width (SpinBox expects int)
        width_val = scipy_config.get('width', 0)
        if width_val is None:
            logging.warning("Config value for 'width' is None, using default 0.")
            width_val = 0
        try:
            self.peak_min_width_spin.setValue(int(width_val))
        except (TypeError, ValueError):
             logging.warning(f"Invalid config value '{width_val}' for 'width', using default 0.")
             self.peak_min_width_spin.setValue(0)


        # Prominence (DoubleSpinBox expects float)
        prominence_val = scipy_config.get('prominence', 0.0)
        if prominence_val is None:
            logging.warning("Config value for 'prominence' is None, using default 0.0.")
            prominence_val = 0.0
        try:
            self.peak_prominence_dspin.setValue(float(prominence_val))
        except (TypeError, ValueError):
             logging.warning(f"Invalid config value '{prominence_val}' for 'prominence', using default 0.0.")
             self.peak_prominence_dspin.setValue(0.0)


        # --- NIST Guided Defaults with None Check (Example) ---
        nist_config = self.config.get('nist_guided', {})

        nist_tol_val = nist_config.get('search_tolerance_nm', 0.5)
        if nist_tol_val is None: nist_tol_val = 0.5
        try:
            self.nist_tolerance_dspin.setValue(float(nist_tol_val))
        except (TypeError, ValueError):
             self.nist_tolerance_dspin.setValue(0.5)


        nist_prom_val = nist_config.get('min_prominence_factor', 0.5)
        if nist_prom_val is None: nist_prom_val = 0.5
        try:
             self.nist_prom_factor_dspin.setValue(float(nist_prom_val))
        except (TypeError, ValueError):
             self.nist_prom_factor_dspin.setValue(0.5)


        # Ensure initial visibility is correct
        self._update_parameter_boxes(method)

    def _update_parameter_boxes(self, method:str):
        """Shows/hides parameter boxes based on selected method."""
        self.scipy_box.setVisible(method=="ScipyFindPeaks")
        self.nist_box.setVisible(method=="NISTGuided")
        # Optionally expand the visible box
        if method == "ScipyFindPeaks" and not self.scipy_box.is_expanded: self.scipy_box.toggle_button.setChecked(True)
        elif method == "NISTGuided" and not self.nist_box.is_expanded: self.nist_box.toggle_button.setChecked(True)


    def get_detection_settings(self) -> dict:
        """Returns the current peak detection settings as a dictionary."""
        method=self.peak_method_combo.currentText(); settings={'method':method}
        if method=='ScipyFindPeaks':
            settings['rel_height_percent']=self.peak_rel_height_dspin.value()
            settings['min_distance_points']=self.peak_min_dist_spin.value()
            settings['min_width_points']=w if (w:=self.peak_min_width_spin.value())>0 else None
            settings['prominence']=p if (p:=self.peak_prominence_dspin.value())>0 else None
        elif method=='NISTGuided':
            settings['search_tolerance_nm']=self.nist_tolerance_dspin.value()
            settings['min_prominence_factor']=self.nist_prom_factor_dspin.value()
            # TODO: Get selected NIST species if added later
            settings['nist_lines_source']="Placeholder - Needs UI for species selection"
        logging.debug(f"Retrieved detection settings: {settings}"); return settings

    def _emit_detect_signal(self):
        settings = self.get_detection_settings()
        logging.info("Detect Peaks clicked.")
        self.detect_peaks_triggered.emit(settings)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        # Also enable/disable the button within the panel
        self.detect_button.setEnabled(enabled)

    # --- Info Callbacks ---
    def _show_detect_method_info(self):
        QMessageBox.information(self,"Detection Method Help","Select the algorithm used to find peaks in the processed spectrum:\n\n- **ScipyFindPeaks:** Standard algorithm finding local maxima based on parameters like relative height, minimum distance, width, and prominence.\n- **NISTGuided:** (Placeholder) Attempts to find peaks specifically around known spectral line wavelengths from the NIST database. Requires successful NIST search results.")