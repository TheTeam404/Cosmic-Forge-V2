# -*- coding: utf-8 -*-
import logging
import os
import sys
import traceback
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Callable, Union
from pathlib import Path

# Third-party Imports
import numpy as np
import pandas as pd
from PyQt6.QtCore import (
    QSize, Qt, pyqtSlot, QSettings, QByteArray, QPoint, QCoreApplication,
    QUrl, QProcess, QStandardPaths, QObject
)
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QCursor, QDesktopServices
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QStatusBar, QMenuBar,
    QMessageBox, QApplication, QFileDialog, QDockWidget, QToolBar, QMenu
)

# --- Constants ---
APP_VERSION = "0.2.5" # Incremented version for refactoring
ORGANIZATION_NAME = "CosmicForgeDev"
APPLICATION_NAME = "LIBSForge"
DEFAULT_THEME = "dark_cosmic"

# --- Application Setup ---
# Configure logging early
logging.basicConfig(
    level=logging.DEBUG, # Set to logging.INFO for production
    format='%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s'
)
log = logging.getLogger(__name__)

# Attempt to add project root to sys.path for local imports
try:
    # Assuming this script is in a subdirectory like 'app' or 'ui'
    _project_root_path = Path(__file__).parent.parent.resolve()
    if str(_project_root_path) not in sys.path:
        sys.path.insert(0, str(_project_root_path))
        log.info(f"Added project root to sys.path: {_project_root_path}")
except Exception as e:
    log.warning(f"Could not determine project root automatically: {e}")
    _project_root_path = Path(".") # Fallback

# --- Local Imports ---
try:
    from utils.helpers import get_project_root
    from ui.theme import ThemeManager
    from core.data_models import Spectrum, Peak, NISTMatch, FitResult
    from core.file_io import (
        load_spectrum_from_file, save_spectrum_data, save_peak_list,
        save_nist_matches, save_dataframe
    )
    from core.session_manager import SessionManager
    from ui.views.plot_widget import SpectrumPlotWidget
    from ui.views.control_panel_view import ProcessingControlPanel
    from ui.views.peak_controls_view import PeakDetectionControlPanel
    from ui.views.peak_fitting_controls_view import PeakFittingControlPanel
    from ui.views.peak_list_view import PeakListView
    from ui.views.nist_search_view import NistSearchView
    from ui.views.boltzmann_plot_view import BoltzmannPlotView
    from ui.views.ml_analysis_view import MLAnalysisView
    from ui.views.cf_libs_view import CfLibsView
    from ui.external_script_runner import ExternalScriptRunnerDialog
    from core.processing import baseline_poly, baseline_snip, smooth_savitzky_golay, denoise_wavelet
    from core.peak_detector import detect_peaks_scipy
    from core.peak_fitter import fit_peak
    from core.nist_manager import search_online_nist # Ensure exists
    from core.cflibs import calculate_electron_density_saha, calculate_cf_libs_conc # Ensure exists
except ImportError as e:
    log.critical(f"Failed to import necessary modules: {e}. Check installation and PYTHONPATH.", exc_info=True)
    # Attempt a more robust project root detection if initial import fails
    try:
        import utils.helpers
        log.info("Successfully imported utils.helpers on second attempt.")
    except ImportError:
        log.critical("Failed to import utils.helpers even after potential path adjustment. Icon loading might fail.")
        def get_project_root(): return _project_root_path # Use calculated fallback
    sys.exit(f"Import Error: {e}")

# --- Enums and Constants for Clarity ---
class DockName(Enum):
    PROCESSING = "processing"
    DETECTION = "detection"
    FITTING = "fitting"
    PEAK_LIST = "peak_list"
    NIST_SEARCH = "nist_search"
    BOLTZMANN = "boltzmann"
    CFLIBS = "cflibs"
    ML_ANALYSIS = "ml_analysis"

class PanelKey(Enum):
    PROCESSING = "processing_settings"
    DETECTION = "detection_settings"
    FITTING = "fitting_settings"
    NIST_SEARCH = "nist_search_settings"
    BOLTZMANN = "boltzmann_settings"
    CFLIBS = "cflibs_settings"
    ML_ANALYSIS = "ml_analysis_settings"

class SaveType(Enum):
    SESSION = "session"
    PROCESSED_SPECTRUM = "processed_spectrum"
    PEAKS = "peaks"
    NIST_MATCHES = "nist_matches"
    BOLTZMANN_DATA = "boltzmann"
    CONCENTRATIONS = "concentrations"
    PLOT = "plot"

# --- Helper Functions ---
def get_icon(name: str) -> QIcon:
    """Loads an icon from the assets folder or falls back to a theme icon."""
    try:
        project_root = get_project_root() # Call the helper function
        icon_path = project_root / "assets" / "icons" / name
        if icon_path.exists():
            return QIcon(str(icon_path))
        else:
            # Fallback: "load_spectrum.png" -> "document-open"
            theme_name = name.split('.')[0].replace('_', '-')
            fallback_icon = QIcon.fromTheme(theme_name) # Simpler fallback
            if fallback_icon.isNull():
                log.warning(f"Icon '{name}' not found in assets path '{icon_path}' or theme '{theme_name}'.")
                return QIcon() # Return empty icon if not found anywhere
            return fallback_icon
    except Exception as e:
         log.error(f"Error getting icon '{name}': {e}. Returning empty icon.", exc_info=True)
         return QIcon()

# --- Main Window Class ---
class MainWindow(QMainWindow):
    """Main application window for LIBS Forge."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the main window, UI components, and internal state."""
        super().__init__()
        log.info(f"Initializing LIBS Forge v{APP_VERSION}...")
        self.config = config

        self._setup_application_info()
        self.settings = QSettings() # Load after setting app info

        # --- Core Components ---
        self.theme_manager = ThemeManager(QApplication.instance(), self.config)
        self.session_manager = SessionManager(self)

        # --- UI Element Placeholders ---
        self._init_ui_elements()

        # --- Application State ---
        self._init_app_state()

        # --- Configuration & Defaults ---
        self._load_config_defaults()

        # --- UI State ---
        self._init_ui_state()

        # --- Build UI ---
        self._init_ui() # Creates menus, toolbars, docks, central widget
        self._connect_signals()
        self._load_persistent_settings() # Load theme, window geom/state, paths

        self.update_status(f"Welcome to LIBS Forge v{APP_VERSION}!")
        self._update_all_ui_states() # Initial enable/disable state for panels/actions
        log.info(f"LIBS Forge v{APP_VERSION} initialization complete.")

    def _setup_application_info(self):
        """Sets application name and organization for QSettings."""
        QCoreApplication.setOrganizationName(ORGANIZATION_NAME)
        QCoreApplication.setApplicationName(APPLICATION_NAME)
        QCoreApplication.setApplicationVersion(APP_VERSION)

    def _init_ui_elements(self):
        """Initializes UI element attributes to None."""
        self.plot_widget: Optional[SpectrumPlotWidget] = None
        self.processing_panel: Optional[ProcessingControlPanel] = None
        self.peak_detection_panel: Optional[PeakDetectionControlPanel] = None
        self.peak_fitting_panel: Optional[PeakFittingControlPanel] = None
        self.peak_list_view: Optional[PeakListView] = None
        self.nist_search_view: Optional[NistSearchView] = None
        self.boltzmann_view: Optional[BoltzmannPlotView] = None
        self.cf_libs_view: Optional[CfLibsView] = None
        self.ml_view: Optional[MLAnalysisView] = None
        self.docks: Dict[DockName, QDockWidget] = {}
        self.status_label: Optional[QLabel] = None
        self.panels_menu: Optional[QMenu] = None
        self.theme_actions: Dict[str, QAction] = {}
        self.main_toolbar: Optional[QToolBar] = None

        # --- Action Attributes ---
        self.load_action: Optional[QAction] = None
        self.load_multi_action: Optional[QAction] = None
        self.load_session_action: Optional[QAction] = None
        self.save_session_action: Optional[QAction] = None
        self.save_processed_action: Optional[QAction] = None
        self.save_peaks_action: Optional[QAction] = None
        self.save_nist_action: Optional[QAction] = None
        self.save_boltzmann_action: Optional[QAction] = None
        self.save_conc_action: Optional[QAction] = None
        self.save_plot_action: Optional[QAction] = None
        self.save_plot_toolbar_action: Optional[QAction] = None
        self.reset_zoom_action: Optional[QAction] = None
        # Add other actions if needed...

    def _init_app_state(self):
        """Initializes core application data state attributes."""
        self.current_spectrum: Optional[Spectrum] = None
        self.multi_spectra: List[Spectrum] = []
        self.detected_peaks: List[Peak] = []
        self.nist_matches: List[NISTMatch] = []
        self.plasma_temp_k: Optional[float] = None
        self.electron_density_cm3: Optional[float] = None
        self.boltzmann_plot_data: Optional[pd.DataFrame] = None
        self.cf_libs_concentrations: Optional[pd.DataFrame] = None
        # Placeholders for data potentially needed by core functions but not saved directly
        # self.partition_functions: Optional[Dict[str, float]] = None
        # self.ionization_energies: Optional[Dict[str, float]] = None

    def _load_config_defaults(self):
        """Loads default settings from the configuration dictionary."""
        file_io_cfg = self.config.get('file_io', {})
        app_cfg = self.config.get('application', {})

        self.default_delimiter = file_io_cfg.get('default_delimiter', '\t')
        self.default_comment_char = file_io_cfg.get('default_comment_char', '#')
        self.remember_window_state = app_cfg.get('remember_window_state', True)
        # Add other config loading here if needed

    def _init_ui_state(self):
        """Initializes UI-related state variables."""
        self._is_busy: bool = False
        try:
            default_dir = self._get_default_directory()
        except Exception as e:
            log.error(f"Error getting default directory: {e}", exc_info=True)
            default_dir = os.path.expanduser("~")
        self._last_save_dir: str = default_dir
        self._last_load_dir: str = default_dir
        self.external_process: Optional[QProcess] = None

    def _get_default_directory(self) -> str:
        """Returns the user's default documents or home directory."""
        try:
            # Use DocumentsLocation first
            docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
            if docs and os.path.isdir(docs):
                log.debug(f"Using default directory: {docs}")
                return docs
            # Fallback to HomeLocation
            home = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation)
            if home and os.path.isdir(home):
                log.debug(f"DocumentsLocation not found or invalid, using home directory: {home}")
                return home
            # Final fallback
            fallback = os.path.expanduser("~")
            log.warning(f"Could not find standard Documents or Home locations. Using fallback: {fallback}")
            return fallback
        except Exception as e:
             log.error(f"Error getting standard paths: {e}. Falling back to home dir.", exc_info=True)
             return os.path.expanduser("~")

    # --- UI Initialization ---

    def _init_ui(self):
        """Initializes the main UI layout, widgets, menus, toolbars, and docks."""
        self.setWindowTitle(f"{APPLICATION_NAME} v{APP_VERSION}")
        self._setup_geometry()
        self._setup_icon()
        self._setup_status_bar()
        self._setup_central_widget()
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_dock_widgets()
        self.setDockNestingEnabled(True)

    def _setup_geometry(self):
        """Sets the initial window size and position based on screen."""
        try:
            screen = QApplication.primaryScreen()
            if not screen:
                log.warning("Could not get primary screen. Using default size.")
                self.resize(1400, 900)
                return

            available_geo = screen.availableGeometry()
            width = int(available_geo.width() * 0.85)
            height = int(available_geo.height() * 0.85)
            x_pos = int(available_geo.left() + (available_geo.width() - width) / 2)
            y_pos = int(available_geo.top() + (available_geo.height() - height) / 2)
            self.setGeometry(x_pos, y_pos, width, height)
            log.debug(f"Initial geometry set based on screen: {x_pos},{y_pos} {width}x{height}")
        except Exception as e:
            log.warning(f"Could not determine screen geometry, using default size. Error: {e}", exc_info=True)
            self.resize(1400, 900)

    def _setup_icon(self):
        """Sets the application window icon."""
        self.setWindowIcon(get_icon("app_icon.png"))

    def _setup_status_bar(self):
        """Creates and configures the status bar."""
        statusBar = QStatusBar(self)
        self.setStatusBar(statusBar)
        self.status_label = QLabel("Initializing...")
        self.status_label.setObjectName("StatusLabel") # For potential styling
        # Add permanent widget to the right side
        statusBar.addPermanentWidget(self.status_label)

    def _setup_central_widget(self):
        """Creates the central plot widget."""
        try:
            self.plot_widget = SpectrumPlotWidget(self, self.config)
            self.setCentralWidget(self.plot_widget)
        except Exception as e:
            log.critical(f"Failed to create central plot widget: {e}", exc_info=True)
            QMessageBox.critical(self, "Initialization Error", f"Could not create the main plot area:\n{e}\n\nApplication cannot continue.")
            # Consider exiting or showing a fallback widget
            self.setCentralWidget(QLabel("Error: Plot widget failed to load."))


    def _create_menu_bar(self):
        """Creates the main menu bar and its submenus."""
        menubar = self.menuBar()
        if not menubar:
            log.error("Could not get menu bar.")
            return
        self._create_file_menu(menubar)
        self._create_view_menu(menubar)
        self._create_tools_menu(menubar)
        self._create_help_menu(menubar)

    def _create_file_menu(self, menubar: QMenuBar):
        """Creates the File menu."""
        file_menu = menubar.addMenu("&File")

        # Load Actions
        self.load_action = self._create_action(
            parent=self,
            text="&Load Spectrum...",
            icon_name="load_spectrum.png",
            shortcut=QKeySequence.StandardKey.Open,
            status_tip="Load a single spectrum file",
            triggered_slot=self.load_spectrum_action
        )
        file_menu.addAction(self.load_action)

        self.load_multi_action = self._create_action(
            parent=self,
            text="Load &Multiple Spectra...",
            icon_name="load_multi_spectra.png",
            status_tip="Load multiple spectra for ML analysis or comparison",
            triggered_slot=self._load_multiple_spectra_action
        )
        file_menu.addAction(self.load_multi_action)

        self.load_session_action = self._create_action(
            parent=self,
            text="Load Session...",
            icon_name="document-open.png", # Theme icon example
            shortcut="Ctrl+L",
            status_tip="Load a previously saved analysis session state",
            triggered_slot=self._on_load_session_triggered
        )
        file_menu.addAction(self.load_session_action)

        file_menu.addSeparator()

        # Save Menu
        save_menu = file_menu.addMenu(get_icon("save_figure.png"), "&Save") # Use icon for submenu

        self.save_session_action = self._create_action(
            parent=self,
            text="Save Session...",
            icon_name="document-save.png", # Theme icon example
            shortcut=QKeySequence.StandardKey.Save,
            status_tip="Save the current analysis state, data, and settings",
            triggered_slot=self._on_save_session_triggered,
            initial_enabled=False # Enabled based on state
        )
        save_menu.addAction(self.save_session_action)
        save_menu.addSeparator()

        # Individual Save Actions (using lambda for type)
        self.save_processed_action = self._create_action(self, "Processed Spectrum (.csv)", None, "Save wavelength, raw intensity, and processed intensity data", lambda: self._save_action(SaveType.PROCESSED_SPECTRUM), False)
        save_menu.addAction(self.save_processed_action)
        self.save_peaks_action = self._create_action(self, "Peak List (.csv)", None, "Save detected and fitted peak parameters", lambda: self._save_action(SaveType.PEAKS), False)
        save_menu.addAction(self.save_peaks_action)
        self.save_nist_action = self._create_action(self, "NIST Matches (.csv)", None, "Save the table of potential NIST line matches found", lambda: self._save_action(SaveType.NIST_MATCHES), False)
        save_menu.addAction(self.save_nist_action)
        self.save_boltzmann_action = self._create_action(self, "Boltzmann Data (.csv)", None, "Save the data points used for the Boltzmann plot calculation", lambda: self._save_action(SaveType.BOLTZMANN_DATA), False)
        save_menu.addAction(self.save_boltzmann_action)
        self.save_conc_action = self._create_action(self, "Concentrations (.csv)", None, "Save calculated CF-LIBS concentrations", lambda: self._save_action(SaveType.CONCENTRATIONS), False)
        save_menu.addAction(self.save_conc_action)
        self.save_plot_action = self._create_action(self, "Plot Image (.png, .svg)...", None, "Save the current main plot view as an image file", lambda: self._save_action(SaveType.PLOT), False)
        save_menu.addAction(self.save_plot_action)

        file_menu.addSeparator()

        # Exit Action
        exit_action = self._create_action(
            parent=self,
            text="E&xit",
            icon_name="exit.png",
            shortcut=QKeySequence.StandardKey.Quit,
            status_tip="Exit the application",
            triggered_slot=self.close # Built-in slot
        )
        file_menu.addAction(exit_action)

    def _create_view_menu(self, menubar: QMenuBar):
        """Creates the View menu."""
        view_menu = menubar.addMenu("&View")

        # Theme Submenu
        theme_menu = view_menu.addMenu("Themes")
        self._populate_theme_menu(theme_menu)

        # Panels Submenu
        self.panels_menu = view_menu.addMenu("Panels")
        # Actions added when docks are created

        view_menu.addSeparator()

        # Reset Zoom Action
        self.reset_zoom_action = self._create_action(
            parent=self,
            text="Reset Zoom",
            icon_name="reset_zoom.png",
            shortcut="Ctrl+H",
            status_tip="Reset plot zoom and pan to the full view"
        )
        # Connect later if plot widget exists
        view_menu.addAction(self.reset_zoom_action)

    def _populate_theme_menu(self, theme_menu: QMenu):
        """Populates the theme selection submenu."""
        theme_menu.clear()
        self.theme_actions.clear()
        available_themes = self.theme_manager.get_available_themes()

        if not available_themes:
            no_themes_action = QAction("No themes found", self)
            no_themes_action.setEnabled(False)
            theme_menu.addAction(no_themes_action)
            return

        current_theme = self.theme_manager.current_theme_name
        for theme_name in available_themes:
            action = QAction(theme_name.replace('_', ' ').title(), self, checkable=True)
            action.setChecked(theme_name == current_theme)
            # Use lambda with default argument to capture current theme_name
            action.triggered.connect(lambda checked=False, name=theme_name: self.change_theme(name))
            self.theme_actions[theme_name] = action
            theme_menu.addAction(action)

    def _create_tools_menu(self, menubar: QMenuBar):
        """Creates the Tools menu."""
        tools_menu = menubar.addMenu("&Tools")

        fetch_nist_action = self._create_action(
            parent=self,
            text="Fetch NIST Data (Script)...",
            icon_name="download.png",
            status_tip="Run the nist_data_fetcher.py script to download atomic data",
            triggered_slot=self.run_nist_fetcher
        )
        tools_menu.addAction(fetch_nist_action)

        build_data_action = self._create_action(
            parent=self,
            text="Build Atomic Data Files (Script)...",
            icon_name="database.png",
            status_tip="Run the atomic_data_builder.py script (placeholder)",
            triggered_slot=self.run_atomic_data_builder
        )
        tools_menu.addAction(build_data_action)
        # Add other tools as needed

    def _create_help_menu(self, menubar: QMenuBar):
        """Creates the Help menu."""
        help_menu = menubar.addMenu("&Help")

        about_act = self._create_action(
            parent=self,
            text=f"About {APPLICATION_NAME}",
            status_tip=f"Show information about {APPLICATION_NAME} v{APP_VERSION}",
            triggered_slot=self.show_about_dialog
        )
        help_menu.addAction(about_act)

        online_docs_act = self._create_action(
            parent=self,
            text="Online Documentation",
            icon_name="help-contents.png", # Theme icon example
            status_tip="Open the online documentation (placeholder)",
            triggered_slot=self._open_online_docs
        )
        help_menu.addAction(online_docs_act)

    def _create_tool_bar(self):
        """Creates the main application toolbar."""
        self.main_toolbar = QToolBar("Main Toolbar", self)
        self.main_toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.main_toolbar)

        # Add actions (check if they were created successfully)
        if self.load_action:
            self.load_action.setToolTip("Load single spectrum (Ctrl+O)")
            self.main_toolbar.addAction(self.load_action)

        # Save plot action for toolbar
        self.save_plot_toolbar_action = self._create_action(
            parent=self,
            text="Save Plot",
            icon_name="save_figure.png",
            tool_tip="Save the current plot as an image", # Use tool_tip for toolbars
            triggered_slot=lambda: self._save_action(SaveType.PLOT),
            initial_enabled=False
        )
        self.main_toolbar.addAction(self.save_plot_toolbar_action)

        self.main_toolbar.addSeparator()

        # Add plot navigation actions if plot widget exists
        if self.plot_widget and hasattr(self.plot_widget, 'toolbar') and self.plot_widget.toolbar:
            nav_toolbar = self.plot_widget.toolbar
            # Try to find actions by object name or tooltip
            home_action = nav_toolbar.actions()[0] # Usually the first action
            pan_action = next((a for a in nav_toolbar.actions() if "pan" in a.toolTip().lower()), None)
            zoom_action = next((a for a in nav_toolbar.actions() if "zoom" in a.toolTip().lower()), None)

            # Use our custom reset zoom action (preferred)
            if self.reset_zoom_action:
                 self.reset_zoom_action.setToolTip("Reset Zoom (Ctrl+H)")
                 self.main_toolbar.addAction(self.reset_zoom_action)
                 # Connect its trigger here where we know plot_widget exists
                 self.reset_zoom_action.triggered.connect(self.plot_widget.toolbar.home)
            elif home_action: # Fallback to matplotlib's home
                home_action.setIcon(get_icon("reset_zoom.png"))
                home_action.setToolTip("Reset Zoom (Ctrl+H)")
                self.main_toolbar.addAction(home_action)
                log.warning("Using plot toolbar's Home action for Reset Zoom.")
            else:
                log.warning("Could not find Reset Zoom/Home action.")

            if pan_action:
                pan_action.setIcon(get_icon("pan.png"))
                pan_action.setToolTip("Pan/Move Plot (Left-Click & Drag or Middle-Click)")
                self.main_toolbar.addAction(pan_action)
            else:
                log.warning("Could not find Pan action in plot toolbar.")

            if zoom_action:
                zoom_action.setIcon(get_icon("zoom.png"))
                zoom_action.setToolTip("Zoom Box (Right-Click & Drag)")
                self.main_toolbar.addAction(zoom_action)
            else:
                log.warning("Could not find Zoom action in plot toolbar.")
        else:
            log.warning("Plot widget or its toolbar not available when creating main toolbar.")
            # Add our reset zoom action anyway, but disable it initially
            if self.reset_zoom_action:
                self.reset_zoom_action.setToolTip("Reset Zoom (Ctrl+H)")
                self.reset_zoom_action.setEnabled(False)
                self.main_toolbar.addAction(self.reset_zoom_action)


    def _create_dock_widgets(self):
        """Creates and arranges all the dockable panels."""
        self.docks = {}
        left_area = Qt.DockWidgetArea.LeftDockWidgetArea
        right_area = Qt.DockWidgetArea.RightDockWidgetArea
        bottom_area = Qt.DockWidgetArea.BottomDockWidgetArea

        # --- Instantiate Panel Widgets ---
        try:
            self.processing_panel = ProcessingControlPanel(self.config)
            self.peak_detection_panel = PeakDetectionControlPanel(self.config)
            self.peak_fitting_panel = PeakFittingControlPanel(self.config)
            self.peak_list_view = PeakListView(self) # Needs main window reference
            self.nist_search_view = NistSearchView(self.config, self) # Needs main window reference
            self.boltzmann_view = BoltzmannPlotView(self.config, self) # Needs main window reference
            self.cf_libs_view = CfLibsView(self.config, self) # Needs main window reference
            self.ml_view = MLAnalysisView(self.config, self) # Needs main window reference
        except Exception as e:
            log.critical(f"Failed to instantiate one or more panel widgets: {e}", exc_info=True)
            QMessageBox.critical(self, "UI Initialization Error", f"Failed to create control panels:\n{e}")
            # Application might be unusable, consider exiting or disabling features
            return # Stop dock creation

        # --- Helper to Add Docks ---
        def add_dock(
            name: DockName, title: str, widget: Optional[QWidget], area: Qt.DockWidgetArea,
            shortcut_num: int, tabify_with: Optional[QDockWidget] = None
        ) -> Optional[QDockWidget]:
            if widget is None:
                log.error(f"Cannot add dock '{name.value}'. Widget is None.")
                return None
            if not isinstance(widget, QWidget):
                log.error(f"Cannot add dock '{name.value}'. Provided widget is not a QWidget (type: {type(widget)}).")
                return None

            dock = QDockWidget(title, self)
            dock.setObjectName(f"{name.value}Dock")
            dock.setWidget(widget)
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            # Allow floating, moving, closing
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                             QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                             QDockWidget.DockWidgetFeature.DockWidgetClosable)

            self.addDockWidget(area, dock)
            self.docks[name] = dock

            # Add toggle action to View -> Panels menu
            toggle_action = dock.toggleViewAction()
            toggle_action.setText(title)
            toggle_action.setShortcut(f"Ctrl+{shortcut_num}")
            if self.panels_menu:
                self.panels_menu.addAction(toggle_action)
            else:
                log.error("Panels menu ('View -> Panels') not initialized before adding dock toggle action.")

            # Tabify if requested
            if tabify_with:
                self.tabifyDockWidget(tabify_with, dock)

            # Start disabled, enable based on state later
            widget.setEnabled(False)
            return dock

        # --- Create and Add Docks ---
        proc_dock = add_dock(DockName.PROCESSING, 'Processing', self.processing_panel, left_area, 1)
        detect_dock = add_dock(DockName.DETECTION, 'Detection', self.peak_detection_panel, left_area, 2, proc_dock)
        fit_dock = add_dock(DockName.FITTING, 'Fitting', self.peak_fitting_panel, left_area, 3, detect_dock)

        list_dock = add_dock(DockName.PEAK_LIST, 'Peak List', self.peak_list_view, right_area, 4)
        nist_dock = add_dock(DockName.NIST_SEARCH, 'NIST Search', self.nist_search_view, right_area, 5, list_dock)

        boltzmann_dock = add_dock(DockName.BOLTZMANN, 'Boltzmann', self.boltzmann_view, bottom_area, 6)
        cflibs_dock = add_dock(DockName.CFLIBS, 'CF-LIBS', self.cf_libs_view, bottom_area, 7, boltzmann_dock)
        ml_dock = add_dock(DockName.ML_ANALYSIS, 'ML Analysis', self.ml_view, bottom_area, 8, cflibs_dock)

        # --- Initial Dock Visibility / Focus ---
        # Raise the first tab in each tabified group to make it visible initially
        if proc_dock: proc_dock.raise_()
        if list_dock: list_dock.raise_()
        if boltzmann_dock: boltzmann_dock.raise_()
        # Optionally hide some docks initially using settings or config
        # Example: if not self.settings.value("DockVisible/Boltzmann", True, type=bool):
        #    if boltzmann_dock: boltzmann_dock.hide()


    def _connect_signals(self):
        """Connects signals from UI elements to slots in the main window."""
        log.debug("Connecting signals...")

        # Processing Panel
        if self.processing_panel and hasattr(self.processing_panel, 'process_triggered'):
            self.processing_panel.process_triggered.connect(self.handle_process_request)
        else:
            log.warning("Processing panel or its 'process_triggered' signal not found for connection.")

        # Peak Detection Panel
        if self.peak_detection_panel and hasattr(self.peak_detection_panel, 'detect_peaks_triggered'):
            self.peak_detection_panel.detect_peaks_triggered.connect(self.handle_peak_detection_request)
        else:
            log.warning("Peak detection panel or its 'detect_peaks_triggered' signal not found.")

        # Peak Fitting Panel
        if self.peak_fitting_panel:
            if hasattr(self.peak_fitting_panel, 'fit_peaks_triggered'):
                self.peak_fitting_panel.fit_peaks_triggered.connect(self.handle_peak_fitting_request)
            if hasattr(self.peak_fitting_panel, 'refit_single_peak_requested'):
                self.peak_fitting_panel.refit_single_peak_requested.connect(self.handle_refit_single_peak)
            if hasattr(self.peak_fitting_panel, 'show_specific_fit'):
                self.peak_fitting_panel.show_specific_fit.connect(self.handle_show_specific_fit)
        else:
            log.warning("Peak fitting panel not found for signal connections.")

        # Peak List View
        if self.peak_list_view and hasattr(self.peak_list_view, 'peak_selected'):
            self.peak_list_view.peak_selected.connect(self.handle_peak_selection)
        else:
            log.warning("Peak list view or its 'peak_selected' signal not found.")

        # Plot Widget
        if self.plot_widget and hasattr(self.plot_widget, 'peak_clicked'):
            self.plot_widget.peak_clicked.connect(self.handle_peak_plot_click)
        else:
            log.warning("Plot widget or its 'peak_clicked' signal not found.")

        # NIST Search View
        if self.nist_search_view:
            if hasattr(self.nist_search_view, 'online_results_obtained'):
                self.nist_search_view.online_results_obtained.connect(self._handle_nist_search_results)
            else:
                 log.warning("NistSearchView exists but missing 'online_results_obtained' signal.")
        else:
            log.warning("NIST search view not found for signal connections.")

        # Boltzmann View
        if self.boltzmann_view:
            if hasattr(self.boltzmann_view, 'populate_lines_requested'):
                self.boltzmann_view.populate_lines_requested.connect(self.handle_boltzmann_populate_request)
            if hasattr(self.boltzmann_view, 'calculation_complete'):
                self.boltzmann_view.calculation_complete.connect(self._handle_boltzmann_result)
        else:
            log.warning("Boltzmann view not found for signal connections.")

        # CF-LIBS View
        if self.cf_libs_view:
            if hasattr(self.cf_libs_view, 'calculate_ne_requested'):
                self.cf_libs_view.calculate_ne_requested.connect(self.handle_ne_calculation_request)
            if hasattr(self.cf_libs_view, 'calculate_conc_requested'):
                self.cf_libs_view.calculate_conc_requested.connect(self.handle_conc_calculation_request)
        else:
            log.warning("CF-LIBS view not found for signal connections.")

        # ML Analysis View
        if self.ml_view and hasattr(self.ml_view, 'status_update'):
            self.ml_view.status_update.connect(self.update_status) # Connect to status bar
        else:
            log.warning("ML Analysis view or its 'status_update' signal not found.")

    # --- Utility / Helper Methods ---

    def _create_action(self, parent: QObject, text: str, icon_name: Optional[str] = None,
                      status_tip: Optional[str] = None, triggered_slot: Optional[Callable] = None,
                      initial_enabled: bool = True, shortcut: Optional[Union[QKeySequence, QKeySequence.StandardKey, str]] = None,
                      tool_tip: Optional[str] = None, checkable: bool = False) -> QAction:
        """Helper to create a QAction with common settings."""
        icon = get_icon(icon_name) if icon_name else QIcon()
        action = QAction(icon, text, parent)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if status_tip:
            action.setStatusTip(status_tip)
        if tool_tip:
            action.setToolTip(tool_tip) # Tooltips often preferred for toolbars
        elif status_tip:
             action.setToolTip(status_tip) # Fallback tooltip
        if triggered_slot:
            action.triggered.connect(triggered_slot)
        action.setEnabled(initial_enabled)
        action.setCheckable(checkable)
        return action

    def set_busy(self, busy: bool, message: str = "Working..."):
        """Sets the application's busy state and updates cursor/status."""
        self._is_busy = busy
        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.update_status(message, timeout=0) # Show indefinitely
        else:
            QApplication.restoreOverrideCursor()
            self.update_status("Ready.") # Or provide a more specific message
        QApplication.processEvents() # Ensure UI updates immediately

    def update_status(self, message: str, timeout: int = 5000):
        """Updates the status bar message."""
        if self.status_label:
            log.debug(f"Status Update: {message}" + (f" (Timeout: {timeout}ms)" if timeout > 0 else ""))
            # Use permanent label for indefinite messages or short fixed messages
            if timeout <= 0:
                self.status_label.setText(message)
                if self.statusBar(): self.statusBar().clearMessage() # Clear temporary message area
            # Use temporary message area for timed messages
            elif self.statusBar():
                 self.statusBar().showMessage(message, timeout)
                 # Maybe clear the permanent label when showing temporary message?
                 # self.status_label.setText("") # Optional: clear permanent label
            else: # Fallback if status bar doesn't exist but label does
                 self.status_label.setText(message)
        else:
            log.warning(f"Status label not initialized. Message ignored: {message}")

    def show_about_dialog(self):
        """Displays the About dialog box."""
        try:
            from PyQt6 import __version__ as PYQT_VERSION_STR
            import scipy
            import matplotlib
            # Add other key library versions if desired
        except ImportError:
            PYQT_VERSION_STR = "N/A"
            scipy = None
            matplotlib = None

        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        project_root = get_project_root()

        about_text = f"""
        <h2>{APPLICATION_NAME} v{APP_VERSION}</h2>
        <p>Advanced LIBS Analysis Suite.</p>
        <p>Developed by: {ORGANIZATION_NAME}</p>
        <hr>
        <p><b>Environment:</b></p>
        <ul>
            <li>Python: {python_version}</li>
            <li>PyQt: {PYQT_VERSION_STR}</li>
        </ul>
        <p><b>Core Libraries:</b></p>
        <ul>
            <li>NumPy: {np.__version__ if np else 'N/A'}</li>
            <li>SciPy: {scipy.__version__ if scipy else 'N/A'}</li>
            <li>Pandas: {pd.__version__ if pd else 'N/A'}</li>
            <li>Matplotlib: {matplotlib.__version__ if matplotlib else 'N/A'}</li>
            <li>Astroquery (for NIST)</li>
            <li>Scikit-learn</li>
            <li>PyYAML</li>
            <li>PyWavelets</li>
            {f'<li>Project Root: {project_root}</li>' if project_root else ''}
        </ul>
        <p><i>Copyright (c) 2024 {ORGANIZATION_NAME}. All rights reserved.</i></p>
        """ # Added more details
        QMessageBox.about(self, f"About {APPLICATION_NAME} v{APP_VERSION}", about_text)

    def _open_online_docs(self):
        """Opens the online documentation URL in the default browser."""
        # Replace with your actual documentation URL
        url_string = "https://github.com/CosmicForge/libs-cosmic-forge#readme"
        url = QUrl(url_string)
        log.info(f"Attempting to open documentation URL: {url.toString()}")
        if not QDesktopServices.openUrl(url):
            log.error(f"Could not open URL: {url.toString()}")
            QMessageBox.warning(self, "Cannot Open Link",
                                f"Could not open the documentation link:\n{url.toString()}\n\nPlease open it manually in your browser.")

    def change_theme(self, theme_name: str):
        """Applies the selected theme and updates UI."""
        log.info(f"Changing theme to: {theme_name}")
        if theme_name not in self.theme_manager.get_available_themes():
            log.warning(f"Attempted to switch to non-existent theme: {theme_name}")
            return

        success = self.theme_manager.apply_theme(theme_name)
        if success:
            # self.config['default_theme'] = theme_name # Update runtime config? Optional.
            self._update_theme_menu_state()
            self._apply_theme_to_plots()
            self._save_theme_setting(theme_name) # Persist choice
        else:
            log.error(f"Theme manager failed to apply theme: {theme_name}")
            QMessageBox.warning(self, "Theme Error", f"Could not apply the theme '{theme_name}'. Check logs.")

    def _apply_theme_to_plots(self):
        """Applies the current theme's colors to relevant plot widgets."""
        log.debug("Applying theme colors to plots.")
        plots_to_update = [
            self.plot_widget,
            getattr(self.boltzmann_view, 'boltzmann_plot_widget', None),
            getattr(self.ml_view, 'results_plot_widget', None)
        ]
        theme_applied_count = 0
        for plot in plots_to_update:
            if plot and hasattr(plot, 'apply_theme_colors') and callable(plot.apply_theme_colors):
                try:
                    plot.apply_theme_colors(self.config) # Pass config if needed by plot
                    theme_applied_count += 1
                except Exception as e:
                    log.error(f"Error applying theme to plot {plot}: {e}", exc_info=True)
        log.debug(f"Applied theme colors to {theme_applied_count} plot widgets.")

    def _update_theme_menu_state(self):
        """Updates the check state of the theme menu actions."""
        current_theme = self.theme_manager.current_theme_name
        log.debug(f"Updating theme menu. Current theme: {current_theme}")
        if not self.theme_actions:
            log.warning("Theme actions dictionary is empty, cannot update menu.")
            return
        for name, action in self.theme_actions.items():
            action.setChecked(name == current_theme)

    # --- Settings Persistence ---

    def _load_persistent_settings(self):
        """Loads window geometry, state, paths, and theme from QSettings."""
        if not self.remember_window_state:
            log.info("Window state persistence is disabled in config. Skipping load.")
            # Apply default theme from config if persistence is off
            default_theme_name = self.config.get('default_theme', DEFAULT_THEME)
            log.info(f"Applying configured default theme: {default_theme_name}")
            self.theme_manager.apply_theme(default_theme_name) # Apply directly
            self._apply_theme_to_plots()
            self._update_theme_menu_state()
            log.info(f"Using default load/save directory: {self._last_load_dir}")
            return

        log.info("Loading persistent window settings using QSettings...")
        try:
            # Geometry
            geometry_data = self.settings.value("MainWindow/geometry", defaultValue=None)
            if isinstance(geometry_data, QByteArray) and not geometry_data.isNull():
                if self.restoreGeometry(geometry_data):
                    log.info("Restored window geometry.")
                else:
                    log.warning("Failed to restore window geometry from settings.")
            else:
                log.info("No valid geometry found in settings, using default layout.") # Default set in _setup_geometry

            # Window State (Docks, Toolbars)
            state_data = self.settings.value("MainWindow/windowState", defaultValue=None)
            if isinstance(state_data, QByteArray) and not state_data.isNull():
                 if self.restoreState(state_data):
                     log.info("Restored window state (docks, toolbars).")
                 else:
                    log.warning("Failed to restore window state from settings.")

            # Paths
            default_dir = self._get_default_directory() # Get a sensible default
            self._last_save_dir = self.settings.value("Paths/lastSaveDir", defaultValue=default_dir)
            self._last_load_dir = self.settings.value("Paths/lastLoadDir", defaultValue=default_dir)
            log.info(f"Restored last load directory: {self._last_load_dir}")
            log.info(f"Restored last save directory: {self._last_save_dir}")

            # Theme
            last_theme_name = self._load_last_theme() # Handles fallback
            log.info(f"Applying restored/default theme: {last_theme_name}")
            self.theme_manager.apply_theme(last_theme_name)
            self._apply_theme_to_plots()
            self._update_theme_menu_state()

        except Exception as e:
            log.error(f"Failed to load persistent window settings: {e}", exc_info=True)
            # Fallback to defaults if loading fails catastrophically
            self._setup_geometry() # Reset geometry
            # Apply default theme
            default_theme_name = self.config.get('default_theme', DEFAULT_THEME)
            self.theme_manager.apply_theme(default_theme_name)
            self._apply_theme_to_plots()
            self._update_theme_menu_state()


    def _save_persistent_settings(self):
        """Saves window geometry, state, paths, and theme to QSettings."""
        if not self.remember_window_state:
            log.info("Window state persistence is disabled. Skipping save.")
            return

        log.info("Saving persistent window settings using QSettings...")
        try:
            self.settings.setValue("MainWindow/geometry", self.saveGeometry())
            self.settings.setValue("MainWindow/windowState", self.saveState())
            self.settings.setValue("Paths/lastSaveDir", self._last_save_dir)
            self.settings.setValue("Paths/lastLoadDir", self._last_load_dir)
            self.settings.setValue("Appearance/lastTheme", self.theme_manager.current_theme_name)

            self.settings.sync() # Explicitly write to storage
            status = self.settings.status()
            if status == QSettings.Status.NoError:
                log.info("Window settings saved successfully.")
            else:
                log.error(f"QSettings sync error while saving window settings: {status}")
        except Exception as e:
            log.error(f"Failed to save persistent window settings: {e}", exc_info=True)

    def _save_theme_setting(self, theme_name: str):
        """Saves only the theme setting immediately."""
        if not self.remember_window_state:
            return
        try:
            self.settings.setValue("Appearance/lastTheme", theme_name)
            self.settings.sync()
            status = self.settings.status()
            if status == QSettings.Status.NoError:
                log.debug(f"Persisted theme setting: {theme_name}")
            else:
                log.error(f"QSettings sync error while saving theme: {status}")
        except Exception as e:
            log.error(f"Failed to save theme setting immediately: {e}", exc_info=True)

    def _load_last_theme(self) -> str:
        """Loads the last used theme from settings, falling back to config default."""
        config_default_theme = self.config.get('default_theme', DEFAULT_THEME)
        # If persistence is off, always return the config default
        if not self.remember_window_state:
            return config_default_theme

        try:
            last_theme = self.settings.value("Appearance/lastTheme", defaultValue=config_default_theme)
            # Validate the loaded theme exists
            if last_theme in self.theme_manager.get_available_themes():
                log.debug(f"Loaded last theme from settings: {last_theme}")
                return last_theme
            else:
                log.warning(f"Saved theme '{last_theme}' not found or invalid. Using default '{config_default_theme}'.")
                return config_default_theme
        except Exception as e:
            log.error(f"Failed to load last theme from settings: {e}. Using default '{config_default_theme}'.", exc_info=True)
            return config_default_theme

    # --- Application State Management ---

    def _reset_state_for_new_spectrum(self, spectrum: Optional[Spectrum]):
        """Resets application state, optionally loading a new spectrum."""
        is_clearing_all = spectrum is None and not self.multi_spectra
        log.info(f"Resetting application state. New single spectrum: {'Yes' if spectrum else 'No'}. Multi-spectra mode: {'Yes' if self.multi_spectra else 'No'}.")

        # Clear core analysis data
        self.current_spectrum = spectrum
        self.detected_peaks = []
        self.nist_matches = []
        self.plasma_temp_k = None
        self.electron_density_cm3 = None
        self.boltzmann_plot_data = None
        self.cf_libs_concentrations = None

        # Clear multi-spectra if loading a single spectrum or clearing all
        if spectrum is not None or is_clearing_all:
            self.multi_spectra = []
            if self.ml_view:
                self.ml_view.set_spectra_list([])
                self.ml_view.clear_all() # Clear results in ML view too

        # Update Title and Status
        status_msg = "Ready."
        window_title = f"{APPLICATION_NAME} v{APP_VERSION}"
        if spectrum:
            try:
                fname = os.path.basename(spectrum.filename) if spectrum.filename else "Untitled Spectrum"
                points = len(spectrum)
                status_msg = f"Loaded: {fname} ({points} points)"
                window_title = f"{APPLICATION_NAME} - {fname}"
                log.info(f"Spectrum loaded: {fname}, {points} points.")
            except AttributeError:
                 log.error("Invalid Spectrum object passed to _reset_state_for_new_spectrum.")
                 status_msg = "Error: Invalid spectrum data."
                 self.current_spectrum = None # Ensure it's None if invalid
        elif self.multi_spectra:
            status_msg = f"Ready for ML analysis ({len(self.multi_spectra)} spectra loaded)."
            log.info("Application state reset for multi-spectra mode.")
        else:
            status_msg = "State cleared. No spectrum loaded."
            log.info("Application state cleared.")

        self.update_status(status_msg)
        self.setWindowTitle(window_title)

        # Clear UI displays
        if self.plot_widget:
            self.plot_widget.clear_plot()
            if spectrum:
                self.plot_widget.plot_spectrum(spectrum) # Plot the new one
            self.plot_widget.plot_peaks([])
            self.plot_widget.clear_nist_matches()
            self.plot_widget.plot_fit_lines([])

        if self.peak_list_view: self.peak_list_view.update_peak_list([])
        if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(None)
        if self.nist_search_view:
            self.nist_search_view.set_peaks_reference([])
            self.nist_search_view.clear_results()
        if self.boltzmann_view: self.boltzmann_view.clear_all()
        if self.cf_libs_view: self.cf_libs_view.clear_all()

        # Update UI element states (panels, actions)
        self._update_all_ui_states()
        self._apply_theme_to_plots() # Reapply theme in case plot was cleared/recreated

    def _clear_downstream_analysis_data(self, clear_peaks: bool = True, clear_nist: bool = True, clear_plasma: bool = True):
        """Clears analysis results that depend on previous steps."""
        log.debug(f"Clearing downstream data: Peaks={clear_peaks}, NIST={clear_nist}, Plasma={clear_plasma}")

        if clear_peaks:
            self.detected_peaks = []
            if self.plot_widget: self.plot_widget.plot_peaks([])
            if self.peak_list_view: self.peak_list_view.update_peak_list([])
            if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(None)
            # If peaks are cleared, NIST/Plasma must also be cleared logically
            clear_nist = True

        if clear_nist:
            self.nist_matches = []
            if self.plot_widget: self.plot_widget.clear_nist_matches()
            if self.nist_search_view:
                self.nist_search_view.set_peaks_reference(self.detected_peaks) # Update reference (might be empty)
                self.nist_search_view.clear_results()
            # If NIST is cleared, Plasma must also be cleared
            clear_plasma = True

        if clear_plasma:
            self.plasma_temp_k = None
            self.electron_density_cm3 = None
            self.boltzmann_plot_data = None
            self.cf_libs_concentrations = None
            if self.boltzmann_view: self.boltzmann_view.clear_all()
            if self.cf_libs_view: self.cf_libs_view.clear_all()

        # Update UI states after clearing
        self._update_all_ui_states()

    def _handle_load_error(self, filepath: str, error: Any):
        """Handles errors during file loading."""
        error_str = str(error)
        log.error(f"Failed to load spectrum from '{filepath}': {error_str}", exc_info=True)
        QMessageBox.critical(
            self, "Spectrum Load Error",
            f"Error loading file:\n{os.path.basename(filepath)}\n\n"
            f"Details:\n{error_str}\n\n"
            "Check file format, delimiter ('{self.default_delimiter}'), "
            "and comment character ('{self.default_comment_char}')."
        )
        self.update_status(f"Error loading {os.path.basename(filepath)}.", 5000)
        self._reset_state_for_new_spectrum(None) # Clear everything on load error


    def _update_all_ui_states(self):
        """Updates the enabled/disabled state of all relevant UI elements."""
        log.debug("Updating all UI element enabled states.")
        self._update_panel_enable_states()
        self._update_save_actions_state()
        # Update other actions or elements if necessary
        if self.reset_zoom_action:
             self.reset_zoom_action.setEnabled(self.plot_widget is not None)


    def _update_panel_enable_states(self):
        """Enables or disables dock panels based on the current application state."""
        spectrum_loaded = self.current_spectrum is not None
        multi_loaded = bool(self.multi_spectra)
        # Peaks detected requires successful fits for some downstream panels
        peaks_exist = bool(self.detected_peaks)
        peaks_fitted = peaks_exist and any(p.best_fit and p.best_fit.success for p in self.detected_peaks)
        # NIST needs peaks fitted AND correlated matches for some downstream panels
        nist_correlated = peaks_fitted and any(p.potential_matches for p in self.detected_peaks)
        # Plasma requires NIST correlation
        plasma_temp_exists = self.plasma_temp_k is not None and np.isfinite(self.plasma_temp_k)
        ne_exists = self.electron_density_cm3 is not None and np.isfinite(self.electron_density_cm3)


        states = {
            DockName.PROCESSING: spectrum_loaded and not multi_loaded,
            DockName.DETECTION: spectrum_loaded and not multi_loaded and self.current_spectrum.processed_intensity is not None,
            DockName.PEAK_LIST: spectrum_loaded and not multi_loaded, # Show list even if empty
            DockName.FITTING: spectrum_loaded and peaks_exist and not multi_loaded,
            DockName.NIST_SEARCH: spectrum_loaded and peaks_fitted and not multi_loaded, # Needs fits to search
            DockName.BOLTZMANN: spectrum_loaded and nist_correlated and not multi_loaded, # Needs correlated NIST data
            DockName.CFLIBS: spectrum_loaded and nist_correlated and plasma_temp_exists and not multi_loaded, # Needs T, potentially Ne
            DockName.ML_ANALYSIS: multi_loaded and not spectrum_loaded,
        }

        log.debug(f"Updating panel states: {states}")

        for name, dock in self.docks.items():
            widget = dock.widget()
            if widget:
                is_enabled = states.get(name, False) # Default to False if name not in states
                widget.setEnabled(is_enabled)
            else:
                log.warning(f"Dock '{name.value}' has no valid widget to enable/disable.")

        # Enable/disable plot interaction based on whether peaks exist
        if self.plot_widget and hasattr(self.plot_widget, 'set_interaction_enabled'):
            self.plot_widget.set_interaction_enabled(peaks_exist)

    def _update_save_actions_state(self):
        """Enables or disables save-related menu actions based on available data."""
        has_spec = self.current_spectrum is not None
        has_proc = has_spec and self.current_spectrum.processed_intensity is not None
        has_peaks = bool(self.detected_peaks)
        has_nist = bool(self.nist_matches)
        has_boltz_data = self.boltzmann_plot_data is not None and not self.boltzmann_plot_data.empty
        has_conc_data = self.cf_libs_concentrations is not None and not self.cf_libs_concentrations.empty
        has_multi = bool(self.multi_spectra)
        can_save_session = has_spec or has_multi # Can save session if any data is loaded

        action_states = {
            self.save_processed_action: has_proc,
            self.save_peaks_action: has_peaks,
            self.save_nist_action: has_nist,
            self.save_boltzmann_action: has_boltz_data,
            self.save_conc_action: has_conc_data,
            self.save_plot_action: has_spec, # Enable if any spectrum is plotted
            self.save_plot_toolbar_action: has_spec,
            self.save_session_action: can_save_session,
        }

        log.debug(f"Updating save actions state: { {a.text().split(' ')[0]: s for a, s in action_states.items() if a} }")

        for action, enabled in action_states.items():
            if action:
                action.setEnabled(enabled)

    # --- Session Load/Save ---

    @pyqtSlot()
    def _on_save_session_triggered(self) -> bool:
        """Handles the Save Session action."""
        if self._is_busy:
            log.warning("Save Session action ignored while busy.")
            return False
        if not self.current_spectrum and not self.multi_spectra:
            QMessageBox.information(self, "Save Session", "No spectrum or analysis data loaded to save.")
            return False

        log.info("Triggered Save Session action.")
        file_filter = f"{APPLICATION_NAME} Session (*{SessionManager.SESSION_FILE_EXTENSION});;All Files (*)"

        # Suggest filename based on current spectrum or default
        suggested_name = f"libs_forge_session{SessionManager.SESSION_FILE_EXTENSION}"
        if self.current_spectrum and self.current_spectrum.filename:
            base_name = os.path.splitext(os.path.basename(self.current_spectrum.filename))[0]
            suggested_name = f"{base_name}_session{SessionManager.SESSION_FILE_EXTENSION}"
        default_path = os.path.join(self._last_save_dir, suggested_name)

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Session As...", default_path, file_filter
        )

        if not filepath:
            log.info("Save Session action cancelled by user.")
            self.update_status("Save session cancelled.", 3000)
            return False

        # Ensure correct extension
        if selected_filter.startswith(f"{APPLICATION_NAME} Session") and not filepath.lower().endswith(SessionManager.SESSION_FILE_EXTENSION):
            filepath += SessionManager.SESSION_FILE_EXTENSION
            log.debug(f"Appended session file extension: {filepath}")

        self._last_save_dir = os.path.dirname(filepath)
        self.set_busy(True, f"Saving session to {os.path.basename(filepath)}...")
        save_successful = False
        try:
            # Gather state from MainWindow and potentially panels
            # The SessionManager should handle gathering state from the main window instance
            save_successful = self.session_manager.save_session(filepath)

            if save_successful:
                self.update_status(f"Session saved: {os.path.basename(filepath)}", 5000)
                log.info(f"Session successfully saved to {filepath}")
                return True
            else:
                # save_session should ideally raise an exception on failure
                QMessageBox.warning(self, "Save Warning", "Session saving reported failure, but no specific error was raised. Check logs.")
                self.update_status("Session save potentially failed.", 5000)
                return False
        except (IOError, PermissionError, Exception) as e:
            log.error(f"Failed to save session to {filepath}: {e}", exc_info=True)
            QMessageBox.critical(self, "Save Session Error", f"Could not save the session:\n{e}")
            self.update_status("Session save failed.", 5000)
            return False
        finally:
            self.set_busy(False)

    @pyqtSlot()
    def _on_load_session_triggered(self):
        """Handles the Load Session action."""
        if self._is_busy:
            log.warning("Load Session action ignored while busy.")
            return

        log.info("Triggered Load Session action.")
        file_filter = f"{APPLICATION_NAME} Session (*{SessionManager.SESSION_FILE_EXTENSION});;All Files (*)"
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Session", self._last_load_dir, file_filter
        )

        if not filepath:
            log.info("Load Session action cancelled by user.")
            self.update_status("Load session cancelled.", 3000)
            return

        self._last_load_dir = os.path.dirname(filepath)
        self.set_busy(True, f"Loading session from {os.path.basename(filepath)}...")

        try:
            # Load data dictionary from file
            session_state = self.session_manager.load_session_data(filepath)

            if session_state:
                # Apply the loaded state (this is the complex part)
                self._apply_loaded_session_state(session_state)
                self.update_status(f"Session loaded: {os.path.basename(filepath)}", 5000)
                log.info(f"Session successfully loaded from {filepath} and applied.")
            else:
                # load_session_data should raise error or return None/empty dict
                QMessageBox.warning(self, "Load Session Warning", f"Session file loaded but contained no valid data or failed validation:\n{filepath}")
                self.update_status("Session load failed (empty or invalid data).", 5000)
                self._reset_state_for_new_spectrum(None) # Reset state if load fails

        except FileNotFoundError:
            log.error(f"Session file not found: {filepath}")
            QMessageBox.critical(self,"Load Session Error", f"Session file not found:\n{filepath}")
            self.update_status("Session load failed (file not found).", 5000)
        except (ValueError, IOError, KeyError, Exception) as e: # Catch KeyError for missing dict keys
            log.error(f"Error reading or parsing session file {filepath}: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Session Error", f"Error loading session file:\n{e}\n\nFile may be corrupted or incompatible.")
            self.update_status("Session load failed (read error).", 5000)
            self._reset_state_for_new_spectrum(None) # Reset on error
        finally:
            self.set_busy(False)

    def _apply_loaded_session_state(self, state: Dict[str, Any]):
        """Applies the loaded session state dictionary to the application."""
        log.info("Applying loaded session state...")
        self._reset_state_for_new_spectrum(None) # Start with a clean slate before applying

        try:
            # 1. Restore Window Geometry and State
            self._restore_window_settings(state)

            # 2. Restore Paths and Theme
            self._restore_paths_theme(state)

            # 3. Reload Spectrum/Spectra Data
            # This is complex: needs to handle single vs multi, potential errors
            reloaded_single, reloaded_multi = self._reload_spectra_from_session(state)
            # Update state based on what was actually reloaded
            self.current_spectrum = reloaded_single
            self.multi_spectra = reloaded_multi

            # 4. Restore Panel Settings
            self._restore_panel_settings(state)

            # 5. Restore Analysis Results (Peaks, NIST, Plasma) - DEPENDS on spectrum reload
            if self.current_spectrum: # Only restore analysis if single spectrum was reloaded
                self._restore_analysis_results(state)
            elif self.multi_spectra and self.ml_view: # Potentially restore ML state
                 # Add ML state restoration if needed
                 log.debug("Restoring ML state (if implemented in session)...")
                 ml_settings = state.get(PanelKey.ML_ANALYSIS.value)
                 if ml_settings and hasattr(self.ml_view, 'set_settings'):
                     try:
                         self.ml_view.set_settings(ml_settings)
                     except Exception as e:
                         log.error(f"Session: Error restoring ML settings: {e}", exc_info=True)


            # 6. Update UI based on restored state
            self._update_ui_from_session()

            log.info("Finished applying loaded session state.")

        except Exception as e:
            log.error(f"Critical error applying loaded session state: {e}", exc_info=True)
            QMessageBox.critical(self, "Session Load Error", f"Failed to apply the loaded session state:\n{e}\n\nThe application state will be reset.")
            self._reset_state_for_new_spectrum(None) # Reset if applying fails

        finally:
            # Final UI state update after applying everything
             self._update_all_ui_states()


    def _restore_window_settings(self, state: Dict[str, Any]):
        """Restores window geometry and state from session data."""
        log.debug("Session: Restoring window geometry and state...")
        if 'window_geometry' in state and state['window_geometry']:
            try:
                geom_bytes = QByteArray.fromBase64(state['window_geometry'].encode('ascii'))
                if not self.restoreGeometry(geom_bytes):
                    log.warning("Session: Failed to restore window geometry.")
            except Exception as e:
                log.error(f"Session: Error restoring geometry: {e}", exc_info=True)

        if 'window_state' in state and state['window_state']:
            try:
                state_bytes = QByteArray.fromBase64(state['window_state'].encode('ascii'))
                if not self.restoreState(state_bytes):
                    log.warning("Session: Failed to restore window state (docks/toolbars).")
            except Exception as e:
                log.error(f"Session: Error restoring state: {e}", exc_info=True)

    def _restore_paths_theme(self, state: Dict[str, Any]):
        """Restores last used paths and theme from session data."""
        log.debug("Session: Restoring paths and theme...")
        # Paths
        default_dir = self._get_default_directory()
        self._last_load_dir = state.get('last_load_dir', default_dir)
        self._last_save_dir = state.get('last_save_dir', default_dir)
        log.debug(f"Session restored load/save dirs: {self._last_load_dir} / {self._last_save_dir}")

        # Theme
        session_theme = state.get('current_theme')
        if session_theme and session_theme in self.theme_manager.get_available_themes():
            log.debug(f"Session: Applying theme from session: {session_theme}")
            self.change_theme(session_theme) # Apply theme (handles UI updates)
        else:
            log.warning(f"Session: Theme '{session_theme}' not found or invalid. Applying last known good theme.")
            last_theme = self._load_last_theme() # Load from settings as fallback
            self.change_theme(last_theme)

    def _reload_spectra_from_session(self, state: Dict[str, Any]) -> Tuple[Optional[Spectrum], List[Spectrum]]:
        """Reloads single or multiple spectra based on paths in the session state."""
        log.debug("Session: Attempting to reload spectra...")
        reloaded_spectrum: Optional[Spectrum] = None
        reloaded_multi_spectra: List[Spectrum] = []

        # --- Try reloading single spectrum first ---
        spectrum_path = state.get('current_spectrum_path')
        if spectrum_path:
            if not os.path.exists(spectrum_path):
                log.error(f"Session: Spectrum path '{spectrum_path}' not found.")
                QMessageBox.warning(self, "Session Load Warning", f"Spectrum file not found:\n{spectrum_path}\n\nCannot fully restore analysis state.")
                # Continue loading other things, but analysis results won't be restored
            else:
                delimiter = state.get('current_spectrum_delimiter', self.default_delimiter)
                comment = state.get('current_spectrum_comment', self.default_comment_char)
                try:
                    log.info(f"Session: Reloading single spectrum from {spectrum_path}")
                    reloaded_spectrum = load_spectrum_from_file(spectrum_path, delimiter=delimiter, comment=comment)
                    log.info(f"Session restored single spectrum: {os.path.basename(spectrum_path)}")
                except Exception as e:
                    log.error(f"Session: Failed to reload spectrum file '{spectrum_path}': {e}", exc_info=True)
                    QMessageBox.critical(self, "Session Load Error", f"Failed to reload the spectrum file referenced in the session:\n{spectrum_path}\n\nError: {e}\n\nAnalysis results cannot be restored.")
                    reloaded_spectrum = None # Ensure it's None if reload fails

        # --- If no single spectrum, try reloading multiple spectra ---
        elif 'multi_spectra_paths' in state:
            multi_paths = state.get('multi_spectra_paths', [])
            if multi_paths:
                log.info(f"Session contained {len(multi_paths)} multi-spectra paths. Attempting reload...")
                errors = []
                delimiter = state.get('multi_spectra_delimiter', self.default_delimiter)
                comment = state.get('multi_spectra_comment', self.default_comment_char)

                for i, fp in enumerate(multi_paths):
                    self.update_status(f"Session: Reloading multi-spectrum {i+1}/{len(multi_paths)}...", 0)
                    QApplication.processEvents()
                    if not os.path.exists(fp):
                        error_msg = f"File not found: {os.path.basename(fp)}"
                        errors.append(error_msg)
                        log.warning(f"Session: Multi-spectrum path not found: {fp}")
                        continue
                    try:
                        spectrum = load_spectrum_from_file(fp, delimiter=delimiter, comment=comment)
                        reloaded_multi_spectra.append(spectrum)
                    except Exception as e:
                        error_msg = f"{os.path.basename(fp)}: {e}"
                        errors.append(error_msg)
                        log.warning(f"Session: Failed to reload multi-spectrum file {fp}: {e}")

                log.info(f"Session: Reloaded {len(reloaded_multi_spectra)}/{len(multi_paths)} multi-spectra.")
                if errors:
                    error_summary = "\n".join([f"- {e}" for e in errors[:5]])
                    if len(errors) > 5: error_summary += f"\n- ... ({len(errors)-5} more)"
                    QMessageBox.warning(self, "Session Load Warning", f"Could not reload all multi-spectra files referenced in session:\n{error_summary}")
            else:
                 log.info("Session: No multi-spectra paths found.")
        else:
            log.info("Session: No single spectrum path or multi-spectra paths saved.")

        return reloaded_spectrum, reloaded_multi_spectra


    def _restore_panel_settings(self, state: Dict[str, Any]):
        """Restores settings for each panel from session data."""
        log.debug("Session: Restoring panel settings...")
        panel_map = {
            PanelKey.PROCESSING: self.processing_panel,
            PanelKey.DETECTION: self.peak_detection_panel,
            PanelKey.FITTING: self.peak_fitting_panel,
            PanelKey.NIST_SEARCH: self.nist_search_view,
            PanelKey.BOLTZMANN: self.boltzmann_view,
            PanelKey.CFLIBS: self.cf_libs_view,
            PanelKey.ML_ANALYSIS: self.ml_view,
        }

        for key, panel_widget in panel_map.items():
            settings = state.get(key.value) # Use PanelKey enum value as key
            if settings and panel_widget:
                if hasattr(panel_widget, 'set_settings') and callable(panel_widget.set_settings):
                    try:
                        panel_widget.set_settings(settings)
                        log.debug(f"Session: Restored settings for panel: {key.name}")
                    except Exception as e:
                        log.error(f"Session: Error restoring settings for panel '{key.name}': {e}", exc_info=True)
                else:
                    log.warning(f"Session: Panel/View '{key.name}' exists but has no 'set_settings' method.")
            elif settings:
                # Log if settings exist but panel doesn't (shouldn't happen if UI init is robust)
                log.warning(f"Session: Settings found for panel '{key.name}', but the panel widget itself was not found or initialized.")


    def _restore_analysis_results(self, state: Dict[str, Any]):
        """Restores peaks, NIST matches, and plasma parameters from session data."""
        log.debug("Session: Restoring analysis results...")
        # --- Restore Peaks ---
        restored_peaks: List[Peak] = []
        if 'detected_peaks' in state:
            peak_data_list = state['detected_peaks']
            log.info(f"Session: Attempting to restore {len(peak_data_list)} peaks...")
            if hasattr(Peak, 'from_dict') and callable(Peak.from_dict):
                valid_peaks = 0
                for i, peak_data in enumerate(peak_data_list):
                    try:
                        p = Peak.from_dict(peak_data)
                        if p:
                            restored_peaks.append(p)
                            valid_peaks += 1
                        else:
                            log.warning(f"Session: Skipped invalid peak data (from_dict returned None) at index {i}: {peak_data}")
                    except (ValueError, TypeError, KeyError) as ve:
                        log.warning(f"Session: Skipping invalid peak data during restore at index {i}: {peak_data}. Error: {ve}")
                self.detected_peaks = restored_peaks
                log.info(f"Session: Successfully restored {valid_peaks}/{len(peak_data_list)} peak objects.")
            else:
                log.error("Session: Peak class does not have 'from_dict' method. Cannot restore peaks.")
                self.detected_peaks = []
        else:
            self.detected_peaks = []

        # --- Restore NIST Matches ---
        restored_matches: List[NISTMatch] = []
        if 'nist_matches' in state and state['nist_matches']:
            match_data_list = state['nist_matches']
            log.info(f"Session: Attempting to restore {len(match_data_list)} NIST matches...")
            if hasattr(NISTMatch, 'from_dict') and callable(NISTMatch.from_dict):
                valid_matches = 0
                for match_data in match_data_list:
                    try:
                        match = NISTMatch.from_dict(match_data)
                        if match:
                            restored_matches.append(match)
                            valid_matches += 1
                        else:
                             log.warning(f"Session: Skipped invalid NIST match data (from_dict returned None): {match_data}")
                    except Exception as e:
                        log.warning(f"Session: Skipping invalid NIST match data: {match_data}. Error: {e}")
                self.nist_matches = restored_matches
                log.info(f"Session: Successfully restored {valid_matches}/{len(match_data_list)} NIST Match objects.")
                # Re-correlate after restoring both peaks and matches
                self._correlate_nist_matches_to_peaks()
            else:
                log.error("Session: NISTMatch class does not have 'from_dict' method. Cannot restore matches.")
                self.nist_matches = []
        else:
            self.nist_matches = []

        # --- Restore Plasma Parameters and Derived Data ---
        self.plasma_temp_k = state.get('plasma_temp_k') # Can be None
        self.electron_density_cm3 = state.get('electron_density_cm3') # Can be None
        log.debug(f"Session: Restored T={self.plasma_temp_k} K, Ne={self.electron_density_cm3} cm⁻³")

        self.boltzmann_plot_data = None
        if 'boltzmann_plot_data_json' in state and state['boltzmann_plot_data_json']:
            try:
                self.boltzmann_plot_data = pd.read_json(state['boltzmann_plot_data_json'], orient='split')
                log.debug(f"Session: Restored Boltzmann plot data DataFrame ({len(self.boltzmann_plot_data)} points).")
            except Exception as e:
                log.error(f"Session: Failed to restore Boltzmann plot data from JSON: {e}")

        self.cf_libs_concentrations = None
        if 'cf_libs_concentrations_json' in state and state['cf_libs_concentrations_json']:
            try:
                self.cf_libs_concentrations = pd.read_json(state['cf_libs_concentrations_json'], orient='split')
                log.debug(f"Session: Restored CF-LIBS concentrations DataFrame ({len(self.cf_libs_concentrations)} points).")
            except Exception as e:
                log.error(f"Session: Failed to restore CF-LIBS concentrations from JSON: {e}")


    def _update_ui_from_session(self):
        """Updates UI elements to reflect the newly loaded session state."""
        log.debug("Session: Updating UI elements with restored data...")

        # Update main plot (if single spectrum was loaded)
        if self.plot_widget:
            self.plot_widget.clear_plot() # Clear first
            if self.current_spectrum:
                self.plot_widget.plot_spectrum(self.current_spectrum, show_raw=True, show_processed=True, show_baseline=True) # Show all layers
                self.plot_widget.plot_peaks(self.detected_peaks)
                self.plot_widget.plot_fit_lines(self.detected_peaks)
                # Plot NIST matches *after* correlation
                self.plot_widget.plot_nist_matches(self.nist_matches, correlate=False) # Already correlated
                self.setWindowTitle(f"{APPLICATION_NAME} - {os.path.basename(self.current_spectrum.filename)}")
            elif self.multi_spectra:
                 # Optionally plot an overview or the first multi-spectrum?
                 # self.plot_widget.plot_spectrum(self.multi_spectra[0]) # Example
                 self.setWindowTitle(f"{APPLICATION_NAME} - Multiple Spectra ({len(self.multi_spectra)})")
            else:
                 self.setWindowTitle(f"{APPLICATION_NAME} v{APP_VERSION}")

        # Update Peak List View
        if self.peak_list_view:
            self.peak_list_view.update_peak_list(self.detected_peaks)

        # Update NIST Search View
        if self.nist_search_view:
             self.nist_search_view.set_peaks_reference(self.detected_peaks) # Set reference peaks
             if hasattr(self.nist_search_view, 'display_results') and self.nist_matches:
                 try:
                     if hasattr(NISTMatch, 'to_dict'): # Check if method exists
                         nist_df = pd.DataFrame([m.to_dict() for m in self.nist_matches if hasattr(m, 'to_dict')])
                         self.nist_search_view.display_results(nist_df)
                     else:
                         log.warning("Cannot update NIST results table: NISTMatch object is missing 'to_dict' method.")
                 except Exception as e:
                     log.error(f"Session: Failed to update NIST view results table: {e}")

        # Update Boltzmann View
        if self.boltzmann_view and hasattr(self.boltzmann_view, 'set_restored_data'):
            self.boltzmann_view.set_restored_data(self.plasma_temp_k, self.boltzmann_plot_data)

        # Update CF-LIBS View
        if self.cf_libs_view and hasattr(self.cf_libs_view, 'set_restored_data'):
            self.cf_libs_view.set_restored_data(self.plasma_temp_k, self.electron_density_cm3, self.cf_libs_concentrations)

        # Update ML View (if multi-spectra were loaded)
        if self.ml_view and self.multi_spectra:
             self.ml_view.set_spectra_list(self.multi_spectra)
             # Potentially trigger re-display or re-calculation based on restored settings


        # Re-apply theme colors to plots
        self._apply_theme_to_plots()


    # --- Action Handler / Slot Implementations ---

    @pyqtSlot()
    def load_spectrum_action(self):
        """Handles the Load Single Spectrum action."""
        if self._is_busy:
            log.warning("Load Spectrum action ignored while busy.")
            return
        log.info("Triggered Load Spectrum action.")
        self.update_status("Opening file dialog...")

        file_filter = "Data Files (*.txt *.csv *.asc);;All Files (*)"
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Spectrum File", self._last_load_dir, file_filter
        )

        if not filepath:
            log.info("Load Spectrum action cancelled by user.")
            self.update_status("Load cancelled.", 3000)
            return

        self._last_load_dir = os.path.dirname(filepath)
        self.set_busy(True, f"Loading {os.path.basename(filepath)}...")
        spectrum = None
        try:
            spectrum = load_spectrum_from_file(
                filepath,
                delimiter=self.default_delimiter,
                comment_char=self.default_comment_char
            )
            # Reset state completely and load the new spectrum
            self._reset_state_for_new_spectrum(spectrum)
            # No need to call _update_panel_enable_states here, reset does it.
        except FileNotFoundError:
            self._handle_load_error(filepath, "File not found.")
        except (IOError, ValueError, IndexError, Exception) as e:
            self._handle_load_error(filepath, e)
        finally:
            self.set_busy(False)

    @pyqtSlot()
    def _load_multiple_spectra_action(self):
        """Handles the Load Multiple Spectra action."""
        if self._is_busy:
            log.warning("Load Multiple Spectra action ignored while busy.")
            return
        log.info("Triggered Load Multiple Spectra action.")
        self.update_status("Opening file dialog for multiple spectra...")

        file_filter = "Data Files (*.txt *.csv *.asc);;All Files (*)"
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Load Multiple Spectra Files", self._last_load_dir, file_filter
        )

        if not filepaths:
            log.info("Load Multiple Spectra action cancelled by user.")
            self.update_status("Load cancelled.", 3000)
            return

        self._last_load_dir = os.path.dirname(filepaths[0])
        self.set_busy(True, f"Loading {len(filepaths)} spectra...")

        loaded_spectra: List[Spectrum] = []
        errors: List[str] = []
        delimiter = self.config.get('file_io', {}).get('default_delimiter', '\t')
        comment = self.config.get('file_io', {}).get('default_comment_char', '#')

        try:
            for i, fp in enumerate(filepaths):
                self.update_status(f"Loading {i+1}/{len(filepaths)}: {os.path.basename(fp)}...", 0)
                QApplication.processEvents() # Keep UI responsive
                try:
                    spectrum = load_spectrum_from_file(fp, delimiter=delimiter, comment=comment)
                    loaded_spectra.append(spectrum)
                except Exception as e:
                    error_msg = f"'{os.path.basename(fp)}': {e}"
                    errors.append(error_msg)
                    log.warning(f"Failed to load spectrum file {fp}: {e}", exc_info=True) # Log with traceback

            # --- Update State and UI ---
            num_loaded = len(loaded_spectra)
            num_attempted = len(filepaths)
            status_msg = f"Loaded {num_loaded}/{num_attempted} spectra for ML."
            if errors: status_msg += " Some errors occurred."
            self.update_status(status_msg, 5000)

            if errors:
                error_summary = "\n".join([f"- {e}" for e in errors[:10]])
                if len(errors) > 10: error_summary += f"\n- ... ({len(errors)-10} more)"
                QMessageBox.warning(self, "Load Issues", f"Could not load all files:\n{error_summary}\n\nSee logs for full details.")

            # Set multi_spectra state *before* resetting
            self.multi_spectra = loaded_spectra
            self._reset_state_for_new_spectrum(None) # Clear single spectrum, reset panels

            if self.ml_view:
                self.ml_view.set_spectra_list(loaded_spectra) # Pass loaded spectra to ML view

            # Optionally activate the ML panel
            if loaded_spectra and DockName.ML_ANALYSIS in self.docks:
                 self.docks[DockName.ML_ANALYSIS].raise_()
                 self.docks[DockName.ML_ANALYSIS].show()

        except Exception as e:
            log.error(f"Critical error during multi-spectrum load: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", f"An unexpected error occurred during loading:\n{e}")
            self.multi_spectra = [] # Clear partial results
            if self.ml_view: self.ml_view.set_spectra_list([])
            self._reset_state_for_new_spectrum(None) # Reset fully on critical error
        finally:
            self.set_busy(False)


    # --- Core Processing/Analysis Slots ---

    @pyqtSlot(dict)
    def handle_process_request(self, settings: dict):
        """Handles baseline subtraction, denoising, and smoothing requests."""
        if self.current_spectrum is None:
            QMessageBox.warning(self, "Processing", "Please load a single spectrum before processing.")
            return
        if self._is_busy:
            log.warning("Process request ignored: Application is busy.")
            return

        log.info(f"Handling process request with settings: {settings}")
        self.set_busy(True, "Applying processing steps...")

        try:
            # Ensure raw data exists
            if self.current_spectrum.raw_intensity is None:
                 log.error("Cannot process: Current spectrum is missing raw_intensity data.")
                 QMessageBox.critical(self, "Processing Error", "Spectrum data is inconsistent (missing raw intensity). Cannot process.")
                 return

            wavelengths = self.current_spectrum.wavelengths
            # IMPORTANT: Always start processing from the original raw intensity
            intensity = self.current_spectrum.raw_intensity.copy()
            processed_intensity = intensity # Initialize processed with raw
            baseline = np.zeros_like(intensity) # Initialize baseline

            # --- 1. Baseline Correction ---
            baseline_method = settings.get('baseline_method', 'None')
            if baseline_method == 'Polynomial':
                log.debug(f"Applying Polynomial Baseline with settings: { {k:v for k,v in settings.items() if k.startswith('baseline_poly_')} }")
                # Pass only relevant settings to the function
                poly_settings = {k.replace('baseline_poly_', ''): v for k, v in settings.items() if k.startswith('baseline_poly_')}
                processed_intensity, baseline = baseline_poly(wavelengths, intensity, **poly_settings)
            elif baseline_method == 'SNIP':
                log.debug(f"Applying SNIP Baseline with settings: { {k:v for k,v in settings.items() if k.startswith('baseline_snip_')} }")
                snip_settings = {k.replace('baseline_snip_', ''): v for k, v in settings.items() if k.startswith('baseline_snip_')}
                processed_intensity, baseline = baseline_snip(wavelengths, intensity, **snip_settings)
            elif baseline_method != 'None':
                 log.warning(f"Unknown baseline method: {baseline_method}. Skipping baseline correction.")
            # If baseline was 'None', processed_intensity is still raw_intensity, baseline is zeros

            # --- 2. Denoising (applied to baseline-corrected signal) ---
            denoising_method = settings.get('denoising_method', 'None')
            if denoising_method == 'Wavelet':
                # Pass only relevant settings
                wavelet_settings = {k.replace('wavelet_', ''): v for k, v in settings.items() if k.startswith('wavelet_')}
                log.debug(f"Applying Wavelet Denoising with settings: {wavelet_settings}")
                processed_intensity = denoise_wavelet(processed_intensity, **wavelet_settings)
            elif denoising_method != 'None':
                 log.warning(f"Unknown denoising method: {denoising_method}. Skipping denoising.")

            # --- 3. Smoothing (applied AFTER baseline & denoising) ---
            smoothing_method = settings.get('smoothing_method', 'None')
            if smoothing_method == 'SavitzkyGolay':
                # Pass only relevant settings
                sg_settings = {k.replace('sg_', ''): v for k, v in settings.items() if k.startswith('sg_')}
                log.debug(f"Applying Savitzky-Golay smoothing with settings: {sg_settings}")
                processed_intensity = smooth_savitzky_golay(processed_intensity, **sg_settings)
            elif smoothing_method != 'None':
                 log.warning(f"Unknown smoothing method: {smoothing_method}. Skipping smoothing.")

            # --- 4. Update Spectrum Object and UI ---
            # Use the final 'processed_intensity' and the calculated 'baseline'
            # Ensure baseline is None if no method was applied
            final_baseline = baseline if baseline_method != 'None' else None
            self.current_spectrum.update_processed(processed_intensity, final_baseline)
            log.info("Processing complete. Spectrum object updated.")
            self.update_status("Processing complete.", 5000)

            if self.plot_widget:
                self.plot_widget.plot_spectrum(
                    self.current_spectrum, show_raw=True,
                    show_processed=True, show_baseline=(final_baseline is not None)
                )
                self._apply_theme_to_plots() # Reapply theme after replot

            # --- 5. Clear Downstream Analysis Results ---
            self._clear_downstream_analysis_data(clear_peaks=True) # Clears peaks, NIST, plasma

        except Exception as e:
            log.error(f"Error during processing: {e}", exc_info=True)
            QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n{e}")
            self.update_status("Processing failed.", 5000)
            # Optionally reset processed data in spectrum on error?
            # if self.current_spectrum: self.current_spectrum.update_processed(None, None)
            # self._update_all_ui_states()
        finally:
            self.set_busy(False)


    @pyqtSlot(dict)
    def handle_peak_detection_request(self, settings: dict):
        """Handles the peak detection request."""
        if self.current_spectrum is None:
            QMessageBox.warning(self, "Peak Detection", "Please load a single spectrum first.")
            return
        if self.current_spectrum.processed_intensity is None:
            QMessageBox.warning(self, "Peak Detection", "Spectrum has not been processed yet. Please run processing first.")
            return
        if self._is_busy:
            log.warning("Peak detection request ignored: Busy.")
            return

        log.info(f"Handling peak detection request with settings: {settings}")
        self.set_busy(True, "Detecting peaks...")

        try:
            # Make a copy of settings to potentially modify for the core function
            detection_settings = settings.copy()
            method = detection_settings.pop('method', 'Unknown') # Get method and remove it

            if method == 'ScipyFindPeaks':
                log.debug(f"Using ScipyFindPeaks with params: {detection_settings}")
                # Pass the current spectrum and remaining settings
                self.detected_peaks = detect_peaks_scipy(self.current_spectrum, **detection_settings)
            else:
                # Handle other methods or raise error
                raise ValueError(f"Unsupported peak detection method selected: {method}")

            num_peaks = len(self.detected_peaks)
            log.info(f"Peak detection complete. Found {num_peaks} peaks.")
            self.update_status(f"Found {num_peaks} peaks.", 5000)

            # Update UI
            if self.plot_widget:
                self.plot_widget.plot_peaks(self.detected_peaks)
                self._apply_theme_to_plots() # Reapply theme after replot
                # Clear fits and NIST matches shown on plot
                self.plot_widget.plot_fit_lines([])
                self.plot_widget.clear_nist_matches()
            if self.peak_list_view:
                self.peak_list_view.update_peak_list(self.detected_peaks)

            # Clear downstream results (NIST, Plasma)
            self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=True)

        except ValueError as ve:
            log.error(f"Peak detection configuration error: {ve}")
            QMessageBox.critical(self, "Peak Detection Error", str(ve))
            self.update_status("Peak detection failed (config error).", 5000)
            self._clear_downstream_analysis_data(clear_peaks=True) # Clear everything if detection fails
        except Exception as e:
            log.error(f"Error during peak detection: {e}", exc_info=True)
            QMessageBox.critical(self, "Peak Detection Error", f"An unexpected error occurred during peak detection:\n{e}")
            self.update_status("Peak detection failed.", 5000)
            self._clear_downstream_analysis_data(clear_peaks=True) # Clear everything if detection fails
        finally:
            self.set_busy(False)


    @pyqtSlot(dict)
    def handle_peak_fitting_request(self, settings: dict):
        """Handles the request to fit all detected peaks."""
        if not self.detected_peaks:
            QMessageBox.warning(self, "Peak Fitting", "No peaks have been detected yet. Please run peak detection first.")
            return
        if self.current_spectrum is None or self.current_spectrum.processed_intensity is None:
             QMessageBox.warning(self, "Peak Fitting", "Cannot fit peaks. Processed spectrum data is missing.")
             return
        if self._is_busy:
            log.warning("Peak fitting request ignored: Busy.")
            return

        num_peaks_to_fit = len(self.detected_peaks)
        log.info(f"Handling request to fit {num_peaks_to_fit} detected peaks with settings: {settings}")
        self.set_busy(True, f"Fitting {num_peaks_to_fit} peaks...")

        # Use processed intensity for fitting
        processed_intensity = self.current_spectrum.processed_intensity
        num_success = 0
        num_fail = 0
        fit_profile = settings.get('profile', 'Unknown') # For status updates

        try:
            # Optimize progress update frequency
            update_interval = max(1, num_peaks_to_fit // 20)

            for i, peak in enumerate(self.detected_peaks):
                # Update status bar periodically
                if i % update_interval == 0:
                    self.update_status(f"Fitting peak {i + 1}/{num_peaks_to_fit} ({fit_profile})...", 0)
                    QApplication.processEvents() # Keep UI responsive

                try:
                    # Pass spectrum, peak index, processed intensity, and fitting settings
                    best_fit, all_fits = fit_peak(
                        spectrum=self.current_spectrum,
                        peak_index=peak.index, # Pass the actual data index
                        processed_intensity=processed_intensity,
                        **settings # Pass fitting parameters (profile, window, etc.)
                    )
                    # Store results in the Peak object
                    peak.best_fit = best_fit
                    peak.alternative_fits = all_fits

                    if best_fit and best_fit.success:
                        num_success += 1
                        # Reduce log level for successful fits unless debugging heavily
                        log.log(logging.DEBUG - 1 if log.isEnabledFor(logging.DEBUG - 1) else logging.DEBUG,
                                f"Fit success: Peak {i} (Idx {peak.index}, Wl={peak.wavelength_detected:.3f}), Profile: {best_fit.profile_type}, Params: {best_fit.params}")
                    else:
                        num_fail += 1
                        reason = 'No best fit found' if not best_fit else best_fit.message
                        log.warning(f"Fit failed: Peak {i} (Idx {peak.index}, Wl={peak.wavelength_detected:.3f}). Reason: {reason}")
                        # Ensure best_fit is None if fitting fails
                        peak.best_fit = None
                except Exception as fit_error:
                    # Catch errors during individual peak fits
                    log.error(f"Error fitting peak {i} (Idx {peak.index}, Wl={peak.wavelength_detected:.3f}): {fit_error}", exc_info=True) # Log traceback for individual errors
                    num_fail += 1
                    peak.best_fit = None # Ensure fit is None on error
                    peak.alternative_fits = {}

            log.info(f"Bulk peak fitting complete. Success: {num_success}, Failed: {num_fail}")
            self.update_status(f"Fitting complete: {num_success}/{num_peaks_to_fit} successful.", 5000)

            # Update UI
            if self.plot_widget:
                self.plot_widget.plot_peaks(self.detected_peaks) # Update peak markers (maybe color?)
                self.plot_widget.plot_fit_lines(self.detected_peaks) # Plot the new fits
                self._apply_theme_to_plots()
            if self.peak_list_view:
                self.peak_list_view.update_peak_list(self.detected_peaks) # Update list with fit results
            # Clear selection in fitting panel as bulk fit finished
            if self.peak_fitting_panel:
                 self.peak_fitting_panel.display_peak_fit_details(None)

            # Clear downstream results (NIST, Plasma) but keep peaks/fits
            self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=True)

            # Re-correlate NIST matches (if any exist) to the new fits
            self._correlate_nist_matches_to_peaks()
            if self.plot_widget:
                 self.plot_widget.plot_nist_matches(self.nist_matches, clear_previous=False) # Display potentially updated correlations

        except Exception as e:
            # Catch errors in the main loop or setup
            log.error(f"Critical error during bulk peak fitting: {e}", exc_info=True)
            QMessageBox.critical(self, "Peak Fitting Error", f"An unexpected error occurred during fitting:\n{e}")
            self.update_status("Peak fitting failed.", 5000)
            # Consider clearing all fits if bulk process fails? Or leave partial?
            # self._clear_downstream_analysis_data(clear_peaks=True) # Safer to clear all if process fails badly
        finally:
            self.set_busy(False)


    @pyqtSlot(int, dict)
    def handle_refit_single_peak(self, peak_list_index: int, settings: dict):
        """Handles the request to refit a single selected peak."""
        if self._is_busy:
            log.warning("Refit single peak ignored: Busy.")
            return

        # Validate index
        if not (0 <= peak_list_index < len(self.detected_peaks)):
            log.error(f"Invalid peak list index provided for refit: {peak_list_index} (List size: {len(self.detected_peaks)})")
            QMessageBox.critical(self, "Refit Error", f"Internal error: Invalid peak index ({peak_list_index}).")
            return

        peak_to_refit = self.detected_peaks[peak_list_index]

        # Validate spectrum state
        if self.current_spectrum is None or self.current_spectrum.processed_intensity is None:
            log.warning("Refit single peak ignored: No processed spectrum available.")
            QMessageBox.warning(self, "Refit Error", "Cannot refit peak: Processed spectrum data is missing.")
            return

        log.info(f"Handling request to refit peak list index {peak_list_index} (Spectrum index {peak_to_refit.index}, Wl={peak_to_refit.wavelength_detected:.3f}) with settings: {settings}")
        self.set_busy(True, f"Refitting peak @ {peak_to_refit.wavelength_detected:.2f} nm...")

        processed_intensity = self.current_spectrum.processed_intensity
        original_fit = peak_to_refit.best_fit # Keep original in case of failure
        best_fit = None # Initialize

        try:
            # Call the core fitting function
            best_fit, all_fits = fit_peak(
                spectrum=self.current_spectrum,
                peak_index=peak_to_refit.index, # Pass the actual data index
                processed_intensity=processed_intensity,
                **settings # Pass fitting parameters
            )

            # Update the specific peak object
            peak_to_refit.best_fit = best_fit
            peak_to_refit.alternative_fits = all_fits

            if best_fit and best_fit.success:
                log.info(f"Refit successful for peak {peak_list_index}. Best profile: {best_fit.profile_type}")
                self.update_status(f"Refit successful for peak {peak_list_index}.", 3000)
            else:
                reason = 'No best fit found' if not best_fit else best_fit.message
                log.warning(f"Refit failed for peak {peak_list_index}. Reason: {reason}")
                self.update_status(f"Refit failed for peak {peak_list_index}.", 3000)
                peak_to_refit.best_fit = None # Ensure fit is None if failed

            # Update UI elements
            if self.peak_list_view:
                self.peak_list_view.update_peak_list(self.detected_peaks) # Update the whole list (easier than single row)
                # Reselect the refitted peak in the list
                self.peak_list_view.select_peak_by_index(peak_list_index)
            if self.peak_fitting_panel:
                self.peak_fitting_panel.display_peak_fit_details(peak_to_refit, peak_list_index) # Update fitting panel
            if self.plot_widget:
                # Redraw all fit lines, highlighting the newly refitted one
                self.plot_widget.plot_fit_lines(self.detected_peaks, highlight_fit=best_fit)
                self.plot_widget.plot_peaks(self.detected_peaks) # Update peak markers potentially
                self._apply_theme_to_plots()

            # Clear downstream results (NIST, Plasma) as a fit changed
            self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=True)

            # Re-correlate NIST matches and update plot
            self._correlate_nist_matches_to_peaks()
            if self.plot_widget:
                 self.plot_widget.plot_nist_matches(self.nist_matches, correlate=False)

        except Exception as e:
            log.error(f"Error refitting peak {peak_list_index} (Index {peak_to_refit.index}): {e}", exc_info=True)
            QMessageBox.critical(self, "Refit Error", f"An error occurred while refitting the peak:\n{e}")
            self.update_status("Refit failed.", 5000)
            # Restore original fit on error? Or leave as None? Let's leave as None.
            peak_to_refit.best_fit = None
            peak_to_refit.alternative_fits = {}
            # Update UI to reflect the failed state
            if self.peak_list_view: self.peak_list_view.update_peak_list(self.detected_peaks)
            if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(peak_to_refit, peak_list_index)
            if self.plot_widget: self.plot_widget.plot_fit_lines(self.detected_peaks, highlight_fit=None) # Remove highlight
        finally:
            self.set_busy(False)


    @pyqtSlot(list) # Expecting a list of NISTMatch objects
    def _handle_nist_search_results(self, matches: List[NISTMatch]):
        """Handles the results received from the NIST search view."""
        num_matches = len(matches)
        log.info(f"Received {num_matches} potential NIST matches from search.")
        self.update_status(f"Received {num_matches} NIST matches.", 3000)

        self.nist_matches = matches # Store the raw matches

        # Correlate these matches to the *currently detected and fitted* peaks
        self._correlate_nist_matches_to_peaks() # This updates peak.potential_matches

        # Update UI
        if self.plot_widget:
            # Plot using the newly correlated matches stored in self.nist_matches
            # The correlation step doesn't modify self.nist_matches, it updates peaks.
            # So, we plot self.nist_matches, but correlation=True ensures it uses peak info.
            self.plot_widget.plot_nist_matches(self.nist_matches, correlate=True)
            self._apply_theme_to_plots()

        # Update NIST results table in the view (if it exists and has the method)
        if self.nist_search_view and hasattr(self.nist_search_view, 'display_results'):
            try:
                if not self.nist_matches:
                    self.nist_search_view.display_results(pd.DataFrame()) # Display empty table
                elif hasattr(NISTMatch, 'to_dict'):
                     # Create DataFrame from the received matches
                    nist_df = pd.DataFrame([m.to_dict() for m in self.nist_matches if hasattr(m, 'to_dict')])
                    self.nist_search_view.display_results(nist_df)
                else:
                     log.warning("Cannot update NIST results table: NISTMatch object missing 'to_dict' method.")
            except Exception as e:
                 log.error(f"Failed to update NIST view results table: {e}", exc_info=True)

        # Update Peak list view to show correlation icons/info
        if self.peak_list_view:
             self.peak_list_view.update_peak_list(self.detected_peaks)


        # Clear downstream results (Plasma parameters)
        self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=False, clear_plasma=True)


    @pyqtSlot(str)
    def handle_boltzmann_populate_request(self, species: str):
        """Populates the Boltzmann view's line table with candidates for the given species."""
        if self._is_busy:
            log.warning("Boltzmann populate request ignored: Busy.")
            return
        if not self.boltzmann_view:
            log.error("Boltzmann view not initialized, cannot populate.")
            return

        # --- Prerequisite Checks ---
        if not self.detected_peaks:
            msg = "Cannot populate Boltzmann lines: No peaks detected."
            log.warning(msg)
            self.update_status(msg, 3000)
            self.boltzmann_view.display_candidate_lines(pd.DataFrame()) # Show empty table
            return
        if not any(p.best_fit and p.best_fit.success for p in self.detected_peaks):
            msg = "Cannot populate Boltzmann lines: Peaks require successful fits."
            log.warning(msg)
            self.update_status(msg, 4000)
            self.boltzmann_view.display_candidate_lines(pd.DataFrame())
            return
        if not any(p.potential_matches for p in self.detected_peaks):
            msg = "Cannot populate Boltzmann lines: Run NIST search and ensure correlation first."
            log.warning(msg)
            self.update_status(msg, 4000)
            self.boltzmann_view.display_candidate_lines(pd.DataFrame())
            return

        log.info(f"Populating Boltzmann candidates for species: '{species}'")
        self.update_status(f"Finding candidate lines for {species}...")
        QApplication.processEvents()

        candidates = []
        try:
            # --- Parse Species Input ---
            species_clean = species.strip()
            parts = species_clean.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid species format: '{species}'. Expected 'Element IonState' (e.g., 'Fe I', 'Al II').")
            element_target, ion_state_target = parts[0].strip().lower(), parts[1].strip().lower()
            if not element_target or not ion_state_target:
                raise ValueError(f"Invalid species format: '{species}'. Element and Ion State cannot be empty.")
            log.debug(f"Targeting Element: '{element_target}', Ion State: '{ion_state_target}'")

            # --- Iterate Through Peaks and Matches ---
            required_match_keys = ['ei', 'gi', 'aki'] # NISTMatch attributes needed
            missing_data_count = 0
            skipped_intensity_count = 0

            for peak in self.detected_peaks:
                # Need a successful fit for intensity
                if not peak.best_fit or not peak.best_fit.success:
                    continue

                # Use fitted intensity (more accurate than detected height)
                # Prioritize area if available and valid, fallback to amplitude
                intensity_val = getattr(peak.best_fit, 'area', None)
                if intensity_val is None or not np.isfinite(intensity_val) or intensity_val <= 0:
                    intensity_val = getattr(peak.best_fit, 'amplitude', None)

                if intensity_val is None or not np.isfinite(intensity_val) or intensity_val <= 0:
                    log.log(logging.DEBUG - 1, f"Skipping peak {peak.index}: Invalid or non-positive fitted intensity/amplitude ({intensity_val}).")
                    skipped_intensity_count += 1
                    continue

                # Check potential NIST matches for this peak
                found_candidate_for_peak = False
                if peak.potential_matches:
                    for match in peak.potential_matches:
                        match_element = getattr(match, 'element', '').strip().lower()
                        match_ion_state = getattr(match, 'ion_state_str', '').strip().lower()

                        # Check if match is the target species
                        if match_element == element_target and match_ion_state == ion_state_target:
                            # Check if required atomic data exists and is valid
                            missing_keys = []
                            valid_data = True
                            atomic_data = {}
                            for key in required_match_keys:
                                val = getattr(match, key, None)
                                if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
                                    missing_keys.append(key)
                                    valid_data = False
                                else:
                                    atomic_data[key] = val # Store valid data

                            if valid_data:
                                # Add candidate if data is valid
                                candidates.append({
                                    'Peak λ (nm)': peak.wavelength_fitted_or_detected, # Use fitted if available
                                    'Intensity': intensity_val,
                                    'Elem': match.element,
                                    'Ion': match.ion_state_str,
                                    'DB λ (nm)': match.wavelength_db,
                                    'E_k (eV)': atomic_data['ei'],
                                    'g_k': atomic_data['gi'],
                                    'A_ki (s⁻¹)': atomic_data['aki'],
                                    'Peak Index': peak.index # Useful for debugging/linking back
                                })
                                found_candidate_for_peak = True
                                break # Found a suitable match for this peak, move to next peak
                            else:
                                missing_data_count += 1
                                log.log(logging.DEBUG - 1, f"Skipping match {match.element} {match.ion_state_str} ({match.wavelength_db:.3f} nm) for peak {peak.index} due to missing/invalid Boltzmann data: {missing_keys}.")
                                # Don't break, maybe another match for the same peak *is* valid

            # --- Process and Display Results ---
            candidate_df = pd.DataFrame(candidates)
            num_candidates = len(candidate_df)

            if not candidate_df.empty:
                 # Remove duplicates based on Peak Index (one line per peak for Boltzmann)
                 initial_count = num_candidates
                 candidate_df = candidate_df.drop_duplicates(subset=['Peak Index'], keep='first') # Keep first valid match found
                 num_candidates = len(candidate_df)
                 if initial_count != num_candidates:
                     log.warning(f"Removed {initial_count - num_candidates} duplicate peak entries during Boltzmann population.")
                 log.info(f"Found {num_candidates} unique candidate lines for {species}.")
            else:
                 log.info(f"No suitable candidate lines found for {species} with required atomic data and successful fits.")

            # Report skipped counts
            if missing_data_count > 0:
                log.warning(f"Skipped {missing_data_count} potential line matches due to missing/invalid atomic data (E_k, g_k, A_ki).")
            if skipped_intensity_count > 0:
                 log.warning(f"Skipped {skipped_intensity_count} peaks due to invalid fitted intensity/amplitude.")

            # Display in the Boltzmann view
            self.boltzmann_view.display_candidate_lines(candidate_df)
            self.update_status(f"Populated {num_candidates} candidates for {species}.", 5000)

        except ValueError as ve: # Catch specific input errors
            log.error(f"Invalid input for Boltzmann population: {ve}")
            QMessageBox.warning(self, "Boltzmann Plot Input Error", str(ve))
            if self.boltzmann_view: self.boltzmann_view.display_candidate_lines(pd.DataFrame())
            self.update_status("Boltzmann population failed (invalid input).", 5000)
        except Exception as e:
            log.error(f"Error populating Boltzmann candidates for {species}: {e}", exc_info=True)
            QMessageBox.critical(self, "Boltzmann Plot Error", f"An error occurred while finding candidate lines:\n{e}")
            if self.boltzmann_view: self.boltzmann_view.display_candidate_lines(pd.DataFrame())
            self.update_status("Boltzmann population failed.", 5000)


    # Type hint 'object' is vague, could be float or None or error object? Let's assume float or None.
    @pyqtSlot(bool, object, object, object)
    def _handle_boltzmann_result(self, success: bool, temperature: Optional[float], r_squared: Optional[float], plot_data: Optional[pd.DataFrame]):
        """Handles the results from the Boltzmann calculation."""
        log.debug(f"Received Boltzmann result: success={success}, T={temperature}, R²={r_squared}, plot_data type={type(plot_data)}")

        # Validate temperature
        valid_temp = success and isinstance(temperature, (float, int)) and np.isfinite(temperature)
        valid_r2 = isinstance(r_squared, (float, int)) and np.isfinite(r_squared) if r_squared is not None else False

        if valid_temp:
            self.plasma_temp_k = float(temperature)
            r2_str = f"(R²={r_squared:.4f})" if valid_r2 else ""
            log.info(f"Boltzmann calculation successful. Stored Plasma Temperature: {self.plasma_temp_k:.2f} K {r2_str}")
            self.update_status(f"Plasma Temperature calculated: {self.plasma_temp_k:.0f} K", 5000)

            # Update CF-LIBS view with the new temperature
            if self.cf_libs_view:
                 self.cf_libs_view.update_temperature(self.plasma_temp_k)
            # Clear subsequent results (Ne, Conc) as Temperature changed
            self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=False, clear_plasma=True)
            # Set T again after clearing, as clear resets it
            self.plasma_temp_k = float(temperature)
            if self.cf_libs_view: self.cf_libs_view.update_temperature(self.plasma_temp_k)


        else:
            # Calculation failed or returned invalid temperature
            self.plasma_temp_k = None
            log.warning("Boltzmann calculation failed or returned invalid/non-finite temperature.")
            self.update_status("Plasma temperature calculation failed.", 5000)
            # Clear subsequent results and update CF-LIBS view
            self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=False, clear_plasma=True)


        # Store plot data regardless of success (might show failed fit)
        if isinstance(plot_data, pd.DataFrame):
            self.boltzmann_plot_data = plot_data.copy()
            log.debug(f"Stored Boltzmann plot data ({len(self.boltzmann_plot_data)} points).")
        else:
            self.boltzmann_plot_data = None
            if success: # Log warning if success was True but data is bad
                log.warning("Boltzmann calculation reported success, but plot data is not a valid DataFrame.")

        # Update save actions and panel states
        self._update_all_ui_states()


    @pyqtSlot(str, str, float)
    def handle_ne_calculation_request(self, species1: str, species2: str, temperature_k: float):
        """Handles the request to calculate electron density (Ne)."""
        if self._is_busy:
            log.warning("Electron density calculation request ignored: Busy.")
            return

        # --- Prerequisite Checks ---
        if temperature_k is None or not np.isfinite(temperature_k):
            QMessageBox.warning(self, "Electron Density", "Cannot calculate Nₑ: Valid plasma temperature is required (from Boltzmann plot).")
            return
        if not self.detected_peaks or not any(p.best_fit and p.best_fit.success for p in self.detected_peaks):
             QMessageBox.warning(self, "Electron Density", "Cannot calculate Nₑ: Successfully fitted peaks are required.")
             return
        # Need correlated NIST data for Saha-Boltzmann approach
        if not any(p.potential_matches for p in self.detected_peaks):
             QMessageBox.warning(self, "Electron Density", "Cannot calculate Nₑ: Correlated NIST data is required.")
             return

        log.info(f"Handling Nₑ calculation request for {species1}/{species2} at T={temperature_k:.0f} K.")
        self.set_busy(True, "Calculating Electron Density (Nₑ)...")
        QApplication.processEvents() # Update UI

        ne_cm3: Optional[float] = None
        try:
            # --- Call Core Calculation Function ---
            # This function needs access to peaks, matches, temperature, and potentially partition functions/ionization energies
            # It needs to find suitable line pairs for species1 and species2 from the detected/matched peaks
            if not callable(calculate_electron_density_saha):
                 raise NotImplementedError("Core function 'calculate_electron_density_saha' not found or not callable.")

            # TODO: Implement the actual call and logic in calculate_electron_density_saha
            # ne_cm3 = calculate_electron_density_saha(
            #     peaks=self.detected_peaks, # Includes fits and matches
            #     temperature_k=temperature_k,
            #     species1_str=species1,
            #     species2_str=species2,
            #     # Potentially pass partition function data if needed internally
            # )
            raise NotImplementedError("Saha-Boltzmann electron density calculation logic in 'calculate_electron_density_saha' is not yet implemented.")

            # --- Update State and UI ---
            if ne_cm3 is not None and np.isfinite(ne_cm3):
                self.electron_density_cm3 = ne_cm3
                log.info(f"Electron density calculated: {ne_cm3:.3e} cm⁻³")
                self.update_status(f"Nₑ calculated: {ne_cm3:.3e} cm⁻³", 5000)
                if self.cf_libs_view:
                     self.cf_libs_view.update_electron_density(ne_cm3)
                # Clear concentrations as Ne changed
                self.cf_libs_concentrations = None
                if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)
            else:
                 # Calculation returned None or invalid value
                 self.electron_density_cm3 = None
                 log.warning("Electron density calculation did not return a valid result.")
                 self.update_status("Nₑ calculation failed or returned invalid value.", 5000)
                 if self.cf_libs_view: self.cf_libs_view.update_electron_density(None)
                 self.cf_libs_concentrations = None
                 if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)

        except NotImplementedError as nie:
            log.warning(f"Nₑ calculation failed: {nie}")
            QMessageBox.warning(self, "Calculation Not Implemented", str(nie))
            self.update_status(f"Nₑ calculation failed: Not implemented.", 5000)
            self.electron_density_cm3 = None
            if self.cf_libs_view: self.cf_libs_view.update_electron_density(None)
            self.cf_libs_concentrations = None
            if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)
        except ValueError as ve: # Catch specific errors from the calculation function
            log.warning(f"Nₑ calculation failed: {ve}")
            QMessageBox.warning(self, "Electron Density Calculation Error", f"Could not calculate Nₑ:\n{ve}")
            self.update_status(f"Nₑ calculation failed: {ve}", 5000)
            self.electron_density_cm3 = None
            if self.cf_libs_view: self.cf_libs_view.update_electron_density(None)
            self.cf_libs_concentrations = None
            if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)
        except Exception as e: # Catch unexpected errors
            log.error(f"Error during electron density calculation: {e}", exc_info=True)
            QMessageBox.critical(self, "Calculation Error", f"An unexpected error occurred during Nₑ calculation:\n{e}")
            self.update_status("Nₑ calculation failed (unexpected error).", 5000)
            self.electron_density_cm3 = None
            if self.cf_libs_view: self.cf_libs_view.update_electron_density(None)
            self.cf_libs_concentrations = None
            if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)
        finally:
            self._update_all_ui_states() # Update panel enables etc.
            self.set_busy(False)


    @pyqtSlot(float, object) # N_e might be None or float
    def handle_conc_calculation_request(self, temperature_k: float, ne_cm3_obj: Optional[object]):
        """Handles the request to calculate CF-LIBS concentrations."""
        if self._is_busy:
            log.warning("Concentration calculation request ignored: Busy.")
            return

        # --- Prerequisite Checks ---
        if not self.detected_peaks or not any(p.best_fit and p.best_fit.success and p.potential_matches for p in self.detected_peaks):
             QMessageBox.warning(self, "CF-LIBS Calculation", "Cannot calculate concentrations. Fitted peaks with NIST correlations are required.")
             return
        if temperature_k is None or not np.isfinite(temperature_k):
             QMessageBox.warning(self, "CF-LIBS Calculation", "Cannot calculate concentrations. A valid plasma temperature (from Boltzmann plot) is required.")
             return

        # Validate and convert N_e input
        ne_cm3: Optional[float] = None
        if ne_cm3_obj is not None:
            try:
                ne_cm3 = float(ne_cm3_obj)
                if not np.isfinite(ne_cm3):
                    QMessageBox.warning(self, "CF-LIBS Calculation", f"Invalid Electron Density provided ({ne_cm3_obj}). Please calculate or enter a valid Nₑ.")
                    return
            except (ValueError, TypeError):
                QMessageBox.warning(self, "CF-LIBS Calculation", f"Invalid Electron Density provided ('{ne_cm3_obj}'). Must be a number.")
                return
        # If ne_cm3_obj was None, ne_cm3 remains None (LTE assumption might be used)

        log.info(f"Handling concentration calculation request. T={temperature_k:.0f} K, Nₑ={f'{ne_cm3:.3e} cm⁻³' if ne_cm3 is not None else 'Not provided (assuming LTE?)'}.")
        self.set_busy(True, "Calculating CF-LIBS Concentrations...")
        QApplication.processEvents() # Update UI

        concentrations_df: Optional[pd.DataFrame] = None
        try:
            # --- Call Core Calculation Function ---
            if not callable(calculate_cf_libs_conc):
                raise NotImplementedError("Core function 'calculate_cf_libs_conc' not found or not callable.")

            # TODO: Implement the actual call and logic in calculate_cf_libs_conc
            # This function needs access to peaks, matches, temperature, Ne (optional),
            # and potentially partition functions, reference composition, etc.
            # concentrations_df = calculate_cf_libs_conc(
            #     peaks=self.detected_peaks,
            #     temperature_k=temperature_k,
            #     electron_density_cm3=ne_cm3, # Pass None if not calculated/provided
            #     # Add other necessary parameters: partition_functions, reference_element, etc.
            # )
            raise NotImplementedError("CF-LIBS concentration calculation logic in 'calculate_cf_libs_conc' is not yet implemented.")

            # --- Update State and UI ---
            if isinstance(concentrations_df, pd.DataFrame) and not concentrations_df.empty:
                self.cf_libs_concentrations = concentrations_df
                log.info(f"CF-LIBS concentrations calculated ({len(concentrations_df)} elements).")
                self.update_status("CF-LIBS concentrations calculated.", 5000)
                if self.cf_libs_view:
                     self.cf_libs_view.display_concentrations(concentrations_df)
            else:
                 # Calculation returned None or empty DataFrame
                 self.cf_libs_concentrations = None
                 log.warning("CF-LIBS calculation did not return a valid DataFrame.")
                 self.update_status("CF-LIBS calculation failed or returned no results.", 5000)
                 if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)

        except NotImplementedError as nie:
            log.warning(f"CF-LIBS calculation failed: {nie}")
            QMessageBox.warning(self, "Calculation Not Implemented", str(nie))
            self.update_status(f"CF-LIBS calculation failed: Not implemented.", 5000)
            self.cf_libs_concentrations = None
            if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)
        except ValueError as ve: # Catch specific errors from calculation
            log.warning(f"CF-LIBS calculation failed: {ve}")
            QMessageBox.warning(self, "CF-LIBS Calculation Error", f"Could not calculate concentrations:\n{ve}")
            self.update_status(f"CF-LIBS calculation failed: {ve}", 5000)
            self.cf_libs_concentrations = None
            if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)
        except Exception as e: # Catch unexpected errors
            log.error(f"Error during concentration calculation: {e}", exc_info=True)
            QMessageBox.critical(self, "Calculation Error", f"An unexpected error occurred during concentration calculation:\n{e}")
            self.update_status("Concentration calculation failed (unexpected error).", 5000)
            self.cf_libs_concentrations = None
            if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)
        finally:
            self._update_all_ui_states() # Update save action states etc.
            self.set_busy(False)

    # --- UI Interaction Slots ---

    @pyqtSlot(int)
    def handle_peak_selection(self, peak_list_index: int):
        """Handles selection changes in the PeakListView."""
        if self._is_busy: return # Ignore UI interaction while busy

        selected_peak: Optional[Peak] = None
        if 0 <= peak_list_index < len(self.detected_peaks):
            selected_peak = self.detected_peaks[peak_list_index]
            log.debug(f"Peak selected from list: Index {peak_list_index} (Wl={selected_peak.wavelength_fitted_or_detected:.3f})")
        else:
            log.debug(f"Peak selection cleared or invalid index from list: {peak_list_index}")
            peak_list_index = -1 # Use -1 to indicate no selection

        # Update fitting panel with selected peak details
        if self.peak_fitting_panel:
            # Pass the index along with the peak object
            self.peak_fitting_panel.display_peak_fit_details(selected_peak, peak_list_index)

        # Highlight peak and its fit (if any) on the plot
        if self.plot_widget:
            self.plot_widget.highlight_peak(peak_list_index if selected_peak else None)
            # Determine which fit to highlight (the best one if it exists and succeeded)
            fit_to_highlight = selected_peak.best_fit if (selected_peak and selected_peak.best_fit and selected_peak.best_fit.success) else None
            self.plot_widget.highlight_fit_line(fit_to_highlight)


    @pyqtSlot(int)
    def handle_peak_plot_click(self, peak_plot_index: int):
        """Handles clicks on peak markers in the plot widget."""
        if self._is_busy: return # Ignore UI interaction while busy

        # Validate index received from plot
        if 0 <= peak_plot_index < len(self.detected_peaks):
            clicked_peak = self.detected_peaks[peak_plot_index]
            log.info(f"Peak clicked on plot: Index {peak_plot_index} (Wl={clicked_peak.wavelength_fitted_or_detected:.3f})")

            # Select the corresponding peak in the list view
            if self.peak_list_view:
                 if hasattr(self.peak_list_view, 'select_peak_by_index'):
                     self.peak_list_view.select_peak_by_index(peak_plot_index)
                 else:
                     log.warning("PeakListView does not have 'select_peak_by_index' method.")
            else:
                 log.warning("PeakListView not available to sync selection.")
            # Trigger the same actions as selecting from the list
            self.handle_peak_selection(peak_plot_index)
        else:
            log.warning(f"Invalid peak index received from plot click: {peak_plot_index}. List length: {len(self.detected_peaks)}")


    @pyqtSlot(object) # FitResult or None
    def handle_show_specific_fit(self, fit_result: Optional[FitResult]):
        """Highlights a specific fit line on the plot, usually from fitting panel interaction."""
        if self._is_busy: return

        if fit_result:
            log.debug(f"Request from fitting panel to highlight fit: Profile={fit_result.profile_type}")
        else:
            log.debug("Request from fitting panel to clear fit highlight.")

        # Highlight the specific fit line on the plot
        if self.plot_widget:
             if hasattr(self.plot_widget, 'highlight_fit_line'):
                 self.plot_widget.highlight_fit_line(fit_result)
             else:
                 log.warning("Plot widget does not have 'highlight_fit_line' method.")


    # --- Data Handling Helpers ---

    def _correlate_nist_matches_to_peaks(self):
        """Associates loaded NIST matches with the closest detected/fitted peaks within a tolerance."""
        if not self.detected_peaks:
            log.debug("Correlation skipped: No peaks detected.")
            return
        if not self.nist_matches:
            log.debug("Correlation skipped: No NIST matches available.")
            # Ensure potential_matches lists are clear if NIST list is empty
            for peak in self.detected_peaks:
                if hasattr(peak, 'potential_matches') and peak.potential_matches:
                    peak.potential_matches.clear()
            # Update list view if necessary
            if self.peak_list_view: self.peak_list_view.update_peak_list(self.detected_peaks)
            return

        # Get tolerance from NIST search view UI (robustly)
        tolerance_nm = 0.1 # Default tolerance
        try:
            if (self.nist_search_view and
                hasattr(self.nist_search_view, 'tolerance_spinbox') and
                self.nist_search_view.tolerance_spinbox is not None):
                tolerance_nm = self.nist_search_view.tolerance_spinbox.value()
            else:
                 log.warning("Could not get tolerance from NIST view UI. Using default %.2f nm.", tolerance_nm)
        except Exception as e:
            log.error(f"Unexpected error getting tolerance: {e}. Using default %.2f nm.", tolerance_nm, exc_info=True)

        log.info(f"Correlating {len(self.nist_matches)} NIST matches to {len(self.detected_peaks)} peaks with tolerance {tolerance_nm:.3f} nm.")

        # --- Clear existing correlations and perform new ones ---
        # Use a temporary dict for efficient lookup: peak_index -> peak object
        peak_dict = {peak.index: peak for peak in self.detected_peaks}
        for peak in self.detected_peaks:
            if hasattr(peak, 'potential_matches'):
                 peak.potential_matches.clear() # Clear previous matches on the peak object
            else:
                 log.warning(f"Peak object at index {peak.index} missing 'potential_matches' list attribute.")


        correlation_count = 0
        unmatched_nist_lines = 0

        for match in self.nist_matches:
            db_wavelength = getattr(match, 'wavelength_db', None)
            if db_wavelength is None or not np.isfinite(db_wavelength):
                log.log(logging.DEBUG - 1, f"Skipping NIST match due to invalid DB wavelength: {match}")
                continue

            best_peak_match: Optional[Peak] = None
            min_diff = tolerance_nm # Start with max allowed difference

            # Find the *closest* peak within the tolerance
            for peak in self.detected_peaks:
                # Use fitted wavelength if available and valid, else detected
                peak_wavelength = peak.wavelength_fitted_or_detected
                if peak_wavelength is None or not np.isfinite(peak_wavelength):
                    continue # Skip peaks without valid wavelength

                diff = abs(peak_wavelength - db_wavelength)
                if diff <= min_diff: # Found a closer peak within tolerance
                    min_diff = diff
                    best_peak_match = peak

            # If a peak within tolerance was found, add the match to that peak
            if best_peak_match:
                if hasattr(best_peak_match, 'add_nist_match') and callable(best_peak_match.add_nist_match):
                    best_peak_match.add_nist_match(match)
                    correlation_count += 1
                    log.log(logging.DEBUG-1, f"Correlated NIST {match.element} {match.ion_state_str} @ {match.wavelength_db:.4f} nm to Peak {best_peak_match.index} @ {best_peak_match.wavelength_fitted_or_detected:.4f} nm (diff={min_diff:.4f} nm)")
                else:
                    # This should not happen if Peak class is defined correctly
                    log.error(f"Peak object (Index: {best_peak_match.index}) is missing the 'add_nist_match' method!")
            else:
                unmatched_nist_lines += 1
                log.log(logging.DEBUG -1, f"NIST line {match.element} {match.ion_state_str} @ {match.wavelength_db:.4f} nm did not correlate to any peak within tolerance {tolerance_nm:.3f} nm.")

        log.info(f"Correlation complete. Associated {correlation_count} NIST match instances to peaks. {unmatched_nist_lines} NIST lines remain uncorrelated.")

        # Update peak list view to reflect correlation status (e.g., icons)
        if self.peak_list_view:
            self.peak_list_view.update_peak_list(self.detected_peaks)

    def get_current_peaks(self) -> List[Peak]:
        """Returns the current list of detected/fitted peaks."""
        return self.detected_peaks

    # --- Save Actions ---

    SAVE_ACTION_CONFIG = {
        SaveType.PROCESSED_SPECTRUM: {
            "description": "processed spectrum data",
            "filename_suffix": "_processed.csv",
            "filter": "CSV Files (*.csv);;Text Files (*.txt)",
            "data_checker": lambda self: self.current_spectrum and self.current_spectrum.processed_intensity is not None,
            "data_getter": lambda self: self.current_spectrum,
            "save_function": save_spectrum_data,
            "save_kwargs": {}
        },
        SaveType.PEAKS: {
            "description": "peak list",
            "filename_suffix": "_peaks.csv",
            "filter": "CSV Files (*.csv)",
            "data_checker": lambda self: bool(self.detected_peaks),
            "data_getter": lambda self: self.detected_peaks,
            "save_function": save_peak_list,
            "save_kwargs": {}
        },
        SaveType.NIST_MATCHES: {
            "description": "NIST match results",
            "filename_suffix": "_nist_matches.csv",
            "filter": "CSV Files (*.csv)",
            "data_checker": lambda self: bool(self.nist_matches),
            "data_getter": lambda self: pd.DataFrame([m.to_dict() for m in self.nist_matches if hasattr(m, 'to_dict')]) if self.nist_matches else pd.DataFrame(),
            "save_function": save_dataframe,
            "save_kwargs": {}
        },
        SaveType.BOLTZMANN_DATA: {
            "description": "Boltzmann plot data",
            "filename_suffix": "_boltzmann_data.csv",
            "filter": "CSV Files (*.csv)",
            "data_checker": lambda self: self.boltzmann_plot_data is not None and not self.boltzmann_plot_data.empty,
            "data_getter": lambda self: self.boltzmann_plot_data,
            "save_function": save_dataframe,
            "save_kwargs": {}
        },
        SaveType.CONCENTRATIONS: {
            "description": "CF-LIBS concentrations",
            "filename_suffix": "_concentrations.csv",
            "filter": "CSV Files (*.csv)",
            "data_checker": lambda self: self.cf_libs_concentrations is not None and not self.cf_libs_concentrations.empty,
            "data_getter": lambda self: self.cf_libs_concentrations,
            "save_function": save_dataframe,
            "save_kwargs": {}
        },
        SaveType.PLOT: {
            "description": "main plot image",
            "filename_suffix": "_spectrum_plot.png",
            "filter": "PNG Image (*.png);;SVG Vector (*.svg);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)",
            "data_checker": lambda self: self.plot_widget and self.plot_widget.figure and (self.plot_widget.ax.lines or self.plot_widget.ax.collections),
            "data_getter": lambda self: self.plot_widget.figure, # Save the figure object itself
            "save_function": lambda fig, filepath, **kwargs: fig.savefig(filepath, **kwargs), # Use savefig method
            "save_kwargs": {'dpi': 300, 'bbox_inches': 'tight'}
        }
    }

    def _save_action(self, save_type: SaveType):
        """Handles saving different types of data using a configuration-driven approach."""
        if self._is_busy:
            log.warning(f"Save action '{save_type.value}' ignored while busy.")
            return

        log.info(f"Triggered save action for: {save_type.name}")

        config = self.SAVE_ACTION_CONFIG.get(save_type)
        if not config:
            log.error(f"Unknown save type requested: {save_type}")
            QMessageBox.critical(self, "Save Error", f"Internal error: Unknown data type '{save_type.value}' requested for saving.")
            return

        data_description = config["description"]
        is_data_available = False
        data_to_save = None

        try:
            # Check if data exists using the checker function
            if config["data_checker"](self):
                 is_data_available = True
                 # Get the data using the getter function
                 data_to_save = config["data_getter"](self)
                 # Additional check for DataFrames obtained from getters
                 if isinstance(data_to_save, pd.DataFrame) and data_to_save.empty:
                     is_data_available = False
                     log.warning(f"Data getter for '{save_type.name}' returned an empty DataFrame.")
                 elif data_to_save is None and save_type != SaveType.PLOT: # Allow figure object to be None briefly
                      is_data_available = False
                      log.warning(f"Data getter for '{save_type.name}' returned None.")

            if not is_data_available:
                log.warning(f"No data available to save for type: {save_type.name}")
                QMessageBox.information(self, "No Data to Save", f"There is no {data_description} available to save.")
                return

            # --- Get Filename ---
            base_name = "libs_forge_output"
            if self.current_spectrum and self.current_spectrum.filename:
                base_name = os.path.splitext(os.path.basename(self.current_spectrum.filename))[0]
            default_filename = f"{base_name}{config['filename_suffix']}"
            default_path = os.path.join(self._last_save_dir, default_filename)
            file_filter = config["filter"]

            filepath, selected_filter = QFileDialog.getSaveFileName(
                self, f"Save {data_description.capitalize()} As...", default_path, file_filter
            )

            if not filepath:
                log.info(f"Save action for '{save_type.name}' cancelled by user.")
                self.update_status("Save cancelled.", 3000)
                return

            self._last_save_dir = os.path.dirname(filepath)
            self.set_busy(True, f"Saving {os.path.basename(filepath)}...")

            # --- Perform Save Operation ---
            save_function = config["save_function"]
            save_kwargs = config["save_kwargs"]

            save_function(data_to_save, filepath=filepath, **save_kwargs)

            log.info(f"{data_description.capitalize()} saved successfully to {filepath}")
            self.update_status(f"Saved: {os.path.basename(filepath)}", 5000)

        except (IOError, PermissionError) as e:
            log.error(f"Failed to save {save_type.name} to {filepath}: {e}", exc_info=True)
            QMessageBox.critical(self, "Save Error", f"Failed to save {data_description} (Permission/IO Error):\n{e}")
            self.update_status("Save failed.", 5000)
        except AttributeError as ae:
             # Catch issues like missing 'to_dict' in NIST matches during data_getter
             log.error(f"Attribute error during save preparation for {save_type.name}: {ae}", exc_info=True)
             QMessageBox.critical(self, "Save Error", f"Failed to prepare data for saving {data_description}:\n{ae}\n(Possibly missing methods like 'to_dict').")
             self.update_status("Save preparation failed.", 5000)
        except Exception as e:
            log.error(f"Error during save action for '{save_type.name}': {e}", exc_info=True)
            QMessageBox.critical(self, "Save Error", f"An unexpected error occurred while saving {data_description}:\n{e}")
            self.update_status("Save failed.", 5000)
        finally:
            # Ensure busy state is reset even if errors occur during prep
            if self._is_busy:
                self.set_busy(False)

    # --- External Script Runner ---

    def run_external_script(self, script_relative_path: str, script_args: Optional[List[str]] = None):
        """Runs an external Python script using a dialog."""
        if self._is_busy:
            log.warning(f"Request to run script '{script_relative_path}' ignored while busy.")
            QMessageBox.warning(self, "Busy", "Cannot start an external script while another operation is in progress.")
            return

        script_args = script_args or []

        try:
            project_root = get_project_root()
            script_absolute_path = project_root / script_relative_path
            script_absolute_path = script_absolute_path.resolve() # Get absolute path

            if not script_absolute_path.is_file():
                log.error(f"External script not found at resolved path: {script_absolute_path}")
                QMessageBox.critical(
                    self, "Script Not Found",
                    f"The required script was not found:\n{script_absolute_path}\n"
                    f"(Relative path: {script_relative_path}, Project Root: {project_root})"
                )
                return

            log.info(f"Preparing to run external script: {script_absolute_path} with args: {script_args}")

            # Use the dedicated dialog
            dialog = ExternalScriptRunnerDialog(parent=self)
            # Pre-fill the command and arguments
            dialog.set_command(f'"{sys.executable}" "{str(script_absolute_path)}"')
            dialog.set_arguments(" ".join(script_args))

            # Execute the dialog modally
            dialog.exec()

            # Dialog handles running and displaying output. We just log that it closed.
            log.info(f"External script dialog closed for: {script_relative_path}.")
            # Could potentially get exit code from dialog if it stores it.

        except ImportError:
            log.error("Failed to import ExternalScriptRunnerDialog. Cannot run external scripts.", exc_info=True)
            QMessageBox.critical(self, "Dependency Error", "Could not find the ExternalScriptRunnerDialog component.")
        except Exception as e:
            log.error(f"Error setting up or running external script runner for '{script_relative_path}': {e}", exc_info=True)
            QMessageBox.critical(self, "Script Runner Error", f"Could not initialize or run the script runner:\n{e}")

    @pyqtSlot()
    def run_nist_fetcher(self):
        """Runs the NIST data fetching script."""
        self.run_external_script('database/nist_data_fetcher.py')

    @pyqtSlot()
    def run_atomic_data_builder(self):
        """Runs the atomic data building script."""
        # Update path if needed
        self.run_external_script('database/atomic_data_builder.py')

    # --- Application Exit ---

    def closeEvent(self, event):
        """Handles the main window close event."""
        log.info("Close event triggered. Preparing to exit application.")

        # --- Stop any running background tasks ---
        log.debug("Stopping background tasks...")
        # Example: Stop NIST search thread if running
        if self.nist_search_view and hasattr(self.nist_search_view, 'stop_search'):
            try:
                self.nist_search_view.stop_search() # Assuming a method to signal stop
            except Exception as e:
                log.warning(f"Error trying to stop NIST search: {e}")

        # Example: Terminate external QProcess if running
        if self.external_process and self.external_process.state() != QProcess.ProcessState.NotRunning:
            log.info("Terminating external process...")
            self.external_process.terminate() # Ask nicely first
            if not self.external_process.waitForFinished(1000): # Wait 1 sec
                log.warning("External process did not terminate gracefully, killing.")
                self.external_process.kill()
            self.external_process = None # Clear reference

        # --- Save Settings ---
        self._save_persistent_settings()

        log.info("Accepting close event. Application will exit.")
        event.accept()


# --- Main Execution Block ---
if __name__ == '__main__':
    # Set up logging (moved to top)
    app = QApplication(sys.argv)

    # Example Configuration (replace with loading from YAML/JSON)
    test_config = {
        'application': {
            'remember_window_state': True,
        },
        'default_theme': DEFAULT_THEME,
        'file_io': {
            'default_delimiter': '\t',
            'default_comment_char': '#',
        },
        'plotting': {
            # Plotting specific configs, e.g., default colors, line styles
        },
        'paths': {
            # Paths for databases, etc.
        },
        'processing': {
            # Default parameters for processing steps
            'baseline_poly': {'degree': 3},
            'baseline_snip': {'iterations': 10},
            'denoising_wavelet': {'wavelet': 'db4', 'level': 5, 'mode': 'soft'},
            'smoothing_sg': {'window_length': 5, 'polyorder': 2},
        },
        'peak_detection': {
             'scipy': {'height': 100, 'prominence': 50, 'distance': 5}
        },
         'peak_fitting': {
             'default_profile': 'Gaussian',
             'fit_window_factor': 3.0 # e.g., +/- 3 * initial FWHM
         },
         'nist_search': {
              'default_tolerance_nm': 0.1,
              'default_min_aki': 1e6
         }
        # Add other sections as needed (boltzmann, cflibs, ml)
    }

    try:
        main_win = MainWindow(test_config)
        main_win.show()
        sys.exit(app.exec())
    except Exception as e:
        log.critical(f"An unhandled exception occurred during application startup or execution: {e}", exc_info=True)
        # Optionally show a critical error message box
        QMessageBox.critical(None, "Fatal Error", f"A critical error occurred:\n{e}\n\nPlease see logs for details.")
        sys.exit(1) # Exit with error code
