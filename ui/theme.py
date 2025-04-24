
"""
Theme Management for the LIBS Cosmic Forge UI.
Loads and applies Qt Stylesheets (QSS) and sets Matplotlib style.
"""
import os
import logging
from typing import List, Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QFile, QTextStream # For reading QSS resource files if needed
from utils.helpers import get_project_root

# Try importing matplotlib safely
try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    logging.warning("Matplotlib not found. Plot styling will be basic.")

class ThemeManager:
    """Handles loading and applying QSS themes and Matplotlib styles."""

    def __init__(self, application: QApplication, config: dict):
        self.app = application
        self.config = config
        self.styles_dir = os.path.join(get_project_root(), "assets", "styles")
        self.available_themes = self._find_available_themes()
        self.current_theme_name = config.get('default_theme', 'dark_cosmic') # Default fallback
        # Ensure default theme exists
        if self.current_theme_name not in self.available_themes and self.available_themes:
             logging.warning(f"Default theme '{self.current_theme_name}' not found. Using '{self.available_themes[0]}'.")
             self.current_theme_name = self.available_themes[0]
        elif not self.available_themes:
             logging.error("No theme files (.qss) found in styles directory!")
             self.current_theme_name = None

        logging.info(f"Available themes: {self.available_themes}")

    def _find_available_themes(self) -> List[str]:
        themes = [];
        if not os.path.isdir(self.styles_dir): logging.warning(f"Styles dir not found: {self.styles_dir}"); return themes
        try: [themes.append(fn[:-4]) for fn in os.listdir(self.styles_dir) if fn.endswith(".qss")]; return sorted(themes)
        except Exception as e: logging.error(f"Error scanning styles dir {self.styles_dir}: {e}"); return []

    def get_available_themes(self) -> List[str]: return self.available_themes

    def _load_stylesheet(self, theme_name: str) -> Optional[str]:
        if theme_name not in self.available_themes: logging.error(f"Theme '{theme_name}' not available."); return None
        qss_path = os.path.join(self.styles_dir, f"{theme_name}.qss")
        if not os.path.exists(qss_path): logging.error(f"Stylesheet not found: {qss_path}"); return None
        try:
            with open(qss_path, 'r', encoding='utf-8') as f: stylesheet = f.read()
            # Simple variable substitution (optional) - Example: Replace ##ICON_PATH##
            icon_path_str = os.path.join(get_project_root(), "assets", "icons").replace("\\", "/") # Use forward slashes
            stylesheet = stylesheet.replace("##ICON_PATH##", icon_path_str)
            logging.debug(f"Loaded stylesheet: {qss_path}"); return stylesheet
        except Exception as e: logging.error(f"Error reading stylesheet {qss_path}: {e}", exc_info=True); return None

    def apply_theme(self, theme_name: Optional[str] = None):
        """ Applies the specified theme and corresponding Matplotlib style. """
        target_theme = theme_name if theme_name and theme_name in self.available_themes else self.current_theme_name
        if not target_theme: logging.error("No valid theme available to apply."); return

        stylesheet = self._load_stylesheet(target_theme)
        if stylesheet is not None:
            try: self.app.setStyleSheet(stylesheet); self.current_theme_name = target_theme; logging.info(f"Applied theme: '{target_theme}'")
            except Exception as e: logging.error(f"Error applying stylesheet '{target_theme}': {e}", exc_info=True)
        else: logging.error(f"Could not apply theme '{target_theme}': Stylesheet failed to load.")

        # --- Apply Matplotlib Style ---
        if MPL_AVAILABLE:
             is_dark = 'dark' in target_theme.lower()
             mpl_style_key = 'matplotlib_style_dark' if is_dark else 'matplotlib_style_light'
             fallback = 'dark_background' if is_dark else 'default'
             mpl_style = self.config.get('plotting', {}).get(mpl_style_key, fallback)
             try:
                  plt.style.use(mpl_style); logging.info(f"Set Matplotlib style: '{mpl_style}' for theme '{target_theme}'")
             except Exception as e: logging.warning(f"Could not set MPL style '{mpl_style}'. Error: {e}")
        else: logging.debug("Matplotlib not available, skipping style setting.")