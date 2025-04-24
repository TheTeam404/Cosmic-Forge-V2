
"""
A reusable Info Button widget using standard icons.
Provides tooltips and calls a function to show detailed help.
"""
import logging
from PyQt6.QtWidgets import QPushButton, QToolTip, QMessageBox, QStyle
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QSize, Qt

class InfoButton(QPushButton):
    """
    A standardized '?' button for showing help/info messages.
    Uses standard Qt icons with a text fallback if the icon fails.
    """
    def __init__(self, detailed_info_func: callable, tooltip_text: str = "Show Help", parent=None):
        """
        Args:
            detailed_info_func (callable): Function to call when clicked
                                            (usually shows a QMessageBox).
            tooltip_text (str, optional): Text for the brief tooltip on hover.
                                           Defaults to "Show Help".
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.detailed_info_func = detailed_info_func

        self.setToolTip(tooltip_text)
        self.clicked.connect(self.show_detailed_info)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Prevent button taking focus
        self.setObjectName("InfoButton") # For QSS styling

        # --- Set Icon ---
        help_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion, None, self)
        icon_size = QSize(16, 16) # Standard small icon size

        if not help_icon.isNull():
            self.setIcon(help_icon)
            self.setText("") # Remove text if standard icon works
            self.setIconSize(icon_size)
            # Set fixed size based on typical icon + padding needs
            margin = 3 # Add margin around icon
            self.setFixedSize(icon_size.width() + 2 * margin, icon_size.height() + 2 * margin)
            # logging.debug("InfoButton using standard Qt SP_MessageBoxQuestion icon.") # Keep logging minimal in final
        else:
            # Fallback to text "?" if standard icon fails
            logging.warning("Standard help icon not found, falling back to text '?'.")
            self.setText("?")
            font_metrics = self.fontMetrics()
            text_width = font_metrics.horizontalAdvance("?")
            text_height = font_metrics.height()
            padding = 8 # Add some padding around the text
            # Ensure minimum size is reasonable
            min_w = max(text_width + padding, 20)
            min_h = max(text_height + padding // 2, 20)
            self.setFixedSize(min_w, min_h)
            # Apply specific QSS styling for text-based button if needed,
            # or rely on the #InfoButton objectName style.

    def show_detailed_info(self):
        """Calls the provided function to show detailed information."""
        if self.detailed_info_func and callable(self.detailed_info_func):
            try:
                self.detailed_info_func()
            except Exception as e:
                 logging.error(f"Error executing info function: {e}", exc_info=True)
                 # Show error to the user, ensuring parent context if available
                 parent_widget = self.parent() if self.parent() is not None else self
                 QMessageBox.warning(parent_widget, "Help Error", f"Could not display help information:\n{e}")
        else:
            logging.warning("No valid detailed info function provided for InfoButton.")
            parent_widget = self.parent() if self.parent() is not None else self
            QMessageBox.information(parent_widget, "No Help", "Detailed help is not available for this item.")

