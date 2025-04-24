
"""
A simple placeholder view widget.
Used to occupy space before specific content (like a plot) is loaded.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class PlaceholderView(QWidget):
    """A basic widget showing a message."""
    def __init__(self, message="Placeholder View", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label = QLabel(message)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Apply a distinct style that can be overridden by themes
        self.label.setStyleSheet("font-size: 14pt; color: #888888;")
        layout.addWidget(self.label)
        self.setLayout(layout)

    def set_message(self, message: str):
        """Updates the message displayed on the label."""
        self.label.setText(message)
