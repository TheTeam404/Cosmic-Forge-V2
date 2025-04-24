
"""
A collapsible QGroupBox widget.
Allows sections of the UI to be hidden/shown with an animation.
Inspired by various online examples.
"""

import logging
from PyQt6.QtWidgets import (QWidget, QGroupBox, QVBoxLayout, QToolButton, QSizePolicy,
                             QScrollArea, QLayout) # Added QLayout
from PyQt6.QtCore import (Qt, pyqtSignal, QPropertyAnimation, QEasingCurve,
                          QParallelAnimationGroup, QTimer, pyqtSlot) # Added QTimer, pyqtSlot

class CollapsibleBox(QGroupBox):
    """
    A QGroupBox subclass that can be collapsed/expanded by clicking the
    title bar button. Content area scrolls if needed.
    """
    toggled = pyqtSignal(bool) # Emits True when expanded, False when collapsed

    def __init__(self, title: str = "", parent: QWidget = None, is_expanded: bool = True):
        """
        Args:
            title (str, optional): The title displayed on the button header. Defaults to "".
            parent (QWidget, optional): Parent widget. Defaults to None.
            is_expanded (bool, optional): Initial state of the box. Defaults to True (expanded).
        """
        super().__init__("", parent) # GroupBox title set via button
        self.setObjectName("CollapsibleBox") # For specific QSS styling

        self.is_expanded = is_expanded
        self.animation_duration = 200 # ms
        self.content_height = 0 # Calculated height of content

        # --- Main layout for this GroupBox ---
        self.box_layout = QVBoxLayout(self)
        self.box_layout.setSpacing(0)
        self.box_layout.setContentsMargins(1, 1, 1, 1)

        # --- Toggle Button (acts as header) ---
        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("CollapsibleBoxToggleButton") # For QSS
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if self.is_expanded else Qt.ArrowType.RightArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(self.is_expanded)
        # Basic styling (can be overridden by theme QSS)
        self.toggle_button.setStyleSheet("QToolButton#CollapsibleBoxToggleButton { border: none; padding: 4px; background-color: transparent; text-align: left; }")
        font = self.font(); font.setBold(True); self.toggle_button.setFont(font)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.box_layout.addWidget(self.toggle_button)

        # --- Content Area (Scrollable) ---
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred) # Allow vertical expansion
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        # self.scroll_area.setStyleSheet("background-color: transparent; border: none;") # Theme handles this

        self.content_widget = QWidget(self.scroll_area)
        self.content_layout = QVBoxLayout(self.content_widget) # Default layout for content widget
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(8)
        # Apply layout to the widget that will be scrolled
        self.content_widget.setLayout(self.content_layout)

        self.scroll_area.setWidget(self.content_widget) # Put content widget in scroll area
        self.box_layout.addWidget(self.scroll_area) # Add scroll area to the main box layout

        # --- Animation Setup ---
        # Animate the maximum height of the scroll area
        self.animation = QPropertyAnimation(self.scroll_area, b"maximumHeight")
        self.animation.setDuration(self.animation_duration)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # --- Connections ---
        self.toggle_button.toggled.connect(self._toggle_content)

        # --- Initial State ---
        # Use QTimer to set initial height after layout calculation
        QTimer.singleShot(0, self._initialize_height)

    def _update_content_height(self):
        """Recalculates the maximum height needed for the content."""
        current_height = 0
        if self.content_layout:
            # Ensure layout is updated
            self.content_layout.activate()
            current_height = self.content_layout.sizeHint().height()
            # Add margins
            margins = self.content_layout.contentsMargins()
            current_height += margins.top() + margins.bottom()
        else:
            # Fallback if no layout set, less reliable
            current_height = self.content_widget.sizeHint().height()

        self.content_height = max(current_height, 10) # Ensure a small minimum height for calculation
        # logging.debug(f"Updated content height for '{self.title()}': {self.content_height}")


    def _initialize_height(self):
        """ Sets the initial expanded/collapsed height after layout settles."""
        self._update_content_height()
        if self.is_expanded:
             self.scroll_area.setMaximumHeight(self.content_height)
             self.scroll_area.setVisible(True)
        else:
             self.scroll_area.setMaximumHeight(0)
             self.scroll_area.setVisible(False)

    def setContentLayout(self, layout: QLayout):
        """ Replaces the existing content layout with a new one. """
        old_layout = self.content_widget.layout()
        if old_layout is not None:
            # Remove widgets safely before deleting layout
            while (item := old_layout.takeAt(0)) is not None:
                if (widget := item.widget()) is not None:
                    widget.setParent(None); widget.deleteLater()
                elif (sub_layout := item.layout()) is not None:
                    # TODO: Handle nested layouts recursively if needed
                    pass # For now, assume simple layouts
            QWidget().setLayout(old_layout) # Reparent to delete

        self.content_widget.setLayout(layout)
        self.content_layout = layout
        QTimer.singleShot(0, self._initialize_height)


    def addWidget(self, widget: QWidget):
         """Adds a widget to the collapsible content area's layout."""
         if self.content_layout:
             self.content_layout.addWidget(widget)
             QTimer.singleShot(0, self._initialize_height) # Update height after adding
         else:
             logging.error(f"Cannot add widget to CollapsibleBox '{self.title()}' - content layout is missing.")


    @pyqtSlot(bool)
    def _toggle_content(self, checked: bool):
        """Handles the collapse/expand animation."""
        if self.is_expanded == checked: return # State already matches

        self.is_expanded = checked
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._update_content_height() # Ensure height is current

        start_height = self.scroll_area.maximumHeight()
        target_height = self.content_height if checked else 0

        # Ensure widget is visible for expansion animation start
        if checked and start_height == 0:
             self.scroll_area.setVisible(True)

        self.animation.setStartValue(start_height)
        self.animation.setEndValue(target_height)

        # Manage finished signal connection
        try: self.animation.finished.disconnect()
        except TypeError: pass # No connection existed
        if not checked:
             # Hide after collapsing animation
             self.animation.finished.connect(self._hide_content_after_collapse)

        self.animation.start()
        self.toggled.emit(self.is_expanded)

    def _hide_content_after_collapse(self):
        """Slot connected to animation finish to hide widget if collapsed."""
        if not self.is_expanded:
             self.scroll_area.setVisible(False)

    def setTitle(self, title): super().setTitle(""); self.toggle_button.setText(title)
    def title(self) -> str: return self.toggle_button.text()