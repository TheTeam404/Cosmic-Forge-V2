
# This file makes 'widgets' a Python package within 'ui'.

# Expose custom widgets for easier import elsewhere
from .info_button import InfoButton
from .collapsible_box import CollapsibleBox

__all__ = [
    "InfoButton",
    "CollapsibleBox",
]
