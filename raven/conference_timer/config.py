"""Configuration constants for the Raven Conference Timer."""

# Font size (pixels) — must match raven's global app standard.
FONT_SIZE = 20

# Large countdown font size (pixels).
COUNTDOWN_FONT_SIZE = 120

# DPG style values (pixels). Used for auto-fit margin calculation.
DPG_WINDOW_PADDING = 8   # mvStyleVar_WindowPadding (default)
DPG_SCROLLBAR_SIZE = 14  # mvStyleVar_ScrollbarSize (default)

# Initial viewport dimensions (before auto-fit to text size).
INITIAL_WIDTH = 500
INITIAL_HEIGHT = 300

# Default thresholds (seconds remaining) for color changes.
YELLOW_THRESHOLD = 300  # 5:00 — counter turns yellow
RED_THRESHOLD = 120     # 2:00 — counter turns red

# Colors (RGBA 0–255).
COLOR_NORMAL = (255, 255, 255, 255)
COLOR_YELLOW = (255, 255, 0, 255)
COLOR_RED = (255, 64, 64, 255)
COLOR_EXPIRED = (255, 0, 0, 255)
