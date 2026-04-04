"""Configuration constants for the Raven Conference Timer."""

# Default countdown font size (pixels).
COUNTDOWN_FONT_SIZE = 120

# Maximum countdown font size (pixels).  DPG rasterizes each Latin-1 glyph
# into a font atlas texture; above ~1200px the atlas silently overflows.
MAX_COUNTDOWN_FONT_SIZE = 1000


# DPG style values (pixels). Used for auto-fit margin calculation.
DPG_WINDOW_PADDING = 8   # mvStyleVar_WindowPadding (default)
DPG_SCROLLBAR_SIZE = 14  # mvStyleVar_ScrollbarSize (default)

# Initial viewport dimensions (before auto-fit to text size).
INITIAL_WIDTH = 500
INITIAL_HEIGHT = 300

# Default thresholds (seconds remaining) for color changes.
YELLOW_THRESHOLD = 300  # 5:00 — counter turns yellow
RED_THRESHOLD = 120     # 2:00 — counter turns red

# Pulsation cycle duration (seconds). Used for expired and paused states.
PULSATION_CYCLE = 2.0

# Colors (RGBA 0–255).
COLOR_NORMAL = (255, 255, 255, 255)
COLOR_YELLOW = (255, 255, 0, 255)
COLOR_RED = (255, 64, 64, 255)
COLOR_EXPIRED = (255, 0, 0, 255)
