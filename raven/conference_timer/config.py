"""Configuration constants for the Raven Conference Timer."""

# Font size (pixels) — must match raven's global app standard.
FONT_SIZE = 20

# Large countdown font size (pixels).
COUNTDOWN_FONT_SIZE = 120

# Default viewport dimensions.
DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 300

# Default thresholds (seconds remaining) for color changes.
YELLOW_THRESHOLD = 300  # 5:00 — counter turns yellow
RED_THRESHOLD = 120     # 2:00 — counter turns red

# Colors (RGBA 0–255).
COLOR_NORMAL = (255, 255, 255, 255)
COLOR_YELLOW = (255, 255, 0, 255)
COLOR_RED = (255, 64, 64, 255)
COLOR_EXPIRED = (255, 0, 0, 255)
