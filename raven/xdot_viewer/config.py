"""Configuration constants for the Raven XDot Viewer."""

# Font size (pixels) — must match raven's global app standard.
FONT_SIZE = 20

# DPG (ImGui) default style values used in layout calculations.
DPG_WINDOW_PADDING_Y = 8    # mvStyleVar_WindowPadding[1]
DPG_FRAME_PADDING_Y = 3     # mvStyleVar_FramePadding[1]
DPG_ITEM_SPACING_Y = 4      # mvStyleVar_ItemSpacing[1]
DPG_SCROLLBAR_SIZE = 14     # mvStyleVar_ScrollbarSize

# Derived layout sizes.
TOOLBAR_H = FONT_SIZE + 2 * DPG_FRAME_PADDING_Y   # tallest toolbar item (button/input)
STATUS_H = FONT_SIZE                                # text line

WIDGET_H_PADDING = DPG_SCROLLBAR_SIZE + 2 * DPG_WINDOW_PADDING_Y - 13  # -13: empirical fudge to align with toolbar search field
WIDGET_V_PADDING = (2 * DPG_WINDOW_PADDING_Y +     # top + bottom window margin
                    TOOLBAR_H +                     # toolbar row
                    4 * DPG_ITEM_SPACING_Y +        # gaps: toolbar group↔widget, widget↔status,
                                                    #        + 2 empirical (ImGui internal leading/rounding)
                    STATUS_H)                       # status bar

# Default viewport dimensions (also used as argparse defaults).
# Fits onto a 1080p screen in Linux Mint (same as Librarian/Visualizer).
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1040

# Arrow key pan amount (pixels per keypress).
PAN_AMOUNT = 10

# Zoom factors.
ZOOM_IN_FACTOR = 1.2           # per keypress / toolbar button click
ZOOM_OUT_FACTOR = 1.2          # applied as 1/factor
MOUSE_WHEEL_ZOOM_FACTOR = 1.1  # per wheel notch (finer than keyboard)

# Dark mode — invert graph lightness for DPG's dark theme.
DARK_MODE = True
DARK_MODE_BACKGROUND = (45, 45, 48, 255)     # DPG default dark gray
LIGHT_MODE_BACKGROUND = (255, 255, 255, 255)  # white

# Interval (seconds) between file modification checks for auto-reload.
FILE_RELOAD_POLL_INTERVAL = 2.0

# Duration (seconds) of highlight fade-out animation.
HIGHLIGHT_FADE_DURATION = 2.0

# Font atlas sizes (px) for graph text rendering.
# The renderer picks whichever is closest to the rendered text size.
GRAPH_TEXT_FONT_SIZES = [4, 8, 16, 32, 64]

# Help card dimensions (pixels). Chosen to fit the content.
HELP_WINDOW_W = 1400
HELP_WINDOW_H = 760
