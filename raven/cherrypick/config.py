"""raven-cherrypick configuration."""

import torch

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

# Uses raven.common.deviceinfo pattern: {"component": {"device_string": ..., "dtype": ...}}.
# Validated at startup by deviceinfo.validate() — auto-falls back to CPU if CUDA unavailable,
# auto-promotes float16 → float32 on CPU.
gpu_config = {
    "thumbnails": {"device_string": "cuda:0",
                   "dtype": torch.float32},
}

# ---------------------------------------------------------------------------
# Font & Layout (derived from DPG ImGui style defaults)
# ---------------------------------------------------------------------------

FONT_SIZE = 20  # must match Raven's global standard

DPG_WINDOW_PADDING_Y = 8
DPG_FRAME_PADDING_Y = 3
DPG_ITEM_SPACING_Y = 4
DPG_SCROLLBAR_SIZE = 14

TOOLBAR_H = FONT_SIZE + 2 * DPG_FRAME_PADDING_Y  # ~26
STATUS_H = FONT_SIZE  # 20

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1040  # fits 1080p Linux Mint with taskbar

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

IMAGE_PANE_RATIO = 0.70  # main image view gets 70% of width

# Padding subtracted from the main window size to get the image view drawlist size.
# Empirical; accounts for DPG window padding, scrollbar, item spacing.
IMAGE_VIEW_H_PADDING = 14
IMAGE_VIEW_V_PADDING = (2 * DPG_WINDOW_PADDING_Y
                        + TOOLBAR_H
                        + 4 * DPG_ITEM_SPACING_Y
                        + STATUS_H)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

DEFAULT_TILE_SIZE = 128
TILE_SIZES = [32, 64, 128, 256, 512]

# Unique VHS noise placeholder tiles per grid tile size.
# Scaled to ~2× visible tiles per screenful; memory stays bounded
# because count × tile_area is roughly constant (~12 MB float32).
PLACEHOLDER_POOL_SIZES = {32: 256, 64: 128, 128: 48, 256: 12, 512: 4}
PLACEHOLDER_TINT = (0.92, 0.92, 1.0)  # subtle cool blue-gray
PLACEHOLDER_BRIGHTNESS = (0.04, 0.40)  # (lo, hi) — keeps placeholders dark; wider range ≈ PAL VHS

THUMBNAIL_BATCH_SIZE = 32  # images per GPU batch
THUMBNAIL_VRAM_FRACTION = 0.5  # fraction of free VRAM to budget for thumbnails
THUMBNAIL_VRAM_BUDGET_MAX_MB = 4096  # hard cap

THUMBNAIL_INTERPOLATION = "lanczos"  # "lanczos" or "area"
THUMBNAIL_LANCZOS_ORDER = 4  # Lanczos kernel order (3, 4, or 5); higher = better stopband, slightly more compute
THUMBNAIL_AREA_THRESHOLD = 64  # auto-switch to "area" at tile sizes ≤ this

# ---------------------------------------------------------------------------
# Colors (colorblind-accessible)
# ---------------------------------------------------------------------------

CHERRY_COLOR = (220, 180, 50, 255)  # golden/amber
LEMON_COLOR = (45, 45, 50, 255)  # muted dark gray (darker than neutral — rejects recede)
NEUTRAL_BORDER_COLOR = (60, 60, 65, 255)  # subtle gray
CURRENT_COLOR = (80, 160, 255, 255)  # bright blue
SELECTION_TINT = (255, 255, 255, 40)  # subtle overlay

# ---------------------------------------------------------------------------
# Zoom
# ---------------------------------------------------------------------------

ZOOM_IN_FACTOR = 1.25
ZOOM_OUT_FACTOR = 1.25
MOUSE_WHEEL_ZOOM_FACTOR = 1.1
PAN_AMOUNT = 30  # pixels per arrow keypress (at 1:1 zoom)
ZOOM_FIT_CAP = True  # cap zoom-to-fit at 100% (no upscale of small images)

# ---------------------------------------------------------------------------
# Preload
# ---------------------------------------------------------------------------

PRELOAD_WINDOW = 2  # ±N tiles in cross neighborhood (horizontal + vertical)
PRELOAD_MAX_SCALE = 0.25  # cap speculative preload at this scale (skip larger mips)
PRELOAD_RAM_FRACTION = 0.25  # fraction of available system RAM for preload cache
PRELOAD_RAM_BUDGET_MIN_MB = 512  # floor (don't starve the cache on low-RAM systems)
PRELOAD_RAM_BUDGET_MAX_MB = 16384  # ceiling (diminishing returns beyond this)
PRELOAD_RAM_BUDGET_FALLBACK_MB = 4096  # used when RAM detection fails

# ---------------------------------------------------------------------------
# Mipmaps
# ---------------------------------------------------------------------------

MIP_MIN_SIZE = 64  # smallest mip level (short edge, pixels)

# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

COMPARE_DEFAULT_FPS = 3.0
COMPARE_MIN_FPS = 0.5
COMPARE_MAX_FPS = 15.0
COMPARE_FPS_STEP = 0.5  # comma/period increment
COMPARE_MAX_IMAGES = 9
COMPARE_FADE_COLOR = (80, 160, 255, 180)  # bright blue tint for active tile
COMPARE_PAUSED_PULSE_DURATION = 2.0  # seconds, full 1→0→1 cycle when paused

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

# Page Up/Down jumps by this many rows (0 = dynamic, based on visible grid rows).
PAGE_JUMP_ROWS = 0

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

HELP_WINDOW_W = 1400
HELP_WINDOW_H = 760

# ---------------------------------------------------------------------------
# Triage mark overlay (main image view)
# ---------------------------------------------------------------------------

TRIAGE_MARK_FONT_SIZE = 2 * FONT_SIZE  # 40px — large enough to read at a glance
TRIAGE_MARK_DEFAULT_VISIBLE = True  # show by default

# ---------------------------------------------------------------------------
# Resize beacon (grid)
# ---------------------------------------------------------------------------

BEACON_DURATION = 0.6  # seconds, fade-out after resize settles
BEACON_COLOR = (80, 160, 255, 160)  # same blue family as CURRENT_COLOR
