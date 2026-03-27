"""Raven Conference Timer - Countdown timer for conference talks.

Usage:
    raven-conference-timer 15:00
    raven-conference-timer 5
    python -m raven.conference_timer 15:00
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .. import __version__

logger.info(f"Raven Conference Timer version {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import argparse
    import math
    import sys
    import time

    import dearpygui.dearpygui as dpg

    from ..common.gui import utils as guiutils

    from . import config
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")


def _parse_duration(text: str) -> int:
    """Parse a duration string into total seconds.

    Accepts "mm:ss" or bare minutes (e.g. "15" -> 900s, "5:00" -> 300s).
    """
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise ValueError(f"Expected mm:ss, got {text!r}")
        minutes, seconds = int(parts[0]), int(parts[1])
    else:
        minutes, seconds = int(text), 0
    total = minutes * 60 + seconds
    if total <= 0:
        raise ValueError(f"Duration must be positive, got {total}s")
    return total


def main() -> int:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Raven Conference Timer - Countdown timer for talks"
    )
    parser.add_argument('-v', '--version', action='version',
                        version=('%(prog)s ' + __version__))
    parser.add_argument(
        "duration",
        help="Duration as mm:ss or bare minutes (e.g. 15:00 or 15)"
    )
    parser.add_argument(
        "--width", type=int, default=config.DEFAULT_WIDTH,
        help=f"Window width (default: {config.DEFAULT_WIDTH})"
    )
    parser.add_argument(
        "--height", type=int, default=config.DEFAULT_HEIGHT,
        help=f"Window height (default: {config.DEFAULT_HEIGHT})"
    )
    args = parser.parse_args()

    try:
        total_seconds = _parse_duration(args.duration)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # --- DPG bootup ---
    dpg.create_context()

    themes_and_fonts = guiutils.bootup(font_size=config.FONT_SIZE)

    # Load the large countdown font.
    _key, countdown_font = guiutils.load_extra_font(
        themes_and_fonts, config.COUNTDOWN_FONT_SIZE, "OpenSans", "Bold")

    dpg.create_viewport(
        title=f"Raven Conference Timer {__version__}",
        width=args.width,
        height=args.height
    )

    dpg.setup_dearpygui()

    # --- Color themes for the countdown text ---
    with dpg.theme() as theme_normal:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, config.COLOR_NORMAL)

    with dpg.theme() as theme_warning:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, config.COLOR_WARNING)

    with dpg.theme() as theme_expired:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, config.COLOR_EXPIRED)

    # --- Build GUI ---
    with dpg.window(tag="main_window"):
        countdown_text = dpg.add_text("", tag="countdown_text")
        dpg.bind_item_font(countdown_text, countdown_font)
        dpg.bind_item_theme(countdown_text, theme_normal)

    # --- Start app ---
    dpg.set_primary_window("main_window", True)
    dpg.show_viewport()

    # --- Timer state ---
    start_time = time.monotonic()
    expired = False

    # --- Render loop ---
    try:
        while dpg.is_dearpygui_running():
            elapsed = time.monotonic() - start_time
            remaining = max(0.0, total_seconds - elapsed)
            remaining_int = math.ceil(remaining)

            minutes = remaining_int // 60
            seconds = remaining_int % 60
            dpg.set_value(countdown_text, f"{minutes:02d}:{seconds:02d}")

            # Update color.
            if remaining <= 0:
                if not expired:
                    dpg.bind_item_theme(countdown_text, theme_expired)
                    expired = True
            elif remaining <= config.WARNING_THRESHOLD:
                dpg.bind_item_theme(countdown_text, theme_warning)

            dpg.render_dearpygui_frame()
    except KeyboardInterrupt:
        pass

    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    sys.exit(main())
