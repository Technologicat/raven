"""Raven Conference Timer - Countdown timer for conference talks.

Usage:
    raven-conference-timer 15:00
    raven-conference-timer 5
    raven-conference-timer 15:00 --size 600
    python -m raven.conference_timer 15:00
"""

import argparse

from .. import __version__
from . import config

parser = argparse.ArgumentParser(description="Raven Conference Timer - Countdown timer for talks")
parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
parser.add_argument("duration", metavar="mm:ss",
                    help="Countdown duration (e.g. 15:00, or bare minutes like 15)")
parser.add_argument("--yellow", metavar="mm:ss", default=None,
                    help=f"Yellow threshold time (default: {config.YELLOW_THRESHOLD // 60}:{config.YELLOW_THRESHOLD % 60:02d})")
parser.add_argument("--red", metavar="mm:ss", default=None,
                    help=f"Red threshold time (default: {config.RED_THRESHOLD // 60}:{config.RED_THRESHOLD % 60:02d})")
parser.add_argument("--size", metavar="PIXELS", default=None, type=int,
                    help=(f"Font size in pixels (default: {config.COUNTDOWN_FONT_SIZE},"
                          f" max: {config.MAX_COUNTDOWN_FONT_SIZE})"))
parser.add_argument('--log', metavar='PATH', default=None,
                    help='mirror stderr log to this file (overwritten each run)')
parser.add_argument('--log-level', default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='root logger level (default: INFO)')
args = parser.parse_args()

import logging
from ..common import logsetup
logsetup.configure(level=getattr(logging, args.log_level),
                   logfile=args.log)
logger = logging.getLogger(__name__)

logger.info(f"Raven Conference Timer version {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import math
    import os
    import pathlib
    import platform
    import sys
    import time

    import dearpygui.dearpygui as dpg

    from unpythonic.env import env

    from ..common.gui import animation as gui_animation
    from ..common.gui import helpcard
    from ..common.gui import utils as guiutils
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
    # `args` was parsed at module top, before heavy imports — see top of file.
    try:
        total_seconds = _parse_duration(args.duration)
        yellow_threshold = _parse_duration(args.yellow) if args.yellow else config.YELLOW_THRESHOLD
        red_threshold = _parse_duration(args.red) if args.red else config.RED_THRESHOLD
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.size is not None:
        if args.size <= 0:
            print(f"Error: --size must be positive, got {args.size}", file=sys.stderr)
            return 1
        font_size = min(args.size, config.MAX_COUNTDOWN_FONT_SIZE)
    else:
        font_size = config.COUNTDOWN_FONT_SIZE

    # --- DPG bootup ---
    #
    # We skip `guiutils.bootup` — its default font (20px with ~11k extended
    # Unicode codepoints) causes a visible flash when switching fonts at large
    # countdown sizes (DPG briefly falls back to the bound default during atlas
    # rebuild).  Instead, we load only the countdown font (Latin-1), and call
    # `setup_markdown` directly for the help card (markdown fonts are lazy-loaded,
    # so no atlas rebuild until F1 is actually pressed).
    dpg.create_context()

    reference_size = config.COUNTDOWN_FONT_SIZE
    countdown_font_path = guiutils.get_font_path("OpenSans", variant="Bold")
    loaded_fonts = {}  # size → DPG font ID

    # For fonts ≤ reference_size, we load the target font directly — it's
    # small enough to measure accurately in the initial 500×300 viewport.
    # For fonts > reference_size, we start with the reference font and switch
    # to the target in a frame callback after estimating the viewport size.
    initial_font_size = min(font_size, reference_size)

    with dpg.font_registry() as font_registry:
        with dpg.font(countdown_font_path, initial_font_size) as initial_font:
            pass  # default Latin-1 is sufficient for "00:00"
        dpg.bind_font(initial_font)
        loaded_fonts[initial_font_size] = initial_font

    countdown_font = initial_font

    # Set up Markdown renderer and a small GUI font for the help card.
    # Markdown fonts are loaded lazily (no atlas rebuild until F1 is pressed).
    # The GUI font is needed because the DPG default is the large countdown font.
    guiutils.setup_markdown(font_registry, font_size=config.GUI_FONT_SIZE)
    guiutils.setup_themes()  # rounded corners for the help card
    gui_font_path = guiutils.get_font_path("OpenSans", variant="Regular")
    with dpg.font(gui_font_path, config.GUI_FONT_SIZE, parent=font_registry) as gui_font:
        pass  # Latin-1 is sufficient for help card text

    def _load_countdown_font(size: int) -> int:
        """Load OpenSans Bold at `size`.  Cached; skips extended Unicode ranges."""
        if size not in loaded_fonts:
            logger.info(f"_load_countdown_font: loading at size {size}")
            with dpg.font(countdown_font_path, size, parent=font_registry) as font:
                pass
            loaded_fonts[size] = font
        return loaded_fonts[size]

    # Start with a small viewport — the startup callback resizes it after
    # measuring the text.  The text widget is positioned offscreen during
    # setup so the user doesn't see intermediate states.
    icon_ext = "ico" if platform.system().upper() == "WINDOWS" else "png"
    icons_dir = pathlib.Path(os.path.dirname(__file__), "..", "icons")
    dpg.create_viewport(
        title=f"Raven Conference Timer {__version__}",
        small_icon=str((icons_dir / f"app_128_notext.{icon_ext}").resolve()),
        large_icon=str((icons_dir / f"app_256.{icon_ext}").resolve()),
        width=config.INITIAL_WIDTH,
        height=config.INITIAL_HEIGHT,
    )

    dpg.setup_dearpygui()

    # --- Color themes for the countdown text ---
    with dpg.theme() as theme_normal:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, config.COLOR_NORMAL)

    with dpg.theme() as theme_yellow:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, config.COLOR_YELLOW)

    with dpg.theme() as theme_red:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, config.COLOR_RED)

    # Pulsating theme for the expired state.
    with dpg.theme() as theme_expired:
        with dpg.theme_component(dpg.mvText):
            expired_color_widget = dpg.add_theme_color(dpg.mvThemeCol_Text, config.COLOR_EXPIRED)
        expired_glow = gui_animation.PulsatingColor(cycle_duration=config.PULSATION_CYCLE,
                                                    theme_color_widget=expired_color_widget)
        gui_animation.animator.add(expired_glow)

    # Pulsating theme for the paused state.
    #
    # This reuses the current countdown color — we replace the glow animation's
    # `.rgb` (which controls the RGB components) when the paused state is entered.
    with dpg.theme() as theme_paused:
        with dpg.theme_component(dpg.mvText):
            paused_color_widget = dpg.add_theme_color(dpg.mvThemeCol_Text, config.COLOR_NORMAL)  # dummy color
        pause_glow = gui_animation.PulsatingColor(cycle_duration=config.PULSATION_CYCLE,
                                                  theme_color_widget=paused_color_widget)
        gui_animation.animator.add(pause_glow)

    # Lookup tables for color state → theme / RGB.
    # The RGB is used by the pause mechanism, overriding the alpha channel.
    solid_themes = {
        "normal": theme_normal,
        "yellow": theme_yellow,
        "red": theme_red,
        "expired": theme_expired,
    }
    color_rgbs = {
        "normal": list(config.COLOR_NORMAL[:3]),
        "yellow": list(config.COLOR_YELLOW[:3]),
        "red": list(config.COLOR_RED[:3]),
        "expired": list(config.COLOR_EXPIRED[:3]),
    }

    # --- Build GUI ---
    # Position the text offscreen during setup so the user doesn't see
    # intermediate states.  The startup callback moves it to (0, 0).
    initial_minutes = total_seconds // 60
    initial_seconds = total_seconds % 60
    with dpg.window(tag="main_window", no_scrollbar=True):
        countdown_text = dpg.add_text(f"{initial_minutes:02d}:{initial_seconds:02d}",
                                      tag="countdown_text", pos=[10000, 10000])
        dpg.bind_item_font(countdown_text, countdown_font)
        dpg.bind_item_theme(countdown_text, theme_normal)

    # --- Start app ---
    dpg.set_primary_window("main_window", True)
    dpg.show_viewport()

    # --- State ---
    start_time = None  # set by _startup; timer doesn't run until then
    color_state = "normal"
    paused = False
    frozen_remaining = None  # remaining seconds snapshot when paused
    _help_window = None  # set up after frame callbacks (below)

    # Compensate for trailing glyph advance width in the text rect — the
    # visible ink is narrower than `get_item_rect_size` reports, making the
    # text look left-aligned.  We widen the viewport by `nudge` and let the
    # centering formula shift the text right naturally.
    nudge = int(math.sqrt(font_size))

    # --- Recentering & fullscreen ---

    def _recenter_text() -> None:
        """Re-center the countdown text in the current viewport."""
        tw, th = guiutils.get_widget_size(countdown_text)
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        pad = 2 * config.DPG_WINDOW_PADDING
        text_x = max(0, int((vw - pad - tw) / 2) + nudge // 2)
        text_y = max(0, int((vh - pad - th) / 2))
        dpg.set_item_pos(countdown_text, [text_x, text_y])

    def _toggle_fullscreen(*_args) -> None:
        """Toggle fullscreen and re-center the text."""
        dpg.toggle_viewport_fullscreen()
        if guiutils.wait_for_resize("main_window"):
            _recenter_text()
            # Re-center help card too, if open.
            if _help_window is not None:
                _help_window.reposition()

    # --- Keyboard ---

    def _on_key(_sender, key, *_args):
        nonlocal paused, frozen_remaining, start_time
        # Help card handles its own Escape key; suppress other keys while visible.
        if _help_window is not None and _help_window.is_visible():
            return
        if key == dpg.mvKey_Escape:
            dpg.stop_dearpygui()
        elif key == dpg.mvKey_F1:
            if _help_window is not None:
                _help_window.show()
        elif key == dpg.mvKey_F11:
            _toggle_fullscreen()
        elif key == dpg.mvKey_Spacebar and start_time is not None:
            paused = not paused
            if paused:
                # Freeze the display and start pulsating.
                elapsed = time.monotonic() - start_time
                frozen_remaining = max(0.0, total_seconds - elapsed)
                pause_glow.rgb = color_rgbs[color_state]
                pause_glow.reset()
                # Expired already pulsates; only swap theme for non-expired states.
                if color_state != "expired":
                    dpg.bind_item_theme(countdown_text, theme_paused)
            else:
                # Resume: shift start_time so remaining stays continuous.
                start_time = time.monotonic() - (total_seconds - frozen_remaining)
                frozen_remaining = None
                dpg.bind_item_theme(countdown_text, solid_themes[color_state])

    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=_on_key)

    # --- Startup callbacks (deferred to frame 10 so DPG has laid out text) ---
    #
    # Frame 10: measure reference text (120px, fits in any viewport), load
    #           the target font.  Don't resize the viewport yet — keep it
    #           small so the user doesn't see a huge empty window.
    #
    # Frame 12: compute the final viewport size from the reference measurement
    #           (font metrics scale ~linearly), resize the viewport.
    #
    # Frame 14: the viewport has settled — read its actual client dimensions,
    #           center the text, reveal it, and start the timer.

    if font_size <= reference_size:
        # Default size — font is already loaded.  Measure, fit, center.
        def _fit_viewport(*_args):
            """Frame 10: measure text, resize viewport."""
            tw, th = guiutils.get_widget_size(countdown_text)
            pad = 2 * config.DPG_WINDOW_PADDING
            target_w = int(tw + pad)
            target_h = int(th + pad)
            dpg.set_viewport_width(target_w)
            dpg.set_viewport_height(target_h)
            dpg.set_viewport_min_width(target_w)
            dpg.set_viewport_min_height(target_h)

        def _center_and_start(*_args):
            """Frame 12: center text in actual viewport, reveal, start."""
            nonlocal start_time
            tw, th = guiutils.get_widget_size(countdown_text)
            vw = dpg.get_viewport_client_width()
            vh = dpg.get_viewport_client_height()
            pad = 2 * config.DPG_WINDOW_PADDING
            text_x = max(0, int((vw - pad - tw) / 2) + nudge // 2)
            text_y = max(0, int((vh - pad - th) / 2))
            dpg.set_item_pos(countdown_text, [text_x, text_y])
            dpg.set_viewport_resizable(False)
            start_time = time.monotonic()

        dpg.set_frame_callback(10, _fit_viewport)
        dpg.set_frame_callback(12, _center_and_start)

    else:
        # Custom --size: load target font, estimate viewport from reference
        # measurement, then center text in the actual viewport.
        ref_dims = [0, 0]  # tw_ref, th_ref

        def _load_font(*_args):
            """Frame 10: measure reference text, load target font."""
            nonlocal countdown_font
            ref_dims[0], ref_dims[1] = guiutils.get_widget_size(countdown_text)
            countdown_font = _load_countdown_font(font_size)
            dpg.bind_item_font(countdown_text, countdown_font)

        def _resize_viewport(*_args):
            """Frame 12: resize viewport from reference estimate."""
            tw_ref, th_ref = ref_dims
            if tw_ref <= 0 or th_ref <= 0:
                return
            pad = 2 * config.DPG_WINDOW_PADDING
            ratio = font_size / reference_size
            target_w = int(tw_ref * ratio + pad)
            target_h = int(th_ref * ratio + pad)
            dpg.set_viewport_width(target_w)
            dpg.set_viewport_height(target_h)

        def _center_and_start(*_args):
            """Frame 14: center text in actual viewport, reveal, start."""
            nonlocal start_time
            tw_ref, th_ref = ref_dims
            ratio = font_size / reference_size
            text_w = tw_ref * ratio
            vw = dpg.get_viewport_client_width()
            vh = dpg.get_viewport_client_height()
            pad = 2 * config.DPG_WINDOW_PADDING
            text_x = max(0, int((vw - pad - text_w) / 2) + nudge // 2)
            text_y = max(0, int((vh - pad - th_ref * ratio) / 2))
            dpg.set_item_pos(countdown_text, [text_x, text_y])
            dpg.set_viewport_resizable(False)
            start_time = time.monotonic()

        dpg.set_frame_callback(10, _load_font)
        dpg.set_frame_callback(12, _resize_viewport)
        dpg.set_frame_callback(14, _center_and_start)

    # --- Help card (F1) ---
    #
    # HelpWindow defers its GUI rendering to first show (frame ≥ 10),
    # so it's safe to create the instance here.
    hotkey_info = (
        env(key_indent=0, key="Space", action_indent=0, action="Pause / resume", notes=""),
        env(key_indent=0, key="Esc", action_indent=0, action="Quit", notes=""),
        helpcard.hotkey_blank_entry,
        env(key_indent=0, key="F1", action_indent=0, action="This help card", notes=""),
        env(key_indent=0, key="F11", action_indent=0, action="Toggle fullscreen", notes=""),
    )
    _help_window = helpcard.HelpWindow(  # noqa: F841 — read by `_on_key` closure
        hotkey_info=hotkey_info,
        width=config.HELP_WINDOW_W,
        height=config.HELP_WINDOW_H,
        reference_window="main_window",
        themes_and_fonts=env(font_size=config.GUI_FONT_SIZE),
        gui_font=gui_font,
    )

    # --- Render loop ---
    try:
        while dpg.is_dearpygui_running():
            if start_time is None or paused:
                # Waiting for startup, or paused — just keep animations ticking.
                gui_animation.animator.render_frame()
                dpg.render_dearpygui_frame()
                continue

            elapsed = time.monotonic() - start_time
            remaining = max(0.0, total_seconds - elapsed)
            remaining_int = math.ceil(remaining)

            minutes = remaining_int // 60
            seconds = remaining_int % 60
            dpg.set_value(countdown_text, f"{minutes:02d}:{seconds:02d}")

            # Update color (only on state change).
            if remaining <= 0 and color_state != "expired":
                dpg.bind_item_theme(countdown_text, theme_expired)
                color_state = "expired"
            elif remaining > 0 and remaining <= red_threshold and color_state != "red":
                dpg.bind_item_theme(countdown_text, theme_red)
                color_state = "red"
            elif remaining > red_threshold and remaining <= yellow_threshold and color_state != "yellow":
                dpg.bind_item_theme(countdown_text, theme_yellow)
                color_state = "yellow"

            gui_animation.animator.render_frame()
            dpg.render_dearpygui_frame()
    except KeyboardInterrupt:
        pass

    gui_animation.animator.clear()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    sys.exit(main())
