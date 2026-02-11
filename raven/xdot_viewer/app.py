"""Raven XDot Viewer - Standalone graph viewer application.

This is a standalone viewer for xdot format graphs. It can open:
- .xdot files (xdot format directly)
- .dot files (filtered through GraphViz)

Usage:
    raven-xdot-viewer [file.xdot|file.dot]
    python -m raven.xdot_viewer [file.xdot|file.dot]
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .. import __version__

logger.info(f"Raven-xdot-viewer version {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import argparse
    import os
    import subprocess
    import sys
    import time
    import pathlib
    from typing import Optional, Union

    import dearpygui.dearpygui as dpg

    from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders

    from ..common import utils as common_utils
    from ..common.gui.xdotwidget import XDotWidget
    from ..common.gui import utils as guiutils
    from ..common.gui import animation as gui_animation
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")


# Application state
_app_state = {
    "widget": None,
    "current_file": None,
    "search_input": None,
    "status_text": None,
    "file_mtime": None,
}

# Layout constants for the main window.
# These account for the toolbar and status bar around the graph widget.
_WIDGET_H_PADDING = 20   # horizontal padding (scrollbar + margins)
_WIDGET_V_PADDING = 100  # vertical: toolbar + status bar + margins


def _load_file(filepath: Union[pathlib.Path, str]) -> Optional[str]:
    """Load an xdot or dot file and return xdot code.

    For .dot files, runs GraphViz to convert to xdot format.
    """
    # Resolve to absolute path so mtime checks remain valid if CWD changes.
    filepath = common_utils.absolutize_filename(filepath)

    if not os.path.exists(filepath):
        logger.error(f"_load_file: File not found: '{filepath}'")  # TODO: additionally use `raven.common.gui.messagebox` to spawn a modal GUI error box that background-threads automatically
        return None

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".xdot":  # has pre-rendered layout
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext in (".dot", ".gv"):  # needs GraphViz rendering
        try:
            result = subprocess.run(
                ["dot", "-Txdot", filepath],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except FileNotFoundError:
            logger.error("_load_file: `dot` command not found, cannot render `.dot` or `.gv` files. Please install GraphViz.")  # TODO: additionally use `raven.common.gui.messagebox` to spawn a modal GUI error box that background-threads automatically
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"_load_file: Error running the `dot` command: {e.stderr}")  # TODO: additionally use `raven.common.gui.messagebox` to spawn a modal GUI error box that background-threads automatically
            return None

    else:
        logger.error(f"_load_file: Unsupported file type: '{ext}'. Supported: `.xdot`, `.dot`, `.gv`.")  # TODO: additionally use `raven.common.gui.messagebox` to spawn a modal GUI error box that background-threads automatically
        return None


def _open_file(filepath: str) -> None:
    """Open and display a graph file."""
    xdotcode = _load_file(filepath)
    if xdotcode is None:
        return

    _app_state["current_file"] = filepath
    _app_state["file_mtime"] = os.path.getmtime(filepath)

    widget = _app_state["widget"]
    if widget is not None:
        widget.set_xdotcode(xdotcode)
        widget.zoom_to_fit(animate=False)

    filename = os.path.basename(filepath)
    dpg.set_viewport_title(f"Raven XDot Viewer - {filename}")

    _set_status(f"Loaded: {filepath}")


def _set_status(text: str) -> None:
    """Set the status bar text."""
    if _app_state["status_text"] is not None:
        dpg.set_value(_app_state["status_text"], text)


def _on_hover(node_id: Optional[str]) -> None:
    """Handle hover callback."""
    if node_id:
        _set_status(f"Node: {node_id}")
    else:
        _set_status("")


def _on_click(node_id: str, button: int) -> None:
    """Handle click callback."""
    _set_status(f"Clicked: {node_id} (button {button})")


def _do_search(*_args) -> None:
    """Execute search from the search input.

    Accepts and discards DPG callback args ``(sender, app_data, user_data)``
    so this can be used directly as both a button callback and an
    ``on_enter`` callback.
    """
    widget = _app_state["widget"]
    search_input = _app_state["search_input"]
    search_input_text_color = _app_state["search_input_text_color"]

    if widget is None or search_input is None:
        return

    query = dpg.get_value(search_input)
    results = widget.search(query)

    if not query:
        _set_status("")
        dpg.set_value(search_input_text_color, (255, 255, 255))  # no search active
    else:
        if len(results):
            _set_status(f"Found {len(results)} matches")
            widget.next_match()  # jump the view to the first match
            dpg.set_value(search_input_text_color, (180, 255, 180))  # found, green
        else:
            _set_status("No matches found")
            dpg.set_value(search_input_text_color, (255, 128, 128))  # not found, red


def _next_match(*_args) -> None:
    """Navigate to next search match."""
    widget = _app_state["widget"]
    if widget is not None:
        node_id = widget.next_match()
        if node_id:
            count = widget.get_search_count()
            _set_status(f"Match: {node_id} ({count} total)")


def _prev_match(*_args) -> None:
    """Navigate to previous search match."""
    widget = _app_state["widget"]
    if widget is not None:
        node_id = widget.prev_match()
        if node_id:
            count = widget.get_search_count()
            _set_status(f"Match: {node_id} ({count} total)")


def _zoom_to_fit(*_args) -> None:
    """Zoom to fit the entire graph."""
    widget = _app_state["widget"]
    if widget is not None:
        widget.zoom_to_fit()


def _zoom_in(*_args) -> None:
    """Zoom in."""
    widget = _app_state["widget"]
    if widget is not None:
        widget.zoom_in()


def _zoom_out(*_args) -> None:
    """Zoom out."""
    widget = _app_state["widget"]
    if widget is not None:
        widget.zoom_out()


def _check_file_reload() -> None:
    """Check if the current file has been modified and reload if so."""
    filepath = _app_state["current_file"]
    if filepath is None:
        return

    try:
        current_mtime = os.path.getmtime(filepath)
        if current_mtime > _app_state["file_mtime"]:
            _open_file(filepath)
            _set_status(f"Reloaded: {filepath}")
    except OSError:
        pass


def _on_key(sender, app_data) -> None:
    """Handle keyboard shortcuts.

    Ctrl+F: Focus search field (Esc to unfocus)
    F3 / Shift+F3: Next / previous search match (like old DOS apps; consistency with rest of Raven)
    +/=: Zoom in
    -: Zoom out
    0: Zoom to fit
    Arrow keys: Pan view
    """
    PAN_AMOUNT = 10  # pixels

    key = app_data
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

    widget = _app_state["widget"]

    if dpg.is_item_focused("search_input"):
        if key == dpg.mvKey_Return:  # accept and unfocus
            dpg.focus_item(widget.get_dpg_widget_id())
        elif key == dpg.mvKey_Escape:  # unfocus
            # TODO: revert to last search field content, update search
            dpg.focus_item(widget.get_dpg_widget_id())
    elif ctrl_pressed:
        if key == dpg.mvKey_F:
            if _app_state["search_input"] is not None:
                dpg.focus_item(_app_state["search_input"])
    else:  # BARE KEYS - BE VERY CAREFUL HERE
        if key == dpg.mvKey_F3:
            if shift_pressed:
                _prev_match()
            else:
                _next_match()
        # TODO: Verify +/= handling on non-US keyboard layouts (Nordic etc.).
        # DPG key handling may or may not be layout-aware.
        elif key in (dpg.mvKey_Plus, ord("=")):
            _zoom_in()
        elif key == dpg.mvKey_Minus:
            _zoom_out()
        elif key == ord("0"):
            _zoom_to_fit()
        elif key == dpg.mvKey_Up:
            widget.pan_by(dx=0, dy=-PAN_AMOUNT)
        elif key == dpg.mvKey_Down:
            widget.pan_by(dx=0, dy=+PAN_AMOUNT)
        elif key == dpg.mvKey_Left:
            widget.pan_by(dx=-PAN_AMOUNT, dy=0)
        elif key == dpg.mvKey_Right:
            widget.pan_by(dx=+PAN_AMOUNT, dy=0)


def _gui_shutdown() -> None:
    """Clean up on app exit. Registered via ``dpg.set_exit_callback``."""
    gui_animation.animator.clear()


def main() -> int:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Raven XDot Viewer - View xdot/dot graph files"
    )
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument(
        "file",
        nargs="?",
        help="GraphViz graph file to open (.xdot, .dot, or .gv)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Window width (default: 1200)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1000,
        help="Window height (default: 1000)"
    )
    args = parser.parse_args()

    # --- DPG bootup ---
    # Order matters: create_context -> bootup (fonts/themes) -> create_viewport -> setup. See `guiutils.bootup`.
    dpg.create_context()

    themes_and_fonts = guiutils.bootup(font_size=14)

    dpg.create_viewport(
        title=f"Raven XDot Viewer {__version__}",
        width=args.width,
        height=args.height
    )

    dpg.setup_dearpygui()

    # --- Build GUI ---
    with dpg.window(tag="main_window"):
        # Toolbar
        with dpg.group(horizontal=True):
            dpg.add_button(label=fa.ICON_SQUARE, tag="zoom_to_fit_button", callback=_zoom_to_fit, width=30)
            dpg.bind_item_font("zoom_to_fit_button", themes_and_fonts.icon_font_regular)  # tag
            with dpg.tooltip("zoom_to_fit_button"):  # tag
                dpg.add_text("Zoom to fit [0]")

            dpg.add_button(label=fa.ICON_MAGNIFYING_GLASS_PLUS, tag="zoom_in_button", callback=_zoom_in, width=30)
            dpg.bind_item_font("zoom_in_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("zoom_in_button"):  # tag
                dpg.add_text("Zoom in [+]")

            dpg.add_button(label=fa.ICON_MAGNIFYING_GLASS_MINUS, tag="zoom_out_button", callback=_zoom_out, width=30)
            dpg.bind_item_font("zoom_out_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("zoom_out_button"):  # tag
                dpg.add_text("Zoom out [-]")

            dpg.add_separator()
            dpg.add_text("Search:")
            _app_state["search_input"] = dpg.add_input_text(
                default_value="",
                tag="search_input",
                hint="[Ctrl+F] [incremental fragment search; 'cat photo' matches 'photocatalytic'; lowercase = case-insensitive]",
                callback=_do_search,  # DPG passes (sender, app_data, user_data); _do_search accepts and discards via *_args.
                width=200,
            )
            with dpg.theme(tag="search_input_theme"):
                with dpg.theme_component(dpg.mvInputText):
                    _app_state["search_input_text_color"] = dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (140, 140, 140))
            dpg.bind_item_theme("search_input", "search_input_theme")  # tag

            dpg.add_button(label=fa.ICON_CIRCLE_UP, tag="prev_match_button", callback=_prev_match, width=30)
            dpg.bind_item_font("prev_match_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("prev_match_button"):  # tag
                dpg.add_text("Previous search match [Shift+F3]")

            dpg.add_button(label=fa.ICON_CIRCLE_DOWN, tag="next_match_button", callback=_next_match, width=30)
            dpg.bind_item_font("next_match_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("next_match_button"):  # tag
                dpg.add_text("Next search match [F3]")

        # Graph view
        # TODO: These sizes don't track viewport resizes. Wire up a resize callback,
        # or find a DPG fill-parent mechanism.
        _app_state["widget"] = XDotWidget(
            parent="main_window",
            width=args.width - _WIDGET_H_PADDING,
            height=args.height - _WIDGET_V_PADDING,
            on_hover=_on_hover,
            on_click=_on_click
        )

        # Status bar
        _app_state["status_text"] = dpg.add_text("Ready")

    # Keyboard handler (global — applies to entire viewport)
    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=_on_key)

    # --- Start app ---
    dpg.set_primary_window("main_window", True)
    dpg.set_exit_callback(_gui_shutdown)
    dpg.show_viewport()

    # Load initial file if provided
    if args.file:
        _open_file(args.file)

    # --- Render loop ---
    # TODO: Add Ctrl+O / file dialog for opening files interactively. See the Avatar pose editor for the cleanest example of how Raven does it (customized file dialog that hooks into Raven's GUI animation system).
    last_check = time.time()
    def _poll_reload():
        nonlocal last_check
        now = time.time()
        if now - last_check > 2.0:
            _check_file_reload()
            last_check = now
    try:
        while dpg.is_dearpygui_running():
            _poll_reload()
            gui_animation.animator.render_frame()
            dpg.render_dearpygui_frame()
    except KeyboardInterrupt:
        pass

    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    sys.exit(main())
