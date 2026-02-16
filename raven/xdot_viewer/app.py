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
    import webbrowser
    from typing import Optional, Union

    import dearpygui.dearpygui as dpg

    from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders

    from ..common import utils as common_utils
    from ..common.gui.xdotwidget import XDotWidget
    from ..common.gui import utils as guiutils
    from ..common.gui import animation as gui_animation
    from ..vendor.file_dialog.fdialog import FileDialog

    from . import config
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")


# Application state
_app_state = {
    "widget": None,
    "current_file": None,
    "search_input": None,
    "status_text": None,
    "file_mtime": None,
}

_filedialog_open = None  # initialized after DPG setup


def _open_file_dialog_callback(selected_files):
    """Callback for the file dialog. `selected_files` is a list of Path objects."""
    _app_state["widget"].input_enabled = True
    if selected_files:
        _open_file(str(selected_files[0]))


def _show_open_dialog(*_args) -> None:
    """Show the file open dialog."""
    if _filedialog_open is not None:
        _app_state["widget"].input_enabled = False
        _filedialog_open.show_file_dialog()


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
    dpg.set_viewport_title(f"Raven XDot Viewer {__version__} - {filename}")

    _set_status(f"Loaded: {filepath}")


def _set_status(text: str) -> None:
    """Set the status bar text."""
    if _app_state["status_text"] is not None:
        dpg.set_value(_app_state["status_text"], text)


def _on_hover(description: Optional[str]) -> None:
    """Handle hover callback.

    `description`: Human-readable element description from the widget, or None.
    """
    if description:
        _set_status(description)
    else:
        _set_status("")


def _on_click(description: str, button: int) -> None:
    """Handle click callback.

    The widget handles zoom-to-element internally; the app just
    updates the status bar.
    """
    _set_status(f"Clicked: {description}")


def _on_open_url(url: str) -> None:
    """Handle double-click on a node with a URL attribute."""
    logger.info(f"Opening URL: {url}")
    _set_status(f"Opening: {url}")
    webbrowser.open(url)


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
    count = widget.get_search_count()  # includes both nodes and edges

    if not query:
        widget.clear_highlights()
        _set_status("")
        dpg.set_value(search_input_text_color, (255, 255, 255))  # no search active
    else:
        if count:
            widget.highlight_search_results()  # highlights both nodes and edges
            _set_status(f"Found {count} matches")
            dpg.set_value(search_input_text_color, (180, 255, 180))  # found, green
        else:
            widget.clear_highlights()
            _set_status("No matches found")
            dpg.set_value(search_input_text_color, (255, 128, 128))  # not found, red


def _next_match(*_args) -> None:
    """Navigate to next search match."""
    widget = _app_state["widget"]
    if widget is not None:
        desc = widget.next_match()
        if desc:
            count = widget.get_search_count()
            _set_status(f"Match: {desc} ({count} total)")


def _prev_match(*_args) -> None:
    """Navigate to previous search match."""
    widget = _app_state["widget"]
    if widget is not None:
        desc = widget.prev_match()
        if desc:
            count = widget.get_search_count()
            _set_status(f"Match: {desc} ({count} total)")


def _zoom_to_fit(*_args) -> None:
    """Zoom to fit the entire graph."""
    widget = _app_state["widget"]
    if widget is not None:
        widget.zoom_to_fit()


def _zoom_in(*_args) -> None:
    """Zoom in."""
    widget = _app_state["widget"]
    if widget is not None:
        widget.zoom_in(factor=config.ZOOM_IN_FACTOR)


def _zoom_out(*_args) -> None:
    """Zoom out."""
    widget = _app_state["widget"]
    if widget is not None:
        widget.zoom_out(factor=config.ZOOM_OUT_FACTOR)


def _toggle_dark_mode(*_args) -> None:
    """Toggle dark mode and update the toolbar button."""
    widget = _app_state["widget"]
    if widget is None:
        return
    widget.dark_mode = not widget.dark_mode
    _update_dark_mode_button()


def _update_dark_mode_button() -> None:
    """Sync the dark mode button icon and tooltip with current state."""
    widget = _app_state["widget"]
    if widget is None:
        return
    if widget.dark_mode:
        dpg.set_item_label("dark_mode_button", fa.ICON_SUN)
        dpg.set_value("dark_mode_tooltip_text", "Switch to light mode")
    else:
        dpg.set_item_label("dark_mode_button", fa.ICON_MOON)
        dpg.set_value("dark_mode_tooltip_text", "Switch to dark mode")


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

    Ctrl+O: Open file dialog
    Ctrl+F: Focus search field (Esc to unfocus)
    F3 / Shift+F3: Next / previous search match (like old DOS apps; consistency with rest of Raven)
    +/=: Zoom in
    -: Zoom out
    0: Zoom to fit
    Arrow keys: Pan view
    """
    key = app_data

    widget = _app_state["widget"]
    if widget is not None and not widget.input_enabled:
        return

    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

    if dpg.is_item_focused("search_input"):
        if key == dpg.mvKey_Return:  # accept and unfocus
            dpg.focus_item(widget.get_dpg_widget_id())
            widget.next_match()  # jump the view to the first match
        elif key == dpg.mvKey_Escape:  # unfocus and cancel current search edit (handled by the text input internally, by sending a change event; but we need to handle the keyboard focus)
            dpg.focus_item(widget.get_dpg_widget_id())
    elif ctrl_pressed and shift_pressed:
        # Some hidden debug features. Mnemonic: "Mr. T Lite" (Ctrl + Shift + M, R, T, L)
        if key == dpg.mvKey_M:
            dpg.show_metrics()
        elif key == dpg.mvKey_R:
            dpg.show_item_registry()
        elif key == dpg.mvKey_T:
            dpg.show_font_manager()
        elif key == dpg.mvKey_L:
            dpg.show_style_editor()
    elif ctrl_pressed:
        if key == dpg.mvKey_O:
            _show_open_dialog()
        elif key == dpg.mvKey_F:
            if _app_state["search_input"] is not None:
                dpg.focus_item(_app_state["search_input"])
    else:  # BARE KEYS - BE VERY CAREFUL HERE
        if key == dpg.mvKey_F3:
            if shift_pressed:
                _prev_match()
            else:
                _next_match()
        # Regular +/- are unreliable on non-US layouts (DPG maps physical
        # keys as if US layout). Numpad +/- always work.
        elif key in (dpg.mvKey_Plus, dpg.mvKey_Add):
            _zoom_in()
        elif key in (dpg.mvKey_Minus, dpg.mvKey_Subtract):
            _zoom_out()
        elif key in (dpg.mvKey_0, dpg.mvKey_NumPad0):
            _zoom_to_fit()
        elif key == dpg.mvKey_Up:
            widget.pan_by(dx=0, dy=+config.PAN_AMOUNT)
        elif key == dpg.mvKey_Down:
            widget.pan_by(dx=0, dy=-config.PAN_AMOUNT)
        elif key == dpg.mvKey_Left:
            widget.pan_by(dx=+config.PAN_AMOUNT, dy=0)
        elif key == dpg.mvKey_Right:
            widget.pan_by(dx=-config.PAN_AMOUNT, dy=0)


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
        default=config.DEFAULT_WIDTH,
        help=f"Window width (default: {config.DEFAULT_WIDTH})"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.DEFAULT_HEIGHT,
        help=f"Window height (default: {config.DEFAULT_HEIGHT})"
    )
    args = parser.parse_args()

    # --- DPG bootup ---
    # Order matters: create_context -> bootup (fonts/themes) -> create_viewport -> setup. See `guiutils.bootup`.
    dpg.create_context()

    themes_and_fonts = guiutils.bootup(font_size=config.FONT_SIZE)

    dpg.create_viewport(
        title=f"Raven XDot Viewer {__version__}",
        width=args.width,
        height=args.height
    )

    dpg.setup_dearpygui()

    # Load extra fonts for graph text at various zoom levels.
    # The renderer picks whichever atlas size is closest to the rendered size.
    graph_text_fonts = []
    for size in config.GRAPH_TEXT_FONT_SIZES:
        _key, font_id = guiutils.load_extra_font(
            themes_and_fonts, size, "OpenSans", "Regular")
        graph_text_fonts.append((size, font_id))

    # Initialize file dialog (must be after dpg.setup_dearpygui)
    global _filedialog_open
    cwd = os.getcwd()
    _filedialog_open = FileDialog(title="Open graph file",
                                  tag="open_file_dialog",
                                  callback=_open_file_dialog_callback,
                                  modal=True,
                                  filter_list=[".xdot", ".dot", ".gv"],
                                  file_filter=".xdot",
                                  multi_selection=False,
                                  allow_drag=False,
                                  default_path=cwd)

    # --- Build GUI ---
    with dpg.window(tag="main_window"):
        # Toolbar
        with dpg.group(horizontal=True):
            dpg.add_button(label=fa.ICON_FOLDER_OPEN, tag="open_file_button", callback=_show_open_dialog, width=30)
            dpg.bind_item_font("open_file_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("open_file_button"):  # tag
                dpg.add_text("Open file [Ctrl+O]")

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

            _dark_mode_initial_icon = fa.ICON_SUN if config.DARK_MODE else fa.ICON_MOON
            dpg.add_button(label=_dark_mode_initial_icon, tag="dark_mode_button", callback=_toggle_dark_mode, width=30)
            dpg.bind_item_font("dark_mode_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("dark_mode_button"):  # tag
                _dark_mode_initial_tip = "Switch to light mode" if config.DARK_MODE else "Switch to dark mode"
                dpg.add_text(_dark_mode_initial_tip, tag="dark_mode_tooltip_text")

            dpg.add_button(label=fa.ICON_CIRCLE_UP, tag="prev_match_button", callback=_prev_match, width=30)
            dpg.bind_item_font("prev_match_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("prev_match_button"):  # tag
                dpg.add_text("Previous search match [Shift+F3]")

            dpg.add_button(label=fa.ICON_CIRCLE_DOWN, tag="next_match_button", callback=_next_match, width=30)
            dpg.bind_item_font("next_match_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("next_match_button"):  # tag
                dpg.add_text("Next search match [F3]")

            dpg.add_separator()
            dpg.add_text("Search:")
            _app_state["search_input"] = dpg.add_input_text(
                default_value="",
                tag="search_input",
                hint="[Ctrl+F] [incremental fragment search; 'cat photo' matches 'photocatalytic'; lowercase = case-insensitive]",
                callback=_do_search,  # DPG passes (sender, app_data, user_data); _do_search accepts and discards via *_args.
                width=-1,  # last item, fill
            )
            with dpg.theme(tag="search_input_theme"):
                with dpg.theme_component(dpg.mvInputText):
                    _app_state["search_input_text_color"] = dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (140, 140, 140))
            dpg.bind_item_theme("search_input", "search_input_theme")  # tag

        # Graph view
        # TODO: These sizes don't track viewport resizes. Wire up a resize callback,
        # or find a DPG fill-parent mechanism.
        _app_state["widget"] = XDotWidget(
            parent="main_window",
            width=args.width - config.WIDGET_H_PADDING,
            height=args.height - config.WIDGET_V_PADDING,
            on_hover=_on_hover,
            on_click=_on_click,
            on_open_url=_on_open_url,
            highlight_fade_duration=config.HIGHLIGHT_FADE_DURATION,
            graph_text_fonts=graph_text_fonts,
            mouse_wheel_zoom_factor=config.MOUSE_WHEEL_ZOOM_FACTOR,
            dark_mode=config.DARK_MODE,
            dark_bg_color=config.DARK_MODE_BACKGROUND,
            light_bg_color=config.LIGHT_MODE_BACKGROUND
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
    last_check = time.time()
    def _poll_reload():
        nonlocal last_check
        now = time.time()
        if now - last_check > config.FILE_RELOAD_POLL_INTERVAL:
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
