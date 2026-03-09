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
    import math
    import os
    import subprocess
    import sys
    import time
    import pathlib
    import webbrowser
    from typing import Optional, Union

    import dearpygui.dearpygui as dpg

    from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders

    from unpythonic.env import env

    from ..common import utils as common_utils
    from ..common.gui.xdotwidget import XDotWidget
    from ..common.gui.xdotwidget.parser import (DotScanner, ID, STR_ID, HTML_ID,
                                                 LSQUARE, RSQUARE, EQUAL, COMMA, SEMI, EOF, SKIP)
    from ..common.gui import utils as guiutils
    from ..common.gui import helpcard
    from ..common.gui import animation as gui_animation
    from ..vendor import DearPyGui_Markdown as dpg_markdown
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
    "dot_source": None,           # layout-stripped DOT source (str)
    "original_xdotcode": None,    # original file contents for .xdot fast path (str or None)
    "current_filter": "[as-is]",  # active layout engine selection
}

_filedialog_open = None  # initialized after DPG setup
_help_window = None  # initialized after DPG setup


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


# Layout attributes to strip from xdot code before re-layout.
# These are all attributes that GraphViz layout engines produce;
# stripping them gives clean DOT that any engine can process from scratch.
_XDOT_LAYOUT_ATTRS = frozenset({
    "bb", "pos", "lp", "rects", "_background",
    "_draw_", "_ldraw_", "_hdraw_", "_tdraw_", "_hldraw_", "_tldraw_",
})

_dot_scanner = DotScanner()


def _strip_xdot_layout_attrs(xdotcode: str) -> str:
    """Remove layout-related attributes from xdot/dot source.

    Uses the existing DotScanner to correctly handle quoted strings,
    HTML labels, and comments. Works from the original buffer (not
    reconstructed token text) to preserve quoting and formatting.
    """
    buf = xdotcode
    pos = 0
    bracket_depth = 0

    # Byte ranges to delete: list of (start, end) tuples.
    deletions = []

    # State machine for tracking `name = value` inside attribute lists.
    # States: None (idle), "saw_name" (name token matched a layout attr),
    #         "saw_equal" (saw `=` after a matching name).
    attr_state = None
    attr_start = 0  # start position of the name token

    while True:
        tok_type, tok_text, end_pos = _dot_scanner.next(buf, pos)
        tok_start = end_pos - len(tok_text)

        if tok_type == EOF:
            break

        if tok_type == SKIP:
            pos = end_pos
            continue

        if tok_type == LSQUARE:
            bracket_depth += 1
            attr_state = None
            pos = end_pos
            continue

        if tok_type == RSQUARE:
            bracket_depth = max(0, bracket_depth - 1)
            attr_state = None
            pos = end_pos
            continue

        if bracket_depth > 0:
            # Inside an attribute list — run the name=value state machine.
            if attr_state is None:
                # Expecting an attribute name.
                if tok_type in (ID, STR_ID, HTML_ID):
                    # Strip quotes/brackets for name comparison (scanner
                    # returns raw text, unlike the lexer's _filter).
                    name = tok_text
                    if tok_type == STR_ID and len(name) >= 2:
                        name = name[1:-1]
                    elif tok_type == HTML_ID and len(name) >= 2:
                        name = name[1:-1]
                    if name.lower() in _XDOT_LAYOUT_ATTRS:
                        attr_state = "saw_name"
                        attr_start = tok_start
                    # else: not a layout attr, ignore
            elif attr_state == "saw_name":
                if tok_type == EQUAL:
                    attr_state = "saw_equal"
                else:
                    # Not a `name = value` pair (e.g. bare attribute flag).
                    attr_state = None
            elif attr_state == "saw_equal":
                # This token is the value — mark the whole name=value for deletion.
                delete_end = end_pos
                # Also consume any trailing separator (comma, semicolon, whitespace).
                probe_pos = delete_end
                while probe_pos < len(buf) and buf[probe_pos] in " \t\r\n":
                    probe_pos += 1
                if probe_pos < len(buf) and buf[probe_pos] in ",;":
                    probe_pos += 1
                    # Eat whitespace after the separator too.
                    while probe_pos < len(buf) and buf[probe_pos] in " \t\r\n":
                        probe_pos += 1
                delete_end = probe_pos
                # Also consume leading whitespace before the attribute name.
                while attr_start > 0 and buf[attr_start - 1] in " \t":
                    attr_start -= 1
                deletions.append((attr_start, delete_end))
                attr_state = None
        else:
            attr_state = None

        pos = end_pos

    if not deletions:
        return buf

    # Build output by copying everything except deletion ranges.
    parts = []
    prev = 0
    for start, end in deletions:
        if start > prev:
            parts.append(buf[prev:start])
        prev = end
    if prev < len(buf):
        parts.append(buf[prev:])
    return "".join(parts)


def _run_graphviz(dot_source: str, engine: str = "dot") -> Optional[str]:
    """Run a GraphViz layout engine on DOT source, return xdot code."""
    try:
        result = subprocess.run(
            [engine, "-Txdot"],
            input=dot_source,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except FileNotFoundError:
        logger.error(f"_run_graphviz: `{engine}` command not found. Please install GraphViz.")  # TODO: additionally use `raven.common.gui.messagebox` to spawn a modal GUI error box that background-threads automatically
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"_run_graphviz: Error running `{engine}`: {e.stderr}")  # TODO: additionally use `raven.common.gui.messagebox` to spawn a modal GUI error box that background-threads automatically
        return None


def _load_file(filepath: Union[pathlib.Path, str]) -> Optional[str]:
    """Load a graph file, store stripped DOT source, and return xdot code.

    For .xdot files with [as-is] filter, returns the original contents directly.
    Otherwise runs the selected (or fallback) GraphViz engine.
    """
    filepath = common_utils.absolutize_filename(filepath)

    if not os.path.exists(filepath):
        logger.error(f"_load_file: File not found: '{filepath}'")  # TODO: additionally use `raven.common.gui.messagebox` to spawn a modal GUI error box that background-threads automatically
        return None

    ext = os.path.splitext(filepath)[1].lower()
    current_filter = _app_state["current_filter"]

    if ext == ".xdot":
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        _app_state["original_xdotcode"] = raw
        _app_state["dot_source"] = _strip_xdot_layout_attrs(raw)
        if current_filter == "[as-is]":
            return raw  # fast path
        else:
            return _run_graphviz(_app_state["dot_source"], current_filter)

    elif ext in (".dot", ".gv"):
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        _app_state["original_xdotcode"] = None
        _app_state["dot_source"] = _strip_xdot_layout_attrs(raw)
        engine = current_filter if current_filter != "[as-is]" else config.GRAPHVIZ_ENGINES[1]
        return _run_graphviz(_app_state["dot_source"], engine)

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

    _set_status(_format_load_status(filepath))


def _format_load_status(filepath: str) -> str:
    """Format the status bar message after loading or re-rendering."""
    current_filter = _app_state["current_filter"]
    if current_filter == "[as-is]" and _app_state["original_xdotcode"] is not None:
        return f"Original layout: {filepath}"
    else:
        engine = current_filter if current_filter != "[as-is]" else config.GRAPHVIZ_ENGINES[1]
        return f"Rendered with {engine}: {filepath}"


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
    results = widget.search(query)  # noqa: F841: kept for documentation purposes; you could access the result set from here
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


def toggle_fullscreen(*_args) -> None:
    """Toggle fullscreen and resize the graph widget to match."""
    dpg.toggle_viewport_fullscreen()
    resize_gui()


def resize_gui(*_args) -> None:
    """Wait for viewport size to settle, then resize the graph widget."""
    if guiutils.wait_for_resize("main_window"):
        _resize_gui()


def _resize_gui(*_args) -> None:
    """Resize the graph widget to fill the main window."""
    w, h = guiutils.get_widget_size("main_window")
    widget = _app_state["widget"]
    if widget is not None:
        widget.set_size(w - config.WIDGET_H_PADDING,
                        h - config.WIDGET_V_PADDING)


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
        dpg.set_value("dark_mode_tooltip_text", "Switch to light mode [F12]")
    else:
        dpg.set_item_label("dark_mode_button", fa.ICON_MOON)
        dpg.set_value("dark_mode_tooltip_text", "Switch to dark mode [F12]")


def _get_focus_anchor() -> tuple[Optional[str], Optional[float]]:
    """Determine whether the viewport is centered on a specific node.

    Returns ``(node_name, zoom)`` if anchored, or ``(None, None)`` if not.
    The anchor condition requires one clearly nearest node within 25% of
    the viewport half-extent.
    """
    widget = _app_state["widget"]
    if widget is None:
        return None, None
    graph = widget.get_graph()
    if graph is None or not graph.nodes:
        return None, None

    cx, cy = widget.get_view_center()
    saved_zoom = widget.get_zoom()
    vx1, vy1, vx2, vy2 = widget.get_visible_bounds()
    threshold = 0.25 * min(vx2 - vx1, vy2 - vy1) / 2

    # Find two nearest nodes by Euclidean distance from viewport center.
    distances = []
    for node in graph.nodes:
        d = math.hypot(node.x - cx, node.y - cy)
        distances.append((d, node))
    distances.sort(key=lambda pair: pair[0])

    d1, nearest = distances[0]
    d2 = distances[1][0] if len(distances) > 1 else float("inf")

    if d1 <= threshold and (d2 > 1.5 * d1 or len(distances) == 1):
        return nearest.internal_name, saved_zoom
    return None, None


def _restore_focus(anchor_name: Optional[str], saved_zoom: Optional[float]) -> None:
    """Restore focus after re-layout. Falls back to zoom-to-fit."""
    widget = _app_state["widget"]
    if widget is None:
        return

    graph = widget.get_graph()
    if anchor_name is not None and graph is not None:
        node = graph.nodes_by_name.get(anchor_name)
        if node is not None:
            widget.pan_to_node(anchor_name, animate=False)
            widget.set_zoom(saved_zoom, animate=False)
            widget.request_render()
            return

    widget.zoom_to_fit(animate=False)


def _apply_filter(engine: str) -> None:
    """Re-render the current graph with a different layout engine."""
    dot_source = _app_state["dot_source"]
    if dot_source is None:
        return

    anchor_name, saved_zoom = _get_focus_anchor()

    # Determine xdot code to use.
    if engine == "[as-is]" and _app_state["original_xdotcode"] is not None:
        xdotcode = _app_state["original_xdotcode"]
    elif engine == "[as-is]":
        xdotcode = _run_graphviz(dot_source, config.GRAPHVIZ_ENGINES[1])
    else:
        xdotcode = _run_graphviz(dot_source, engine)

    if xdotcode is None:
        return

    widget = _app_state["widget"]
    if widget is not None:
        widget.set_xdotcode(xdotcode)

    _restore_focus(anchor_name, saved_zoom)

    _app_state["current_filter"] = engine

    filepath = _app_state["current_file"]
    if filepath:
        _set_status(_format_load_status(filepath))


def _on_filter_changed(sender, app_data, user_data=None):
    """Callback for the layout engine combo."""
    _apply_filter(app_data)


# Combobox keyboard navigation map: {widget_tag: (choices_list, callback)}
_combobox_choice_map = {"filter_combo": (config.GRAPHVIZ_ENGINES, _on_filter_changed)}


def _browse_combo(combo_tag, key):
    """Handle Up/Down/Home/End/Esc for a focused combo widget.

    Returns True if the key was consumed.
    """
    if combo_tag not in _combobox_choice_map:
        return False

    choices, callback = _combobox_choice_map[combo_tag]

    if key == dpg.mvKey_Escape:
        widget = _app_state["widget"]
        if widget is not None:
            dpg.focus_item(widget.get_dpg_widget_id())
        return True

    current = dpg.get_value(combo_tag)
    try:
        index = choices.index(current)
    except ValueError:
        index = 0

    if key == dpg.mvKey_Down:
        new_index = min(index + 1, len(choices) - 1)
    elif key == dpg.mvKey_Up:
        new_index = max(index - 1, 0)
    elif key == dpg.mvKey_Home:
        new_index = 0
    elif key == dpg.mvKey_End:
        new_index = len(choices) - 1
    else:
        return False

    dpg.set_value(combo_tag, choices[new_index])
    if callback is not None:
        callback(combo_tag, choices[new_index])
    return True


def _check_file_reload() -> None:
    """Check if the current file has been modified and reload if so."""
    filepath = _app_state["current_file"]
    if filepath is None:
        return

    try:
        current_mtime = os.path.getmtime(filepath)
        if current_mtime > _app_state["file_mtime"]:
            _open_file(filepath)
            _set_status(f"Reloaded: {_format_load_status(filepath)}")
    except OSError:
        pass


def _on_key(sender, app_data) -> None:
    """Handle keyboard shortcuts."""
    key = app_data

    widget = _app_state["widget"]
    if widget is not None and not widget.input_enabled:
        return

    # Help card handles its own hotkeys (Escape to close)
    if _help_window is not None and _help_window.is_visible():
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
        elif key == dpg.mvKey_E:
            dpg.focus_item("filter_combo")
    else:  # BARE KEYS - BE VERY CAREFUL HERE
        # Combo keyboard navigation (Up/Down/Home/End/Esc while combo focused)
        focused_alias = dpg.get_item_alias(dpg.get_focused_item())
        if focused_alias in _combobox_choice_map:
            if _browse_combo(focused_alias, key):
                return
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
        elif key == dpg.mvKey_F1:
            if _help_window is not None:
                _help_window.show()
        elif key == dpg.mvKey_F11:
            toggle_fullscreen()
        elif key == dpg.mvKey_F12:
            _toggle_dark_mode()


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
                _dark_mode_initial_tip = "Switch to light mode [F12]" if config.DARK_MODE else "Switch to dark mode [F12]"
                dpg.add_text(_dark_mode_initial_tip, tag="dark_mode_tooltip_text")

            dpg.add_combo(
                items=config.GRAPHVIZ_ENGINES,
                default_value=config.GRAPHVIZ_ENGINES[0],
                tag="filter_combo",  # tag
                callback=_on_filter_changed,
                width=80,
            )
            with dpg.tooltip("filter_combo"):  # tag
                dpg.add_text("GraphViz layout engine\n(Ctrl+E; then Up, Down, Home, End to jump; Esc to return)")

            dpg.add_button(label=fa.ICON_EXPAND, tag="fullscreen_button", callback=toggle_fullscreen, width=30)
            dpg.bind_item_font("fullscreen_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("fullscreen_button"):  # tag
                dpg.add_text("Toggle fullscreen [F11]")

            dpg.add_button(label=fa.ICON_CIRCLE_QUESTION, tag="help_button", width=30)
            dpg.bind_item_font("help_button", themes_and_fonts.icon_font_regular)  # tag
            with dpg.tooltip("help_button"):  # tag
                dpg.add_text("Help [F1]")

            dpg.add_button(label=fa.ICON_CIRCLE_UP, tag="prev_match_button", callback=_prev_match, width=30)
            dpg.bind_item_font("prev_match_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("prev_match_button"):  # tag
                dpg.add_text("Previous search match [Shift+F3]")

            dpg.add_button(label=fa.ICON_CIRCLE_DOWN, tag="next_match_button", callback=_next_match, width=30)
            dpg.bind_item_font("next_match_button", themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("next_match_button"):  # tag
                dpg.add_text("Next search match [F3]")

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

    # --- Help card ---
    hotkey_info = (
        # Column 1: search & file
        env(key_indent=0, key="Ctrl+O", action_indent=0, action="Open file", notes=""),
        env(key_indent=0, key="Ctrl+F", action_indent=0, action="Focus search field", notes=""),
        env(key_indent=1, key="Enter", action_indent=0, action="Accept and jump to first match", notes="When focused"),
        env(key_indent=1, key="Esc", action_indent=0, action="Unfocus and revert", notes="When focused"),
        env(key_indent=0, key="F3", action_indent=0, action="Next search match", notes=""),
        env(key_indent=0, key="Shift+F3", action_indent=0, action="Previous search match", notes=""),
        env(key_indent=0, key="Ctrl+E", action_indent=0, action="Focus layout engine selector", notes=""),
        env(key_indent=1, key="Up / Down", action_indent=0, action="Previous / next engine", notes="While focused"),
        env(key_indent=1, key="Home / End", action_indent=0, action="First / last engine", notes="While focused"),
        env(key_indent=1, key="Esc", action_indent=0, action="Focus graph view", notes="While engine selector focused"),

        helpcard.hotkey_new_column,

        # Column 2: navigation & app
        env(key_indent=0, key="+  / Numpad +", action_indent=0, action="Zoom in", notes=""),
        env(key_indent=0, key="-  / Numpad -", action_indent=0, action="Zoom out", notes=""),
        env(key_indent=0, key="0  / Numpad 0", action_indent=0, action="Zoom to fit", notes=""),
        env(key_indent=0, key="Arrow keys", action_indent=0, action="Pan view", notes=""),
        env(key_indent=0, key="Mouse wheel", action_indent=0, action="Zoom at cursor", notes=""),
        env(key_indent=0, key="Mouse drag", action_indent=0, action="Pan view", notes=""),
        helpcard.hotkey_blank_entry,
        env(key_indent=0, key="F1", action_indent=0, action="Open this Help card", notes=""),
        env(key_indent=0, key="F11", action_indent=0, action="Toggle fullscreen", notes=""),
        env(key_indent=0, key="F12", action_indent=0, action="Toggle dark mode", notes=""),
    )

    def render_help_extras(self: helpcard.HelpWindow,
                           gui_parent) -> None:
        """Render app-specific extra information into the help card."""
        dpg_markdown.add_text(f"{self.c_hed}**Interaction modes**{self.c_end}", parent=gui_parent)
        g = dpg.add_group(parent=gui_parent)
        dpg_markdown.add_text(f"{self.c_txt}**Click** a node or edge to focus the view on it. Clicking an edge cycles: zoom-to-fit -> source -> destination -> zoom-to-fit.{self.c_end}",
                              parent=g)
        dpg_markdown.add_text(f"{self.c_txt}**Right-click** a node to open its URL (if it has one) in the browser.{self.c_end}",
                              parent=g)
        dpg_markdown.add_text(f"{self.c_txt}**Shift+hover** (**Ctrl+hover**) over a node to highlight its outgoing (incoming) connections, respectively.{self.c_end}",
                              parent=g)
        dpg_markdown.add_text(f"{self.c_txt}**Hover near an edge endpoint** to reveal a follow indicator; **click** it to jump to the node at the other end.{self.c_end}",
                              parent=g)

        dpg.add_spacer(width=1, height=themes_and_fonts.font_size // 2, parent=gui_parent)
        dpg_markdown.add_text(f"{self.c_hed}**How search works**{self.c_end}", parent=gui_parent)
        g = dpg.add_group(parent=gui_parent)
        dpg_markdown.add_text(f"{self.c_txt}Each space-separated search term is a **fragment**. For a match, **all** fragments must match. Order does not matter. Results live-update as you type.{self.c_end}",
                              parent=g)
        dpg_markdown.add_text(f'- {self.c_txt}A **lowercase** fragment matches {self.c_end}{self.c_hig}case-insensitively{self.c_end}{self.c_txt}. E.g. *"cat photo"* matches *"photocatalytic"*.{self.c_end}',
                              parent=g)
        dpg_markdown.add_text(f'- {self.c_txt}A fragment with **at least one uppercase** letter matches {self.c_end}{self.c_hig}case-sensitively{self.c_end}{self.c_txt}. E.g. *"TiO"* matches titanium oxide, not *"bastion"*.{self.c_end}',
                              parent=g)

        dpg.add_spacer(width=1, height=themes_and_fonts.font_size // 2, parent=gui_parent)
        dpg_markdown.add_text(f"{self.c_hed}**Auto-reload**{self.c_end}", parent=gui_parent)
        g = dpg.add_group(parent=gui_parent)
        dpg_markdown.add_text(f"{self.c_txt}The currently open file is polled for changes and reloaded automatically.{self.c_end}",
                              parent=g)

    def _help_on_show():
        _app_state["widget"].input_enabled = False

    def _help_on_hide():
        _app_state["widget"].input_enabled = True

    global _help_window
    _help_window = helpcard.HelpWindow(hotkey_info=hotkey_info,
                                       width=config.HELP_WINDOW_W,
                                       height=config.HELP_WINDOW_H,
                                       reference_window="main_window",
                                       themes_and_fonts=themes_and_fonts,
                                       on_render_extras=render_help_extras,
                                       on_show=_help_on_show,
                                       on_hide=_help_on_hide)
    dpg.set_item_callback("help_button", _help_window.show)  # tag

    # --- Start app ---
    dpg.set_primary_window("main_window", True)
    dpg.set_viewport_resize_callback(_resize_gui)
    dpg.set_exit_callback(_gui_shutdown)
    dpg.show_viewport()

    # Load initial file if provided
    if args.file:
        _open_file(args.file)

    # The widget was created with the *requested* viewport size, but the window
    # manager may have constrained the window (e.g. on a 1080p screen with a
    # taskbar). Sync the widget to the realized window size once DPG has
    # settled its layout (needs several frames). Re-fit the graph too, since
    # _open_file's zoom_to_fit used the old (pre-correction) dimensions.
    def _initial_resize(*_args):
        _resize_gui()
        widget = _app_state["widget"]
        if widget is not None and _app_state["current_file"] is not None:
            widget.zoom_to_fit(animate=False)
    dpg.set_frame_callback(10, _initial_resize)

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
