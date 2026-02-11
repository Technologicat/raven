"""Raven XDot Viewer - Standalone graph viewer application.

This is a standalone viewer for xdot format graphs. It can open:
- .xdot files (xdot format directly)
- .dot files (filtered through GraphViz)

Usage:
    raven-xdot-viewer [file.xdot|file.dot]
    python -m raven.xdot_viewer [file.xdot|file.dot]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg

from ..common.gui.xdotwidget import XDotWidget, parse_xdot
from ..common.gui import utils as gui_utils


# Application state
_app_state = {
    "widget": None,
    "current_file": None,
    "search_input": None,
    "status_text": None,
    "file_mtime": None,
}


def _load_file(filepath: str) -> Optional[str]:
    """Load an xdot or dot file and return xdot code.

    For .dot files, runs GraphViz to convert to xdot format.
    """
    filepath = os.path.expanduser(filepath)

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return None

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".xdot":
        # Direct xdot file
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext in (".dot", ".gv"):
        # Run through GraphViz
        try:
            result = subprocess.run(
                ["dot", "-Txdot", filepath],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except FileNotFoundError:
            print("Error: 'dot' command not found. Install GraphViz.", file=sys.stderr)
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error running dot: {e.stderr}", file=sys.stderr)
            return None

    else:
        print(f"Error: Unsupported file type: {ext}", file=sys.stderr)
        print("Supported: .xdot, .dot, .gv", file=sys.stderr)
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

    # Update window title
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


def _do_search() -> None:
    """Execute search from the search input."""
    widget = _app_state["widget"]
    search_input = _app_state["search_input"]

    if widget is None or search_input is None:
        return

    query = dpg.get_value(search_input)
    results = widget.search(query)

    if results:
        _set_status(f"Found {len(results)} matches")
        widget.next_match()  # Jump to first match
    else:
        _set_status("No matches found")


def _next_match() -> None:
    """Navigate to next search match."""
    widget = _app_state["widget"]
    if widget is not None:
        node_id = widget.next_match()
        if node_id:
            count = widget.get_search_count()
            _set_status(f"Match: {node_id} ({count} total)")


def _prev_match() -> None:
    """Navigate to previous search match."""
    widget = _app_state["widget"]
    if widget is not None:
        node_id = widget.prev_match()
        if node_id:
            count = widget.get_search_count()
            _set_status(f"Match: {node_id} ({count} total)")


def _zoom_to_fit() -> None:
    """Zoom to fit the entire graph."""
    widget = _app_state["widget"]
    if widget is not None:
        widget.zoom_to_fit()


def _zoom_in() -> None:
    """Zoom in."""
    widget = _app_state["widget"]
    if widget is not None:
        widget.zoom_in()


def _zoom_out() -> None:
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
        if current_mtime != _app_state["file_mtime"]:
            _open_file(filepath)
            _set_status(f"Reloaded: {filepath}")
    except OSError:
        pass


def _on_key(sender, app_data) -> None:
    """Handle keyboard shortcuts."""
    key = app_data

    # Ctrl+F: Focus search
    # N: Next match
    # Shift+N: Previous match
    # +/=: Zoom in
    # -: Zoom out
    # 0: Zoom to fit

    if key == dpg.mvKey_F and dpg.is_key_down(dpg.mvKey_Control):
        if _app_state["search_input"] is not None:
            dpg.focus_item(_app_state["search_input"])
    elif key == dpg.mvKey_N:
        if dpg.is_key_down(dpg.mvKey_Shift):
            _prev_match()
        else:
            _next_match()
    elif key in (dpg.mvKey_Plus, ord("=")):
        _zoom_in()
    elif key == dpg.mvKey_Minus:
        _zoom_out()
    elif key == ord("0"):
        _zoom_to_fit()


def main() -> int:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Raven XDot Viewer - View xdot/dot graph files"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Graph file to open (.xdot, .dot, or .gv)"
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
        default=800,
        help="Window height (default: 800)"
    )
    args = parser.parse_args()

    # Initialize DPG
    dpg.create_context()
    dpg.create_viewport(
        title="Raven XDot Viewer",
        width=args.width,
        height=args.height
    )

    # Set up fonts and themes
    try:
        gui_utils.bootup(font_size=14)
    except Exception as e:
        print(f"Warning: Could not load custom fonts: {e}", file=sys.stderr)

    # Create main window
    with dpg.window(tag="main_window"):
        # Toolbar
        with dpg.group(horizontal=True):
            dpg.add_button(label="Fit", callback=_zoom_to_fit)
            dpg.add_button(label="+", callback=_zoom_in, width=30)
            dpg.add_button(label="-", callback=_zoom_out, width=30)
            dpg.add_separator()
            dpg.add_text("Search:")
            _app_state["search_input"] = dpg.add_input_text(
                width=200,
                on_enter=True,
                callback=lambda: _do_search()
            )
            dpg.add_button(label="Find", callback=_do_search)
            dpg.add_button(label="<", callback=_prev_match, width=30)
            dpg.add_button(label=">", callback=_next_match, width=30)

        # Graph view
        _app_state["widget"] = XDotWidget(
            parent="main_window",
            width=args.width - 20,
            height=args.height - 100,
            on_hover=_on_hover,
            on_click=_on_click
        )

        # Status bar
        _app_state["status_text"] = dpg.add_text("Ready")

    # Keyboard handler
    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=_on_key)

    # Set as primary window
    dpg.set_primary_window("main_window", True)

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Load initial file if provided
    if args.file:
        _open_file(args.file)

    # Main loop
    last_check = time.time()
    while dpg.is_dearpygui_running():
        # Update widget animations
        widget = _app_state["widget"]
        if widget is not None:
            widget.update()

        # Check for file changes every 2 seconds
        now = time.time()
        if now - last_check > 2.0:
            _check_file_reload()
            last_check = now

        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    sys.exit(main())
