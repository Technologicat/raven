"""XDot graph viewer widget for DearPyGUI.

This module provides an interactive graph visualization widget that renders
graphs in xdot format with support for:
- Pan/zoom with smooth animations
- Find/search with fragment matching
- Hover highlights with fade-out
- Programmatic highlighting
- Text compaction for zoomed-out views

Based on xdottir (https://github.com/Technologicat/xdottir), adapted for DearPyGUI.

Example usage::

    import dearpygui.dearpygui as dpg
    from raven.common.gui.xdotwidget import XDotWidget, parse_xdot

    dpg.create_context()
    dpg.create_viewport(title="XDot Viewer", width=800, height=600)
    dpg.setup_dearpygui()

    with dpg.window(label="Graph", tag="main_window"):
        widget = XDotWidget(
            parent="main_window",
            width=780,
            height=560,
            on_hover=lambda node_id: print(f"Hovering: {node_id}"),
            on_click=lambda node_id, button: print(f"Clicked: {node_id}")
        )

        # Load xdot code (from GraphViz output)
        xdotcode = '''digraph { a -> b -> c }'''
        widget.set_xdotcode(xdotcode)
        widget.zoom_to_fit()

    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
"""

__all__ = ["XDotWidget", "parse_xdot"]

from .parser import parse_xdot
from .widget import XDotWidget
