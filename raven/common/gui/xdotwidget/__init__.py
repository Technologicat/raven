"""XDot graph viewer widget for DearPyGUI.

This module provides an interactive graph visualization widget that renders
graphs in xdot format with support for:
- Pan/zoom with smooth animations
- Find/search with fragment matching
- Hover highlights with fade-out
- Programmatic highlighting
- Text compaction for zoomed-out views

Based on xdottir (https://github.com/Technologicat/xdottir), adapted for DearPyGUI.

**This package is licensed under the GNU LGPL, not under Raven's usual 2-clause BSD.** It is a derivative
work of xdottir, which is itself a fork of Jose Fonseca's `xdot.py`; the parser especially carries its lexer
across substantially. See `LICENSE`, and `LICENSE_colorbrewer_color_schemes` for the ColorBrewer palettes.

    Copyright 2008 Jose Fonseca
    Copyright 2012-2019 Juha Jeronen
    Copyright 2026 Juha Jeronen and Jamk University of Applied Sciences (the DearPyGUI adaptation)

    and the xdottir contributors named in its AUTHORS file: Marius Gedminas (animated jumping between
    nodes, original highlight code, additional xdot language features), Jaap Karssenberg (Unicode input
    and returncode fixes), michael.hliao (ColorBrewer color scheme support), Robert Meerman (auto-reload
    for changed file), lodatom (motion-notify fix).

    Licensed under the GNU Lesser General Public License, either version 3 of the License, or (at your
    option) any later version. `xdot.py`'s own header grants the "or later", so this is LGPL-3.0-or-later
    rather than -only.

LGPL rather than GPL is what makes this usable from the rest of Raven: BSD-licensed modules may import and
use this package as a library, which is the whole point of the "Lesser". What stays under the LGPL is this
package and any modification to it.

**Do not confuse this with the AGPL rule elsewhere in Raven.** `raven.avatar.pose_editor` carries the strict
constraint that no BSD-licensed module may import from it. The opposite holds here: importing is expressly
permitted, and the obligations that come with it are ones this tree already meets — keep the copyright
notices above, ship `LICENSE` alongside, and leave the source editable in place, which satisfies the
requirement that a user be able to replace the library. That last clause is a real difficulty for statically
linked C and costs nothing for Python shipped as source.

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


def __getattr__(name):
    """Lazy import for XDotWidget — avoids pulling in DPG at package import time."""
    if name == "XDotWidget":
        from .widget import XDotWidget
        return XDotWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
