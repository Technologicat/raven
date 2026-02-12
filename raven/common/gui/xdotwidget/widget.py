"""Main XDotWidget class for DearPyGUI.

This module provides the XDotWidget class, which is an interactive graph
viewer widget that renders xdot format graphs.

The widget registers itself with Raven's GUI animator for smooth animations.
"""

__all__ = ["XDotWidget"]

import threading
from typing import Callable, List, Optional, Set, Union

import dearpygui.dearpygui as dpg

from unpythonic import sym

from .. import animation as gui_animation
from .. import utils as gui_utils

from .graph import Graph, Node
from .highlight import HighlightState
from .hitdetect import hit_test_screen
from .parser import parse_xdot
from .renderer import render_graph
from .search import SearchState
from .viewport import Viewport


class XDotWidget(gui_animation.Animation):
    """Interactive graph viewer widget for DearPyGUI.

    This widget displays xdot format graphs with support for:
    - Pan/zoom with smooth animations
    - Find/search with fragment matching
    - Hover highlights with fade-out
    - Programmatic highlighting
    - Text compaction for zoomed-out views

    Example usage::

        widget = XDotWidget(parent="my_window", width=800, height=600)
        widget.set_xdotcode(xdot_string)
        widget.zoom_to_fit()

    The widget registers itself with Raven's animator for smooth animations.
    """

    def __init__(self,
                 parent: Union[int, str],
                 width: int,
                 height: int,
                 tag: Optional[str] = None,
                 on_hover: Optional[Callable[[Optional[str]], None]] = None,
                 on_click: Optional[Callable[[str, int], None]] = None,
                 text_compaction_callback: Optional[Callable[[str, float], str]] = None):
        """Create an XDotWidget.

        `parent`: DPG parent (child window, group, etc.)
        `width`, `height`: Widget dimensions in pixels.
        `tag`: Optional DPG tag for the widget group.
        `on_hover`: Callback when hovering changes. Receives node ID or None.
        `on_click`: Callback when a node is clicked. Receives (node_id, button).
        `text_compaction_callback`: Callback for text compaction, used while rendering,
                                     when zoomed out so far that the full label text of
                                     a node won't fit inside that node visually.
                                     Receives (text, available_width_px).
                                     Must return compacted text.
        """
        self._width = width
        self._height = height
        self._on_hover = on_hover
        self._on_click = on_click
        self._text_compaction_callback = text_compaction_callback

        self._graph: Optional[Graph] = None
        self._viewport = Viewport(width, height)
        self._highlight = HighlightState()
        self._search = SearchState()

        self._render_lock = threading.RLock()
        self._needs_render = True

        # Mouse state
        self._dragging = False
        self._last_mouse_pos = (0.0, 0.0)
        self._last_hover_id: Optional[str] = None

        # Build DPG structure
        kwargs = {"parent": parent}
        if tag is not None:
            kwargs["tag"] = tag
        self.group = dpg.add_group(**kwargs)
        self.drawlist = dpg.add_drawlist(width=width, height=height, parent=self.group)

        # Register mouse handlers
        with dpg.handler_registry() as self._handler_registry:
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
            dpg.add_mouse_click_handler(callback=self._on_mouse_click)
            dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                       callback=self._on_mouse_drag)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                          callback=self._on_mouse_release)

        # Register to Raven's GUI animator. This handles calling the frame update.
        gui_animation.animator.add(self)

    def set_xdotcode(self, xdotcode: str) -> None:
        """Load a graph from xdot format code.

        `xdotcode`: The xdot format string (output of GraphViz with xdot format).
        """
        with self._render_lock:
            self._graph = parse_xdot(xdotcode)
            self._viewport.set_graph_bounds(self._graph.width, self._graph.height)
            self._search.set_graph(self._graph)
            self._needs_render = True

    def set_graph(self, graph: Graph) -> None:
        """Set a pre-parsed Graph object."""
        with self._render_lock:
            self._graph = graph
            self._viewport.set_graph_bounds(graph.width, graph.height)
            self._search.set_graph(graph)
            self._needs_render = True

    def get_graph(self) -> Optional[Graph]:
        """Return the current Graph, or None."""
        return self._graph

    def get_dpg_widget_id(self):
        """Return the DPG ID of the top-level group of this graph widget.

        Useful e.g. for programmatically focusing the graph view.
        """
        return self.group

    # -------------------------------------------------------------------------
    # Public API: View control

    def zoom_to_fit(self, animate: bool = True) -> None:
        """Adjust pan/zoom to show the entire graph.

        `animate`: If True, animate the transition.
        """
        if self._graph is not None:
            self._viewport.zoom_to_fit(self._graph, animate=animate)
            self._needs_render = True

    def zoom_to_node(self, node_id: str, animate: bool = True) -> None:
        """Center the view on a specific node.

        `node_id`: The internal name of the node.
        `animate`: If True, animate the transition.
        """
        if self._graph is None:
            return

        node = self._graph.get_node_by_name(node_id)
        if node is not None:
            self._viewport.zoom_to_point(node.x, node.y, animate=animate)
            self._needs_render = True

    def zoom_in(self, factor: float = 1.2) -> None:
        """Zoom in by a factor."""
        self._viewport.zoom_by(factor)
        self._needs_render = True

    def zoom_out(self, factor: float = 1.2) -> None:
        """Zoom out by a factor."""
        self._viewport.zoom_by(1.0 / factor)
        self._needs_render = True

    def pan_by(self, dx, dy):
        """Pan the view by (dx, dy) pixels."""
        self._viewport.pan_by(dx, dy)
        self._needs_render = True

    # -------------------------------------------------------------------------
    # Public API: Highlighting

    def set_highlighted_nodes(self, node_ids: Set[str]) -> None:
        """Set programmatic highlighting for a set of nodes.

        `node_ids`: Set of node internal names to highlight.
        """
        self._highlight.set_highlighted_nodes(node_ids)
        self._needs_render = True

    def get_highlighted_nodes(self) -> Set[str]:
        """Return the set of programmatically highlighted node IDs."""
        return self._highlight.get_highlighted_node_ids()

    def clear_highlights(self) -> None:
        """Clear all programmatic highlights."""
        self._highlight.clear_programmatic()
        self._needs_render = True

    # -------------------------------------------------------------------------
    # Public API: Search

    def search(self, query: str) -> List[str]:
        """Search for nodes/edges containing the query text.

        `query`: Search string (space-separated fragments).

        Returns a list of matching node IDs.
        """
        self._search.search(query)
        return self._search.get_result_ids()

    def next_match(self) -> Optional[str]:
        """Navigate to the next search match.

        Returns the node ID of the next match, or None.
        Also centers the view on the match.
        """
        element = self._search.next_match()
        if element is not None:
            if isinstance(element, Node) and element.internal_name:
                self.zoom_to_node(element.internal_name)
                return element.internal_name
        return None

    def prev_match(self) -> Optional[str]:
        """Navigate to the previous search match.

        Returns the node ID of the previous match, or None.
        Also centers the view on the match.
        """
        element = self._search.prev_match()
        if element is not None:
            if isinstance(element, Node) and element.internal_name:
                self.zoom_to_node(element.internal_name)
                return element.internal_name
        return None

    def clear_search(self) -> None:
        """Clear the current search."""
        self._search.clear()

    def get_search_count(self) -> int:
        """Return the number of search results."""
        return self._search.get_result_count()

    # -------------------------------------------------------------------------
    # Size management

    def set_size(self, width: int, height: int) -> None:
        """Update the widget size."""
        with self._render_lock:
            self._width = width
            self._height = height
            self._viewport.set_size(width, height)
            dpg.configure_item(self.drawlist, width=width, height=height)
            self._needs_render = True

    def get_size(self) -> tuple:
        """Return (width, height) in pixels."""
        return self._width, self._height

    # -------------------------------------------------------------------------
    # Animation and rendering

    def render_frame(self, t: int) -> sym:
        """Adapter; hook for Raven's GUI animation system.

        See `raven.common.gui.animation.Animation` for details.
        """
        # We don't need the `t` parameter here. Because we never reset it, it just auto-tracks time (in nanoseconds) since this instance was created.

        # This actually animates only when needed; otherwise, this is a no-op, so we can afford to run this every DPG frame.
        # The return value isn't needed here (we don't need to know if anything was actually animated or not), so we discard it.
        self.update()

        # Persistent updatable; the animation keeps running as long as this object is alive.
        return gui_animation.action_continue

    def update(self) -> bool:
        """Update animations and render if needed.

        Call this once per frame if not using Raven's animator.

        Returns True if still animating (needs more frames).
        """
        animating = False

        # Update viewport animations
        if self._viewport.update():
            animating = True
            self._needs_render = True

        # Update highlight animations
        if self._highlight.update():
            animating = True
            self._needs_render = True

        # Render if needed
        if self._needs_render:
            self._render()
            self._needs_render = False

        return animating

    def _render(self) -> None:
        """Render the graph to the drawlist."""
        with self._render_lock:
            if self._graph is None:
                dpg.delete_item(self.drawlist, children_only=True)
                return

            highlighted = self._highlight.get_all_highlighted(self._graph)

            # Calculate average highlight intensity for animation
            if highlighted:
                intensities = [self._highlight.get_intensity(e, self._graph)
                               for e in highlighted]
                highlight_t = sum(intensities) / len(intensities)
            else:
                highlight_t = 1.0

            render_graph(
                self.drawlist,
                self._graph,
                self._viewport,
                highlighted=highlighted,
                highlight_t=highlight_t,
                text_compaction_cb=self._text_compaction_callback
            )

    def request_render(self) -> None:
        """Request (force) a re-render on the next update."""
        self._needs_render = True

    # -------------------------------------------------------------------------
    # Mouse handling

    def _is_mouse_inside(self) -> bool:
        """Check if the mouse is inside this widget."""
        return gui_utils.is_mouse_inside_widget(self.drawlist)

    def _get_local_mouse_pos(self) -> tuple:
        """Get mouse position relative to this widget."""
        return gui_utils.get_mouse_relative_pos(self.drawlist)

    def _on_mouse_move(self, sender, app_data) -> None:
        """Handle mouse movement, updating highlights, and triggering the custom callback if set."""
        if not self._is_mouse_inside():
            # Mouse left widget - always clear hover on the highlight state
            # (set_hover has its own early-return if already None, so this is cheap).
            # The _last_hover_id guard only controls the user callback.
            self._highlight.set_hover(None)
            if self._last_hover_id is not None:
                self._last_hover_id = None
                if self._on_hover:
                    self._on_hover(None)
            self._needs_render = True
            return
        if self._graph is None:
            return

        # Hit test
        sx, sy = self._get_local_mouse_pos()
        element = hit_test_screen(self._graph, self._viewport, sx, sy)

        # Update hover
        self._highlight.set_hover(element)

        # Notify callback if hover changed
        new_hover_id = None
        if isinstance(element, Node) and element.internal_name:
            new_hover_id = element.internal_name

        if new_hover_id != self._last_hover_id:
            self._last_hover_id = new_hover_id
            self._needs_render = True
            if self._on_hover:
                self._on_hover(new_hover_id)

    def _on_mouse_click(self, sender, app_data) -> None:
        """Handle mouse click, triggering the custom callback if set."""
        if not self._is_mouse_inside():
            return
        if self._graph is None:
            return

        button = app_data  # 0=left, 1=right, 2=middle

        sx, sy = self._get_local_mouse_pos()
        element = hit_test_screen(self._graph, self._viewport, sx, sy)

        if isinstance(element, Node) and element.internal_name:
            if self._on_click:
                self._on_click(element.internal_name, button)

    def _on_mouse_wheel(self, sender, app_data) -> None:
        """Handle mouse wheel for zooming."""
        if not self._is_mouse_inside():
            return

        delta = app_data  # positive = scroll up (zoom in)

        sx, sy = self._get_local_mouse_pos()

        if delta > 0:
            self._viewport.zoom_by(1.1, sx, sy)
        else:
            self._viewport.zoom_by(1.0 / 1.1, sx, sy)

        self._needs_render = True

    def _on_mouse_drag(self, sender, app_data) -> None:
        """Handle mouse drag for panning.

        DPG's drag handler reports *cumulative* delta from the drag start
        point, not per-frame delta. We track the previous cumulative value
        and compute the per-frame increment.
        """
        if not self._is_mouse_inside() and not self._dragging:
            return

        button, dx, dy = app_data

        if not self._dragging:
            self._dragging = True
            self._drag_cumulative = (dx, dy)
            return

        # Per-frame delta from cumulative
        frame_dx = dx - self._drag_cumulative[0]
        frame_dy = dy - self._drag_cumulative[1]
        self._drag_cumulative = (dx, dy)

        self.pan_by(frame_dx, frame_dy)

    def _on_mouse_release(self, sender, app_data) -> None:
        """Handle mouse release."""
        self._dragging = False

    # -------------------------------------------------------------------------
    # Cleanup

    def destroy(self) -> None:
        """Clean up resources."""
        try:
            dpg.delete_item(self._handler_registry)
        except SystemError:
            pass

        try:
            dpg.delete_item(self.group)
        except SystemError:
            pass
