"""Main XDotWidget class for DearPyGUI.

This module provides the XDotWidget class, which is an interactive graph
viewer widget that renders xdot format graphs.

The widget registers itself with Raven's GUI animator for smooth animations.
"""

__all__ = ["XDotWidget"]

import threading
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

import dearpygui.dearpygui as dpg

from unpythonic import sym

from .. import animation as gui_animation
from .. import utils as gui_utils

from .constants import DPGColor, Point
from .graph import Graph, Node, Edge, PolygonShape, get_highlight_colors
from .highlight import HighlightState
from .hitdetect import hit_test_screen
from .parser import parse_xdot
from .renderer import render_graph, color_to_dpg, set_dark_mode
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
                 text_compaction_callback: Optional[Callable[[str, float], str]] = None,
                 highlight_fade_duration: float = 2.0,
                 graph_text_fonts: Optional[Sequence[Tuple[float, Union[int, str]]]] = None,
                 mouse_wheel_zoom_factor: float = 1.1,
                 dark_mode: bool = False,
                 dark_bg_color: DPGColor = (45, 45, 48, 255),
                 light_bg_color: DPGColor = (255, 255, 255, 255)):
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
        `mouse_wheel_zoom_factor`: Zoom factor per mouse wheel notch.
        `dark_mode`: If True, invert graph lightness for dark backgrounds.
        `dark_bg_color`: Background color in dark mode (DPG format, [0,255]).
        `light_bg_color`: Background color in light mode (DPG format, [0,255]).
        """
        self._width = width
        self._height = height
        self._on_hover = on_hover
        self._on_click = on_click
        self._text_compaction_callback = text_compaction_callback
        self._graph_text_fonts = graph_text_fonts
        self._mouse_wheel_zoom_factor = mouse_wheel_zoom_factor
        self._dark_mode = dark_mode
        self._dark_bg_color = dark_bg_color
        self._light_bg_color = light_bg_color

        set_dark_mode(dark_mode)

        self._graph: Optional[Graph] = None
        self._viewport = Viewport(width, height)
        self._highlight = HighlightState(fade_duration=highlight_fade_duration)
        self._search = SearchState()

        self._render_lock = threading.RLock()
        self._needs_render = True

        # Input suppression (e.g. while a modal dialog is open)
        self._input_enabled = True

        # Mouse state
        self._dragging = False
        self._last_mouse_pos = (0.0, 0.0)
        self._last_hover_desc: Optional[str] = None

        # Modifier key state for link highlights (tracked to detect changes per-frame)
        self._last_shift = False
        self._last_ctrl = False

        # Follow-edge indicator: screen coords of the endpoint to highlight, or None
        self._follow_indicator_pos: Optional[Point] = None

        # Edge click cycle: repeated clicks on same edge body cycle
        # through midpoint → src → dst → midpoint → ...
        self._edge_click_edge: Optional[Edge] = None
        self._edge_click_cycle: int = 0  # 0=midpoint, 1=src, 2=dst

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

    def pan_to_node(self, node_id: str, animate: bool = True) -> None:
        """Pan the view to center on a specific node.

        Pan only — does not change the zoom level.

        `node_id`: The internal name of the node.
        `animate`: If True, animate the transition.
        """
        if self._graph is None:
            return

        node = self._graph.get_node_by_name(node_id)
        if node is not None:
            self._viewport.pan_to_point(node.x, node.y, animate=animate)
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

        Returns a list of matching node IDs (for backward compat).
        """
        self._search.search(query)
        return self._search.get_result_ids()

    def highlight_search_results(self) -> None:
        """Highlight all current search results (nodes and edges)."""
        results = self._search.get_results()
        self._highlight.set_highlighted(set(results))
        self._needs_render = True

    def next_match(self) -> Optional[str]:
        """Navigate to the next search match.

        Returns a description of the match (node ID, or "edge: src → dst"),
        or None if no results. Also centers the view on the match.
        """
        element = self._search.next_match()
        return self._pan_to_element(element)

    def prev_match(self) -> Optional[str]:
        """Navigate to the previous search match.

        Returns a description of the match (node ID, or "edge: src → dst"),
        or None if no results. Also centers the view on the match.
        """
        element = self._search.prev_match()
        return self._pan_to_element(element)

    def _pan_to_element(self, element) -> Optional[str]:
        """Pan the view to center on `element` (Node or Edge).

        Pan only — does not change the zoom level.

        Returns a human-readable description of the element, or None.
        """
        if element is None:
            return None
        if isinstance(element, Node):
            if element.internal_name:
                self.pan_to_node(element.internal_name)
        elif isinstance(element, Edge):
            mx = (element.src.x + element.dst.x) / 2
            my = (element.src.y + element.dst.y) / 2
            self._viewport.pan_to_point(mx, my, animate=True)
            self._needs_render = True
        return self._describe_element(element)

    def _navigate_to_element(self, element) -> Optional[str]:
        """Navigate the view to center on `element` (Node or Edge).

        For nodes, pans to center on the node.
        For edges, repeated clicks on the same edge cycle through:
        zoom-to-fit → src node → dst node → zoom-to-fit → ...

        Returns a human-readable description of the element, or None.
        """
        if element is None:
            return None
        if isinstance(element, Node):
            self._edge_click_edge = None  # reset edge cycle
            if element.internal_name:
                self.pan_to_node(element.internal_name)
        elif isinstance(element, Edge):
            # Advance cycle if clicking the same edge again
            if element is self._edge_click_edge:
                self._edge_click_cycle = (self._edge_click_cycle + 1) % 3
            else:
                self._edge_click_edge = element
                self._edge_click_cycle = 0

            if self._edge_click_cycle == 0:
                # Zoom to fit the whole edge (both endpoints visible)
                bbox = element.get_bounding_box()
                if bbox is not None:
                    self._viewport.zoom_to_bbox(*bbox, animate=True)
                else:
                    mx = (element.src.x + element.dst.x) / 2
                    my = (element.src.y + element.dst.y) / 2
                    self._viewport.pan_to_point(mx, my, animate=True)
            elif self._edge_click_cycle == 1:
                # Source node
                self._viewport.pan_to_point(element.src.x, element.src.y, animate=True)
            else:
                # Destination node
                self._viewport.pan_to_point(element.dst.x, element.dst.y, animate=True)
            self._needs_render = True
        return self._describe_element(element)

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

    @property
    def input_enabled(self) -> bool:
        """Whether mouse/keyboard input is processed.

        Set to False to suppress input (e.g. while a modal dialog is open).
        """
        return self._input_enabled

    @input_enabled.setter
    def input_enabled(self, value: bool) -> None:
        self._input_enabled = value

    @property
    def dark_mode(self) -> bool:
        """Whether dark mode (HSL lightness inversion) is active."""
        return self._dark_mode

    @dark_mode.setter
    def dark_mode(self, value: bool) -> None:
        self._dark_mode = value
        set_dark_mode(value)
        self._needs_render = True

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
            # Viewport moved — what's under the cursor changed even though
            # the mouse didn't move.  Re-evaluate hover so the old highlight
            # starts fading instead of staying stuck at full intensity.
            self._refresh_hover()

        # Update highlight animations
        if self._highlight.update():
            animating = True
            self._needs_render = True

        # Re-evaluate link highlights when modifier keys change (without mouse move)
        if self._input_enabled:
            shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
            ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
            if shift != self._last_shift or ctrl != self._last_ctrl:
                self._update_link_highlights()

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

            # Build per-element intensity dict for the renderer.
            highlighted = self._highlight.get_all_highlighted(self._graph)
            highlight_intensities = {
                e: self._highlight.get_intensity(e, self._graph)
                for e in highlighted
            }

            bg_color = self._dark_bg_color if self._dark_mode else self._light_bg_color

            render_graph(
                self.drawlist,
                self._graph,
                self._viewport,
                highlight_intensities=highlight_intensities,
                text_compaction_cb=self._text_compaction_callback,
                graph_text_fonts=self._graph_text_fonts,
                background_color=bg_color
            )

            # Draw follow-edge indicator ring.
            # Recalculate from current mouse position so the indicator
            # stays correct during zoom/pan (screen coords shift).
            if self._is_mouse_inside():
                sx, sy = self._get_local_mouse_pos()
                self._follow_indicator_pos = self._get_follow_indicator_pos(sx, sy)
            else:
                self._follow_indicator_pos = None
            if self._follow_indicator_pos is not None:
                ix, iy = self._follow_indicator_pos
                base_color, _light = get_highlight_colors()
                ring_color = color_to_dpg(base_color)
                r = self._EDGE_ENDPOINT_RADIUS_PX
                dpg.draw_circle((ix, iy), r, color=ring_color,
                                thickness=2, parent=self.drawlist)

    def request_render(self) -> None:
        """Request (force) a re-render on the next update."""
        self._needs_render = True

    # -------------------------------------------------------------------------
    # Element descriptions (for status bar, callbacks)

    @staticmethod
    def _describe_element(element) -> Optional[str]:
        """Return a human-readable description of a graph element, or None.

        Uses the display label text (from TextShapes), not the internal graph ID.
        """
        if element is None:
            return None
        if isinstance(element, Node):
            texts = element.get_texts()
            label = ", ".join(texts) if texts else element.internal_name
            return f"Node: {label}"
        elif isinstance(element, Edge):
            src_texts = element.src.get_texts()
            dst_texts = element.dst.get_texts()
            src_label = ", ".join(src_texts) if src_texts else (element.src.internal_name or "?")
            dst_label = ", ".join(dst_texts) if dst_texts else (element.dst.internal_name or "?")
            edge_texts = element.get_texts()
            label = f"{src_label} -> {dst_label}"
            if edge_texts:
                label += f" ({', '.join(edge_texts)})"
            return f"Edge: {label}"
        return None

    # -------------------------------------------------------------------------
    # Mouse handling

    def _is_mouse_inside(self) -> bool:
        """Check if the mouse is inside this widget."""
        return gui_utils.is_mouse_inside_widget(self.drawlist)

    def _get_local_mouse_pos(self) -> tuple:
        """Get mouse position relative to this widget."""
        return gui_utils.get_mouse_relative_pos(self.drawlist)

    def _update_link_highlights(self) -> None:
        """Update Shift/Ctrl link highlights based on current hover and modifier state."""
        element = self._highlight.get_hover()
        shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
        ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

        if isinstance(element, Node) and shift and not ctrl:
            linked = self._graph.get_linked_elements(element, "outgoing")
            linked.add(element)
            self._highlight.set_link_highlights(linked)
            self._needs_render = True
        elif isinstance(element, Node) and ctrl and not shift:
            linked = self._graph.get_linked_elements(element, "incoming")
            linked.add(element)
            self._highlight.set_link_highlights(linked)
            self._needs_render = True
        elif self._highlight.has_link_highlights():
            self._highlight.clear_link_highlights()
            self._needs_render = True

        self._last_shift = shift
        self._last_ctrl = ctrl

    def _refresh_hover(self) -> None:
        """Re-evaluate hover state from current mouse position.

        Updates the hover highlight, follow-edge indicator, and notifies
        the hover callback. Called by `_on_mouse_move` (mouse moved) and
        by `update` (viewport moved during pan/zoom animation).
        """
        if not self._input_enabled or not self._is_mouse_inside():
            # Mouse left widget or input suppressed — clear hover.
            self._highlight.set_hover(None)
            if self._highlight.has_link_highlights():
                self._highlight.clear_link_highlights()
            if self._last_hover_desc is not None:
                self._last_hover_desc = None
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

        # Update follow-edge indicator: show ring near the endpoint
        # that would be the follow origin (i.e. the end you're near).
        # This is independent of the hit test, so the indicator works
        # even when a node's bounding box overlaps the edge endpoint.
        old_indicator = self._follow_indicator_pos
        self._follow_indicator_pos = self._get_follow_indicator_pos(sx, sy)
        if self._follow_indicator_pos != old_indicator:
            self._needs_render = True

        # Notify callback if hover changed.
        # Build a human-readable description (using labels, not internal names).
        new_hover_desc = self._describe_element(element)

        if new_hover_desc != self._last_hover_desc:
            self._last_hover_desc = new_hover_desc
            self._needs_render = True
            if self._on_hover:
                self._on_hover(new_hover_desc)

    def _on_mouse_move(self, sender, app_data) -> None:
        """Handle mouse movement, updating highlights, and triggering the custom callback if set."""
        self._refresh_hover()
        self._update_link_highlights()

    _EDGE_ENDPOINT_RADIUS_PX = 15  # pixel radius for follow-edge-on-click

    def _on_mouse_click(self, sender, app_data) -> None:
        """Handle mouse click: zoom to element, then trigger callback.

        For edges, clicking near an endpoint follows the edge to the node
        at the other end (xdottir-style navigation). Clicking elsewhere
        on the edge centers on the edge midpoint.
        """
        if not self._input_enabled or not self._is_mouse_inside():
            return
        if self._graph is None:
            return

        button = app_data  # 0=left, 1=right, 2=middle

        sx, sy = self._get_local_mouse_pos()

        # Clear follow indicator on click (we're about to navigate away)
        self._follow_indicator_pos = None

        # Check edge-follow first (independent of hit test, so it works
        # even when a node's bounding box overlaps the edge endpoint).
        follow_target = self._get_edge_follow_target(sx, sy)
        if follow_target is not None:
            self._navigate_to_element(follow_target)
            if self._on_click:
                desc = self._describe_element(follow_target)
                self._on_click(desc, button)
            return

        # Normal hit test
        element = hit_test_screen(self._graph, self._viewport, sx, sy)
        if element is not None:
            self._navigate_to_element(element)
            if self._on_click:
                desc = self._describe_element(element)
                self._on_click(desc, button)

    def _nearest_edge_endpoint(self, sx: float, sy: float) -> Optional[Tuple[Edge, str]]:
        """Find the nearest edge endpoint within follow radius.

        Searches all edges in the graph, independent of the hover hit test.
        This ensures the follow feature works even when a node's bounding
        box overlaps the edge endpoint.

        Returns ``(edge, "src")`` or ``(edge, "dst")`` for the nearest
        endpoint within the follow radius, or None.
        """
        if self._graph is None:
            return None

        r_sq = self._EDGE_ENDPOINT_RADIUS_PX ** 2
        best = None
        best_dist_sq = r_sq  # must be within radius

        for edge in self._graph.edges:
            if len(edge.points) < 2:
                continue
            for which in ("src", "dst"):
                # Use arrowhead centroid as the detection point when available,
                # so the clickable region matches the indicator ring position.
                centroid = self._arrowhead_centroid(edge, which)
                if centroid is not None:
                    pt_sx, pt_sy = self._viewport.graph_to_screen(*centroid)
                else:
                    pt = edge.points[0] if which == "src" else edge.points[-1]
                    pt_sx, pt_sy = self._viewport.graph_to_screen(*pt)

                d = (sx - pt_sx) ** 2 + (sy - pt_sy) ** 2
                if d <= best_dist_sq:
                    best_dist_sq = d
                    best = (edge, which)

        return best

    @staticmethod
    def _arrowhead_centroid(edge: Edge, which: str) -> Optional[Point]:
        """Find the centroid of the arrowhead polygon nearest to an endpoint.

        `which`: one of "src", "dst"

        Returns the centroid in graph coordinates, or None if no filled
        polygon is found in the edge's shapes.
        """
        endpoint = edge.points[0] if which == "src" else edge.points[-1]
        best_centroid = None
        best_dist_sq = float("inf")
        for shape in edge.shapes:
            if isinstance(shape, PolygonShape) and shape.filled and shape.points:
                n = len(shape.points)
                cx = sum(p[0] for p in shape.points) / n
                cy = sum(p[1] for p in shape.points) / n
                # Only consider polygons actually near this endpoint.
                # Use the polygon's own size as threshold (3x its radius),
                # so we don't pick up arrowheads at the other end of the edge.
                poly_radius_sq = max((p[0] - cx) ** 2 + (p[1] - cy) ** 2
                                     for p in shape.points)
                d_sq = (cx - endpoint[0]) ** 2 + (cy - endpoint[1]) ** 2
                if d_sq < best_dist_sq and d_sq <= 9 * poly_radius_sq:
                    best_dist_sq = d_sq
                    best_centroid = (cx, cy)
        return best_centroid

    def _get_follow_indicator_pos(self, sx: float, sy: float) -> Optional[Point]:
        """Return the screen position for the follow-edge indicator ring, or None.

        Centers the ring on the arrowhead centroid if one exists near the
        endpoint, otherwise on the endpoint itself.
        """
        result = self._nearest_edge_endpoint(sx, sy)
        if result is None:
            return None
        edge, which = result
        # Prefer arrowhead centroid over raw endpoint
        centroid = self._arrowhead_centroid(edge, which)
        if centroid is not None:
            return self._viewport.graph_to_screen(*centroid)
        pt = edge.points[0] if which == "src" else edge.points[-1]
        return self._viewport.graph_to_screen(*pt)

    def _get_edge_follow_target(self, sx: float, sy: float) -> Optional[Node]:
        """If the cursor is near an edge endpoint, return the node at the
        *other* end (for follow-edge navigation).

        Returns None if not near any edge endpoint.
        """
        result = self._nearest_edge_endpoint(sx, sy)
        if result is None:
            return None
        edge, which = result
        # Near src end → follow to dst; near dst end → follow to src.
        return edge.dst if which == "src" else edge.src

    def _on_mouse_wheel(self, sender, app_data) -> None:
        """Handle mouse wheel for zooming."""
        if not self._input_enabled or not self._is_mouse_inside():
            return

        delta = app_data  # positive = scroll up (zoom in)

        sx, sy = self._get_local_mouse_pos()

        f = self._mouse_wheel_zoom_factor
        if delta > 0:
            self._viewport.zoom_by(f, sx, sy)
        else:
            self._viewport.zoom_by(1.0 / f, sx, sy)

        self._needs_render = True

    def _on_mouse_drag(self, sender, app_data) -> None:
        """Handle mouse drag for panning.

        DPG's drag handler reports *cumulative* delta from the drag start
        point, not per-frame delta. We track the previous cumulative value
        and compute the per-frame increment.
        """
        if not self._input_enabled:
            return
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
