"""Main XDotWidget class for DearPyGUI.

This module provides the XDotWidget class, which is an interactive graph
viewer widget that renders xdot format graphs.

The widget registers itself with Raven's GUI animator for smooth animations.
"""

__all__ = ["XDotWidget"]

import logging
import threading
import time
import uuid
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

import dearpygui.dearpygui as dpg

from unpythonic import sym

from .. import animation as gui_animation
from .. import utils as guiutils

from .constants import DPGColor, Point
from .graph import Graph, Node, Edge, Element, PolygonShape, get_highlight_colors
from .highlight import HighlightState
from .hitdetect import hit_test_screen
from .parser import parse_xdot
from .renderer import render_graph, color_to_dpg, set_dark_mode
from .search import SearchState
from .viewport import Viewport

# How much room to leave between a zoom's anchor node and the edge of the view, in screen pixels. Enough
# that a node held on screen reads as being on screen rather than as having got stuck against the side.
_ANCHOR_MARGIN = 8


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
                 on_hover: Optional[Callable[[Optional[Element]], None]] = None,
                 on_click: Optional[Callable[[Element, int], None]] = None,
                 on_open_url: Optional[Callable[[str], None]] = None,
                 input_blocked: Optional[Callable[[], bool]] = None,
                 text_compaction_callback: Optional[Callable[[str, float], str]] = None,
                 highlight_fade_duration: float = 1.0,
                 graph_text_fonts: Optional[Sequence[Tuple[float, Union[int, str]]]] = None,
                 mouse_wheel_zoom_factor: float = 1.25,
                 clamp_pan_to_graph: bool = False,
                 dark_mode: bool = False,
                 dark_bg_color: DPGColor = (45, 45, 48, 255),
                 light_bg_color: DPGColor = (255, 255, 255, 255)):
        """Create an XDotWidget.

        `parent`: DPG parent (child window, group, etc.)
        `width`, `height`: Widget dimensions in pixels.
        `tag`: Optional DPG tag for the widget group.
        `on_hover`: Callback when the hovered element changes. Receives the `Node` or `Edge` under the
                    cursor, or `None` when the cursor leaves everything.
        `on_click`: Callback when an element is clicked. Receives `(element, button)`, where `element` is
                    the `Node` or `Edge` that was hit.

                    The element itself rather than a description of it, because a caller that has to *act*
                    on the click needs to know which element it was, and a label cannot be turned back
                    into one -- two nodes may carry the same text, and an `Edge` has no name at all. For a
                    caption, pass it to `describe_element`.
        `input_blocked`: Predicate answering "is something on top of me, so I should ignore input?".
                          Consulted on every mouse event, so keep it cheap.

                          The widget cannot answer this itself, which is why the app supplies it. The mouse
                          handlers are registered globally — DPG's `handler_registry` fires them wherever
                          the cursor happens to be — and the widget's own "is the mouse over me" test is
                          geometric, so a dialog drawn on top is invisible to it. Without this, the click
                          that dismisses an error dialog also lands on the graph behind it. What counts as
                          "on top" is the app's business: a messagebox, a file dialog and a help card have
                          nothing in common that the widget could test for. Raven's apps pass their own
                          `is_any_modal_window_visible`.
        `text_compaction_callback`: Callback for text compaction, used while rendering,
                                     when zoomed out so far that the full label text of
                                     a node won't fit inside that node visually.
                                     Receives (text, available_width_px).
                                     Must return compacted text.
        `mouse_wheel_zoom_factor`: Zoom factor per mouse wheel notch. 1.25 needs three notches to double,
                                   which is about the coarsest that still feels controllable; the earlier
                                   1.1 took seven and read as an unresponsive wheel.
        `clamp_pan_to_graph`: If True, the view cannot be moved to show empty space beyond the graph;
                              an axis on which the graph is smaller than the viewport is centred instead.
                              Off by default: a viewer of arbitrary graphs may want the space, to compare
                              two distant parts or simply to have somewhere to put the pointer. Worth
                              switching on where the graph *is* the content, and space beside it is only
                              distance the reader has to pan back across.
        `on_open_url`: Callback when a node with a URL is right-clicked.
                        Receives the URL string.
        `dark_mode`: If True, invert graph lightness for dark backgrounds.
        `dark_bg_color`: Background color in dark mode (DPG format, [0,255]).
        `light_bg_color`: Background color in light mode (DPG format, [0,255]).
        """
        # The widget is its own animation — it registers itself with the animator below, and the animator
        # reads base-class state on every animation it holds, so this has to run whatever else we do here.
        #
        # Ambient to begin with, because a widget with no graph in it yet is not animating anything.
        # `render_frame` takes the flag over from there, per frame — see the note there.
        super().__init__(ambient=True)
        self.gui_uuid = str(uuid.uuid4())  # used in GUI widget tags
        self._width = width
        self._height = height
        self._on_hover = on_hover
        self._on_click = on_click
        self._on_open_url = on_open_url
        self._input_blocked = input_blocked
        self._text_compaction_callback = text_compaction_callback
        self._graph_text_fonts = graph_text_fonts
        self._mouse_wheel_zoom_factor = mouse_wheel_zoom_factor
        self._dark_mode = dark_mode
        self._dark_bg_color = dark_bg_color
        self._light_bg_color = light_bg_color

        set_dark_mode(dark_mode)

        self._graph: Optional[Graph] = None
        self._viewport = Viewport(width, height)
        self._viewport.clamp_pan = clamp_pan_to_graph
        self._highlight = HighlightState(fade_duration=highlight_fade_duration)
        self._search = SearchState()

        self._render_lock = threading.RLock()
        self._needs_render = True

        # Input suppression (e.g. while a modal dialog is open)
        self._input_enabled = True

        # Mouse state
        self._dragging = False
        self._last_mouse_pos = (0.0, 0.0)
        # The hovered element itself, for change detection. Comparing captions would miss a move between
        # two elements that happen to be labelled the same.
        self._last_hover_element = None

        # Modifier key state for link highlights (tracked to detect changes per-frame)
        self._last_shift = False
        self._last_ctrl = False

        # Follow-edge indicator: screen coords of the endpoint to highlight, or None
        self._follow_indicator_pos: Optional[Point] = None

        # Tooltip window for node annotations (e.g. pyan3 tooltips).
        # Created here (before the render loop) so it gets correct z-order
        # (DPG renders windows in creation order; primary window is background).
        #
        # The tags are per-instance. They were fixed strings until 2026-08-25, which made a second widget
        # in one DPG context die on "Alias already exists" - an app holding one never met it, and the first
        # test to build two did so immediately.
        self._tooltip_window = dpg.add_window(
            tag=f"xdot_tooltip_window_{self.gui_uuid}",
            show=False, modal=False, no_title_bar=True,
            autosize=True, no_move=True, no_resize=True,
            no_scrollbar=True, no_collapse=True,
            no_focus_on_appearing=True,
            min_size=[1, 1])
        # Tight padding so the tooltip doesn't have excess whitespace.
        # Explicit parents, no `with`: a widget is built mid-session, and DPG's container stack is one
        # process-wide global shared by themes and widgets alike. See `dpg-notes.md`, "DPG parent
        # management".
        tooltip_theme = dpg.add_theme()
        tooltip_theme_component = dpg.add_theme_component(dpg.mvAll, parent=tooltip_theme)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 6, 0, category=dpg.mvThemeCat_Core, parent=tooltip_theme_component)
        dpg.bind_item_theme(self._tooltip_window, tooltip_theme)
        self._tooltip_group = dpg.add_group(tag=f"xdot_tooltip_group_{self.gui_uuid}",
                                            parent=self._tooltip_window)
        self._tooltip_node: Optional[Node] = None  # which node the tooltip is currently showing for
        self._tooltip_hover_start: int = 0  # monotonic_ns when hover on current node began
        self._tooltip_visible: bool = False  # whether the tooltip window is currently shown

        # Edge click cycle: repeated clicks on same edge body cycle
        # through midpoint → src → dst → midpoint → ...
        self._edge_click_edge: Optional[Edge] = None
        self._edge_click_cycle: int = 0  # 0=midpoint, 1=src, 2=dst

        # Intentional focus: the node the user last navigated to
        # (click, search match, follow-edge). Cleared by manual pan/zoom.
        # Used by the app layer to decide whether to preserve focus
        # across layout engine switches.
        self._focus_node_name: Optional[str] = None
        # Where a left button went down, until it comes up again. A click is a press and a release with no
        # drag between them, and only the release can know which it was.
        self._pending_click_pos: Optional[Point] = None

        # Build DPG structure
        kwargs = {"parent": parent}
        if tag is not None:
            kwargs["tag"] = tag
        self.group = dpg.add_group(**kwargs)
        self.drawlist = dpg.add_drawlist(width=width, height=height, parent=self.group)

        # Register mouse handlers
        self._handler_registry = dpg.add_handler_registry()
        dpg.add_mouse_move_handler(callback=self._on_mouse_move, parent=self._handler_registry)
        dpg.add_mouse_click_handler(callback=self._on_mouse_click, parent=self._handler_registry)

        dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel, parent=self._handler_registry)
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                   callback=self._on_mouse_drag,
                                   parent=self._handler_registry)
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                      callback=self._on_mouse_release,
                                      parent=self._handler_registry)

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
            self._focus_node_name = None
            self._viewport.zoom_to_fit(self._graph, animate=animate)
            self._needs_render = True

    def zoom_to_bbox(self, x1: float, y1: float, x2: float, y2: float,
                     margin: int = 12, animate: bool = True) -> None:
        """Adjust pan/zoom to fit a rectangle of the graph, in graph coordinates.

        The zoom that fits *both* dimensions is chosen, so a box taller than it is wide -- against a
        widget that is wider than it is tall -- comes out fitted by height with the width free to overflow.
        That is the way to frame one part of a graph too wide to show whole.
        """
        self._viewport.zoom_to_bbox(x1, y1, x2, y2, margin=margin, animate=animate)
        self._needs_render = True

    def pan_to_point(self, gx: float, gy: float, animate: bool = True) -> None:
        """Centre the view on a point in graph coordinates. Pan only; the zoom is left alone.

        `pan_to_node` centres a node; this is for putting something somewhere other than the middle, which
        needs a point the caller has worked out rather than a node.
        """
        self._viewport.pan_to_point(gx, gy, animate=animate)
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
            self._focus_node_name = node_id
            self._viewport.pan_to_point(node.x, node.y, animate=animate)
            self._needs_render = True

    def zoom_in(self, factor: float = 1.2, anchor_node: Optional[str] = None) -> None:
        """Zoom in by a factor.

        `anchor_node`: Internal name of a node to zoom about, which then keeps its place on screen while
                       everything else grows away from it, and stays whole in the view. `None` zooms about
                       the middle of the view, and so does a node that is not in the graph or is not
                       currently on screen — see `_usable_anchor`.
        """
        self._zoom_about_node(anchor_node, lambda sx, sy: self._viewport.zoom_by(factor, sx, sy))

    def zoom_out(self, factor: float = 1.2, anchor_node: Optional[str] = None) -> None:
        """Zoom out by a factor. `anchor_node` is as for `zoom_in`."""
        self._zoom_about_node(anchor_node, lambda sx, sy: self._viewport.zoom_by(1.0 / factor, sx, sy))

    def _zoom_about_node(self, anchor_node: Optional[str],
                         zoom: Callable[[Optional[float], Optional[float]], None],
                         animate: bool = True) -> None:
        """Run a zoom about `anchor_node`, and leave that node whole on screen.

        `anchor_node`: As for `zoom_in`.
        `zoom`: Called with the screen point to turn about, or `(None, None)` for the middle of the view.
        `animate`: Whether `zoom` was asked to animate, so the correction below can match it.

        **Turning about a node's centre keeps a point, and a node is a box.** Zoomed in far enough about a
        centre near the edge of the view, the far half of the box leaves it — and the zoom that lands the
        centre itself outside is the one after which the node stops being a usable anchor at all, so the
        next press silently reverts to the middle of the view. Following the zoom with the smallest pan
        that puts the whole box back inside the view is what keeps the promise the anchor is making.

        Applied in both directions rather than only when zooming in. Zooming out cannot clip a node that
        was whole, so the correction is free there; what it buys is the simpler invariant — an anchored
        node is whole after a zoom, whichever way the zoom went — and the recovery of a node left clipped
        by something else.
        """
        node = self._usable_anchor(anchor_node)
        if node is None:
            zoom(None, None)
        else:
            zoom(*self._viewport.graph_to_screen(node.x, node.y))
            self._viewport.keep_box_visible(*node.get_bounding_box(),
                                            margin=_ANCHOR_MARGIN, animate=animate)
        self._needs_render = True

    def _usable_anchor(self, anchor_node: Optional[str]) -> Optional[Node]:
        """Return the node a zoom should turn about, or `None` to turn about the middle of the view.

        `None` for no node, for a name the graph does not have, and for a node that is currently off
        screen.

        **That last one is the case worth having.** The transform is defined about any point and answers
        an off-screen one uselessly: a node out of sight pulls the view further away from itself with
        every press, and away from the graph with it. So a node anchors the zoom only while it is
        somewhere the reader can see it, and the middle of the view stands in when it is not.
        """
        if anchor_node is None or self._graph is None:
            return None
        node = self._graph.get_node_by_name(anchor_node)
        if node is None:
            return None
        sx, sy = self._viewport.graph_to_screen(node.x, node.y)
        if not (0.0 <= sx <= self._viewport.width and 0.0 <= sy <= self._viewport.height):
            return None
        return node

    def pan_by(self, dx, dy):
        """Pan the view by (dx, dy) pixels."""
        self._focus_node_name = None
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

    def flash_nodes(self, node_ids: Set[str]) -> None:
        """Light the named nodes and let them fade, the way a hover fades when the cursor leaves.

        For saying "here" about a view that has just moved. Nothing has to switch it off again: it is the
        same fade-out hover uses, seeded at full intensity.
        """
        if self._graph is None:
            return
        elements = {node for node in (self._graph.get_node_by_name(name) for name in node_ids)
                    if node is not None}
        self._highlight.flash(elements)
        self._needs_render = True

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
        return self.describe_element(element)

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
                self._focus_node_name = None
                bbox = element.get_bounding_box()
                if bbox is not None:
                    self._viewport.zoom_to_bbox(*bbox, animate=True)
                else:
                    mx = (element.src.x + element.dst.x) / 2
                    my = (element.src.y + element.dst.y) / 2
                    self._viewport.pan_to_point(mx, my, animate=True)
            elif self._edge_click_cycle == 1:
                # Source node
                self._focus_node_name = element.src.internal_name
                self._viewport.pan_to_point(element.src.x, element.src.y, animate=True)
            else:
                # Destination node
                self._focus_node_name = element.dst.internal_name
                self._viewport.pan_to_point(element.dst.x, element.dst.y, animate=True)
            self._needs_render = True
        return self.describe_element(element)

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

    def _input_allowed(self) -> bool:
        """Whether to act on mouse input right now: input is enabled, and nothing is covering the widget.

        The second condition is the app's to answer — see the `input_blocked` constructor argument.
        """
        if not self._input_enabled:
            return False
        return self._input_blocked is None or not self._input_blocked()

    # -------------------------------------------------------------------------
    # Public API: Viewport state

    def get_view_center(self) -> Tuple[float, float]:
        """Return the viewport center in graph coordinates as ``(pan_x, pan_y)``."""
        return self._viewport.pan_x.current, self._viewport.pan_y.current

    def get_zoom(self) -> float:
        """Return the current zoom level."""
        return self._viewport.zoom.current

    def set_zoom(self, zoom: float, animate: bool = True, anchor_node: Optional[str] = None) -> None:
        """Set the zoom level.

        `zoom`: Target zoom level.
        `animate`: If True, animate the transition.
        `anchor_node`: As for `zoom_in` — the node to keep in place while the scale changes around it.

        Obeyed as given, where `zoom_in` and `zoom_out` decline to leave the graph behind. A caller naming
        a scale means that scale; the incremental pair is the one being steered by how it looks.
        """
        self._zoom_about_node(anchor_node,
                              lambda sx, sy: self._viewport.zoom_to(zoom, sx, sy, animate=animate),
                              animate=animate)

    def get_visible_bounds(self) -> Tuple[float, float, float, float]:
        """Return the visible area in graph coordinates as ``(x1, y1, x2, y2)``."""
        return self._viewport.get_visible_bounds()

    def get_focus_node(self) -> Optional[str]:
        """Return the internal name of the intentionally focused node, or None.

        Set by navigation actions (click, search match, follow-edge).
        Cleared by manual pan/zoom.
        """
        return self._focus_node_name

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

    def is_animating(self) -> bool:
        """Return True if any viewport or highlight animation is in progress."""
        return self._viewport.is_animating() or self._highlight.is_animating()

    def render_frame(self, t: int) -> sym:
        """Adapter; hook for Raven's GUI animation system.

        See `raven.common.gui.animation.Animation` for details.
        """
        # We don't need the `t` parameter here. Because we never reset it, it just auto-tracks time (in nanoseconds) since this instance was created.

        # This actually animates only when needed; otherwise, this is a no-op, so we can afford to run this every DPG frame.
        animating = self.update()

        # Ambient while resting, transient while a pan, zoom or highlight is in flight.
        #
        # This registration is permanent -- the widget needs a per-frame hook for as long as it exists --
        # so a fixed `ambient=False` would tell every app with an idle throttle that its GUI is busy
        # forever, and the throttle would never engage while a graph was on screen. `ambient=True` would
        # be the opposite mistake: a pan is exactly when the frames are wanted, and 12 fps is visible in
        # one. What the flag has to track is not the hook but the work, and `update` has just said whether
        # there was any.
        #
        # `Animator.transient_count` reads this attribute on every call rather than at registration, which
        # is what makes a per-frame answer possible at all. It also makes `transient_count` agree with
        # `is_animating`, which is the same question `raven-xdot-viewer` asks the widget by hand.
        self.ambient = not animating

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

        # Re-evaluate hover each frame. Needed because several situations
        # change what's under the cursor without a mouse-move event:
        # viewport animation, Alt+Tab back from another window, and
        # right-click-to-open-URL clearing the hover.
        self._refresh_hover()

        # Update highlight animations
        if self._highlight.update():
            animating = True
            self._needs_render = True

        # Re-evaluate link highlights when modifier keys change (without mouse move)
        if self._input_allowed():
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
    def describe_element(element) -> Optional[str]:
        """Return a human-readable description of a graph element, or `None` if there is none.

        Uses the display label text (from TextShapes), not the internal graph ID. This is what a status
        bar wants; `on_click` and `on_hover` hand over the element itself, and this turns one into a
        caption.
        """
        if element is None:
            return None
        if isinstance(element, Node):
            texts = element.get_texts()
            label = ", ".join(texts) if texts else element.internal_name
            if element.url:
                return f"Node: {label}  —  {element.url} (right-click to open)"
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

    @staticmethod
    def _get_node_tooltip_text(node: Node) -> Optional[str]:
        """Return tooltip text for a node, or None.

        Uses the explicit ``tooltip`` attribute from the dot file
        (e.g. pyan3 always populates this for non-top-level nodes).
        """
        return node.tooltip or None

    _TOOLTIP_OFFSET = 20  # pixels away from cursor, to prevent hover-over-tooltip
    _TOOLTIP_SHOW_DELAY_NS = 300_000_000  # 300ms before tooltip appears

    def _update_tooltip(self, node: Node) -> None:
        """Update tooltip state for `node`. Called each frame while hovering.

        Handles the show delay: the tooltip appears only after the cursor
        has dwelled on the same annotated node for `_TOOLTIP_SHOW_DELAY` seconds.
        Once visible, the tooltip follows the cursor.
        """
        text = self._get_node_tooltip_text(node)
        if text is None:
            self._hide_tooltip()
            return

        now = time.monotonic_ns()

        if self._tooltip_node is not node:
            # Hovering a new node — reset the delay timer, hide if currently showing.
            self._tooltip_node = node
            self._tooltip_hover_start = now
            if self._tooltip_visible:
                self._tooltip_visible = False
                dpg.configure_item(self._tooltip_window, show=False)
            return

        if not self._tooltip_visible:
            # Still waiting for the dwell delay.
            if now - self._tooltip_hover_start < self._TOOLTIP_SHOW_DELAY_NS:
                return
            # Delay elapsed — build and show the tooltip.
            dpg.delete_item(self._tooltip_group, children_only=True)
            dpg.add_text(text, parent=self._tooltip_group)
            dpg.add_spacer(height=2, parent=self._tooltip_group)  # bottom padding (top comes from font ascender)
            self._tooltip_visible = True
            self._position_tooltip()
            dpg.configure_item(self._tooltip_window, show=True)
            logger.debug("_update_tooltip: showing for '%s'", node.internal_name)
        else:
            # Already visible — just reposition to follow the cursor.
            self._position_tooltip()

    def _hide_tooltip(self) -> None:
        """Hide the tooltip window and reset state."""
        if self._tooltip_node is not None:
            self._tooltip_node = None
        if self._tooltip_visible:
            self._tooltip_visible = False
            dpg.configure_item(self._tooltip_window, show=False)

    def _position_tooltip(self) -> None:
        """Position the tooltip window near the mouse cursor."""
        mouse_pos = dpg.get_mouse_pos(local=False)
        tooltip_size = guiutils.get_widget_size(self._tooltip_window)
        vp_w = dpg.get_viewport_client_width()
        vp_h = dpg.get_viewport_client_height()
        offset = self._TOOLTIP_OFFSET

        xpos = guiutils.compute_tooltip_position_scalar(
            algorithm="snap",
            cursor_pos=mouse_pos[0],
            tooltip_size=tooltip_size[0],
            viewport_size=vp_w,
            offset=offset)
        ypos = guiutils.compute_tooltip_position_scalar(
            algorithm="snap",
            cursor_pos=mouse_pos[1],
            tooltip_size=tooltip_size[1],
            viewport_size=vp_h,
            offset=offset)
        dpg.set_item_pos(self._tooltip_window, [xpos, ypos])

    # -------------------------------------------------------------------------
    # Mouse handling

    def _is_mouse_inside(self) -> bool:
        """Check if the mouse is inside this widget."""
        return guiutils.is_mouse_inside_widget(self.drawlist)

    def _get_local_mouse_pos(self) -> tuple:
        """Get mouse position relative to this widget."""
        return guiutils.get_mouse_relative_pos(self.drawlist)

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
        if not self._input_allowed() or not self._is_mouse_inside():
            # Mouse left widget or input suppressed — clear hover state.
            # Only flip `_needs_render` when something visible actually changed;
            # in the common steady-state (cursor outside, nothing to update), an
            # unconditional re-render here costs O(edges) per frame on dense graphs.
            visual_changed = False
            if self._highlight.get_hover() is not None:
                self._highlight.set_hover(None)  # starts fade; `highlight.update()` drives renders during the fade
                visual_changed = True
            self._hide_tooltip()  # tooltip is its own DPG window, not part of the drawlist — no render needed
            if self._highlight.has_link_highlights():
                self._highlight.clear_link_highlights()  # instant clear, no fade — render once to make it take effect
                visual_changed = True
            if self._last_hover_element is not None:
                self._last_hover_element = None
                if self._on_hover:
                    self._on_hover(None)
            if visual_changed:
                self._needs_render = True
            return
        if self._graph is None:
            return

        # Hit test
        sx, sy = self._get_local_mouse_pos()
        element = hit_test_screen(self._graph, self._viewport, sx, sy)

        # Update hover
        self._highlight.set_hover(element)

        # Update annotation tooltip (follows cursor while hovering on annotated nodes)
        if isinstance(element, Node):
            self._update_tooltip(element)
        else:
            self._hide_tooltip()

        # Update follow-edge indicator: show ring near the endpoint
        # that would be the follow origin (i.e. the end you're near).
        # This is independent of the hit test, so the indicator works
        # even when a node's bounding box overlaps the edge endpoint.
        old_indicator = self._follow_indicator_pos
        self._follow_indicator_pos = self._get_follow_indicator_pos(sx, sy)
        if self._follow_indicator_pos != old_indicator:
            self._needs_render = True

        # Notify the callback if the hovered element changed.
        if element is not self._last_hover_element:
            self._last_hover_element = element
            self._needs_render = True
            if self._on_hover:
                self._on_hover(element)

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
        if not self._input_allowed() or not self._is_mouse_inside():
            return
        if self._graph is None:
            return

        button = app_data  # 0=left, 1=right, 2=middle

        sx, sy = self._get_local_mouse_pos()

        # Right-click on a URL node: open in browser, don't navigate. Acted on at the press, since it
        # begins no gesture a drag could turn into something else.
        if button == 1 and self._on_open_url is not None:
            element = hit_test_screen(self._graph, self._viewport, sx, sy)
            if isinstance(element, Node) and element.url:
                self._on_open_url(element.url)
                return

        # A left press is not yet a click. Acting here navigates on the press that *begins a drag*, and
        # the drag then pans a view that has already jumped elsewhere. Worst when the press lands on an
        # edge, whose click behaviour zooms to the edge's own bounding box -- two endpoints, so an enormous
        # zoom -- leaving the reader with no idea where they are. Remembered here, acted on at release, and
        # only if no drag happened in between.
        if button == 0:
            self._pending_click_pos = (sx, sy)

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
        if not self._input_allowed() or not self._is_mouse_inside():
            return

        delta = app_data  # positive = scroll up (zoom in)

        sx, sy = self._get_local_mouse_pos()

        self._focus_node_name = None

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
        if not self._input_allowed():
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
        """Handle mouse release: where a left click is finally decided, and acted on."""
        was_dragging = self._dragging
        self._dragging = False
        pending = self._pending_click_pos
        self._pending_click_pos = None

        if was_dragging or pending is None or self._graph is None:
            return  # a drag, or a press this widget never saw
        if not self._input_allowed():
            return

        sx, sy = pending
        self._follow_indicator_pos = None  # about to navigate away

        # Edge-follow first, and independent of the hit test, so it works even where a node's bounding box
        # overlaps the edge endpoint.
        follow_target = self._get_edge_follow_target(sx, sy)
        if follow_target is not None:
            self._navigate_to_element(follow_target)
            if self._on_click:
                self._on_click(follow_target, 0)
            return

        element = hit_test_screen(self._graph, self._viewport, sx, sy)
        if element is not None:
            self._navigate_to_element(element)
            if self._on_click:
                self._on_click(element, 0)

    # -------------------------------------------------------------------------
    # Cleanup

    def destroy(self) -> None:
        """Clean up resources."""
        # Reverse of the order they were acquired in, so the animator stops calling `render_frame` before
        # the items that frame draws into go away. Left registered, a destroyed widget keeps being ticked
        # for the life of the process, drawing into deleted items.
        gui_animation.animator.cancel(self)

        with guiutils.nonexistent_ok():
            dpg.delete_item(self._handler_registry)

        with guiutils.nonexistent_ok():
            dpg.delete_item(self._tooltip_window)

        with guiutils.nonexistent_ok():
            dpg.delete_item(self.group)
