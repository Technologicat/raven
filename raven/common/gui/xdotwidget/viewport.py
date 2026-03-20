"""Viewport transforms for pan/zoom.

Coordinate transformation between graph space and screen space,
with smooth animation support via `SmoothValue` for panning and zooming.

The interaction with Raven's GUI animation system, to actually run the
animated parts, is handled by `widget.py`; this is an internal class.
"""

__all__ = ["SmoothValue", "Viewport"]

from typing import Optional, Tuple

from ... import numutils
from ...smoothvalue import SmoothValue
from .. import utils as guiutils

from .constants import Point
from .graph import Graph


class Viewport:
    """Manages pan/zoom state and coordinate transforms.

    The viewport transforms between graph coordinates (the coordinate system
    used in the xdot data) and screen coordinates (pixels in the DPG drawlist).

    Transform formula:
        screen_x = (graph_x - pan_x) * zoom + widget_w / 2
        screen_y = (graph_y - pan_y) * zoom + widget_h / 2

    Where (pan_x, pan_y) is the graph point at the center of the viewport.
    """

    def __init__(self, width: int = 512, height: int = 512):
        """Initialize viewport.

        `width`, `height`: Widget dimensions in pixels.
        """
        self.width = width
        self.height = height

        # Pan position (center of view in graph coordinates)
        self.pan_x = SmoothValue(0.0)
        self.pan_y = SmoothValue(0.0)

        # Zoom level (>1 = zoomed in, <1 = zoomed out)
        self.zoom = SmoothValue(1.0)

        # Limits
        self.min_zoom = 0.01
        self.max_zoom = 100.0

        # Graph bounds (for clamping pan)
        self._graph_width = 1.0
        self._graph_height = 1.0

    def set_size(self, width: int, height: int) -> None:
        """Update the widget size."""
        self.width = width
        self.height = height

    def set_graph_bounds(self, width: float, height: float) -> None:
        """Set the graph dimensions for pan clamping."""
        self._graph_width = width
        self._graph_height = height

    def graph_to_screen(self, gx: float, gy: float) -> Point:
        """Convert graph coordinates to screen coordinates."""
        return guiutils.content_to_screen(gx, gy,
                                          self.pan_x.current, self.pan_y.current,
                                          self.zoom.current,
                                          self.width, self.height)

    def screen_to_graph(self, sx: float, sy: float) -> Point:
        """Convert screen coordinates to graph coordinates."""
        return guiutils.screen_to_content(sx, sy,
                                          self.pan_x.current, self.pan_y.current,
                                          self.zoom.current,
                                          self.width, self.height)

    def zoom_to_fit(self, graph: Graph, margin: int = 12, animate: bool = True) -> None:
        """Adjust pan and zoom to fit the entire graph in the viewport.

        `graph`: The Graph to fit.
        `margin`: Margin in pixels around the graph.
        `animate`: If True, animate the transition. If False, jump immediately.
        """
        gw, gh = graph.get_size()
        self._graph_width = gw
        self._graph_height = gh
        self.zoom_to_bbox(0, 0, gw, gh, margin=margin, animate=animate)

    def zoom_to_bbox(self, x1: float, y1: float, x2: float, y2: float,
                     margin: int = 20, animate: bool = True) -> None:
        """Adjust pan and zoom to fit a bounding box in the viewport.

        `x1`, `y1`, `x2`, `y2`: Bounding box in graph coordinates.
        `margin`: Margin in pixels around the content.
        `animate`: If True, animate the transition.
        """
        gw = abs(x2 - x1)
        gh = abs(y2 - y1)

        available_w = self.width - 2 * margin
        available_h = self.height - 2 * margin

        if gw > 0 and gh > 0:
            zoom_w = available_w / gw
            zoom_h = available_h / gh
            new_zoom = min(zoom_w, zoom_h)
        elif gw > 0:
            new_zoom = available_w / gw
        elif gh > 0:
            new_zoom = available_h / gh
        else:
            new_zoom = self.zoom.current  # degenerate bbox, keep current zoom

        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        new_pan_x = (x1 + x2) / 2
        new_pan_y = (y1 + y2) / 2

        if animate:
            self.pan_x.target = new_pan_x
            self.pan_y.target = new_pan_y
            self.zoom.target = new_zoom
        else:
            self.pan_x.set_immediate(new_pan_x)
            self.pan_y.set_immediate(new_pan_y)
            self.zoom.set_immediate(new_zoom)

    def pan_to_point(self, gx: float, gy: float, animate: bool = True) -> None:
        """Center the viewport on a specific graph point.

        `gx`, `gy`: Point in graph coordinates.
        `animate`: If True, animate the transition.
        """
        if animate:
            self.pan_x.target = gx
            self.pan_y.target = gy
        else:
            self.pan_x.set_immediate(gx)
            self.pan_y.set_immediate(gy)

    def zoom_by(self, factor: float, center_sx: Optional[float] = None,
                center_sy: Optional[float] = None) -> None:
        """Zoom by a multiplicative factor.

        `factor`: Zoom multiplier (>1 = zoom in, <1 = zoom out).
        `center_sx`, `center_sy`: Screen coordinates to zoom toward.
                                   If None, zoom toward viewport center.
        """
        old_zoom = self.zoom.current
        new_zoom = old_zoom * factor  # NOTE: multiplicative drift possible but bounded by clamp
        new_zoom = numutils.clamp(new_zoom, self.min_zoom, self.max_zoom)

        if center_sx is not None and center_sy is not None:
            # Adjust pan so the point at (center_sx, center_sy) stays in place.
            new_pan_x, new_pan_y = guiutils.zoom_keep_point(
                old_zoom, new_zoom,
                center_sx, center_sy,
                self.pan_x.current, self.pan_y.current,
                self.width, self.height)

            self.pan_x.target = new_pan_x
            self.pan_y.target = new_pan_y

        self.zoom.target = new_zoom

    def pan_by(self, dx: float, dy: float) -> None:
        """Pan by a screen offset.

        `dx`, `dy`: Screen pixels to pan (positive = content moves right/down).
        """
        z = self.zoom.current
        if z == 0:
            z = 1.0

        # Convert screen delta to graph delta (inverted because panning
        # the view left means moving the pan position right in graph coords)
        self.pan_x.target = self.pan_x.target - dx / z
        self.pan_y.target = self.pan_y.target - dy / z

    def update(self) -> bool:
        """Advance all animations by one frame.

        Returns True if any animation is still running.
        """
        animating = False
        if self.pan_x.update():
            animating = True
        if self.pan_y.update():
            animating = True
        if self.zoom.update():
            animating = True
        return animating

    def is_animating(self) -> bool:
        """Return True if any value is still animating."""
        return (self.pan_x.is_animating() or
                self.pan_y.is_animating() or
                self.zoom.is_animating())

    def get_visible_bounds(self) -> Tuple[float, float, float, float]:
        """Return the visible area in graph coordinates as (x1, y1, x2, y2)."""
        gx1, gy1 = self.screen_to_graph(0, 0)
        gx2, gy2 = self.screen_to_graph(self.width, self.height)
        return min(gx1, gx2), min(gy1, gy2), max(gx1, gx2), max(gy1, gy2)

    def is_visible(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if a bounding box (in graph coords) is visible in the viewport."""
        vx1, vy1, vx2, vy2 = self.get_visible_bounds()
        # Check for intersection
        return not (x2 < vx1 or x1 > vx2 or y2 < vy1 or y1 > vy2)
