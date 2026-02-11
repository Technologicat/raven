"""Viewport transforms and smooth animation for pan/zoom.

This module provides coordinate transformation between graph space and
screen space, with smooth animation support for panning and zooming.

The interaction with Raven's GUI animation system, to actually run the
animated parts, is handled by `widget.py`; this is an internal class.
"""

__all__ = ["SmoothValue", "Viewport"]

import math
import time
from typing import Optional, Tuple

from ... import numutils

from .constants import Point
from .graph import Graph


# TODO: Unify the interpolator implementations: this one; `raven.server.modules.avatar` for pose interpolation; and `raven.common.gui.animation` for scroll position animation. Should have just one with a specifiable datatype (float or int) and an optional subpixel flag for the int case.
class SmoothValue:
    """An animated float value using first-order ODE solution.

    The animation depends only on current and target positions, using
    the analytical solution to Newton's law of cooling (exponential decay
    toward target). Hence this can adapt to sudden target value changes.

    This provides smooth transitions between values, without a fixed duration.

    The `rate` parameter, in the half-open interval (0, 1], controls how quickly
    the value approaches the target. Higher values mean faster animation.

    For example, the default rate of 0.8 means that 80% of the remaining distance
    toward the target value is covered in one frame at *calibration FPS*, which is 25 FPS.
    The animator internally compensates for the render framerate automatically,
    so that the animation speed remains the same (w.r.t. wall clock time)
    regardless of the actual render FPS.
    """

    CALIBRATION_FPS = 25  # FPS for which `rate` was calibrated
    EPSILON = 1e-3  # Range from which to snap to target; denormal guard

    def __init__(self, value: float = 0.0, rate: float = 0.8):
        """Initialize with a starting value.

        `value`: Initial value.
        `rate`: Animation rate, in (0, 1]. Higher = faster. Default 0.8.
        """
        self._current = value
        self._target = value
        self._rate = rate
        self._last_time = time.time()

    # TODO: Use the same style for defining properties as in the rest of the Raven codebase.
    @property
    def current(self) -> float:
        """The current (animated) value."""
        return self._current

    @property
    def target(self) -> float:
        """The target value the animation is moving toward."""
        return self._target

    @target.setter
    def target(self, value: float) -> None:
        """Set a new target value."""
        self._target = value

    def set_immediate(self, value: float) -> None:
        """Set both current and target value immediately (no animation)."""
        self._current = value
        self._target = value

    def is_animating(self) -> bool:
        """Return True if the value is still animating toward target."""
        return abs(self._target - self._current) > 0.001

    def update(self, dt: Optional[float] = None) -> bool:
        """Advance the animation by one frame.

        `dt`: Time delta in seconds. If None, computed from wall clock.

        Returns True if still animating, False if reached target.
        """
        if not self.is_animating():
            self._current = self._target
            return False

        # Compute time delta
        now = time.time()
        if dt is None:
            dt = now - self._last_time
        self._last_time = now

        # Framerate correction for rate-based animation
        # (from raven.common.gui.animation.SmoothScrolling)
        alpha_orig = 1.0 - self._rate
        if 0 < alpha_orig < 1:
            # Estimate FPS from dt
            if dt > 0:
                avg_fps = 1.0 / dt
            else:
                avg_fps = self.CALIBRATION_FPS

            # Compute number of frames to cover 50% of distance at calibration FPS
            xrel = 0.5
            n_orig = math.log(1.0 - xrel) / math.log(alpha_orig)
            # Scale for actual FPS
            n_scaled = (avg_fps / self.CALIBRATION_FPS) * n_orig
            if n_scaled > 0:
                alpha_scaled = (1.0 - xrel) ** (1 / n_scaled)
            else:
                alpha_scaled = alpha_orig
        else:
            alpha_scaled = alpha_orig

        step_scaled = 1.0 - alpha_scaled

        # Calculate new position
        remaining = self._target - self._current
        delta = step_scaled * remaining
        new_value = self._current + delta

        # Snap to target when close enough, to avoid denormal floats (which are software-emulated, very slow).
        if abs(self._target - new_value) <= self.EPSILON:
            self._current = self._target
            return False
        self._current = new_value
        return True


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
        z = self.zoom.current
        sx = (gx - self.pan_x.current) * z + self.width / 2
        sy = (gy - self.pan_y.current) * z + self.height / 2
        return sx, sy

    def screen_to_graph(self, sx: float, sy: float) -> Point:
        """Convert screen coordinates to graph coordinates."""
        z = self.zoom.current
        if z == 0:
            z = 1.0
        gx = (sx - self.width / 2) / z + self.pan_x.current
        gy = (sy - self.height / 2) / z + self.pan_y.current
        return gx, gy

    def zoom_to_fit(self, graph: Graph, margin: int = 12, animate: bool = True) -> None:
        """Adjust pan and zoom to fit the entire graph in the viewport.

        `graph`: The Graph to fit.
        `margin`: Margin in pixels around the graph.
        `animate`: If True, animate the transition. If False, jump immediately.
        """
        gw, gh = graph.get_size()
        self._graph_width = gw
        self._graph_height = gh

        # Calculate zoom to fit
        available_w = self.width - 2 * margin
        available_h = self.height - 2 * margin

        if gw > 0 and gh > 0:
            zoom_w = available_w / gw
            zoom_h = available_h / gh
            new_zoom = min(zoom_w, zoom_h)
        else:
            new_zoom = 1.0

        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        # Center on graph
        new_pan_x = gw / 2
        new_pan_y = gh / 2

        if animate:
            self.pan_x.target = new_pan_x
            self.pan_y.target = new_pan_y
            self.zoom.target = new_zoom
        else:
            self.pan_x.set_immediate(new_pan_x)
            self.pan_y.set_immediate(new_pan_y)
            self.zoom.set_immediate(new_zoom)

    def zoom_to_point(self, gx: float, gy: float, animate: bool = True) -> None:
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
            # Adjust pan so the point at (center_sx, center_sy) stays in place on the screen.
            #
            # We have, from `screen_to_graph`:
            #   gx = (sx - w / 2) / zoom + pan_x
            #   gy = (sy - h / 2) / zoom + pan_y
            #
            # so for the x component,
            #   sx = (gx - pan_x) * zoom + w / 2
            #
            # and after zoom,
            #   sx = (gx - new_pan_x) * new_zoom + w / 2
            #
            # Solving for new_pan_x:
            #   new_pan_x = gx - (sx - self.width / 2) / new_zoom
            #
            # y component similarly.
            #
            gx, gy = self.screen_to_graph(center_sx, center_sy)
            new_pan_x = gx - (center_sx - self.width / 2) / new_zoom
            new_pan_y = gy - (center_sy - self.height / 2) / new_zoom

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
