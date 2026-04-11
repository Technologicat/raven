"""Tests for the viewport transforms.

Skips in CI: viewport → gui.utils → DPG (decoupling TODO).
"""
import pytest
pytest.importorskip("dearpygui", reason="transitive DPG dependency via gui.utils")

from raven.common.tests import approx

from ..viewport import Viewport
from ..graph import Graph


# ---------------------------------------------------------------------------
# Tests: Viewport
# ---------------------------------------------------------------------------

class TestViewport:
    """Test the Viewport class."""

    def test_initial_state(self):
        """Viewport stores its dimensions."""
        vp = Viewport(width=800, height=600)
        assert vp.width == 800
        assert vp.height == 600

    # --- Coordinate transforms ---

    def test_graph_to_screen_identity(self):
        """At origin pan and zoom 1, graph origin maps to screen center."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(0)
        vp.pan_y.set_immediate(0)
        vp.zoom.set_immediate(1.0)

        sx, sy = vp.graph_to_screen(0, 0)
        assert sx == 50  # width / 2
        assert sy == 50  # height / 2

    def test_screen_to_graph_roundtrip(self):
        """graph_to_screen and screen_to_graph are inverses."""
        vp = Viewport(width=800, height=600)
        vp.pan_x.set_immediate(100)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(2.0)

        gx, gy = 75.0, 25.0
        sx, sy = vp.graph_to_screen(gx, gy)
        gx2, gy2 = vp.screen_to_graph(sx, sy)

        assert approx(gx, gx2, tol=0.001)
        assert approx(gy, gy2, tol=0.001)

    def test_zoom_scales_coordinates(self):
        """Doubling zoom doubles screen distance from center."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)

        vp.zoom.set_immediate(1.0)
        sx1, sy1 = vp.graph_to_screen(60, 60)

        vp.zoom.set_immediate(2.0)
        sx2, sy2 = vp.graph_to_screen(60, 60)

        center = 50
        dist1 = abs(sx1 - center)
        dist2 = abs(sx2 - center)
        assert approx(dist2, 2 * dist1, tol=0.001)

    # --- Zoom to fit ---

    def test_zoom_to_fit_centers_pan(self):
        """zoom_to_fit centers the pan on the graph."""
        vp = Viewport(width=400, height=300)
        graph = Graph(width=200, height=100)

        vp.zoom_to_fit(graph, margin=0, animate=False)

        assert approx(vp.pan_x.current, 100)  # graph_width / 2
        assert approx(vp.pan_y.current, 50)   # graph_height / 2

    def test_zoom_to_fit_zoom_level(self):
        """zoom_to_fit sets zoom to fit the graph in the viewport.

        Viewport 400x300, graph 200x100, margin=0.
        Zoom = min(400/200, 300/100) = min(2.0, 3.0) = 2.0.
        """
        vp = Viewport(width=400, height=300)
        graph = Graph(width=200, height=100)

        vp.zoom_to_fit(graph, margin=0, animate=False)

        assert approx(vp.zoom.current, 2.0)

    def test_zoom_to_fit_with_margin(self):
        """zoom_to_fit accounts for margin.

        With margin, the effective viewport area is reduced, so the
        zoom level should be smaller than without margin.
        """
        vp_no_margin = Viewport(width=400, height=300)
        vp_with_margin = Viewport(width=400, height=300)
        graph = Graph(width=200, height=100)

        vp_no_margin.zoom_to_fit(graph, margin=0, animate=False)
        vp_with_margin.zoom_to_fit(graph, margin=20, animate=False)

        assert vp_with_margin.zoom.current < vp_no_margin.zoom.current

    def test_zoom_to_fit_degenerate_graph(self):
        """zoom_to_fit handles a zero-size graph without crashing."""
        vp = Viewport(width=400, height=300)
        # Graph with default dimensions (width=1, height=1)
        graph = Graph()

        # Should not raise
        vp.zoom_to_fit(graph, margin=0, animate=False)
        assert vp.zoom.current > 0

    # --- Visibility ---

    def test_is_visible_center(self):
        """A box at the center of the viewport is visible."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        assert vp.is_visible(40, 40, 60, 60)

    def test_is_visible_far_outside(self):
        """A box far outside the viewport is not visible."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        assert not vp.is_visible(1000, 1000, 1010, 1010)

    def test_is_visible_partial_overlap(self):
        """A box partially overlapping the viewport edge is visible."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        # Screen shows graph coords [0, 100] x [0, 100] at zoom 1 with pan at center.
        # A box that straddles the right edge:
        assert vp.is_visible(90, 40, 110, 60)

    def test_is_visible_fully_enclosing(self):
        """A box larger than the viewport that fully encloses it is visible."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        assert vp.is_visible(-100, -100, 200, 200)

    # --- Pan ---

    def test_pan_by(self):
        """Panning by a screen offset moves the pan target in the opposite graph direction."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        initial_x = vp.pan_x.target
        initial_y = vp.pan_y.target

        vp.pan_by(10, 20)

        # Screen right/down -> graph pan left/up
        assert vp.pan_x.target < initial_x
        assert vp.pan_y.target < initial_y

    def test_pan_by_zoom_scaling(self):
        """Pan amount in graph space is inversely proportional to zoom.

        At zoom 2, a 10-pixel screen pan should move half as far in
        graph space as at zoom 1.
        """
        def pan_delta_x(zoom):
            vp = Viewport(width=100, height=100)
            vp.pan_x.set_immediate(50)
            vp.pan_y.set_immediate(50)
            vp.zoom.set_immediate(zoom)
            initial = vp.pan_x.target
            vp.pan_by(10, 0)
            return abs(vp.pan_x.target - initial)

        delta_z1 = pan_delta_x(zoom=1.0)
        delta_z2 = pan_delta_x(zoom=2.0)

        assert approx(delta_z2, delta_z1 / 2, tol=0.001)

    # --- Zoom ---

    def test_zoom_by(self):
        """zoom_by multiplies the current zoom."""
        vp = Viewport(width=100, height=100)
        vp.zoom.set_immediate(1.0)

        vp.zoom_by(2.0)
        assert vp.zoom.target == 2.0

        vp.zoom.set_immediate(vp.zoom.target)
        vp.zoom_by(0.5)
        assert vp.zoom.target == 1.0

    def test_zoom_min_limit(self):
        """Zoom does not go below min_zoom."""
        vp = Viewport(width=100, height=100)
        vp.min_zoom = 0.1
        vp.zoom.set_immediate(1.0)

        vp.zoom_by(0.001)
        assert vp.zoom.target >= vp.min_zoom

    def test_zoom_max_limit(self):
        """Zoom does not exceed max_zoom."""
        vp = Viewport(width=100, height=100)
        vp.max_zoom = 10.0
        vp.zoom.set_immediate(1.0)

        vp.zoom_by(1000)
        assert vp.zoom.target <= vp.max_zoom

    # --- Size ---

    def test_set_size(self):
        """set_size updates the viewport dimensions."""
        vp = Viewport(width=100, height=100)
        vp.set_size(800, 600)
        assert vp.width == 800
        assert vp.height == 600

    # --- zoom_to_bbox ---

    def test_zoom_to_bbox_centers(self):
        """zoom_to_bbox centers the pan on the bbox midpoint."""
        vp = Viewport(width=400, height=300)
        vp.zoom_to_bbox(20, 30, 80, 90, margin=0, animate=False)
        assert approx(vp.pan_x.current, 50)   # (20 + 80) / 2
        assert approx(vp.pan_y.current, 60)    # (30 + 90) / 2

    def test_zoom_to_bbox_correct_zoom(self):
        """zoom_to_bbox computes the correct zoom level.

        Viewport 400×300, bbox 60×60, margin=0.
        Zoom = min(400/60, 300/60) = min(6.67, 5.0) = 5.0.
        """
        vp = Viewport(width=400, height=300)
        vp.zoom_to_bbox(20, 30, 80, 90, margin=0, animate=False)
        assert approx(vp.zoom.current, 5.0)

    def test_zoom_to_bbox_degenerate_zero_area(self):
        """zoom_to_bbox with zero-area bbox doesn't crash and keeps current zoom."""
        vp = Viewport(width=400, height=300)
        vp.zoom.set_immediate(2.0)
        vp.zoom_to_bbox(50, 50, 50, 50, margin=0, animate=False)
        # Degenerate → zoom should not change
        assert approx(vp.zoom.current, 2.0)
        # But pan should center on the point
        assert approx(vp.pan_x.current, 50)
        assert approx(vp.pan_y.current, 50)

    # --- pan_to_point ---

    def test_pan_to_point_immediate(self):
        """pan_to_point with animate=False sets current immediately."""
        vp = Viewport(width=100, height=100)
        vp.pan_to_point(75.0, 25.0, animate=False)
        assert approx(vp.pan_x.current, 75.0)
        assert approx(vp.pan_y.current, 25.0)

    def test_pan_to_point_animated(self):
        """pan_to_point with animate=True sets target only (current unchanged)."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(0)
        vp.pan_y.set_immediate(0)
        vp.pan_to_point(75.0, 25.0, animate=True)
        # Target set
        assert approx(vp.pan_x.target, 75.0)
        assert approx(vp.pan_y.target, 25.0)
        # Current unchanged
        assert approx(vp.pan_x.current, 0)
        assert approx(vp.pan_y.current, 0)
