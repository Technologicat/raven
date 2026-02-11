"""Tests for the viewport transforms."""

import pytest
import math

from ..viewport import Viewport, SmoothValue
from ..graph import Graph


class TestSmoothValue:
    """Test the SmoothValue animation class."""

    def test_initial_value(self):
        """Test initial value setup."""
        sv = SmoothValue(value=5.0)
        assert sv.current == 5.0
        assert sv.target == 5.0

    def test_set_target(self):
        """Test setting a target value."""
        sv = SmoothValue(value=0.0)
        sv.target = 10.0
        assert sv.target == 10.0
        assert sv.current == 0.0  # Not yet animated

    def test_set_immediate(self):
        """Test immediate value setting."""
        sv = SmoothValue(value=0.0)
        sv.set_immediate(10.0)
        assert sv.current == 10.0
        assert sv.target == 10.0

    def test_is_animating(self):
        """Test animation detection."""
        sv = SmoothValue(value=0.0)
        assert not sv.is_animating()

        sv.target = 10.0
        assert sv.is_animating()

    def test_update_moves_toward_target(self):
        """Test that update moves value toward target."""
        sv = SmoothValue(value=0.0, rate=0.5)
        sv.target = 10.0

        initial = sv.current
        sv.update(dt=0.1)

        # Should have moved toward target
        assert sv.current > initial
        assert sv.current < sv.target

    def test_update_reaches_target(self):
        """Test that repeated updates eventually reach target."""
        sv = SmoothValue(value=0.0, rate=0.9)
        sv.target = 10.0

        # Run many updates
        for _ in range(100):
            sv.update(dt=0.05)

        # Should have reached target
        assert abs(sv.current - sv.target) < 0.01


class TestViewport:
    """Test the Viewport class."""

    def test_initial_state(self):
        """Test viewport initial state."""
        vp = Viewport(width=800, height=600)
        assert vp.width == 800
        assert vp.height == 600

    def test_graph_to_screen_identity(self):
        """Test transform with identity zoom at origin."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(0)
        vp.pan_y.set_immediate(0)
        vp.zoom.set_immediate(1.0)

        # Point at origin should map to center
        sx, sy = vp.graph_to_screen(0, 0)
        assert sx == 50  # width/2
        assert sy == 50  # height/2

    def test_screen_to_graph_roundtrip(self):
        """Test that graph->screen->graph is identity."""
        vp = Viewport(width=800, height=600)
        vp.pan_x.set_immediate(100)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(2.0)

        gx, gy = 75.0, 25.0
        sx, sy = vp.graph_to_screen(gx, gy)
        gx2, gy2 = vp.screen_to_graph(sx, sy)

        assert abs(gx - gx2) < 0.001
        assert abs(gy - gy2) < 0.001

    def test_zoom_scales_coordinates(self):
        """Test that zoom affects coordinate transform."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)

        # At zoom 1
        vp.zoom.set_immediate(1.0)
        sx1, sy1 = vp.graph_to_screen(60, 60)

        # At zoom 2
        vp.zoom.set_immediate(2.0)
        sx2, sy2 = vp.graph_to_screen(60, 60)

        # Screen distance from center should be 2x with double zoom
        center = 50
        dist1 = abs(sx1 - center)
        dist2 = abs(sx2 - center)
        assert abs(dist2 - 2 * dist1) < 0.001

    def test_zoom_to_fit(self):
        """Test zoom_to_fit centers and scales appropriately."""
        vp = Viewport(width=400, height=300)

        # Create a graph of known size
        graph = Graph(width=200, height=100)

        vp.zoom_to_fit(graph, margin=0, animate=False)

        # Pan should be centered on graph
        assert abs(vp.pan_x.current - 100) < 0.001  # graph_width / 2
        assert abs(vp.pan_y.current - 50) < 0.001  # graph_height / 2

    def test_is_visible(self):
        """Test visibility checking."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        # Box in the center should be visible
        assert vp.is_visible(40, 40, 60, 60)

        # Box far outside should not be visible
        assert not vp.is_visible(1000, 1000, 1010, 1010)

    def test_pan_by(self):
        """Test panning by screen offset."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        initial_x = vp.pan_x.target
        initial_y = vp.pan_y.target

        vp.pan_by(10, 20)

        # Panning screen right/down moves pan position left/up in graph coords
        assert vp.pan_x.target < initial_x
        assert vp.pan_y.target < initial_y

    def test_zoom_by(self):
        """Test zooming by factor."""
        vp = Viewport(width=100, height=100)
        vp.zoom.set_immediate(1.0)

        vp.zoom_by(2.0)
        assert vp.zoom.target == 2.0

        vp.zoom.set_immediate(vp.zoom.target)
        vp.zoom_by(0.5)
        assert vp.zoom.target == 1.0

    def test_zoom_limits(self):
        """Test that zoom respects min/max limits."""
        vp = Viewport(width=100, height=100)
        vp.min_zoom = 0.1
        vp.max_zoom = 10.0
        vp.zoom.set_immediate(1.0)

        # Try to zoom way out
        vp.zoom_by(0.001)
        assert vp.zoom.target >= vp.min_zoom

        vp.zoom.set_immediate(1.0)

        # Try to zoom way in
        vp.zoom_by(1000)
        assert vp.zoom.target <= vp.max_zoom
