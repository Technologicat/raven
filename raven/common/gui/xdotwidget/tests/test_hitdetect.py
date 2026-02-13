"""Tests for hit detection: point-in-polygon, point-to-segment, get_edge."""

import pytest

from ..hitdetect import _point_in_polygon, _point_to_segment_dist_sq, get_edge
from ..graph import (Graph, Node, Edge, Pen,
                     TextShape, BezierShape)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approx(a, b, tol=0.01):
    """Check approximate float equality."""
    return abs(a - b) < tol


def _make_node(name, x=0, y=0):
    """Create a minimal Node for testing."""
    pen = Pen()
    text = TextShape(pen, x, y, 0, 10, name)
    return Node(x, y, 20, 10, [text], internal_name=name)


# ---------------------------------------------------------------------------
# Tests: _point_in_polygon
# ---------------------------------------------------------------------------

class TestPointInPolygon:
    """Test the ray-casting point-in-polygon test."""

    def test_inside_square(self):
        """Point inside a unit square."""
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert _point_in_polygon(0.5, 0.5, square)

    def test_outside_square(self):
        """Point outside a unit square."""
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert not _point_in_polygon(2.0, 2.0, square)

    def test_inside_triangle(self):
        """Point inside a triangle."""
        tri = [(0, 0), (4, 0), (2, 3)]
        assert _point_in_polygon(2.0, 1.0, tri)

    def test_outside_triangle(self):
        """Point outside a triangle."""
        tri = [(0, 0), (4, 0), (2, 3)]
        assert not _point_in_polygon(0.0, 3.0, tri)


# ---------------------------------------------------------------------------
# Tests: _point_to_segment_dist_sq
# ---------------------------------------------------------------------------

class TestPointToSegmentDistSq:
    """Test squared distance from point to line segment."""

    def test_perpendicular_projection(self):
        """Point projects perpendicularly onto the segment.

        Segment from (0,0) to (4,0), point at (2,3).
        Distance = 3, squared = 9.
        """
        d = _point_to_segment_dist_sq(2, 3, 0, 0, 4, 0)
        assert _approx(d, 9.0)

    def test_beyond_segment_start(self):
        """Point nearest to the start endpoint.

        Segment from (2,0) to (4,0), point at (0,0).
        Nearest point is (2,0), distance = 2, squared = 4.
        """
        d = _point_to_segment_dist_sq(0, 0, 2, 0, 4, 0)
        assert _approx(d, 4.0)

    def test_beyond_segment_end(self):
        """Point nearest to the end endpoint.

        Segment from (0,0) to (2,0), point at (4,0).
        Nearest point is (2,0), distance = 2, squared = 4.
        """
        d = _point_to_segment_dist_sq(4, 0, 0, 0, 2, 0)
        assert _approx(d, 4.0)

    def test_zero_length_segment(self):
        """Degenerate segment (point). Distance is just point-to-point.

        Segment at (1,1), point at (4,5).
        Distance = sqrt(9+16) = 5, squared = 25.
        """
        d = _point_to_segment_dist_sq(4, 5, 1, 1, 1, 1)
        assert _approx(d, 25.0)

    def test_point_on_segment(self):
        """Point exactly on the segment → distance 0."""
        d = _point_to_segment_dist_sq(2, 0, 0, 0, 4, 0)
        assert _approx(d, 0.0)


# ---------------------------------------------------------------------------
# Tests: get_edge (with BezierShape)
# ---------------------------------------------------------------------------

class TestGetEdge:
    """Test edge hit detection with tessellated bezier curves."""

    @pytest.fixture
    def bezier_graph(self):
        """Graph with one edge that has a bezier curve from (0,0) to (100,0).

        The bezier control points curve upward, so the midpoint of the curve
        is approximately at (50, ~37.5).
        """
        src = _make_node("src", 0, 0)
        dst = _make_node("dst", 100, 0)
        pen = Pen()
        # Cubic bezier: (0,0) → (33,50) → (66,50) → (100,0)
        bezier = BezierShape(pen, [(0, 0), (33, 50), (66, 50), (100, 0)])
        edge = Edge(src, dst, [(0, 0), (100, 0)], [bezier])
        graph = Graph(width=100, height=100,
                      nodes=[src, dst], edges=[edge])
        return graph, edge

    def test_point_on_curve_detected(self, bezier_graph):
        """A point near the tessellated curve is detected."""
        graph, edge = bezier_graph
        # The curve midpoint (t=0.5): B(0.5) = 0.125*(0,0) + 0.375*(33,50)
        #   + 0.375*(66,50) + 0.125*(100,0) = (49.625, 37.5)
        # Use a generous radius to account for tessellation approximation.
        result = get_edge(graph, 50, 37, radius_sq=5**2)
        assert result is edge

    def test_far_point_not_detected(self, bezier_graph):
        """A point far from the curve is not detected."""
        graph, edge = bezier_graph
        result = get_edge(graph, 50, -100, radius_sq=5**2)
        assert result is None

    def test_near_endpoint_detected(self, bezier_graph):
        """A point near the bezier start endpoint is detected."""
        graph, edge = bezier_graph
        result = get_edge(graph, 1, 1, radius_sq=5**2)
        assert result is edge
