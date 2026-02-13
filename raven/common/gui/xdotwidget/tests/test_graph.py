"""Tests for graph data model: tessellate_bezier and get_linked_elements."""

import pytest

from ..graph import (Graph, Node, Edge, TextShape, Pen,
                     tessellate_bezier)


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


def _make_edge(src, dst):
    """Create a minimal Edge for testing."""
    points = [(src.x, src.y), (dst.x, dst.y)]
    return Edge(src, dst, points, [])


# ---------------------------------------------------------------------------
# Tests: tessellate_bezier
# ---------------------------------------------------------------------------

class TestTessellateBezier:
    """Test cubic bezier tessellation."""

    def test_basic_cubic(self):
        """A single cubic segment with n=10 produces 11 points."""
        # Straight-ish bezier from (0,0) to (3,0)
        points = [(0, 0), (1, 1), (2, 1), (3, 0)]
        result = tessellate_bezier(points, n=10)
        assert len(result) == 11

    def test_endpoints_match(self):
        """First and last tessellation points match the bezier endpoints."""
        points = [(0, 0), (1, 2), (2, 2), (3, 0)]
        result = tessellate_bezier(points, n=10)
        assert _approx(result[0][0], 0)
        assert _approx(result[0][1], 0)
        assert _approx(result[-1][0], 3)
        assert _approx(result[-1][1], 0)

    def test_two_segments(self):
        """Two cubic segments: [P0, C1, C2, P1, C3, C4, P2].

        First segment: 11 points (indices 0..10).
        Second segment: 10 points (index 0 of second segment is same as P1, skipped).
        Total: 21 points.
        """
        points = [(0, 0), (1, 1), (2, 1), (3, 0),
                  (4, -1), (5, -1), (6, 0)]
        result = tessellate_bezier(points, n=10)
        assert len(result) == 21
        # Endpoints
        assert _approx(result[0][0], 0)
        assert _approx(result[-1][0], 6)

    def test_degenerate_empty(self):
        """Empty point list returns empty."""
        result = tessellate_bezier([])
        assert result == []

    def test_degenerate_single_point(self):
        """Single point returns that point (not enough for a segment)."""
        result = tessellate_bezier([(5, 7)])
        assert len(result) == 1
        assert result[0] == (5, 7)

    def test_degenerate_three_points(self):
        """Three points (not enough for a cubic segment) returns them as-is."""
        pts = [(0, 0), (1, 1), (2, 0)]
        result = tessellate_bezier(pts)
        assert len(result) == 3

    def test_straight_line_bezier(self):
        """Collinear control points → all tessellated points lie on the line."""
        # All control points on the x-axis (y=0)
        points = [(0, 0), (1, 0), (2, 0), (3, 0)]
        result = tessellate_bezier(points, n=10)
        for x, y in result:
            assert _approx(y, 0, tol=1e-10), f"y={y} should be 0 for collinear controls"


# ---------------------------------------------------------------------------
# Tests: Graph.get_linked_elements
# ---------------------------------------------------------------------------

class TestGetLinkedElements:
    """Test Graph.get_linked_elements for outgoing/incoming directions."""

    @pytest.fixture
    def diamond_graph(self):
        """Diamond-shaped graph: A → B, A → C, B → D, C → D."""
        a = _make_node("A", 0, 0)
        b = _make_node("B", 50, 50)
        c = _make_node("C", -50, 50)
        d = _make_node("D", 0, 100)
        e_ab = _make_edge(a, b)
        e_ac = _make_edge(a, c)
        e_bd = _make_edge(b, d)
        e_cd = _make_edge(c, d)
        graph = Graph(width=100, height=100,
                      nodes=[a, b, c, d],
                      edges=[e_ab, e_ac, e_bd, e_cd])
        return graph, a, b, c, d, e_ab, e_ac, e_bd, e_cd

    def test_outgoing_from_root(self, diamond_graph):
        """Outgoing from A → edges to B and C, plus nodes B and C."""
        graph, a, b, c, d, e_ab, e_ac, e_bd, e_cd = diamond_graph
        linked = graph.get_linked_elements(a, "outgoing")
        assert e_ab in linked
        assert e_ac in linked
        assert b in linked
        assert c in linked
        # Should not include the queried node itself or unrelated elements
        assert a not in linked
        assert d not in linked
        assert e_bd not in linked
        assert e_cd not in linked

    def test_incoming_to_leaf(self, diamond_graph):
        """Incoming to D → edges from B and C, plus nodes B and C."""
        graph, a, b, c, d, e_ab, e_ac, e_bd, e_cd = diamond_graph
        linked = graph.get_linked_elements(d, "incoming")
        assert e_bd in linked
        assert e_cd in linked
        assert b in linked
        assert c in linked
        assert a not in linked
        assert d not in linked

    def test_outgoing_from_middle(self, diamond_graph):
        """Outgoing from B → edge to D, plus node D."""
        graph, a, b, c, d, e_ab, e_ac, e_bd, e_cd = diamond_graph
        linked = graph.get_linked_elements(b, "outgoing")
        assert e_bd in linked
        assert d in linked
        assert len(linked) == 2

    def test_incoming_to_middle(self, diamond_graph):
        """Incoming to B → edge from A, plus node A."""
        graph, a, b, c, d, e_ab, e_ac, e_bd, e_cd = diamond_graph
        linked = graph.get_linked_elements(b, "incoming")
        assert e_ab in linked
        assert a in linked
        assert len(linked) == 2

    def test_outgoing_from_leaf(self, diamond_graph):
        """Outgoing from D (leaf, no outgoing edges) → empty set."""
        graph, a, b, c, d, e_ab, e_ac, e_bd, e_cd = diamond_graph
        linked = graph.get_linked_elements(d, "outgoing")
        assert len(linked) == 0

    def test_incoming_to_root(self, diamond_graph):
        """Incoming to A (root, no incoming edges) → empty set."""
        graph, a, b, c, d, e_ab, e_ac, e_bd, e_cd = diamond_graph
        linked = graph.get_linked_elements(a, "incoming")
        assert len(linked) == 0
