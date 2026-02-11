"""Tests for the xdot parser."""

import pytest

from ..parser import parse_xdot, XDotParser, ParseError
from ..graph import (Graph, Node, Edge,
                     TextShape, EllipseShape, BezierShape, PolygonShape)


# ---------------------------------------------------------------------------
# Test fixtures: xdot format graph strings
#
# These are minimal but complete xdot snippets. The draw commands follow
# the xdot format spec (https://graphviz.org/docs/outputs/canon/#xdot).
# ---------------------------------------------------------------------------

# Simple two-node graph with one edge.
# bb = "0,0,200,150" so the Y transform is: y' = -(y - 150) = 150 - y.
# Node a at (50, 100) -> transformed (50, 50).
# Node b at (150, 100) -> transformed (150, 50).
# Uses lowercase 'e' (unfilled ellipse) and 'T' (text).
SIMPLE_XDOT = """
digraph G {
    graph [bb="0,0,200,150"];
    a [pos="50,100", width="0.75", height="0.5",
       _draw_="c 7 -#000000 e 50 100 27 18 ",
       _ldraw_="F 14 11 -Times-Roman c 7 -#000000 T 50 100 0 5 1 -a "];
    b [pos="150,100", width="0.75", height="0.5",
       _draw_="c 7 -#000000 e 150 100 27 18 ",
       _ldraw_="F 14 11 -Times-Roman c 7 -#000000 T 150 100 0 5 1 -b "];
    a -> b [pos="77,100 95,100 113,100 123,100",
            _draw_="c 7 -#000000 B 4 77 100 95 100 113 100 123 100 ",
            _hdraw_="S 5 -solid c 7 -#000000 C 7 -#000000 P 3 123 104 131 100 123 96 "];
}
"""

# Graph for testing hex color parsing.
# Uses #1a4b7c — all components nonzero and distinct, so we can verify
# there's no channel swizzle (RGB vs BGR, etc.).
COLOR_HEX_XDOT = """
digraph G {
    graph [bb="0,0,100,100"];
    a [pos="50,50", width="0.5", height="0.5",
       _draw_="c 7 -#1a4b7c e 50 50 18 18 "];
}
"""

# Graph for testing hex color with alpha channel.
COLOR_HEX_ALPHA_XDOT = """
digraph G {
    graph [bb="0,0,100,100"];
    a [pos="50,50", width="0.5", height="0.5",
       _draw_="c 9 -#1a4b7c80 e 50 50 18 18 "];
}
"""

# Graph for testing HSV color parsing.
# HSV (0.583333, 0.790323, 0.486275) is the same color as #1a4b7c,
# so we can verify the HSV->RGB conversion against known values.
COLOR_HSV_XDOT = """
digraph G {
    graph [bb="0,0,100,100"];
    a [pos="50,50", width="0.5", height="0.5",
       _draw_="c 26 -0.583333,0.790323,0.486275 e 50 50 18 18 "];
}
"""

# Graph for testing named color parsing.
COLOR_NAMED_XDOT = """
digraph G {
    graph [bb="0,0,100,100"];
    a [pos="50,50", width="0.5", height="0.5",
       _draw_="c 3 -red e 50 50 18 18 "];
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shapes_of_type(element, shape_class):
    """Return all shapes of a given type from an element."""
    return [s for s in element.shapes if isinstance(s, shape_class)]


def _approx(a, b, tol=0.01):
    """Check approximate float equality."""
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Tests: Parser basics
# ---------------------------------------------------------------------------

class TestParser:
    """Test the XDot parser."""

    def test_parse_returns_graph(self):
        """parse_xdot returns a Graph instance."""
        graph = parse_xdot(SIMPLE_XDOT)
        assert isinstance(graph, Graph)

    def test_graph_dimensions(self):
        """Graph dimensions match the bounding box."""
        graph = parse_xdot(SIMPLE_XDOT)
        assert graph.width == 200
        assert graph.height == 150

    def test_node_count(self):
        """Correct number of nodes parsed."""
        graph = parse_xdot(SIMPLE_XDOT)
        assert len(graph.nodes) == 2

    def test_node_names(self):
        """Node internal names are preserved."""
        graph = parse_xdot(SIMPLE_XDOT)
        names = {n.internal_name for n in graph.nodes}
        assert names == {"a", "b"}

    def test_edge_count(self):
        """Correct number of edges parsed."""
        graph = parse_xdot(SIMPLE_XDOT)
        assert len(graph.edges) == 1

    def test_edge_connectivity(self):
        """Edge connects the correct source and destination nodes."""
        graph = parse_xdot(SIMPLE_XDOT)
        edge = graph.edges[0]
        assert edge.src.internal_name == "a"
        assert edge.dst.internal_name == "b"

    def test_empty_graph(self):
        """Parsing an empty graph produces correct dimensions and no elements."""
        xdot = 'digraph G { graph [bb="0,0,100,100"]; }'
        graph = parse_xdot(xdot)
        assert graph.width == 100
        assert graph.height == 100
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_nodes_by_name_lookup(self):
        """get_node_by_name returns the correct node or None."""
        graph = parse_xdot(SIMPLE_XDOT)
        assert graph.get_node_by_name("a") is not None
        assert graph.get_node_by_name("b") is not None
        assert graph.get_node_by_name("nonexistent") is None


# ---------------------------------------------------------------------------
# Tests: Y flip transform
# ---------------------------------------------------------------------------

class TestYTransform:
    """Test that the parser's Y axis flip is applied correctly.

    GraphViz uses a Y-up coordinate system; the parser flips to Y-down
    for screen rendering. For bb="0,0,200,150", the transform is:

        x' = x
        y' = -(y + yoffset) = -(y - 150) = 150 - y
    """

    def test_node_positions_x(self):
        """X coordinates are preserved (no flip)."""
        graph = parse_xdot(SIMPLE_XDOT)
        node_a = graph.get_node_by_name("a")
        node_b = graph.get_node_by_name("b")
        assert node_a.x == 50
        assert node_b.x == 150

    def test_node_positions_y(self):
        """Y coordinates are flipped: raw y=100, bb height=150, so y' = 50."""
        graph = parse_xdot(SIMPLE_XDOT)
        node_a = graph.get_node_by_name("a")
        node_b = graph.get_node_by_name("b")
        assert node_a.y == 50
        assert node_b.y == 50

    def test_ellipse_center_matches_node(self):
        """Ellipse shape center matches the node's transformed position."""
        graph = parse_xdot(SIMPLE_XDOT)
        node_a = graph.get_node_by_name("a")
        ellipses = _shapes_of_type(node_a, EllipseShape)
        assert len(ellipses) >= 1
        e = ellipses[0]
        assert e.x0 == node_a.x
        assert e.y0 == node_a.y

    def test_edge_points_transformed(self):
        """Edge Bézier control points have Y flipped."""
        graph = parse_xdot(SIMPLE_XDOT)
        edge = graph.edges[0]
        beziers = _shapes_of_type(edge, BezierShape)
        assert len(beziers) >= 1
        # All edge points in SIMPLE_XDOT have raw y=100, so transformed y=50.
        for point in beziers[0].points:
            assert point[1] == 50


# ---------------------------------------------------------------------------
# Tests: Shape parsing
# ---------------------------------------------------------------------------

class TestShapeParsing:
    """Test that draw commands produce the correct shape objects."""

    def test_node_has_ellipse(self):
        """Node with 'e' draw command produces an EllipseShape."""
        graph = parse_xdot(SIMPLE_XDOT)
        node_a = graph.get_node_by_name("a")
        ellipses = _shapes_of_type(node_a, EllipseShape)
        assert len(ellipses) == 1  # lowercase 'e' -> one unfilled ellipse

    def test_node_ellipse_dimensions(self):
        """Ellipse has correct radii from the draw command."""
        graph = parse_xdot(SIMPLE_XDOT)
        node_a = graph.get_node_by_name("a")
        ellipses = _shapes_of_type(node_a, EllipseShape)
        e = ellipses[0]
        assert e.w == 27
        assert e.h == 18

    def test_node_ellipse_not_filled(self):
        """Lowercase 'e' produces an unfilled ellipse."""
        graph = parse_xdot(SIMPLE_XDOT)
        node_a = graph.get_node_by_name("a")
        ellipses = _shapes_of_type(node_a, EllipseShape)
        assert not ellipses[0].filled

    def test_node_has_text(self):
        """Node with 'T' draw command produces a TextShape."""
        graph = parse_xdot(SIMPLE_XDOT)
        node_a = graph.get_node_by_name("a")
        texts = _shapes_of_type(node_a, TextShape)
        assert len(texts) == 1

    def test_node_text_content(self):
        """TextShape contains the correct text string."""
        graph = parse_xdot(SIMPLE_XDOT)
        node_a = graph.get_node_by_name("a")
        texts = _shapes_of_type(node_a, TextShape)
        assert texts[0].t == "a"

        node_b = graph.get_node_by_name("b")
        texts = _shapes_of_type(node_b, TextShape)
        assert texts[0].t == "b"

    def test_edge_has_bezier(self):
        """Edge with 'B' draw command produces a BezierShape."""
        graph = parse_xdot(SIMPLE_XDOT)
        edge = graph.edges[0]
        beziers = _shapes_of_type(edge, BezierShape)
        assert len(beziers) >= 1

    def test_edge_bezier_control_points(self):
        """Edge Bézier has 4 control points matching the pos attribute."""
        graph = parse_xdot(SIMPLE_XDOT)
        edge = graph.edges[0]
        beziers = _shapes_of_type(edge, BezierShape)
        b = beziers[0]
        assert len(b.points) == 4
        # X coords should match: 77, 95, 113, 123
        xs = [p[0] for p in b.points]
        assert xs == [77, 95, 113, 123]

    def test_edge_has_arrowhead(self):
        """Edge with 'P' draw command (in _hdraw_) produces a PolygonShape arrowhead."""
        graph = parse_xdot(SIMPLE_XDOT)
        edge = graph.edges[0]
        polygons = _shapes_of_type(edge, PolygonShape)
        # _hdraw_ contains "P 3 ..." -> filled polygon + outline = 2 PolygonShapes
        assert len(polygons) == 2

    def test_edge_arrowhead_vertices(self):
        """Arrowhead polygon has 3 vertices (triangle)."""
        graph = parse_xdot(SIMPLE_XDOT)
        edge = graph.edges[0]
        polygons = _shapes_of_type(edge, PolygonShape)
        filled = [p for p in polygons if p.filled]
        assert len(filled) == 1
        assert len(filled[0].points) == 3


# ---------------------------------------------------------------------------
# Tests: Color parsing
# ---------------------------------------------------------------------------

class TestColorParsing:
    """Test color parsing from xdot draw commands.

    Colors in the xdot format can be specified as:
    - Hex: #RRGGBB or #RRGGBBAA
    - Named: X11 color names
    - HSV: "H,S,V"
    - ColorBrewer: /scheme/index (not tested here; rare in practice)
    """

    def test_hex_color_rgb(self):
        """Hex #RRGGBB is parsed to correct RGBA floats.

        Uses #1a4b7c where all components are nonzero and distinct,
        to catch channel swizzle bugs (RGB vs BGR, etc.).
        """
        graph = parse_xdot(COLOR_HEX_XDOT)
        node = graph.nodes[0]
        ellipses = _shapes_of_type(node, EllipseShape)
        color = ellipses[0].pen.color

        assert len(color) == 4
        assert _approx(color[0], 0x1a / 255.0)  # R
        assert _approx(color[1], 0x4b / 255.0)  # G
        assert _approx(color[2], 0x7c / 255.0)  # B
        assert _approx(color[3], 1.0)            # A (default, no alpha in input)

    def test_hex_color_rgba(self):
        """Hex #RRGGBBAA is parsed with correct alpha."""
        graph = parse_xdot(COLOR_HEX_ALPHA_XDOT)
        node = graph.nodes[0]
        ellipses = _shapes_of_type(node, EllipseShape)
        color = ellipses[0].pen.color

        assert _approx(color[0], 0x1a / 255.0)
        assert _approx(color[1], 0x4b / 255.0)
        assert _approx(color[2], 0x7c / 255.0)
        assert _approx(color[3], 0x80 / 255.0)  # ~0.502

    def test_hsv_color(self):
        """HSV color is converted to correct RGB.

        Uses the HSV equivalent of #1a4b7c to verify the conversion
        produces the same result as the hex path.
        """
        graph = parse_xdot(COLOR_HSV_XDOT)
        node = graph.nodes[0]
        ellipses = _shapes_of_type(node, EllipseShape)
        color = ellipses[0].pen.color

        # Should match #1a4b7c after HSV->RGB conversion.
        assert _approx(color[0], 0x1a / 255.0)  # R
        assert _approx(color[1], 0x4b / 255.0)  # G
        assert _approx(color[2], 0x7c / 255.0)  # B
        assert _approx(color[3], 1.0)            # A (HSV has no alpha; defaults to 1.0)

    def test_named_color(self):
        """Named X11 color 'red' is parsed correctly."""
        graph = parse_xdot(COLOR_NAMED_XDOT)
        node = graph.nodes[0]
        ellipses = _shapes_of_type(node, EllipseShape)
        color = ellipses[0].pen.color

        assert color == (1.0, 0.0, 0.0, 1.0)

    def test_fill_color(self):
        """The 'C' opcode sets fillcolor (distinct from stroke 'c')."""
        # The SIMPLE_XDOT arrowhead uses both: "c 7 -#000000 C 7 -#000000"
        graph = parse_xdot(SIMPLE_XDOT)
        edge = graph.edges[0]
        polygons = _shapes_of_type(edge, PolygonShape)
        filled = [p for p in polygons if p.filled]
        assert len(filled) >= 1
        assert filled[0].pen.fillcolor == (0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------

class TestParseError:
    """Test parser error handling."""

    def test_malformed_input_raises(self):
        """Completely invalid input raises ParseError."""
        with pytest.raises(ParseError):
            parse_xdot("this is not a graph at all {{{")

    def test_missing_bounding_box(self):
        """Graph without bb attribute produces a valid (default-sized) graph."""
        # No bb -> parser should still produce a Graph, just with default dimensions.
        xdot = "digraph G { a; }"
        graph = parse_xdot(xdot)
        assert isinstance(graph, Graph)

    def test_parse_error_str(self):
        """ParseError has a meaningful string representation."""
        err = ParseError("bad token", "test.dot", 5, 10)
        s = str(err)
        assert "bad token" in s
        assert "test.dot" in s


# ---------------------------------------------------------------------------
# Tests: Graph text search
#
# These test Graph.filter_items_by_text, which is part of the graph model
# rather than the parser per se, but lives here because it operates on
# parsed graph data.
# ---------------------------------------------------------------------------

class TestGraphSearch:
    """Test graph text search functionality."""

    def test_search_finds_node(self):
        """Searching for a node's text label returns that node."""
        graph = parse_xdot(SIMPLE_XDOT)
        results = graph.filter_items_by_text("a")
        # Node "a" should be in results
        node_names = {r.internal_name for r in results if isinstance(r, Node)}
        assert "a" in node_names

    def test_empty_search(self):
        """Empty search string returns no results."""
        graph = parse_xdot(SIMPLE_XDOT)
        results = graph.filter_items_by_text("")
        assert len(results) == 0

    def test_case_insensitive_lowercase_query(self):
        """Lowercase query matches case-insensitively (Emacs smart-case)."""
        graph = parse_xdot(SIMPLE_XDOT)
        results = graph.filter_items_by_text("a")
        node_names = {r.internal_name for r in results if isinstance(r, Node)}
        assert "a" in node_names

    def test_case_sensitive_uppercase_query(self):
        """Query containing uppercase is case-sensitive (Emacs smart-case).

        Node labels in SIMPLE_XDOT are lowercase "a" and "b", so an
        uppercase query "A" should not match.

        NOTE: This test documents the intended behavior. The current
        implementation unconditionally lowercases both sides, so this
        will fail until filter_items_by_text is fixed to use smart-case.
        """
        graph = parse_xdot(SIMPLE_XDOT)
        results = graph.filter_items_by_text("A")
        assert len(results) == 0

    def test_no_match(self):
        """Search for nonexistent text returns empty list."""
        graph = parse_xdot(SIMPLE_XDOT)
        results = graph.filter_items_by_text("xyznonexistent")
        assert len(results) == 0
