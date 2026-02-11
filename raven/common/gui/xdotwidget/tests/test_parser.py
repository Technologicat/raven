"""Tests for the xdot parser."""

import pytest

from ..parser import parse_xdot, XDotParser, ParseError
from ..graph import Graph, Node, Edge, TextShape, EllipseShape


# Simple test graph in xdot format
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


class TestParser:
    """Test the XDot parser."""

    def test_parse_simple_graph(self):
        """Test parsing a simple graph."""
        graph = parse_xdot(SIMPLE_XDOT)

        assert isinstance(graph, Graph)
        assert graph.width == 200
        assert graph.height == 150

    def test_parse_nodes(self):
        """Test that nodes are parsed correctly."""
        graph = parse_xdot(SIMPLE_XDOT)

        assert len(graph.nodes) == 2

        # Check node names
        names = {n.internal_name for n in graph.nodes}
        assert names == {"a", "b"}

    def test_parse_edges(self):
        """Test that edges are parsed correctly."""
        graph = parse_xdot(SIMPLE_XDOT)

        assert len(graph.edges) == 1

        edge = graph.edges[0]
        assert edge.src.internal_name == "a"
        assert edge.dst.internal_name == "b"

    def test_parse_node_positions(self):
        """Test that node positions are correct."""
        graph = parse_xdot(SIMPLE_XDOT)

        node_a = graph.get_node_by_name("a")
        node_b = graph.get_node_by_name("b")

        assert node_a is not None
        assert node_b is not None

        # Note: Y is flipped in the transform
        assert node_a.x == 50
        assert node_b.x == 150

    def test_parse_shapes(self):
        """Test that shapes are parsed."""
        graph = parse_xdot(SIMPLE_XDOT)

        node_a = graph.get_node_by_name("a")
        assert node_a is not None

        # Should have shapes (ellipse and text)
        assert len(node_a.shapes) > 0

    def test_empty_graph(self):
        """Test parsing an empty graph."""
        xdot = """digraph G { graph [bb="0,0,100,100"]; }"""
        graph = parse_xdot(xdot)

        assert graph.width == 100
        assert graph.height == 100
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_nodes_by_name(self):
        """Test the nodes_by_name lookup."""
        graph = parse_xdot(SIMPLE_XDOT)

        assert graph.get_node_by_name("a") is not None
        assert graph.get_node_by_name("b") is not None
        assert graph.get_node_by_name("nonexistent") is None


class TestColorParsing:
    """Test color parsing."""

    def test_hex_color(self):
        """Test parsing hex colors."""
        xdot = """
        digraph G {
            graph [bb="0,0,100,100"];
            a [pos="50,50", width="0.5", height="0.5",
               _draw_="c 7 -#ff0000 e 50 50 18 18 "];
        }
        """
        graph = parse_xdot(xdot)
        assert len(graph.nodes) == 1

    def test_named_color(self):
        """Test parsing named colors."""
        xdot = """
        digraph G {
            graph [bb="0,0,100,100"];
            a [pos="50,50", width="0.5", height="0.5",
               _draw_="c 3 -red e 50 50 18 18 "];
        }
        """
        graph = parse_xdot(xdot)
        assert len(graph.nodes) == 1


class TestGraphSearch:
    """Test graph search functionality."""

    def test_filter_items_by_text(self):
        """Test fragment search."""
        graph = parse_xdot(SIMPLE_XDOT)

        # Search for "a"
        results = graph.filter_items_by_text("a")
        assert len(results) >= 1

    def test_empty_search(self):
        """Test empty search returns empty list."""
        graph = parse_xdot(SIMPLE_XDOT)
        results = graph.filter_items_by_text("")
        assert len(results) == 0

    def test_case_insensitive_search(self):
        """Test that search is case-insensitive."""
        graph = parse_xdot(SIMPLE_XDOT)

        results_lower = graph.filter_items_by_text("a")
        results_upper = graph.filter_items_by_text("A")

        # Both should find the same elements
        assert len(results_lower) == len(results_upper)
