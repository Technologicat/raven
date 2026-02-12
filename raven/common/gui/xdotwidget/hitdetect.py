"""Hit detection for mouse interaction with graph elements.

This module provides functions for determining which graph element
(if any) is under the mouse cursor.
"""

__all__ = ["hit_test", "hit_test_screen"]

from typing import List, Optional

from .graph import Graph, Element, Node, Edge, PolygonShape
from .viewport import Viewport

# --------------------------------------------------------------------------------
# in graph coordinates

def hit_test(graph: Graph, gx: float, gy: float) -> Optional[Element]:
    """Find the element at a point in graph coordinates.

    `graph`: The Graph to test.
    `gx`, `gy`: Point in graph coordinates.

    Returns the topmost Element (Node or Edge) at the point, or None.
    Nodes are checked first (they are drawn on top of edges).
    """
    # Check nodes first (they appear on top)
    if node := get_node(graph, gx, gy):
        return node
    # Check edges (endpoint detection)
    if edge := get_edge(graph, gx, gy):
        return edge
    return None

def get_node(graph: Graph, gx: float, gy: float) -> Optional[Node]:
    for node in graph.nodes:
        if node.is_inside(gx, gy):
            return node
    return None

_EDGE_HIT_RADIUS_PX = 10  # detection radius in screen pixels


def _point_to_segment_dist_sq(px: float, py: float,
                               x1: float, y1: float,
                               x2: float, y2: float) -> float:
    """Squared distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return (px - x1) ** 2 + (py - y1) ** 2
    # Project point onto segment, clamped to [0, 1]
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


def _point_in_polygon(px: float, py: float, polygon: List[tuple]) -> bool:
    """Ray-casting point-in-polygon test.

    Returns True if (px, py) is inside the polygon defined by the vertex list.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Check if the ray from (px, py) going rightward crosses this edge
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def get_edge(graph: Graph, gx: float, gy: float,
             radius_sq: float = _EDGE_HIT_RADIUS_PX ** 2) -> Optional[Edge]:
    """Find an edge near point (gx, gy).

    Checks proximity to the edge path (polyline/bezier control points)
    and also whether the point is inside any filled polygon shapes
    (e.g. arrowheads).

    `radius_sq`: Squared hit radius, in the same coordinate system as (gx, gy).
    """
    for edge in graph.edges:
        # Check proximity to edge path
        pts = edge.points
        for i in range(len(pts) - 1):
            if _point_to_segment_dist_sq(gx, gy,
                                          pts[i][0], pts[i][1],
                                          pts[i + 1][0], pts[i + 1][1]) <= radius_sq:
                return edge
        # Check if point is inside any filled polygon (arrowheads)
        for shape in edge.shapes:
            if isinstance(shape, PolygonShape) and shape.filled and shape.points:
                if _point_in_polygon(gx, gy, shape.points):
                    return edge
    return None

# --------------------------------------------------------------------------------
# same, in screen coordinates

def hit_test_screen(graph: Graph,
                    viewport: Viewport,
                    sx: float,
                    sy: float) -> Optional[Element]:
    """Find the element at a point in screen coordinates.

    `graph`: The Graph to test.
    `viewport`: Viewport for coordinate transformation.
    `sx`, `sy`: Point in screen coordinates (pixels).

    Returns the topmost Element (Node or Edge) at the point, or None.
    """
    gx, gy = viewport.screen_to_graph(sx, sy)
    # Check nodes first (they appear on top)
    if node := get_node(graph, gx, gy):
        return node
    # Check edges with zoom-corrected pixel radius.
    # Convert pixel radius to graph coordinates: graph_dist = px_dist / zoom.
    z = viewport.zoom.current
    if z > 0:
        graph_radius = _EDGE_HIT_RADIUS_PX / z
    else:
        graph_radius = _EDGE_HIT_RADIUS_PX
    if edge := get_edge(graph, gx, gy, radius_sq=graph_radius ** 2):
        return edge
    return None


def get_node_screen(graph: Graph,
                    viewport: Viewport,
                    sx: float,
                    sy: float) -> Optional[Node]:
    """Find a node at a point in screen coordinates.

    `graph`: The Graph to test.
    `viewport`: Viewport for coordinate transformation.
    `sx`, `sy`: Point in screen coordinates (pixels).

    Returns the Node at the point, or None if no node is there.
    """
    gx, gy = viewport.screen_to_graph(sx, sy)
    return get_node(graph, gx, gy)


def get_edge_screen(graph: Graph,
                    viewport: Viewport,
                    sx: float,
                    sy: float) -> Optional[Edge]:
    """Find an edge at a point in screen coordinates.

    `graph`: The Graph to test.
    `viewport`: Viewport for coordinate transformation.
    `sx`, `sy`: Point in screen coordinates (pixels).

    Returns the Edge near the point (within pixel radius), or None.
    """
    gx, gy = viewport.screen_to_graph(sx, sy)
    z = viewport.zoom.current
    if z > 0:
        graph_radius = _EDGE_HIT_RADIUS_PX / z
    else:
        graph_radius = _EDGE_HIT_RADIUS_PX
    return get_edge(graph, gx, gy, radius_sq=graph_radius ** 2)
