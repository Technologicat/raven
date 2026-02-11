"""Graph data model for xdot visualization.

This module defines the data structures for representing xdot graphs:
- Pen: Drawing state (colors, line width, font)
- Shape classes: TextShape, EllipseShape, PolygonShape, LineShape, BezierShape
- Element classes: Node, Edge (graph components)
- Graph: Container for nodes, edges, and background shapes

Adapted from xdottir (https://github.com/Technologicat/xdottir),
which in turn was adapted from XDot by José Fonseca.
"""

__all__ = ["Pen", "Shape", "TextShape", "EllipseShape", "PolygonShape",
           "LineShape", "BezierShape", "CompoundShape",
           "Element", "Node", "Edge", "Graph",
           "Url", "Jump",
           "mix_colors", "square_distance"]

from itertools import chain
from typing import Dict, List, Optional, Set, Tuple

from .constants import Color, Point


def mix_colors(rgb1: Color, rgb2: Color, t: float) -> Color:
    """Mix two RGBA colors.

    The formula is::

        out = (1 - t) * rgb1  +  t * rgb2

    where `t` is in [0, 1].

    This is Porter-Duff 'over' with opaque background.
    """
    R1, G1, B1, A1 = rgb1
    R2, G2, B2, A2 = rgb2
    R = (1.0 - t) * R1 + t * R2
    G = (1.0 - t) * G1 + t * G2
    B = (1.0 - t) * B1 + t * B2
    A = (1.0 - t) * A1 + t * A2
    return (R, G, B, A)


def square_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return squared Euclidean distance between two points."""
    deltax = x2 - x1
    deltay = y2 - y1
    return deltax * deltax + deltay * deltay


# Default highlight color (GNOME 2.30.2 blue)
# Can be overridden via `set_highlight_colors`.
# Currently global for the whole process.
_highlight_base: Color = (0.5256, 0.6708, 0.8511, 1.0)
_highlight_light: Color = (0.8234, 0.8871, 0.9662, 1.0)


def set_highlight_colors(base: Color, light: Color) -> None:
    """Set the highlight colors used for selected/hovered elements.

    `base`: The primary highlight color (for strokes).
    `light`: A lighter version (for fills).

    Both are RGBA tuples with values in [0, 1].
    """
    global _highlight_base, _highlight_light
    _highlight_base = base
    _highlight_light = light


def get_highlight_colors() -> Tuple[Color, Color]:
    """Return the current (base, light) highlight colors."""
    return _highlight_base, _highlight_light


class Pen:
    """Store pen attributes for drawing.

    Attributes:
        color: Stroke color, RGBA tuple with values in [0, 1].
        fillcolor: Fill color, RGBA tuple with values in [0, 1].
        linewidth: Line width in points.
        fontsize: Font size in pixels.
        dash: Dash pattern tuple (empty for solid line).
    """

    def __init__(self):
        self.color: Color = (0.0, 0.0, 0.0, 1.0)
        self.fillcolor: Color = (0.0, 0.0, 0.0, 1.0)
        self.linewidth: float = 1.0
        self.fontsize: float = 14.0
        self.dash: Tuple[float, ...] = ()

    def copy(self) -> "Pen":
        """Create and return a copy of this pen."""
        pen = Pen()
        pen.color = self.color
        pen.fillcolor = self.fillcolor
        pen.linewidth = self.linewidth
        pen.fontsize = self.fontsize
        pen.dash = self.dash
        return pen

    def highlighted_initial(self) -> "Pen":
        """Return a new pen with the initial (start-of-animation) highlight color."""
        pen = self.copy()
        pen.color = _highlight_base
        pen.fillcolor = _highlight_light
        return pen

    def highlighted_final(self) -> "Pen":
        """Return a new pen with the final (end-of-animation) highlight color.

        This mixes the app highlight color with the pen's own color,
        so the original color is still recognizable.
        """
        pen = self.copy()
        pen.color = mix_colors(_highlight_base, self.color, 0.3)
        pen.fillcolor = mix_colors(_highlight_light, self.fillcolor, 0.3)
        return pen

    @staticmethod
    def mix(tgt: "Pen", pen1: "Pen", pen2: "Pen", t: float) -> None:
        """Mix colors of pen1 and pen2, saving result to tgt.

        t in [0, 1]: mix result is (1 - t) * pen1 + t * pen2.
        """
        tgt.color = mix_colors(pen1.color, pen2.color, t)
        tgt.fillcolor = mix_colors(pen1.fillcolor, pen2.fillcolor, t)


class Shape:
    """Abstract base class for all drawing shapes."""

    def __init__(self):
        self.pen: Optional[Pen] = None

    def get_bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        """Return (x1, y1, x2, y2) bounding box, or None if not applicable."""
        return None


class TextShape(Shape):
    """Text label shape.

    Attributes:
        pen: Drawing pen.
        x, y: Position (baseline).
        j: Justification (-1=left, 0=center, 1=right). Parsed, but not supported; we always left-justify.
        w: Expected width (for scaling to fit).
        t: The text content.
    """

    LEFT, CENTER, RIGHT = -1, 0, 1

    def __init__(self, pen: Pen, x: float, y: float, j: int, w: float, t: str):
        super().__init__()
        self.pen = pen.copy()
        self.x = x
        self.y = y
        self.j = j
        self.w = w
        self.t = t

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        # Approximate bounding box based on position and width
        # Height is estimated from font size
        h = self.pen.fontsize
        if self.j == self.LEFT:
            x1 = self.x
            x2 = self.x + self.w
        elif self.j == self.CENTER:
            x1 = self.x - self.w / 2
            x2 = self.x + self.w / 2
        else:  # RIGHT
            x1 = self.x - self.w
            x2 = self.x
        y1 = self.y - h
        y2 = self.y
        return (x1, y1, x2, y2)


class EllipseShape(Shape):
    """Ellipse shape.

    Attributes:
        pen: Drawing pen.
        x0, y0: Center position.
        w, h: Width and height (radii).
        filled: Whether to fill the ellipse.
    """

    def __init__(self, pen: Pen, x0: float, y0: float, w: float, h: float, filled: bool = False):
        super().__init__()
        self.pen = pen.copy()
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.filled = filled

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        return (self.x0 - self.w, self.y0 - self.h,
                self.x0 + self.w, self.y0 + self.h)


class PolygonShape(Shape):
    """Polygon shape.

    Attributes:
        pen: Drawing pen.
        points: List of (x, y) vertices.
        filled: Whether to fill the polygon.
    """

    def __init__(self, pen: Pen, points: List[Point], filled: bool = False):
        super().__init__()
        self.pen = pen.copy()
        self.points = points
        self.filled = filled

    def get_bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


class LineShape(Shape):
    """Polyline shape (not closed).

    Attributes:
        pen: Drawing pen.
        points: List of (x, y) vertices.
    """

    def __init__(self, pen: Pen, points: List[Point]):
        super().__init__()
        self.pen = pen.copy()
        self.points = points

    def get_bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


class BezierShape(Shape):
    """Bezier curve shape.

    Attributes:
        pen: Drawing pen.
        points: Control points [start, ctrl1, ctrl2, end, ctrl1, ctrl2, end, ...].
        filled: Whether to fill the shape.
    """

    def __init__(self, pen: Pen, points: List[Point], filled: bool = False):
        super().__init__()
        self.pen = pen.copy()
        self.points = points
        self.filled = filled

    def get_bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


class CompoundShape(Shape):
    """Container for multiple shapes."""

    def __init__(self, shapes: List[Shape]):
        super().__init__()
        self.shapes = shapes

    def get_bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        boxes = [s.get_bounding_box() for s in self.shapes]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return (x1, y1, x2, y2)


class Url:
    """URL associated with a graph element."""

    def __init__(self, item: "Element", url: str, highlight: Optional[Set["Element"]] = None):
        self.item = item
        self.url = url
        if highlight is None:
            highlight = {item}
        self.highlight = highlight


class Jump:
    """Jump target for graph navigation."""

    def __init__(self, item: "Element", x: float, y: float, highlight: Optional[Set["Element"]] = None):
        self.item = item
        self.x = x
        self.y = y
        if highlight is None:
            highlight = {item}
        self.highlight = highlight


class Element(CompoundShape):
    """Base class for graph nodes and edges."""

    def __init__(self, shapes: List[Shape]):
        super().__init__(shapes)

    def get_url(self, x: float, y: float) -> Optional[Url]:
        """Return URL if point (x, y) is inside this element, else None."""
        return None

    def get_jump(self, x: float, y: float, **kwargs) -> Optional[Jump]:
        """Return Jump target if point (x, y) is inside this element, else None."""
        return None

    def get_texts(self) -> List[str]:
        """Return text content of any TextShapes in this element."""
        return [s.t for s in self.shapes if isinstance(s, TextShape)]


class Node(Element):
    """Graph node.

    Attributes:
        x, y: Center position.
        x1, y1, x2, y2: Bounding box corners (computed from x, y, w, h).
        shapes: Drawing shapes.
        url: Optional URL associated with this node.
        internal_name: Node ID from the graph.
    """

    def __init__(self, x: float, y: float, w: float, h: float,
                 shapes: List[Shape], url: Optional[str] = None,
                 internal_name: Optional[str] = None):
        super().__init__(shapes)
        self.x = x
        self.y = y
        self.x1 = x - 0.5 * w
        self.y1 = y - 0.5 * h
        self.x2 = x + 0.5 * w
        self.y2 = y + 0.5 * h
        self.url = url
        self.internal_name = internal_name

    def is_inside(self, x: float, y: float) -> bool:
        """Return whether point (x, y) is inside this node's bounding box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def get_url(self, x: float, y: float) -> Optional[Url]:
        if self.url is None:
            return None
        if self.is_inside(x, y):
            return Url(self, self.url)
        return None

    def get_jump(self, x: float, y: float, **kwargs) -> Optional[Jump]:
        if self.is_inside(x, y):
            return Jump(self, self.x, self.y)
        return None

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class Edge(Element):
    """Graph edge.

    Attributes:
        src: Source node.
        dst: Destination node.
        points: Edge path coordinates.
        shapes: Drawing shapes (line, arrows, labels).
    """

    CLICK_RADIUS = 10  # Click detection radius for endpoints

    def __init__(self, src: Node, dst: Node, points: List[Point], shapes: List[Shape]):
        super().__init__(shapes)
        self.src = src
        self.dst = dst
        self.points = points

    def get_jump(self, x: float, y: float, **kwargs) -> Optional[Jump]:
        """Return Jump to connected node if clicking near an endpoint."""
        if self.points:
            # Click near start -> jump to destination
            if square_distance(x, y, *self.points[0]) <= self.CLICK_RADIUS * self.CLICK_RADIUS:
                return Jump(self, self.dst.x, self.dst.y,
                            highlight={self, self.dst})
            # Click near end -> jump to source
            if square_distance(x, y, *self.points[-1]) <= self.CLICK_RADIUS * self.CLICK_RADIUS:
                return Jump(self, self.src.x, self.src.y,
                            highlight={self, self.src})
        return None


class Graph:
    """Container for a complete graph.

    Attributes:
        width, height: Graph dimensions.
        shapes: Background shapes.
        nodes: List of Node objects.
        edges: List of Edge objects.
        nodes_by_name: Dict mapping node names to Node objects.
    """

    def __init__(self, width: float = 1, height: float = 1,
                 shapes: Optional[List[Shape]] = None,
                 nodes: Optional[List[Node]] = None,
                 edges: Optional[List[Edge]] = None):
        self.width = width
        self.height = height
        self.shapes = shapes or []
        self.nodes = nodes or []
        self.edges = edges or []

        # Build lookup tables
        self.nodes_by_name: Dict[str, Node] = {}
        for n in self.nodes:
            if n.internal_name:
                self.nodes_by_name[n.internal_name] = n

        # Pre-compute search data (lowercase text for case-insensitive search)
        # Format: [(element, "all text in element lowercase")]
        self._items_and_texts: List[Tuple[Element, str]] = [
            (x, " ".join(x.get_texts()).lower())
            for x in chain(self.nodes, self.edges)
        ]

    def get_size(self) -> Tuple[float, float]:
        """Return (width, height) of the graph."""
        return self.width, self.height

    def get_url(self, x: float, y: float) -> Optional[Url]:
        """Return URL at point (x, y), or None."""
        for node in self.nodes:
            url = node.get_url(x, y)
            if url is not None:
                return url
        return None

    def get_jump(self, x: float, y: float,
                 highlight_linked_nodes: Optional[str] = None) -> Optional[Jump]:
        """Return Jump target at point (x, y), or None.

        `highlight_linked_nodes`: One of None, "from", "to", "to_links_only".
            Controls which linked nodes/edges to include in the highlight set
            of the Jump.
        """
        # Check edges first (for endpoint clicks)
        for edge in self.edges:
            jump = edge.get_jump(x, y)
            if jump is not None:
                return jump

        # Check nodes
        for node in self.nodes:
            jump = node.get_jump(x, y)
            if jump is not None:
                # Optionally highlight linked edges
                if highlight_linked_nodes is None:
                    pass
                elif highlight_linked_nodes == "from":
                    linked_edges = [e for e in self.edges if e.src == node]
                else:  # "to" or "to_links_only"
                    linked_edges = [e for e in self.edges if e.dst == node]
                jump.highlight.update(linked_edges)

                # Optionally highlight linked nodes
                if highlight_linked_nodes == "from":
                    linked_nodes = [e.dst for e in linked_edges]
                    jump.highlight.update(linked_nodes)
                elif highlight_linked_nodes == "to":
                    linked_nodes = [e.src for e in linked_edges]
                    jump.highlight.update(linked_nodes)

                return jump
        return None

    def filter_items_by_text(self, text: str) -> List[Element]:
        """Return nodes/edges containing all fragments of the search text.

        Uses fragment search (like Emacs HELM): "cat photo" matches "photocatalytic".
        Search is case-insensitive.
        """
        if not text:
            return []

        fragments = [fragment.lower() for fragment in text.split()]

        def match_text(texts: str) -> bool:
            return all(texts.find(fragment) != -1 for fragment in fragments)

        return [item for item, texts in self._items_and_texts if match_text(texts)]

    def get_node_by_name(self, name: str) -> Optional[Node]:
        """Return node by its internal name, or None."""
        return self.nodes_by_name.get(name)

    def get_all_elements(self) -> List[Element]:
        """Return all nodes and edges."""
        return list(chain(self.nodes, self.edges))
