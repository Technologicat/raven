"""Graph data model for xdot visualization.

This module defines the data structures for representing xdot graphs:
- Pen: Drawing state (colors, line width, font)
- Shape classes: TextShape, EllipseShape, PolygonShape, LineShape, BezierShape, ImageShape
- Element classes: Node, Edge (graph components)
- Graph: Container for nodes, edges, and background shapes

Adapted from xdottir (https://github.com/Technologicat/xdottir). Copyright 2008 Jose Fonseca, 2012-2019 Juha
Jeronen, and the xdottir contributors. LGPL-3.0-or-later, not Raven's usual BSD - see the package docstring
in `__init__.py` and `LICENSE`.

Adapted from xdottir (https://github.com/Technologicat/xdottir),
which in turn was adapted from XDot by José Fonseca.
"""

__all__ = ["mix_colors",
           "tessellate_bezier",
           "set_highlight_colors",
           "get_highlight_colors",
           "Pen",
           "Shape",
           "TextShape",
           "EllipseShape",
           "PolygonShape",
           "LineShape",
           "BezierShape",
           "MipLevel",
           "ImageShape",
           "union_of_boxes",
           "CompoundShape",
           "Element",
           "Node",
           "Edge",
           "Graph"]

from collections.abc import Iterable, Iterator, Sequence
from itertools import chain
from typing import NamedTuple

from ... import utils as common_utils

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


def tessellate_bezier(points: list["Point"], n: int = 10) -> list["Point"]:
    """Tessellate cubic bezier segments into a polyline.

    `points`: Bezier control points [P0, C1, C2, P1, C1, C2, P1, ...].
    `n`: Number of line segments per cubic bezier segment.

    Returns a list of points along the curve.
    """
    if len(points) < 4:
        return list(points)
    result = []
    p0 = points[0]
    for i in range(1, len(points), 3):
        if i + 2 >= len(points):
            break
        c1 = points[i]
        c2 = points[i + 1]
        p1 = points[i + 2]
        # B(t) = (1-t)^3 P0 + 3(1-t)^2 t C1 + 3(1-t) t^2 C2 + t^3 P1
        start = 0 if not result else 1  # skip first point if continuing
        for j in range(start, n + 1):
            t = j / n
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt
            t2 = t * t
            t3 = t2 * t
            x = mt3 * p0[0] + 3 * mt2 * t * c1[0] + 3 * mt * t2 * c2[0] + t3 * p1[0]
            y = mt3 * p0[1] + 3 * mt2 * t * c1[1] + 3 * mt * t2 * c2[1] + t3 * p1[1]
            result.append((x, y))
        p0 = p1
    return result


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


def get_highlight_colors() -> tuple[Color, Color]:
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
        bold, italic: Which face to draw text in. A renderer that was given no font for the face falls
                      back to the regular one, so these are safe to set whatever the caller loaded.
                      A parsed xdot graph never sets them: the format's font opcode carries a family
                      name, and this widget does not resolve one.
        keep_color: Draw text in exactly `color`, rather than in whatever dark mode would pick for
                    contrast against the element's fill. Set it where the colour *is* the message — a
                    search match painted red, a label quoting something other than what was said — and
                    where the caller has already chosen that colour knowing what it will sit on.

                    Off by default, because the contrast rule earns its place on the graphs it was
                    written for: a parsed one carries whatever colours its author chose for paper, and
                    inverting those can put near-white text on a mid-lightness fill.
    """

    DEFAULT_COLOR: Color = (0.0, 0.0, 0.0, 1.0)
    DEFAULT_FILLCOLOR: Color = (0.0, 0.0, 0.0, 1.0)

    def __init__(self):
        self.color: Color = Pen.DEFAULT_COLOR
        self.fillcolor: Color = Pen.DEFAULT_FILLCOLOR
        self.linewidth: float = 1.0
        self.fontsize: float = 14.0
        self.dash: tuple[float, ...] = ()
        self.bold: bool = False
        self.italic: bool = False
        self.keep_color: bool = False

    def copy(self) -> "Pen":
        """Create and return a copy of this pen."""
        pen = Pen()
        pen.color = self.color
        pen.fillcolor = self.fillcolor
        pen.linewidth = self.linewidth
        pen.fontsize = self.fontsize
        pen.dash = self.dash
        pen.bold = self.bold
        pen.italic = self.italic
        pen.keep_color = self.keep_color
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
        """Mix pen1 and pen2, saving result to tgt.

        t in [0, 1]: mix result is (1 - t) * pen1 + t * pen2, for the colors, the line width and the font
        size. The dash pattern has no in-between, and is left as `tgt` has it.
        """
        tgt.color = mix_colors(pen1.color, pen2.color, t)
        tgt.fillcolor = mix_colors(pen1.fillcolor, pen2.fillcolor, t)
        tgt.linewidth = pen1.linewidth + (pen2.linewidth - pen1.linewidth) * t
        tgt.fontsize = pen1.fontsize + (pen2.fontsize - pen1.fontsize) * t


class Shape:
    """Abstract base class for all drawing shapes."""

    def __init__(self):
        self.pen: Pen | None = None

    def get_bounding_box(self) -> tuple[float, float, float, float] | None:
        """Return (x1, y1, x2, y2) bounding box, or None if not applicable."""
        return None


class TextShape(Shape):
    """Text label shape.

    Attributes:
        pen: Drawing pen.
        x, y: Position (baseline).
        justify: `LEFT`, `CENTER` or `RIGHT` (-1, 0, 1), applied against `width`.
        width: Width of the text, in graph coordinates.
        text: The text content.
    """

    LEFT, CENTER, RIGHT = -1, 0, 1

    def __init__(self, pen: Pen, x: float, y: float, justify: int, width: float, text: str):
        super().__init__()
        self.pen = pen.copy()
        self.x = x
        self.y = y
        self.justify = justify
        self.width = width
        self.text = text

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        # Approximate bounding box based on position and width
        # Height is estimated from font size
        h = self.pen.fontsize
        if self.justify == self.LEFT:
            x1 = self.x
            x2 = self.x + self.width
        elif self.justify == self.CENTER:
            x1 = self.x - self.width / 2
            x2 = self.x + self.width / 2
        else:  # RIGHT
            x1 = self.x - self.width
            x2 = self.x
        y1 = self.y - h
        y2 = self.y
        return (x1, y1, x2, y2)


class EllipseShape(Shape):
    """Ellipse shape.

    Attributes:
        pen: Drawing pen.
        x0, y0: Center position.
        rx, ry: Horizontal and vertical radius.
        filled: Whether to fill the ellipse.
    """

    def __init__(self, pen: Pen, x0: float, y0: float, rx: float, ry: float, filled: bool = False):
        super().__init__()
        self.pen = pen.copy()
        self.x0 = x0
        self.y0 = y0
        self.rx = rx
        self.ry = ry
        self.filled = filled

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        return (self.x0 - self.rx, self.y0 - self.ry,
                self.x0 + self.rx, self.y0 + self.ry)


class PolygonShape(Shape):
    """Polygon shape.

    Attributes:
        pen: Drawing pen.
        points: List of (x, y) vertices.
        filled: Whether to fill the polygon.
    """

    def __init__(self, pen: Pen, points: list[Point], filled: bool = False):
        super().__init__()
        self.pen = pen.copy()
        self.points = points
        self.filled = filled

    def get_bounding_box(self) -> tuple[float, float, float, float] | None:
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

    def __init__(self, pen: Pen, points: list[Point]):
        super().__init__()
        self.pen = pen.copy()
        self.points = points

    def get_bounding_box(self) -> tuple[float, float, float, float] | None:
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

    def __init__(self, pen: Pen, points: list[Point], filled: bool = False):
        super().__init__()
        self.pen = pen.copy()
        self.points = points
        self.filled = filled

    def get_bounding_box(self) -> tuple[float, float, float, float] | None:
        if not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


class MipLevel(NamedTuple):
    """One level of an `ImageShape`'s mip chain: a texture, and the pixel size it holds.

    The size is here because the renderer chooses a level by comparing it against the size the picture is
    about to be drawn at, and this package holds no DPG at the data-model layer — so whoever prepared the
    texture says how big it is. Stated per level rather than as a scale against a native size, which is
    the other way Raven spells a chain (`raven.cherrypick.imageview`), because it saves every reader of a
    level the arithmetic.
    """

    width: int
    height: int
    texture: int | str


class ImageShape(Shape):
    """A bitmap, drawn from textures the caller has already registered with DPG.

    Attributes:
        levels: The mip chain, finest first, as `MipLevel`s. Empty to draw nothing.
        x1, y1, x2, y2: The rectangle to draw into, in graph coordinates.
        max_screen_size: Longest side the image may be drawn at, in screen pixels, or `None` for no cap.

    The rectangle is in graph coordinates like every other shape here, so the image pans and zooms with
    the drawing. `max_screen_size` then puts a ceiling on how large it is allowed to get. Shrinking to
    obey it is uniform and about the rectangle's centre, so the picture keeps both its proportions and
    its place.

    **A chain rather than one texture, because DPG samples nearest-neighbour.** Anything drawn at a size
    its texture was not prepared for aliases, and a graph zooms continuously, so there is no one size to
    prepare at: the renderer draws whichever level suits the size on screen. A Lanczos resampler is the
    usual way to build the levels; in this constellation that is `raven.common.image.lanczos.mipchain`.

    An asset shipped at one display size is the degenerate case and is spelled the same way: a chain of
    one, plus a `max_screen_size` that stops it being drawn larger than it is. A 64x64 icon is that.

    Prepare the finest level at the largest size the picture could reasonably be wanted at rather than at
    the size it is usually drawn: past that the renderer has nothing finer to reach for and DPG upsamples.
    **A larger preparation without the rest of the chain is worse than neither**, since drawing a 1024 px
    texture into a 55 px card is an eighteen-fold nearest-neighbour downsample — aliasing at the zoom
    people read at, to cure a softness at one they rarely reach.

    An empty chain is an ordinary state rather than a fault: it is what a caller draws while an image is
    still being prepared on a background thread, and it lets the shape carry the rectangle the image will
    occupy so that whatever the caller draws in the meantime is the right size.
    """

    def __init__(self, levels: Sequence[MipLevel],
                 x1: float, y1: float, x2: float, y2: float,
                 max_screen_size: float | None = None):
        super().__init__()
        self.levels = tuple(levels)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.max_screen_size = max_screen_size

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        return (min(self.x1, self.x2), min(self.y1, self.y2),
                max(self.x1, self.x2), max(self.y1, self.y2))


def union_of_boxes(boxes: Iterable[tuple[float, float, float, float] | None]
                   ) -> tuple[float, float, float, float] | None:
    """Return the smallest box enclosing all of `boxes`, ignoring `None`s. `None` if none is left."""
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


class CompoundShape(Shape):
    """Container for multiple shapes."""

    def __init__(self, shapes: list[Shape]):
        super().__init__()
        self.shapes = shapes

    def get_bounding_box(self) -> tuple[float, float, float, float] | None:
        return union_of_boxes(s.get_bounding_box() for s in self.shapes)


class Element(CompoundShape):
    """Base class for graph nodes and edges."""

    def __init__(self, shapes: list[Shape]):
        super().__init__(shapes)

    def get_texts(self) -> list[str]:
        """Return text content of any TextShapes in this element."""
        return [s.text for s in self.shapes if isinstance(s, TextShape)]

    def get_drawn_bounding_box(self) -> tuple[float, float, float, float] | None:
        """Return the box enclosing everything this element puts on screen.

        Which is a different question from `get_bounding_box`, and the one to ask about whether an
        element is worth drawing. A `Node`'s own box is the layout cell it occupies — what hit testing,
        anchoring and edge routing reason about — and a decoration deliberately drawn in the margin lies
        outside it: an icon straddling an edge, a deck of thumbnails hanging past a corner. Culling on the
        cell makes those vanish while they are still on screen, which is visible as soon as anything is
        zoomed in far enough for the margin to fill the view.

        For an `Edge`, whose box already is what it draws, the two agree.
        """
        return union_of_boxes([CompoundShape.get_bounding_box(self), self.get_bounding_box()])


class Node(Element):
    """Graph node.

    Attributes:
        x, y: Center position.
        x1, y1, x2, y2: Bounding box corners (computed from x, y, w, h).
        shapes: Drawing shapes.
        url: Optional URL associated with this node.
        internal_name: Node ID from the graph.
        tooltip: Optional tooltip text (from dot `tooltip` attribute).
    """

    def __init__(self, x: float, y: float, w: float, h: float,
                 shapes: list[Shape], url: str | None = None,
                 internal_name: str | None = None,
                 tooltip: str | None = None):
        super().__init__(shapes)
        self.x = x
        self.y = y
        self.x1 = x - 0.5 * w
        self.y1 = y - 0.5 * h
        self.x2 = x + 0.5 * w
        self.y2 = y + 0.5 * h
        self.url = url
        self.internal_name = internal_name
        self.tooltip = tooltip
        # The fill this node's text sits on, where the node does not draw that fill itself. `None` means
        # "ask my shapes", which is the ordinary case and what a parsed graph always wants.
        #
        # For a node mid-change: a transition splits one node into several, each drawing part of it, and
        # only one of those parts carries the filled shape. The others are the same box to a reader and
        # have to be told what they are standing on, or text in them is coloured for a background that is
        # not there — which, with a rule that picks text colour from the fill, means an invisible label
        # for exactly as long as the transition lasts.
        self.fillcolor_hint: Color | None = None

    def is_inside(self, x: float, y: float) -> bool:
        """Return whether point (x, y) is inside this node's bounding box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class Edge(Element):
    """Graph edge.

    Attributes:
        src: Source node.
        dst: Destination node.
        points: Edge path coordinates.
        shapes: Drawing shapes (line, arrows, labels).
    """

    def __init__(self, src: Node, dst: Node, points: list[Point], shapes: list[Shape]):
        super().__init__(shapes)
        self.src = src
        self.dst = dst
        self.points = points


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
                 shapes: list[Shape] | None = None,
                 nodes: list[Node] | None = None,
                 edges: list[Edge] | None = None):
        self.width = width
        self.height = height
        self.shapes = shapes or []
        self.nodes = nodes or []
        self.edges = edges or []

        # Build lookup tables
        self.nodes_by_name: dict[str, Node] = {}
        for n in self.nodes:
            if n.internal_name:
                self.nodes_by_name[n.internal_name] = n

        # Pre-compute search data (lowercase text for case-insensitive search)
        # Format: [(element, "all text in element lowercase")]
        self._items_and_texts: list[tuple[Element, str]] = [
            (x, " ".join(x.get_texts()))
            for x in chain(self.nodes, self.edges)
        ]

    def get_size(self) -> tuple[float, float]:
        """Return (width, height) of the graph."""
        return self.width, self.height

    def filter_items_by_text(self, text: str) -> list[Element]:
        """Return nodes/edges containing all fragments of the search text.

        Uses fragment search (like Emacs HELM): "cat photo" matches "photocatalytic".
        A lowercase fragment is case-insensitive; a fragment with at least one uppercase
        letter is case-sensitive.
        """
        if not text:
            return []
        # Simple O(n) scan for exact matches, ANDed across all fragments. No stopwording, lemmatization or anything fancy.
        matches_search = common_utils.make_search_matcher(text)
        return [item for item, item_text in self._items_and_texts if matches_search(item_text)]

    def get_node_by_name(self, name: str) -> Node | None:
        """Return node by its internal name, or None."""
        return self.nodes_by_name.get(name)

    def get_linked_elements(self, node: Node, direction: str) -> set[Element]:
        """Return elements linked to `node` via edges.

        `node`: The node to find links for.
        `direction`: "outgoing" (edges where `node` is src, plus dst nodes)
                     or "incoming" (edges where `node` is dst, plus src nodes).

        Returns a set of Elements (edges and their endpoint nodes),
        not including the queried node itself.
        """
        result: set[Element] = set()
        if direction == "outgoing":
            for edge in self.edges:
                if edge.src is node:
                    result.add(edge)
                    result.add(edge.dst)
        elif direction == "incoming":
            for edge in self.edges:
                if edge.dst is node:
                    result.add(edge)
                    result.add(edge.src)
        return result

    def get_all_elements(self) -> list[Element]:
        """Return all nodes and edges."""
        return list(chain(self.nodes, self.edges))

    def iter_shapes(self) -> Iterator[Shape]:
        """Yield every shape in the graph, descending into compounds.

        Background shapes first, then the nodes, then the edges. A `CompoundShape` is yielded before the
        shapes it holds, so a caller after the leaves alone can skip it by type — and `Node` and `Edge`
        are compounds, so they appear here too.

        For asking a question of the whole drawing: how many distinct textures it references, what fonts
        it uses, whether anything was left at the origin.
        """
        def walk(shapes: Iterable[Shape]) -> Iterator[Shape]:
            for shape in shapes:
                yield shape
                if isinstance(shape, CompoundShape):
                    yield from walk(shape.shapes)
        yield from walk(chain(self.shapes, self.nodes, self.edges))
