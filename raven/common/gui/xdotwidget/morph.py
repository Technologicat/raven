"""The picture partway through a change from one graph to another.

A graph widget that replaces its graph between one frame and the next makes the reader work out what
happened. This module computes the frames in between, as pure data: which element to draw where, and how
faded. Drawing them is `renderer.render_scene`'s job, and driving the progress over time is the widget's.

Elements correspond across the change by `Node.internal_name`, and edges by the names of their endpoints:

- **A node in both** moves from where it was to where it is going.
- **A node only in the new graph** starts where its *stand-in* was, and fades in. A stand-in is whichever
  node of the old picture represents it — for a graph that folds nodes into a summary box, that box — and is
  named by a caller-supplied function, since only the caller knows what its boxes stand for.
- **A node only in the old picture** travels to its stand-in in the new graph, and fades out.

With no stand-in, a node fades where it is.

**The old picture is not a graph.** A change can be retargeted mid-flight, and the picture on screen at that
moment holds elements of two graphs, some partly faded. So the source of a change is a `Picture`: whatever
is drawn, where, and how visibly. A graph at rest is the special case `still(graph)`, and the frame of a
change at any progress is itself a `Picture`, which a new change can start from.

**A node present on both sides changes shape as well as place.** Its old and new shapes are paired, in
order, by what cannot be interpolated — the kind of shape, whether it is filled, how many points it has,
its dash pattern, its text, its texture — and each pair is drawn *once*, with geometry, colours, line width
and font size interpolated between the two. Only shapes without a partner fade: a selection ring moving to
another box, a label whose text changed. So a box and everything drawn with it read as one item throughout,
which two whole copies cross-fading would not — alpha is per shape, and a half-faded copy of the box would
wash over the other copy's decorations wherever they overlap.

**An edge present on both sides is drawn once, the same way**, its shapes paired and interpolated. Two
copies cross-fading would stack two anti-aliased strokes on the same pixels, which draws the line visibly
heavier until the old copy has faded. An edge between the same two names is the same edge, matched in
order where a graph has several.

Where the caller rebuilds edges between wherever their endpoints are drawn (`scene`'s `edge_between`), every
copy of one edge comes out the same, so the copies are drawn once, at their combined opacity.
"""

__all__ = ["Placement", "IN_PLACE",
           "Picture", "Scene",
           "still", "shifted", "frame", "scene"]

import difflib
import weakref
from collections.abc import Callable, Mapping
from typing import NamedTuple

from .constants import Color, Point
from .graph import (Edge, Element, Graph, Node, Pen, Shape,
                    BezierShape, CompoundShape, EllipseShape, ImageShape, LineShape, PolygonShape, TextShape)


class Placement(NamedTuple):
    """Where, and how visibly, to draw an element, relative to what its shapes say.

    `dx`, `dy`: displacement in graph coordinates, added to every point of every shape.
    `opacity`: in [0, 1], multiplied into the alpha of everything the element draws.
    """
    dx: float = 0.0
    dy: float = 0.0
    opacity: float = 1.0


IN_PLACE = Placement()  # where the shapes say, fully opaque


class Picture(NamedTuple):
    """What is on screen: elements, possibly of several graphs, each at a position and an opacity.

    `graph`: the graph this picture is converging on, or at rest in. Stand-ins are looked up in it.
    `nodes`: `(node, x, y, opacity)`, with `(x, y)` where the node's centre is drawn. In drawing order.
    `edges`: `(edge, opacity)`. An edge is drawn between wherever its endpoints are, by name.
    `positions`: node name -> drawn centre, for every named node in `nodes`. Nodes sharing a name are always
                 drawn at the same place, so one position per name is enough.
    """
    graph: Graph
    nodes: tuple[tuple[Node, float, float, float], ...]
    edges: tuple[tuple[Edge, float], ...]
    positions: Mapping[str, Point]


class Scene(NamedTuple):
    """A `Picture` resolved into what `renderer.render_scene` takes."""
    shapes: list[Shape]
    edges: list[tuple[Edge, Placement]]
    nodes: list[tuple[Node, Placement]]


# Below this, an element is not worth carrying into the next picture. Far below one level of 8-bit alpha.
_INVISIBLE = 1e-4

StandIn = Callable[[str, Graph], str | None]
"""`(name, graph) -> name of the node in `graph` that represents `name`, or None`. Asked only about names
that `graph` does not draw itself."""

EdgeBetween = Callable[[Node, Node], Edge]
"""`(src, dst) -> an edge drawn between those two nodes`. The nodes carry positions and sizes and no shapes."""


def still(graph: Graph) -> Picture:
    """The picture of `graph` at rest: every element where its shapes say, fully opaque."""
    return Picture(graph=graph,
                   nodes=tuple((node, node.x, node.y, 1.0) for node in graph.nodes),
                   edges=tuple((edge, 1.0) for edge in graph.edges),
                   positions={node.internal_name: (node.x, node.y)
                              for node in graph.nodes if node.internal_name})


def shifted(picture: Picture, dx: float, dy: float) -> Picture:
    """`picture`, with everything drawn `(dx, dy)` graph units away from where it was.

    For putting a picture into the coordinates of a new graph whose layout has moved as a whole: shift the
    old picture by the same amount the camera moves, and nothing on screen moves at all.
    """
    return picture._replace(nodes=tuple((node, x + dx, y + dy, opacity)
                                        for node, x, y, opacity in picture.nodes),
                            positions={name: (x + dx, y + dy) for name, (x, y) in picture.positions.items()})


def _lerp(p: Point, q: Point, t: float) -> Point:
    return (p[0] + (q[0] - p[0]) * t,
            p[1] + (q[1] - p[1]) * t)


def _edge_key(edge: Edge) -> tuple[str | None, str | None]:
    return (edge.src.internal_name, edge.dst.internal_name)


def _coverage(opacities: list[float]) -> float:
    """The combined opacity of layers drawn over each other: `1 - prod(1 - a_i)`."""
    uncovered = 1.0
    for a in opacities:
        uncovered *= 1.0 - a
    return 1.0 - uncovered


def frame(source: Picture, target: Graph, t: float, stand_in: StandIn | None = None) -> Picture:
    """The picture at progress `t` in [0, 1] of the change from `source` to `target`.

    At `t = 0` it looks exactly like `source`; at `t = 1`, exactly like `still(target)` plus elements at
    zero opacity, which draw nothing.

    `stand_in`: see `StandIn`. Consulted for nodes on one side only; `None` fades them where they are.
    """
    def represented_by(name: str | None, graph: Graph) -> str | None:
        if name is None or stand_in is None:
            return None
        return stand_in(name, graph)

    # For each name the source draws, the index of its most visible copy. That copy is what a survivor
    # changes shape *from*; any other copies of the name are leftovers of an earlier change, and fade.
    main_copy: dict[str, int] = {}
    for index, (node, _, _, opacity) in enumerate(source.nodes):
        name = node.internal_name
        if name and (name not in main_copy or opacity > source.nodes[main_copy[name]][3]):
            main_copy[name] = index
    # Likewise for edges, where a name pair can occur more than once: each target edge takes over the most
    # visible copy of its pair that no earlier one has taken.
    edge_copies: dict[tuple[str | None, str | None], list[int]] = {}
    for index, (edge, _) in enumerate(source.edges):
        edge_copies.setdefault(_edge_key(edge), []).append(index)
    for indices in edge_copies.values():
        indices.sort(key=lambda index: source.edges[index][1], reverse=True)

    positions: dict[str, Point] = {}
    nodes = []
    consumed: set[int] = set()  # source copies a survivor has taken over

    # The target's nodes, underneath.
    for node in target.nodes:
        name = node.internal_name
        end = (node.x, node.y)

        index = main_copy.get(name) if name else None
        if index is not None:  # a survivor: change shape and place together
            consumed.add(index)
            old, x0, y0, was = source.nodes[index]
            x, y = _lerp((x0, y0), end, t)
            positions[name] = (x, y)
            paired, old_only, new_only = _pair_shapes(old, node)
            # Old shapes are in the old node's coordinates; moved into the new one's, they interpolate
            # against their partners directly, and every part comes out placed relative to `node`.
            dx, dy = node.x - old.x, node.y - old.y
            between = [_shape_between(old_shape, new_shape, dx, dy, t) for old_shape, new_shape in paired]
            nodes.append((_carrier(node, between), x, y, was + (1.0 - was) * t))
            if new_only:
                nodes.append((_carrier(node, new_only), x, y, t))
            if old_only and was * (1.0 - t) > _INVISIBLE:
                nodes.append((_carrier(old, old_only), x, y, was * (1.0 - t)))
            continue

        start = None
        stand = represented_by(name, source.graph)
        if stand is not None:
            start = source.positions.get(stand)
        x, y = _lerp(start if start is not None else end, end, t)
        nodes.append((node, x, y, t))
        if name:
            positions[name] = (x, y)

    # The source's nodes a survivor did not take over, on top, fading out.
    for index, (node, x0, y0, opacity) in enumerate(source.nodes):
        if index in consumed or opacity * (1.0 - t) <= _INVISIBLE:
            continue
        name = node.internal_name
        end = (x0, y0)
        if name and name in target.nodes_by_name:
            x, y = positions[name]  # a leftover copy of a survivor moves with it
        else:
            stand = represented_by(name, target)
            if stand is not None and stand in target.nodes_by_name:
                stand_node = target.nodes_by_name[stand]
                end = (stand_node.x, stand_node.y)
            x, y = _lerp((x0, y0), end, t)
            if name:
                positions[name] = (x, y)
        nodes.append((node, x, y, opacity * (1.0 - t)))

    edges = []
    consumed_edges: set[int] = set()
    for edge in target.edges:
        copies = edge_copies.get(_edge_key(edge))
        if not copies:
            edges.append((edge, t))
            continue
        index = copies.pop(0)
        consumed_edges.add(index)
        old, was = source.edges[index]
        paired, old_only, new_only = _pair_shapes(old, edge)
        between = [_shape_between(old_shape, new_shape, 0.0, 0.0, t) for old_shape, new_shape in paired]
        edges.append((Edge(edge.src, edge.dst, edge.points, between), was + (1.0 - was) * t))
        if new_only:
            edges.append((Edge(edge.src, edge.dst, edge.points, new_only), t))
        if old_only and was * (1.0 - t) > _INVISIBLE:
            edges.append((Edge(old.src, old.dst, old.points, old_only), was * (1.0 - t)))
    for index, (edge, opacity) in enumerate(source.edges):
        if index not in consumed_edges and opacity * (1.0 - t) > _INVISIBLE:
            edges.append((edge, opacity * (1.0 - t)))

    return Picture(graph=target, nodes=tuple(nodes), edges=tuple(edges), positions=positions)


def scene(picture: Picture, edge_between: EdgeBetween | None = None) -> Scene:
    """Resolve `picture` into placed elements, ready to render.

    `edge_between`: see `EdgeBetween`. Given, every edge is rebuilt between wherever its endpoints are drawn,
                    so edges stay attached while nodes move. `None` draws each edge where its own points say,
                    which is right at rest and detached while its endpoints are travelling.
    """
    nodes = [(node, Placement(x - node.x, y - node.y, opacity))
             for node, x, y, opacity in picture.nodes
             if opacity > 0.0]

    edges = []
    rebuilt: dict[tuple[str | None, str | None], tuple[Edge, list[float]]] = {}  # in first-seen order
    for edge, opacity in picture.edges:
        if opacity <= 0.0:
            continue
        if edge_between is None:
            edges.append((edge, Placement(opacity=opacity)))
            continue
        src_at = picture.positions.get(edge.src.internal_name)
        dst_at = picture.positions.get(edge.dst.internal_name)
        if src_at is None or dst_at is None:  # an endpoint with no name has nowhere to be looked up
            edges.append((edge, Placement(opacity=opacity)))
            continue
        key = _edge_key(edge)
        if key not in rebuilt:
            rebuilt[key] = (edge_between(_moved(edge.src, src_at), _moved(edge.dst, dst_at)), [])
        rebuilt[key][1].append(opacity)
    # Every copy of one name pair rebuilds to the same edge, so each is drawn once rather than stacked.
    edges.extend((edge, Placement(opacity=_coverage(opacities))) for edge, opacities in rebuilt.values())

    return Scene(shapes=picture.graph.shapes, edges=edges, nodes=nodes)


def _carrier(like: Node, shapes: list[Shape]) -> Node:
    """A node with `like`'s box, name and coordinates, drawing `shapes` — one part of a node mid-change."""
    part = Node(x=like.x, y=like.y, w=like.x2 - like.x1, h=like.y2 - like.y1, shapes=shapes,
                url=like.url, internal_name=like.internal_name, tooltip=like.tooltip)
    # What the whole node is filled with, which only the part carrying the filled shape would otherwise
    # know. A renderer choosing text colour from the fill needs it in every part: a box gaining a run --
    # a label that has just matched a search, say -- puts that run in a part of its own, and without this
    # the run is coloured for no background at all until the change finishes.
    part.fillcolor_hint = _fill_of(like)
    return part


def _fill_of(node: Node) -> Color | None:
    """The fill colour of `node`'s first filled shape, or `None` if it draws none."""
    for shape in node.shapes:
        if isinstance(shape, (EllipseShape, PolygonShape)) and shape.filled and shape.pen is not None:
            return shape.pen.fillcolor
    return None


def _signature(shape: Shape) -> tuple:
    """What two shapes must share to be drawn as one shape changing, rather than one fading into another.

    Everything that has no in-between — the kind, fill, point count, dash, text, texture. What does have one
    (positions, sizes, colours, line width, font size) is left out, since interpolating it is the point.
    """
    dash = shape.pen.dash if shape.pen is not None else ()
    if isinstance(shape, TextShape):
        return (TextShape, shape.justify, shape.text, dash)
    if isinstance(shape, EllipseShape):
        return (EllipseShape, shape.filled, dash)
    if isinstance(shape, (PolygonShape, BezierShape)):
        return (type(shape), shape.filled, len(shape.points), dash)
    if isinstance(shape, LineShape):
        return (LineShape, len(shape.points), dash)
    if isinstance(shape, ImageShape):
        return (ImageShape, shape.levels, shape.max_screen_size)
    if isinstance(shape, CompoundShape):
        return (type(shape), tuple(_signature(child) for child in shape.shapes))
    return (type(shape), id(shape))  # a kind this module does not know how to interpolate never pairs


# `(old element, new element) -> pairing`. A change asks on every frame about the same two nodes or edges,
# and the answer depends only on them. Weak both ways, so a pairing goes when either picture does.
_pairings: "weakref.WeakKeyDictionary[Element, weakref.WeakKeyDictionary]" = weakref.WeakKeyDictionary()


def _pair_shapes(old: Element, new: Element) -> tuple[list[tuple[Shape, Shape]], list[Shape], list[Shape]]:
    """Pair `old`'s shapes with `new`'s: `(pairs, only in old, only in new)`.

    Pairs are matched in order, as the longest common run of `_signature`s, so a shape inserted or removed
    in the middle of a list — a ring appearing between a box's outline and its label — leaves the shapes on
    either side of it paired.
    """
    by_new = _pairings.get(old)
    if by_new is None:
        by_new = _pairings[old] = weakref.WeakKeyDictionary()
    cached = by_new.get(new)
    if cached is not None:
        return cached

    matcher = difflib.SequenceMatcher(None,
                                      [_signature(shape) for shape in old.shapes],
                                      [_signature(shape) for shape in new.shapes],
                                      autojunk=False)
    pairs, old_only, new_only = [], [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend(zip(old.shapes[i1:i2], new.shapes[j1:j2]))
        else:
            old_only.extend(old.shapes[i1:i2])
            new_only.extend(new.shapes[j1:j2])
    result = (pairs, old_only, new_only)
    by_new[new] = result
    return result


def _lerp_number(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _pen_between(old: Pen, new: Pen, t: float) -> Pen:
    pen = new.copy()  # for the dash, which `mix` leaves alone and pairing guarantees the two share
    Pen.mix(pen, old, new, t)
    return pen


def _shape_between(old: Shape, new: Shape, dx: float, dy: float, t: float) -> Shape:
    """The shape partway from `old`, moved by `(dx, dy)`, to `new`. The two must share a `_signature`.

    What a shape stands on it stands on throughout. An in-between is a fresh object, and one built without
    that answer falls back to the element's -- the single fill a carrier can offer, which is exactly what
    a shape-level answer exists to overrule. A gap box's label stands on nothing and is drawn quiet
    because of it; given the pill backing beside it instead, it is corrected against that for as long as
    the transition runs, and reads correctly the moment it settles.
    """
    made = _interpolated(old, new, dx, dy, t)
    if made is not new:  # the fallback below hands `new` straight back, and it already knows
        made.background_hint = new.background_hint
    return made


def _interpolated(old: Shape, new: Shape, dx: float, dy: float, t: float) -> Shape:
    """`_shape_between`'s geometry and pens, without the bookkeeping that belongs to every kind alike."""
    def point(p: Point, q: Point) -> Point:
        return _lerp((p[0] + dx, p[1] + dy), q, t)

    if isinstance(new, TextShape):
        return TextShape(_pen_between(old.pen, new.pen, t),
                         _lerp_number(old.x + dx, new.x, t), _lerp_number(old.y + dy, new.y, t),
                         new.justify, _lerp_number(old.width, new.width, t), new.text)
    if isinstance(new, EllipseShape):
        return EllipseShape(_pen_between(old.pen, new.pen, t),
                            _lerp_number(old.x0 + dx, new.x0, t), _lerp_number(old.y0 + dy, new.y0, t),
                            _lerp_number(old.rx, new.rx, t), _lerp_number(old.ry, new.ry, t), new.filled)
    if isinstance(new, (PolygonShape, BezierShape)):
        return type(new)(_pen_between(old.pen, new.pen, t),
                         [point(p, q) for p, q in zip(old.points, new.points)], new.filled)
    if isinstance(new, LineShape):
        return LineShape(_pen_between(old.pen, new.pen, t), [point(p, q) for p, q in zip(old.points, new.points)])
    if isinstance(new, ImageShape):
        return ImageShape(new.levels,
                          _lerp_number(old.x1 + dx, new.x1, t), _lerp_number(old.y1 + dy, new.y1, t),
                          _lerp_number(old.x2 + dx, new.x2, t), _lerp_number(old.y2 + dy, new.y2, t),
                          new.max_screen_size)
    if isinstance(new, CompoundShape):
        return CompoundShape([_shape_between(o, n, dx, dy, t) for o, n in zip(old.shapes, new.shapes)])
    return new


def _moved(node: Node, at: Point) -> Node:
    """A shapeless copy of `node`'s box, centred on `at` — what `EdgeBetween` is given."""
    return Node(x=at[0], y=at[1], w=node.x2 - node.x1, h=node.y2 - node.y1, shapes=[],
                internal_name=node.internal_name)
