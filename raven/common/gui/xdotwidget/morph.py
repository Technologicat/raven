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

A node present on both sides is drawn twice during the change: the new version underneath, the old copies
on top of it fading out. Where they look alike the stack is indistinguishable from one box; where they
differ — a decoration that moved to another node — the difference cross-fades. The new version's opacity
is chosen so that the stack's *combined* visibility goes linearly from what it was to fully opaque: an
unchanged box stays fully visible throughout, and a box that was fading out and is wanted back carries on
from where it was.
"""

__all__ = ["Placement", "IN_PLACE",
           "Picture", "Scene",
           "still", "shifted", "frame", "scene"]

from collections.abc import Callable, Mapping
from typing import NamedTuple

from .constants import Point
from .graph import Edge, Graph, Node, Shape


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


def _underneath(was: list[float], t: float) -> float:
    """Opacity for the new version of an element whose old copies, at opacities `was`, fade out over it.

    Chosen so the stack's combined opacity is `C = a + (1 - a) t`, linear from the old copies' combined
    opacity `a` to 1. The old copies fading as `was_i (1 - t)` combine to `G`, and a layer `n` under them
    gives `1 - (1 - n)(1 - G)`; setting that equal to `C` gives `n = 1 - (1 - C) / (1 - G)`.

    `was = []` gives `n = t`, a plain fade-in. Any old copy fully opaque gives `n = 1` from the first frame
    on: the stack is then opaque throughout, which is what keeps an unchanged box from dimming mid-change.
    """
    a = _coverage(was)
    covered = _coverage([opacity * (1.0 - t) for opacity in was])
    if covered >= 1.0:  # an opaque old copy hides whatever is underneath; be ready for when it fades
        return 1.0
    return min(1.0, max(0.0, 1.0 - (1.0 - (a + (1.0 - a) * t)) / (1.0 - covered)))


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

    # The opacities each name and each edge is already drawn at, copy by copy.
    node_opacities: dict[str, list[float]] = {}
    for node, _, _, opacity in source.nodes:
        if node.internal_name:
            node_opacities.setdefault(node.internal_name, []).append(opacity)
    edge_opacities: dict[tuple[str | None, str | None], list[float]] = {}
    for edge, opacity in source.edges:
        edge_opacities.setdefault(_edge_key(edge), []).append(opacity)

    positions: dict[str, Point] = {}
    nodes = []

    # The target's nodes, underneath.
    for node in target.nodes:
        name = node.internal_name
        end = (node.x, node.y)
        start = source.positions.get(name) if name else None
        if start is None:
            stand = represented_by(name, source.graph)
            start = source.positions.get(stand, end) if stand is not None else end
        x, y = _lerp(start, end, t)
        nodes.append((node, x, y, _underneath(node_opacities.get(name, []) if name else [], t)))
        if name:
            positions[name] = (x, y)

    # The source's nodes, on top, fading out.
    for node, x0, y0, opacity in source.nodes:
        if opacity * (1.0 - t) <= _INVISIBLE:
            continue
        name = node.internal_name
        end = (x0, y0)
        if name and name in target.nodes_by_name:
            end = positions[name]  # already interpolated; a survivor's two copies move as one
            x, y = end
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
    for edge in target.edges:
        edges.append((edge, _underneath(edge_opacities.get(_edge_key(edge), []), t)))
    for edge, opacity in source.edges:
        if opacity * (1.0 - t) > _INVISIBLE:
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
        edges.append((edge_between(_moved(edge.src, src_at), _moved(edge.dst, dst_at)),
                      Placement(opacity=opacity)))

    return Scene(shapes=picture.graph.shapes, edges=edges, nodes=nodes)


def _moved(node: Node, at: Point) -> Node:
    """A shapeless copy of `node`'s box, centred on `at` — what `EdgeBetween` is given."""
    return Node(x=at[0], y=at[1], w=node.x2 - node.x1, h=node.y2 - node.y1, shapes=[],
                internal_name=node.internal_name)
