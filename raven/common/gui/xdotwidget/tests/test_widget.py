"""`XDotWidget.set_graph` — driving the widget from a `Graph` built in memory, with no xdot text involved.

Every production path reaches the widget through `set_xdotcode`, so this entry point had no callers and no
coverage. It is the one Librarian's chat graph view will use: a chat tree is already a graph in memory, and
serializing it to positioned xdot only to parse it back would be a round trip through a format neither end
needs.

The graph here is chat-shaped on purpose — a root, two children, and a branch under one of them — so what
it exercises is the shape the caller will actually build.

Nothing maps a window. Creating DPG draw items needs no rendered frame, which is what lets the render
assertions run headless; anything about *layout* would not.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.common.gui.xdotwidget.graph import (Graph, Node, Edge, Pen,  # noqa: E402 -- after importorskip by design
                                               TextShape, EllipseShape, LineShape)
from raven.common.gui.xdotwidget.widget import XDotWidget  # noqa: E402 -- after importorskip by design

# Drawlist children live in slot 2. Slot 1 holds none and reads as "the renderer drew nothing", which is a
# convincing wrong answer — it cost one round of diagnosis before the slots were enumerated.
DRAWLIST_SLOT = 2

NODE_W, NODE_H = 120.0, 40.0


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport.

    Module-scoped, which is the house pattern for every DPG test here: a context is not cheap, DPG keeps
    global state, and recreating one per test is what took the `--run-gui` group down for eleven days.
    """
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


def chat_shaped_graph() -> Graph:
    """A root, two children, and two more under the second — a branching chat tree in miniature."""
    pen = Pen()

    def node(name: str, label: str, x: float, y: float) -> Node:
        return Node(x=x, y=y, w=NODE_W, h=NODE_H,
                    internal_name=name,
                    shapes=[EllipseShape(pen, x, y, NODE_W / 2, NODE_H / 2),
                            TextShape(pen, x, y, 0, len(label) * 7.0, label)])

    root = node("root", "system prompt", 200.0, 40.0)
    a = node("a", "user: hello", 100.0, 140.0)
    b = node("b", "user: goodbye", 300.0, 140.0)
    b1 = node("b1", "AI: reply one", 240.0, 240.0)
    b2 = node("b2", "AI: reply two", 360.0, 240.0)

    edges = [Edge(src, dst,
                  [(src.x, src.y), (dst.x, dst.y)],
                  [LineShape(pen, [(src.x, src.y), (dst.x, dst.y)])])
             for src, dst in [(root, a), (root, b), (b, b1), (b, b2)]]

    return Graph(width=460.0, height=300.0, nodes=[root, a, b, b1, b2], edges=edges)


@pytest.fixture
def widget(dpg_context):
    """A widget over a chat-shaped graph, one per test so no view or highlight state carries over.

    Per-test instantiation is itself a regression guard: the tooltip window and group used fixed tags
    until 2026-08-25, so the second `XDotWidget` in a context died with "Alias already exists" and this
    fixture could not exist as written.
    """
    with dpg.window() as window:
        instance = XDotWidget(parent=window, width=600, height=400)
    instance.set_graph(chat_shaped_graph())
    yield instance
    dpg.delete_item(window)


def drawn(instance: XDotWidget) -> int:
    """How many draw items the last render emitted."""
    return len(dpg.get_item_children(instance.drawlist, DRAWLIST_SLOT) or [])


def test_set_graph_takes_a_graph_built_in_memory(widget):
    graph = widget.get_graph()
    assert sorted(graph.nodes_by_name) == ["a", "b", "b1", "b2", "root"]
    assert len(graph.edges) == 4


def test_set_graph_hands_the_bounds_to_the_viewport(widget):
    """Panning and zoom-to-fit both work off these, so a graph whose bounds never arrived is unusable."""
    assert (widget._viewport._graph_width, widget._viewport._graph_height) == (460.0, 300.0)


def test_the_renderer_draws_every_element_of_a_hand_built_graph(widget):
    """Five nodes of two shapes each, four edges of one, and the background: fifteen items.

    The exact count is asserted rather than "more than zero" because this is the question step zero of
    brief 16 asks — whether the renderer is as happy with a constructed `Graph` as with a parsed one — and
    a partial render would answer it wrongly while still being non-empty.
    """
    widget.zoom_to_fit(animate=False)
    widget._render()
    assert drawn(widget) == 2 * 5 + 4 + 1


def test_elements_outside_the_view_are_culled(widget):
    """The negative control for the assertion above.

    Without this, a renderer that ignored the viewport entirely and drew everything unconditionally would
    pass the fifteen-item test, and the count would be telling us nothing about culling. The default view
    is centred on the origin at zoom 1, so part of this graph lies outside it.
    """
    widget._render()
    default_view = drawn(widget)
    widget.zoom_to_fit(animate=False)
    widget._render()
    assert default_view < drawn(widget), ("nothing was culled in the default view, so this fixture cannot "
                                          "tell a culling renderer from one that draws everything")


def test_pan_to_node_changes_what_is_drawn(widget):
    """One of the hooks the chat view needs: follow the conversation to a given node."""
    widget.zoom_to_fit(animate=False)
    widget._render()
    fitted = drawn(widget)
    widget.pan_to_node("b2", animate=False)
    widget._render()
    assert drawn(widget) != fitted


def test_highlighting_a_node_still_renders(widget):
    """The other hook: marking where you are. Highlighting feeds per-element intensities into the render."""
    widget.zoom_to_fit(animate=False)
    widget.set_highlighted_nodes(["b1"])
    widget._render()
    assert drawn(widget) == 2 * 5 + 4 + 1


def test_search_indexes_a_hand_built_graph(widget):
    """`Graph` builds its search index in `__init__`, so this works without the parser having run."""
    assert len(widget._search.search("reply") or []) == 2
