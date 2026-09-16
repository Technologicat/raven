"""`morph` — the frames between two graphs, as data. No DPG here: nothing is drawn."""

import pytest

from raven.common.gui.xdotwidget.graph import (Edge, Graph, ImageShape, LineShape, MipLevel, Node, Pen,
                                               PolygonShape, TextShape)
from raven.common.gui.xdotwidget.morph import Placement, frame, scene, shifted, still


def box(name: str, x: float, y: float) -> Node:
    return Node(x=x, y=y, w=20.0, h=10.0, shapes=[], internal_name=name)


def link(src: Node, dst: Node) -> Edge:
    points = [(src.x, src.y2), (dst.x, dst.y1)]
    return Edge(src, dst, points, [LineShape(Pen(), points)])


def graph(*nodes: Node, edges: tuple[tuple[str, str], ...] = ()) -> Graph:
    by_name = {node.internal_name: node for node in nodes}
    return Graph(width=1000.0, height=1000.0, nodes=list(nodes),
                 edges=[link(by_name[s], by_name[d]) for s, d in edges])


def drawn(picture, name: str) -> list[tuple[float, float, float]]:
    """Every copy of `name` in `picture`, as `(x, y, opacity)`, in drawing order."""
    return [(x, y, opacity) for node, x, y, opacity in picture.nodes if node.internal_name == name]


def visible(picture, name: str) -> list[tuple[float, float, float]]:
    return [copy for copy in drawn(picture, name) if copy[2] > 0.0]


def stacked(picture, name: str) -> float:
    """How opaque the copies of `name` are together, layered one over another."""
    uncovered = 1.0
    for _, _, opacity in drawn(picture, name):
        uncovered *= 1.0 - opacity
    return 1.0 - uncovered


# The stand-in function a folding graph would supply: "gap" represents "hidden", in whichever graph draws it.
def gap_stands_in(name, g):
    return "gap" if name == "hidden" and "gap" in g.nodes_by_name else None


class TestTheEnds:
    def test_at_zero_it_looks_like_the_source(self):
        a = graph(box("r", 0.0, 0.0), box("x", 100.0, 0.0))
        b = graph(box("r", 50.0, 50.0), box("y", 200.0, 0.0))
        at_zero = frame(still(a), b, 0.0)
        assert visible(at_zero, "r") == [(0.0, 0.0, 1.0)]
        assert visible(at_zero, "x") == [(100.0, 0.0, 1.0)]
        assert visible(at_zero, "y") == [], "a node that is only arriving is already visible at the start"

    def test_at_one_it_looks_like_the_target_at_rest(self):
        a = graph(box("r", 0.0, 0.0), box("x", 100.0, 0.0))
        b = graph(box("r", 50.0, 50.0), box("y", 200.0, 0.0))
        at_one = frame(still(a), b, 1.0)
        assert visible(at_one, "r") == [(50.0, 50.0, 1.0)]
        assert visible(at_one, "y") == [(200.0, 0.0, 1.0)]
        assert visible(at_one, "x") == []


class TestASurvivor:
    def test_it_moves_in_a_straight_line(self):
        a = graph(box("r", 0.0, 0.0))
        b = graph(box("r", 100.0, 40.0))
        for t in (0.25, 0.5, 0.75):
            assert drawn(frame(still(a), b, t), "r") == [pytest.approx((100.0 * t, 40.0 * t, 1.0))]


def styled_box(name: str, x: float, *extras, width: float = 40.0, color=(0.0, 0.0, 0.0, 1.0)) -> Node:
    """A box at `(x, 0)`: a fill, an outline and a label, then `extras` — each a function of the box's left
    edge returning one more shape."""
    pen = Pen()
    pen.color = color
    pen.fillcolor = (0.5, 0.5, 0.5, 1.0)
    x1, x2 = x - width / 2, x + width / 2
    corners = [(x1, -5.0), (x2, -5.0), (x2, 5.0), (x1, 5.0)]
    shapes = [PolygonShape(pen, corners, filled=True),
              PolygonShape(pen, corners, filled=False),
              TextShape(pen, x, 0.0, TextShape.CENTER, 20.0, "label")]
    shapes.extend(make(x1) for make in extras)
    return Node(x=x, y=0.0, w=width, h=10.0, shapes=shapes, internal_name=name)


def ring(x1: float) -> PolygonShape:
    pen = Pen()
    pen.dash = (1.0, 2.0)
    return PolygonShape(pen, [(x1 - 3, -8.0), (x1 + 3, -8.0), (x1 + 3, 8.0)], filled=False)


def icon(texture: str):
    return lambda x1: ImageShape([MipLevel(64, 64, texture)], x1 - 5.0, -5.0, x1 + 5.0, 5.0)


def parts(picture, name: str) -> list[tuple[list[str], float]]:
    """The copies of `name`, each as `(the kinds of shape it draws, its opacity)`."""
    return [([type(shape).__name__ for shape in node.shapes], opacity)
            for node, _, _, opacity in picture.nodes if node.internal_name == name]


class TestABoxChangesShapeAsOneItem:
    """A surviving box is drawn once, its shapes paired with their new versions, so the box and its
    decorations cannot wash over each other mid-change. Only shapes without a partner fade."""

    def test_an_unchanged_box_draws_each_shape_once(self):
        old, new = styled_box("r", 0.0, icon("glyph")), styled_box("r", 100.0, icon("glyph"))
        halfway = frame(still(graph(old)), graph(new), 0.5)
        assert parts(halfway, "r") == [(["PolygonShape", "PolygonShape", "TextShape", "ImageShape"], 1.0)]

    def test_a_ring_arriving_fades_in_and_the_rest_stays_paired(self):
        """The ring sits between the outline and the icon in drawing order, so this also checks that an
        insertion in the middle of the list leaves the shapes on either side of it paired."""
        old = styled_box("r", 0.0, icon("glyph"))
        new = styled_box("r", 0.0, ring, icon("glyph"))
        assert parts(frame(still(graph(old)), graph(new), 0.25), "r") == [
            (["PolygonShape", "PolygonShape", "TextShape", "ImageShape"], 1.0),
            (["PolygonShape"], 0.25)]

    def test_a_ring_leaving_fades_out(self):
        old = styled_box("r", 0.0, ring)
        new = styled_box("r", 0.0)
        assert parts(frame(still(graph(old)), graph(new), 0.25), "r") == [
            (["PolygonShape", "PolygonShape", "TextShape"], 1.0),
            (["PolygonShape"], 0.75)]

    def test_geometry_and_pen_are_interpolated(self):
        """A box widening as it is marked, and its outline changing colour, both happen gradually."""
        old = styled_box("r", 0.0, width=40.0, color=(0.0, 0.0, 0.0, 1.0))
        new = styled_box("r", 0.0, width=80.0, color=(1.0, 0.0, 0.0, 1.0))
        (node, _, _, _), = frame(still(graph(old)), graph(new), 0.5).nodes
        fill = node.shapes[0]
        assert [p[0] for p in fill.points] == pytest.approx([-30.0, 30.0, 30.0, -30.0])
        assert node.shapes[1].pen.color == pytest.approx((0.5, 0.0, 0.0, 1.0))

    def test_a_moved_box_keeps_its_shapes_where_they_belong(self):
        """Shapes are paired in the box's own coordinates, so moving the whole box moves them with it."""
        old, new = styled_box("r", 0.0), styled_box("r", 100.0)
        placed = scene(frame(still(graph(old)), graph(new), 0.5))
        (node, placement), = placed.nodes
        assert [p[0] + placement.dx for p in node.shapes[0].points] == pytest.approx([30.0, 70.0, 70.0, 30.0])

    def test_changed_text_crosses_over_rather_than_morphing(self):
        old = styled_box("r", 0.0)
        new = styled_box("r", 0.0)
        new.shapes[2] = TextShape(Pen(), 0.0, 0.0, TextShape.CENTER, 20.0, "another label")
        assert parts(frame(still(graph(old)), graph(new), 0.25), "r") == [
            (["PolygonShape", "PolygonShape"], 1.0),
            (["TextShape"], 0.25),
            (["TextShape"], 0.75)]

    def test_an_image_pairs_only_with_the_same_texture(self):
        old, new = styled_box("r", 0.0, icon("glyph")), styled_box("r", 0.0, icon("another glyph"))
        assert parts(frame(still(graph(old)), graph(new), 0.25), "r") == [
            (["PolygonShape", "PolygonShape", "TextShape"], 1.0),
            (["ImageShape"], 0.25),
            (["ImageShape"], 0.75)]

    def test_the_ends_match_the_two_pictures(self):
        old = styled_box("r", 0.0, icon("glyph"))
        new = styled_box("r", 100.0, ring, icon("glyph"), width=60.0)
        start = scene(frame(still(graph(old)), graph(new), 0.0))
        end = scene(frame(still(graph(old)), graph(new), 1.0))

        def drawn_points(placed):
            return sorted((round(p[0] + placement.dx, 6), round(p[1] + placement.dy, 6))
                          for node, placement in placed.nodes if placement.opacity > 0.0
                          for shape in node.shapes if isinstance(shape, PolygonShape)
                          for p in shape.points)

        def points_of(node):
            return sorted((round(p[0], 6), round(p[1], 6))
                          for shape in node.shapes if isinstance(shape, PolygonShape) for p in shape.points)
        assert drawn_points(start) == points_of(old)
        assert drawn_points(end) == points_of(new)


class TestANodeOnOneSideOnly:
    def test_an_arrival_starts_where_its_stand_in_was(self):
        a = graph(box("r", 0.0, 0.0), box("gap", 300.0, 100.0))
        b = graph(box("r", 0.0, 0.0), box("hidden", 500.0, 100.0))
        start = frame(still(a), b, 0.0, stand_in=gap_stands_in)
        halfway = frame(still(a), b, 0.5, stand_in=gap_stands_in)
        assert drawn(start, "hidden") == [(300.0, 100.0, 0.0)]
        assert drawn(halfway, "hidden") == [pytest.approx((400.0, 100.0, 0.5))]

    def test_a_departure_travels_to_its_stand_in_and_fades(self):
        a = graph(box("r", 0.0, 0.0), box("hidden", 500.0, 100.0))
        b = graph(box("r", 0.0, 0.0), box("gap", 300.0, 100.0))
        halfway = frame(still(a), b, 0.5, stand_in=gap_stands_in)
        assert drawn(halfway, "hidden") == [pytest.approx((400.0, 100.0, 0.5))]

    def test_with_no_stand_in_it_fades_where_it_is(self):
        a = graph(box("r", 0.0, 0.0), box("hidden", 500.0, 100.0))
        b = graph(box("r", 0.0, 0.0), box("gap", 300.0, 100.0))
        halfway = frame(still(a), b, 0.5)
        assert drawn(halfway, "hidden") == [pytest.approx((500.0, 100.0, 0.5))]
        assert drawn(frame(still(b), a, 0.5), "hidden") == [pytest.approx((500.0, 100.0, 0.5))]


class TestRetargetingMidFlight:
    def test_a_new_change_starts_from_wherever_the_old_one_had_got_to(self):
        a = graph(box("r", 0.0, 0.0))
        b = graph(box("r", 100.0, 0.0))
        c = graph(box("r", 100.0, 200.0))
        midway = frame(still(a), b, 0.5)
        restart = frame(midway, c, 0.0)
        assert {(x, y) for x, y, _ in visible(restart, "r")} == {(50.0, 0.0)}, \
            "the node jumped when the change was retargeted"
        assert {(x, y) for x, y, _ in visible(frame(midway, c, 1.0), "r")} == {(100.0, 200.0)}

    def test_a_node_fading_out_that_is_wanted_back_resumes_rather_than_pops(self):
        """It carries on from the opacity it had faded to, rather than popping back to full."""
        a = graph(box("r", 0.0, 0.0), box("x", 100.0, 0.0))
        b = graph(box("r", 0.0, 0.0))
        midway = frame(still(a), b, 0.5)
        assert stacked(midway, "x") == pytest.approx(0.5), "the fixture is not fading x out"
        back = [stacked(frame(midway, a, t), "x") for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert back == pytest.approx([0.5, 0.625, 0.75, 0.875, 1.0]), \
            "x did not come back linearly from where it had faded to"

    def test_a_survivor_stays_fully_visible_throughout(self):
        a = graph(box("r", 0.0, 0.0))
        b = graph(box("r", 100.0, 0.0))
        assert [stacked(frame(still(a), b, t), "r") for t in (0.0, 0.3, 0.7, 1.0)] == pytest.approx([1.0] * 4)

    def test_invisible_leftovers_are_not_carried_forward(self):
        a = graph(box("r", 0.0, 0.0), box("x", 100.0, 0.0))
        b = graph(box("r", 0.0, 0.0))
        done = frame(still(a), b, 1.0)
        assert drawn(frame(done, b, 0.0), "x") == [], "a fully faded node is still being carried along"


class TestEdges:
    def test_opacity_follows_whether_the_edge_is_arriving_staying_or_leaving(self):
        a = graph(box("r", 0.0, 0.0), box("x", 0.0, 50.0), box("y", 50.0, 50.0), edges=(("r", "x"), ("r", "y")))
        b = graph(box("r", 0.0, 0.0), box("x", 0.0, 50.0), box("z", 90.0, 50.0), edges=(("r", "x"), ("r", "z")))
        halfway = frame(still(a), b, 0.25)
        by_key = {}
        for edge, opacity in halfway.edges:
            by_key.setdefault((edge.src.internal_name, edge.dst.internal_name), []).append(opacity)
        assert by_key[("r", "x")] == [1.0, pytest.approx(0.75)]  # the new copy, the old fading on top
        assert by_key[("r", "z")] == [pytest.approx(0.25)]
        assert by_key[("r", "y")] == [pytest.approx(0.75)]

    def test_given_a_builder_edges_stay_attached_to_moving_nodes(self):
        a = graph(box("r", 0.0, 0.0), box("x", 0.0, 50.0), edges=(("r", "x"),))
        b = graph(box("r", 0.0, 0.0), box("x", 100.0, 50.0), edges=(("r", "x"),))
        calls = []

        def edge_between(src, dst):
            calls.append((src.internal_name, src.x, src.y, dst.internal_name, dst.x, dst.y))
            return link(src, dst)

        placed = scene(frame(still(a), b, 0.5), edge_between=edge_between)
        assert calls and all(call == ("r", 0.0, 0.0, "x", 50.0, 50.0) for call in calls)
        assert all(edge.dst.x == 50.0 for edge, _ in placed.edges)

    def test_without_a_builder_an_edge_stays_where_its_points_say(self):
        a = graph(box("r", 0.0, 0.0), box("x", 0.0, 50.0), edges=(("r", "x"),))
        b = graph(box("r", 0.0, 0.0), box("x", 100.0, 50.0), edges=(("r", "x"),))
        placed = scene(frame(still(a), b, 0.5))
        assert {(edge.dst.x, placement[:2]) for edge, placement in placed.edges} == {(100.0, (0.0, 0.0)),
                                                                                     (0.0, (0.0, 0.0))}


class TestScene:
    def test_a_node_is_placed_by_its_offset_from_its_own_coordinates(self):
        a = graph(box("r", 0.0, 0.0))
        b = graph(box("r", 100.0, 0.0))
        placed = scene(frame(still(a), b, 0.5))
        assert [(node.x, placement) for node, placement in placed.nodes] == [(100.0, Placement(-50.0, 0.0, 1.0))]

    def test_invisible_elements_are_left_out(self):
        a = graph(box("r", 0.0, 0.0))
        b = graph(box("r", 100.0, 0.0), box("y", 0.0, 50.0))
        assert [node.internal_name for node, _ in scene(frame(still(a), b, 0.0)).nodes] == ["r"]


def test_shifting_a_picture_moves_everything_in_it_and_nothing_else():
    a = graph(box("r", 0.0, 0.0), box("x", 100.0, 0.0))
    moved = shifted(still(a), 10.0, -5.0)
    assert moved.positions == {"r": (10.0, -5.0), "x": (110.0, -5.0)}
    assert [(x, y) for _, x, y, _ in moved.nodes] == [(10.0, -5.0), (110.0, -5.0)]
    assert [node.x for node, *_ in moved.nodes] == [0.0, 100.0], "the nodes themselves were edited"
