"""`morph` — the frames between two graphs, as data. No DPG here: nothing is drawn."""

import pytest

from raven.common.gui.xdotwidget.graph import Edge, Graph, LineShape, Node, Pen
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
        assert visible(at_zero, "r") == [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]  # the new copy, the old on top
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
            copies = drawn(frame(still(a), b, t), "r")
            assert len(copies) == 2
            for x, y, _ in copies:
                assert (x, y) == pytest.approx((100.0 * t, 40.0 * t)), "a copy left the straight line"

    def test_its_new_version_is_opaque_underneath_and_the_old_fades_on_top(self):
        """Two copies at half opacity each would dim an unchanged box halfway through."""
        a = graph(box("r", 0.0, 0.0))
        b = graph(box("r", 100.0, 0.0))
        (_, _, new), (_, _, old) = drawn(frame(still(a), b, 0.5), "r")
        assert new == 1.0
        assert old == pytest.approx(0.5)


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
        """What the reader sees of a name is its copies stacked, so that is what must not jump.

        Asserting on any one copy cannot catch this: the old copy is at the right opacity on its own, and a
        new copy drawn under it at that same opacity makes the stack *more* visible than it was.
        """
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
        new, old = [(node.x, placement) for node, placement in placed.nodes]
        assert new == (100.0, Placement(-50.0, 0.0, 1.0))
        assert old == (0.0, Placement(50.0, 0.0, pytest.approx(0.5)))

    def test_invisible_elements_are_left_out(self):
        a = graph(box("r", 0.0, 0.0))
        b = graph(box("r", 100.0, 0.0), box("y", 0.0, 50.0))
        assert [node.internal_name for node, _ in scene(frame(still(a), b, 0.0)).nodes] == ["r", "r"]


def test_shifting_a_picture_moves_everything_in_it_and_nothing_else():
    a = graph(box("r", 0.0, 0.0), box("x", 100.0, 0.0))
    moved = shifted(still(a), 10.0, -5.0)
    assert moved.positions == {"r": (10.0, -5.0), "x": (110.0, -5.0)}
    assert [(x, y) for _, x, y, _ in moved.nodes] == [(10.0, -5.0), (110.0, -5.0)]
    assert [node.x for node, *_ in moved.nodes] == [0.0, 100.0], "the nodes themselves were edited"
