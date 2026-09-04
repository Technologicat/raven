"""Tests for the viewport transforms."""

from raven.common.tests import approx

from ..viewport import Viewport
from ..graph import Graph


# ---------------------------------------------------------------------------
# Tests: Viewport
# ---------------------------------------------------------------------------

class TestViewport:
    """Test the Viewport class."""

    def test_initial_state(self):
        """Viewport stores its dimensions."""
        vp = Viewport(width=800, height=600)
        assert vp.width == 800
        assert vp.height == 600

    # --- Coordinate transforms ---

    def test_graph_to_screen_identity(self):
        """At origin pan and zoom 1, graph origin maps to screen center."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(0)
        vp.pan_y.set_immediate(0)
        vp.zoom.set_immediate(1.0)

        sx, sy = vp.graph_to_screen(0, 0)
        assert sx == 50  # width / 2
        assert sy == 50  # height / 2

    def test_screen_to_graph_roundtrip(self):
        """graph_to_screen and screen_to_graph are inverses."""
        vp = Viewport(width=800, height=600)
        vp.pan_x.set_immediate(100)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(2.0)

        gx, gy = 75.0, 25.0
        sx, sy = vp.graph_to_screen(gx, gy)
        gx2, gy2 = vp.screen_to_graph(sx, sy)

        assert approx(gx, gx2, tol=0.001)
        assert approx(gy, gy2, tol=0.001)

    def test_zoom_scales_coordinates(self):
        """Doubling zoom doubles screen distance from center."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)

        vp.zoom.set_immediate(1.0)
        sx1, sy1 = vp.graph_to_screen(60, 60)

        vp.zoom.set_immediate(2.0)
        sx2, sy2 = vp.graph_to_screen(60, 60)

        center = 50
        dist1 = abs(sx1 - center)
        dist2 = abs(sx2 - center)
        assert approx(dist2, 2 * dist1, tol=0.001)

    # --- Zoom to fit ---

    def test_zoom_to_fit_centers_pan(self):
        """zoom_to_fit centers the pan on the graph."""
        vp = Viewport(width=400, height=300)
        graph = Graph(width=200, height=100)

        vp.zoom_to_fit(graph, margin=0, animate=False)

        assert approx(vp.pan_x.current, 100)  # graph_width / 2
        assert approx(vp.pan_y.current, 50)   # graph_height / 2

    def test_zoom_to_fit_zoom_level(self):
        """zoom_to_fit sets zoom to fit the graph in the viewport.

        Viewport 400x300, graph 200x100, margin=0.
        Zoom = min(400/200, 300/100) = min(2.0, 3.0) = 2.0.
        """
        vp = Viewport(width=400, height=300)
        graph = Graph(width=200, height=100)

        vp.zoom_to_fit(graph, margin=0, animate=False)

        assert approx(vp.zoom.current, 2.0)

    def test_zoom_to_fit_with_margin(self):
        """zoom_to_fit accounts for margin.

        With margin, the effective viewport area is reduced, so the
        zoom level should be smaller than without margin.
        """
        vp_no_margin = Viewport(width=400, height=300)
        vp_with_margin = Viewport(width=400, height=300)
        graph = Graph(width=200, height=100)

        vp_no_margin.zoom_to_fit(graph, margin=0, animate=False)
        vp_with_margin.zoom_to_fit(graph, margin=20, animate=False)

        assert vp_with_margin.zoom.current < vp_no_margin.zoom.current

    def test_zoom_to_fit_degenerate_graph(self):
        """zoom_to_fit handles a zero-size graph without crashing."""
        vp = Viewport(width=400, height=300)
        # Graph with default dimensions (width=1, height=1)
        graph = Graph()

        # Should not raise
        vp.zoom_to_fit(graph, margin=0, animate=False)
        assert vp.zoom.current > 0

    # --- Visibility ---

    def test_is_visible_center(self):
        """A box at the center of the viewport is visible."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        assert vp.is_visible(40, 40, 60, 60)

    def test_is_visible_far_outside(self):
        """A box far outside the viewport is not visible."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        assert not vp.is_visible(1000, 1000, 1010, 1010)

    def test_is_visible_partial_overlap(self):
        """A box partially overlapping the viewport edge is visible."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        # Screen shows graph coords [0, 100] x [0, 100] at zoom 1 with pan at center.
        # A box that straddles the right edge:
        assert vp.is_visible(90, 40, 110, 60)

    def test_is_visible_fully_enclosing(self):
        """A box larger than the viewport that fully encloses it is visible."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        assert vp.is_visible(-100, -100, 200, 200)

    # --- Pan ---

    def test_pan_by(self):
        """Panning by a screen offset moves the pan target in the opposite graph direction."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(50)
        vp.pan_y.set_immediate(50)
        vp.zoom.set_immediate(1.0)

        initial_x = vp.pan_x.target
        initial_y = vp.pan_y.target

        vp.pan_by(10, 20)

        # Screen right/down -> graph pan left/up
        assert vp.pan_x.target < initial_x
        assert vp.pan_y.target < initial_y

    def test_pan_by_zoom_scaling(self):
        """Pan amount in graph space is inversely proportional to zoom.

        At zoom 2, a 10-pixel screen pan should move half as far in
        graph space as at zoom 1.
        """
        def pan_delta_x(zoom):
            vp = Viewport(width=100, height=100)
            vp.pan_x.set_immediate(50)
            vp.pan_y.set_immediate(50)
            vp.zoom.set_immediate(zoom)
            initial = vp.pan_x.target
            vp.pan_by(10, 0)
            return abs(vp.pan_x.target - initial)

        delta_z1 = pan_delta_x(zoom=1.0)
        delta_z2 = pan_delta_x(zoom=2.0)

        assert approx(delta_z2, delta_z1 / 2, tol=0.001)

    # --- Zoom ---

    def test_zoom_by(self):
        """zoom_by multiplies the current zoom."""
        vp = Viewport(width=100, height=100)
        vp.zoom.set_immediate(1.0)

        vp.zoom_by(2.0)
        assert vp.zoom.target == 2.0

        vp.zoom.set_immediate(vp.zoom.target)
        vp.zoom_by(0.5)
        assert vp.zoom.target == 1.0

    def test_zoom_min_limit(self):
        """Zoom does not go below min_zoom."""
        vp = Viewport(width=100, height=100)
        vp.min_zoom = 0.1
        vp.zoom.set_immediate(1.0)

        vp.zoom_by(0.001)
        assert vp.zoom.target >= vp.min_zoom

    def test_zoom_max_limit(self):
        """Zoom does not exceed max_zoom."""
        vp = Viewport(width=100, height=100)
        vp.max_zoom = 10.0
        vp.zoom.set_immediate(1.0)

        vp.zoom_by(1000)
        assert vp.zoom.target <= vp.max_zoom

    # --- Size ---

    def test_set_size(self):
        """set_size updates the viewport dimensions."""
        vp = Viewport(width=100, height=100)
        vp.set_size(800, 600)
        assert vp.width == 800
        assert vp.height == 600

    # --- zoom_to_bbox ---

    def test_zoom_to_bbox_centers(self):
        """zoom_to_bbox centers the pan on the bbox midpoint."""
        vp = Viewport(width=400, height=300)
        vp.zoom_to_bbox(20, 30, 80, 90, margin=0, animate=False)
        assert approx(vp.pan_x.current, 50)   # (20 + 80) / 2
        assert approx(vp.pan_y.current, 60)    # (30 + 90) / 2

    def test_zoom_to_bbox_correct_zoom(self):
        """zoom_to_bbox computes the correct zoom level.

        Viewport 400×300, bbox 60×60, margin=0.
        Zoom = min(400/60, 300/60) = min(6.67, 5.0) = 5.0.
        """
        vp = Viewport(width=400, height=300)
        vp.zoom_to_bbox(20, 30, 80, 90, margin=0, animate=False)
        assert approx(vp.zoom.current, 5.0)

    def test_zoom_to_bbox_degenerate_zero_area(self):
        """zoom_to_bbox with zero-area bbox doesn't crash and keeps current zoom."""
        vp = Viewport(width=400, height=300)
        vp.zoom.set_immediate(2.0)
        vp.zoom_to_bbox(50, 50, 50, 50, margin=0, animate=False)
        # Degenerate → zoom should not change
        assert approx(vp.zoom.current, 2.0)
        # But pan should center on the point
        assert approx(vp.pan_x.current, 50)
        assert approx(vp.pan_y.current, 50)

    # --- pan_to_point ---

    def test_pan_to_point_immediate(self):
        """pan_to_point with animate=False sets current immediately."""
        vp = Viewport(width=100, height=100)
        vp.pan_to_point(75.0, 25.0, animate=False)
        assert approx(vp.pan_x.current, 75.0)
        assert approx(vp.pan_y.current, 25.0)

    def test_pan_to_point_animated(self):
        """pan_to_point with animate=True sets target only (current unchanged)."""
        vp = Viewport(width=100, height=100)
        vp.pan_x.set_immediate(0)
        vp.pan_y.set_immediate(0)
        vp.pan_to_point(75.0, 25.0, animate=True)
        # Target set
        assert approx(vp.pan_x.target, 75.0)
        assert approx(vp.pan_y.target, 25.0)
        # Current unchanged
        assert approx(vp.pan_x.current, 0)
        assert approx(vp.pan_y.current, 0)


class TestPanClamping:
    """Keeping the view over the graph, so a pan cannot buy empty space the reader must cross back."""

    def _over_a_tall_graph(self, clamp: bool = True) -> Viewport:
        """A 400x300 viewport at 1:1 over a 600x2000 graph — taller than the view, narrower than it."""
        vp = Viewport(width=400, height=300)
        vp.set_graph_bounds(600.0, 2000.0)
        vp.clamp_pan = clamp
        return vp

    def test_off_by_default(self):
        # A viewer of arbitrary graphs may want the space; the clamp is for a view whose graph *is* the
        # content. This is also the control for everything below: without it they would pass against a
        # viewport that simply refused to pan.
        vp = Viewport(width=400, height=300)
        vp.set_graph_bounds(600.0, 2000.0)
        vp.pan_to_point(0.0, 99999.0, animate=False)
        assert approx(vp.pan_y.current, 99999.0)

    def test_a_pan_past_the_bottom_stops_at_the_bottom(self):
        vp = self._over_a_tall_graph()
        vp.pan_to_point(300.0, 99999.0, animate=False)
        # Half the viewport is 150 graph units at 1:1, so the lowest centre that shows no void is 2000-150.
        assert approx(vp.pan_y.current, 1850.0)

    def test_a_pan_past_the_top_stops_at_the_top(self):
        vp = self._over_a_tall_graph()
        vp.pan_to_point(300.0, -99999.0, animate=False)
        assert approx(vp.pan_y.current, 150.0)

    def test_a_pan_inside_the_graph_is_left_alone(self):
        # The other control: a clamp that pinned every pan would satisfy the two above perfectly.
        vp = self._over_a_tall_graph()
        vp.pan_to_point(300.0, 900.0, animate=False)
        assert approx(vp.pan_y.current, 900.0)

    def test_an_axis_the_graph_does_not_fill_is_centred(self):
        # 600 wide against 400 of viewport is wider, so x clamps; but zoomed out to 0.5 the viewport
        # spans 800 units and the graph cannot fill it, so there is nothing to clamp *to*. Centre it,
        # which is the arrangement that looks deliberate rather than stuck against one edge.
        vp = self._over_a_tall_graph()
        vp.zoom.target = 0.5
        vp.pan_to_point(99999.0, 900.0, animate=False)
        assert approx(vp.pan_x.current, 300.0)

    def test_the_bound_follows_the_zoom(self):
        # How much of the graph fits is what the bound depends on, so a zoom change re-clamps. Without
        # this, zooming out at the bottom edge shows void below the graph and the pan never recovers.
        vp = self._over_a_tall_graph()
        vp.pan_to_point(300.0, 99999.0, animate=False)
        assert approx(vp.pan_y.current, 1850.0)
        vp.zoom.target = 0.5           # the viewport now spans 600 units, so the bound moves up
        vp.update()
        assert approx(vp.pan_y.target, 1700.0)

    def test_zoom_to_bbox_is_clamped_against_the_zoom_it_is_going_to(self):
        # The bbox sits at the very bottom of the graph, so centring on it would show void below. The
        # clamp has to read the *new* zoom: against the old one it would compute the wrong bound.
        vp = self._over_a_tall_graph()
        vp.zoom_to_bbox(0.0, 1900.0, 600.0, 2000.0, margin=0, animate=False)
        half_view = 0.5 * vp.height / vp.zoom.current
        assert approx(vp.pan_y.current, 2000.0 - half_view)

    # --- Zooming out only as far as the graph goes ---
    #
    # The other half of the same promise. `clamp_pan` says the view holds no empty space beyond the graph,
    # and below the fill zoom the space appears whatever the pan does.

    def test_zooming_out_stops_where_the_whole_graph_fits(self):
        # 400x300 over 600x2000: the height is the binding dimension, so the graph is whole at 300/2000
        # and no smaller. Note this is the *smaller* of the two ratios — the larger one, 400/600, is
        # where the graph stops filling the view, and a floor there would make this graph impossible to
        # look at whole.
        vp = self._over_a_tall_graph()
        floor = 300.0 / 2000.0
        assert floor < 400.0 / 600.0, "this graph's proportions match the view's, so the two floors agree"
        vp.zoom.set_immediate(floor * 1.05)  # just above it, so one step must overshoot
        vp.zoom_by(0.5)
        assert approx(vp.zoom.target, floor)

    def test_zooming_out_is_not_capped_without_the_clamp(self):
        # The control, and the reason the cap is tied to `clamp_pan` rather than applied to every viewer:
        # without it the assertion above would hold for a viewport that simply refused to zoom.
        vp = self._over_a_tall_graph(clamp=False)
        vp.zoom.set_immediate(1.0)
        vp.zoom_by(0.5)
        assert approx(vp.zoom.target, 0.5)

    def test_zooming_in_is_not_capped(self):
        vp = self._over_a_tall_graph()
        vp.zoom.set_immediate(1.0)
        vp.zoom_by(2.0)
        assert approx(vp.zoom.target, 2.0)

    def test_a_view_already_below_the_floor_holds_rather_than_jumping_in(self):
        # An absolute zoom is obeyed as given, so the view can legitimately sit below the floor. Raising
        # it to meet one would answer a zoom-*out* press by zooming in, which is worse than not moving.
        vp = self._over_a_tall_graph()
        vp.zoom_to(0.1, animate=False)
        assert approx(vp.zoom.current, 0.1), "the absolute zoom was capped, so there is no case here"
        vp.zoom_by(0.5)
        assert approx(vp.zoom.target, 0.1)

    def test_a_graph_smaller_than_the_view_cannot_be_zoomed_further_out(self):
        # The floor is above 1:1 here — there is no zoom at which this graph fills the view, so every
        # step out only adds more of what the clamp exists to prevent. Holding still is the answer.
        vp = Viewport(width=400, height=300)
        vp.set_graph_bounds(100.0, 100.0)
        vp.clamp_pan = True
        vp.zoom.set_immediate(1.0)
        assert vp._fit_zoom() > 1.0, "this graph is bigger than the view, so it poses a different question"
        vp.zoom_by(0.5)
        assert approx(vp.zoom.target, 1.0)

    # --- Zooming about a point ---

    def test_zoom_to_keeps_the_named_point_where_it_is(self):
        vp = Viewport(width=400, height=300)
        vp.set_graph_bounds(600.0, 2000.0)
        vp.pan_x.set_immediate(300.0)
        vp.pan_y.set_immediate(1000.0)
        vp.zoom.set_immediate(1.0)
        anchor = (120.0, 60.0)  # a corner of the view, so centring instead would be obvious
        before = vp.screen_to_graph(*anchor)

        vp.zoom_to(2.0, anchor[0], anchor[1], animate=False)

        after = vp.screen_to_graph(*anchor)
        assert approx(after[0], before[0])
        assert approx(after[1], before[1])

    # --- Keeping a box on screen ---

    def test_a_box_already_on_screen_is_left_alone(self):
        vp = self._over_a_tall_graph()
        vp.pan_to_point(300.0, 1000.0, animate=False)
        vp.keep_box_visible(280.0, 980.0, 320.0, 1020.0, animate=False)
        assert approx(vp.pan_x.current, 300.0)
        assert approx(vp.pan_y.current, 1000.0)

    def test_a_box_over_an_edge_is_brought_back_and_no_further(self):
        # 400 wide at 1:1, so the view spans 100..500 about a pan of 300, and a box reaching to 560 hangs
        # 60 over the right edge. The pan moves by exactly that, not to the box's own centre.
        vp = self._over_a_tall_graph()
        vp.pan_to_point(300.0, 1000.0, animate=False)
        vp.keep_box_visible(460.0, 980.0, 560.0, 1020.0, animate=False)
        assert approx(vp.pan_x.current, 360.0)
        assert approx(vp.pan_y.current, 1000.0), "the vertical axis needed nothing and was moved anyway"

    def test_the_margin_is_kept_clear(self):
        vp = self._over_a_tall_graph()
        vp.pan_to_point(300.0, 1000.0, animate=False)
        vp.keep_box_visible(460.0, 980.0, 560.0, 1020.0, margin=10.0, animate=False)
        assert approx(vp.pan_x.current, 370.0)

    def test_a_box_too_big_for_the_view_is_centred(self):
        # Nothing can show it whole, so the two bounds cross. Its own middle is the most of it there is
        # room for, and it is the answer that does not depend on which edge it went over first.
        vp = self._over_a_tall_graph()
        vp.pan_to_point(300.0, 1000.0, animate=False)
        vp.keep_box_visible(0.0, 980.0, 600.0, 1020.0, animate=False)
        assert approx(vp.pan_x.current, 300.0)

    def test_zoom_to_without_a_point_turns_about_the_middle(self):
        # The control: the pan is what a fixed point moves, so an implementation that ignored the
        # argument would pass the test above and this one would not tell them apart without it.
        vp = Viewport(width=400, height=300)
        vp.set_graph_bounds(600.0, 2000.0)
        vp.pan_x.set_immediate(300.0)
        vp.pan_y.set_immediate(1000.0)
        vp.zoom.set_immediate(1.0)

        vp.zoom_to(2.0, animate=False)

        assert approx(vp.pan_x.current, 300.0)
        assert approx(vp.pan_y.current, 1000.0)
