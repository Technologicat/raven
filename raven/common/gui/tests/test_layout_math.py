"""Tests for the layout coordinate math in raven.common.gui.layout_math.

These functions are pure math — no DPG dependency.
"""

from raven.common.tests import approx
from raven.common.gui.layout_math import (screen_to_content, content_to_screen,
                                          zoom_keep_point, compute_zoom_to_fit,
                                          decide_tail_follow)


# ---------------------------------------------------------------------------
# Tests: screen_to_content / content_to_screen
# ---------------------------------------------------------------------------

class TestCoordinateTransforms:
    """Test the screen ↔ content coordinate conversions."""

    def test_origin_at_center(self):
        """With pan at origin and zoom 1, screen center maps to content origin."""
        cx, cy = screen_to_content(50, 50, pan_cx=0, pan_cy=0,
                                   zoom=1.0, view_w=100, view_h=100)
        assert approx(cx, 0)
        assert approx(cy, 0)

    def test_roundtrip(self):
        """content_to_screen and screen_to_content are inverses."""
        pan_cx, pan_cy = 100, 50
        zoom = 2.0
        view_w, view_h = 800, 600

        cx, cy = 75.0, 25.0
        sx, sy = content_to_screen(cx, cy, pan_cx, pan_cy, zoom, view_w, view_h)
        cx2, cy2 = screen_to_content(sx, sy, pan_cx, pan_cy, zoom, view_w, view_h)

        assert approx(cx, cx2, tol=0.001)
        assert approx(cy, cy2, tol=0.001)

    def test_zoom_scales_distance(self):
        """Doubling zoom doubles screen distance from center."""
        pan_cx, pan_cy = 50, 50
        view_w, view_h = 100, 100

        sx1, _ = content_to_screen(60, 60, pan_cx, pan_cy,
                                   zoom=1.0, view_w=view_w, view_h=view_h)
        sx2, _ = content_to_screen(60, 60, pan_cx, pan_cy,
                                   zoom=2.0, view_w=view_w, view_h=view_h)

        center = view_w / 2
        assert approx(abs(sx2 - center), 2 * abs(sx1 - center), tol=0.001)

    def test_pan_shifts_content(self):
        """Changing pan shifts which content point appears at screen center."""
        view_w, view_h = 100, 100
        zoom = 1.0

        # Screen center should map to the pan point.
        cx, cy = screen_to_content(50, 50, pan_cx=30, pan_cy=40,
                                   zoom=zoom, view_w=view_w, view_h=view_h)
        assert approx(cx, 30)
        assert approx(cy, 40)

    def test_zero_zoom_guard(self):
        """Zero zoom doesn't crash (falls back to zoom=1)."""
        cx, cy = screen_to_content(50, 50, pan_cx=0, pan_cy=0,
                                   zoom=0, view_w=100, view_h=100)
        assert approx(cx, 0)
        assert approx(cy, 0)


# ---------------------------------------------------------------------------
# Tests: zoom_keep_point
# ---------------------------------------------------------------------------

class TestZoomKeepPoint:
    """Test the zoom-while-keeping-a-screen-point-stationary function."""

    def test_point_stays_stationary(self):
        """The screen point maps to the same content coord before and after."""
        pan_cx, pan_cy = 50, 50
        view_w, view_h = 800, 600
        old_zoom = 1.0
        new_zoom = 2.0
        sx, sy = 300, 200

        # Content coord at (sx, sy) before zoom.
        gx_before, gy_before = screen_to_content(sx, sy, pan_cx, pan_cy,
                                                 old_zoom, view_w, view_h)

        # New pan after zoom.
        new_pan_cx, new_pan_cy = zoom_keep_point(old_zoom, new_zoom,
                                                 sx, sy,
                                                 pan_cx, pan_cy,
                                                 view_w, view_h)

        # Content coord at (sx, sy) after zoom with new pan.
        gx_after, gy_after = screen_to_content(sx, sy, new_pan_cx, new_pan_cy,
                                               new_zoom, view_w, view_h)

        assert approx(gx_before, gx_after, tol=0.001)
        assert approx(gy_before, gy_after, tol=0.001)

    def test_center_zoom_preserves_pan(self):
        """Zooming toward the screen center shouldn't change pan."""
        pan_cx, pan_cy = 50, 50
        view_w, view_h = 100, 100
        # Screen center = (50, 50)
        new_pan_cx, new_pan_cy = zoom_keep_point(1.0, 2.0,
                                                 50, 50,
                                                 pan_cx, pan_cy,
                                                 view_w, view_h)
        assert approx(new_pan_cx, pan_cx, tol=0.001)
        assert approx(new_pan_cy, pan_cy, tol=0.001)

    def test_zoom_out(self):
        """Also works for zooming out (factor < 1)."""
        pan_cx, pan_cy = 100, 200
        view_w, view_h = 640, 480
        sx, sy = 400, 300

        gx_before, gy_before = screen_to_content(sx, sy, pan_cx, pan_cy,
                                                 2.0, view_w, view_h)
        new_pan_cx, new_pan_cy = zoom_keep_point(2.0, 0.5,
                                                 sx, sy,
                                                 pan_cx, pan_cy,
                                                 view_w, view_h)
        gx_after, gy_after = screen_to_content(sx, sy, new_pan_cx, new_pan_cy,
                                               0.5, view_w, view_h)

        assert approx(gx_before, gx_after, tol=0.001)
        assert approx(gy_before, gy_after, tol=0.001)


# ---------------------------------------------------------------------------
# Tests: compute_zoom_to_fit
# ---------------------------------------------------------------------------

class TestComputeZoomToFit:
    """Test the zoom-to-fit computation."""

    def test_landscape_content_in_landscape_view(self):
        """Wide content in a wide view — width-limited."""
        zoom, pan_cx, pan_cy = compute_zoom_to_fit(200, 100, 400, 300, margin=0)
        # min(400/200, 300/100) = min(2, 3) = 2
        assert approx(zoom, 2.0)
        assert approx(pan_cx, 100)  # content_w / 2
        assert approx(pan_cy, 50)   # content_h / 2

    def test_portrait_content_in_landscape_view(self):
        """Tall content in a wide view — height-limited."""
        zoom, pan_cx, pan_cy = compute_zoom_to_fit(100, 300, 400, 300, margin=0)
        # min(400/100, 300/300) = min(4, 1) = 1
        assert approx(zoom, 1.0)

    def test_margin_reduces_zoom(self):
        """Margin reduces available space, so zoom is smaller."""
        zoom_no_margin, _, _ = compute_zoom_to_fit(200, 100, 400, 300, margin=0)
        zoom_with_margin, _, _ = compute_zoom_to_fit(200, 100, 400, 300, margin=20)
        assert zoom_with_margin < zoom_no_margin

    def test_centers_on_content(self):
        """Pan is always at the content center."""
        _, pan_cx, pan_cy = compute_zoom_to_fit(300, 200, 800, 600, margin=10)
        assert approx(pan_cx, 150)
        assert approx(pan_cy, 100)

    def test_zero_content_fallback(self):
        """Zero-size content returns safe defaults."""
        zoom, pan_cx, pan_cy = compute_zoom_to_fit(0, 0, 800, 600)
        assert zoom == 1.0
        assert pan_cx == 0.0
        assert pan_cy == 0.0

    def test_zero_view_fallback(self):
        """Zero-size view returns safe defaults."""
        zoom, pan_cx, pan_cy = compute_zoom_to_fit(200, 100, 0, 0)
        assert zoom == 1.0
        assert pan_cx == 0.0
        assert pan_cy == 0.0


# ---------------------------------------------------------------------------
# Tests: decide_tail_follow
# ---------------------------------------------------------------------------

class TestDecideTailFollow:
    """Whether a growing log keeps following its end, told apart from the reader having scrolled away.

    Each case that turns on one input is paired with a control changing only that input, so that a passing
    assertion cannot be explained by the branch under test never having engaged.
    """

    TOLERANCE = 40

    def decide(self, **kwargs):
        args = dict(maybe_target_y_scroll=None, last_step=0.0, maybe_commanded_y_scroll=None,
                    commanded_to_end=False, tolerance=self.TOLERANCE)
        args.update(kwargs)
        return decide_tail_follow(**args)

    def test_content_arriving_below_a_followed_view_keeps_it_following(self):
        # The case the function exists for: the maximum grew by 300 px, the position is where we put it.
        grown = dict(y_scroll=1000, max_y_scroll=1300, maybe_commanded_y_scroll=1000)
        assert not self.decide(**grown).at_end, "the gap should exceed the tolerance, or this tests nothing"
        assert self.decide(**grown, commanded_to_end=True).follow
        assert not self.decide(**grown, commanded_to_end=False).follow, \
            "a view we were not following should not start following because content arrived"

    def test_a_reader_scrolling_up_stops_the_following(self):
        moved = self.decide(y_scroll=600, max_y_scroll=1300, maybe_commanded_y_scroll=1000, commanded_to_end=True)
        assert moved.maybe_drift == 400
        assert not moved.undisturbed
        assert not moved.follow

    def test_a_reader_scrolling_back_to_the_end_resumes_it(self):
        # How the following recovers: at the end by position, however it got there.
        assert self.decide(y_scroll=1290, max_y_scroll=1300, maybe_commanded_y_scroll=600).follow

    def test_a_scroll_in_flight_is_judged_by_where_it_is_heading(self):
        # The reader pressed Up: the view has barely moved, but the animation is heading 400 px up.
        in_flight = dict(y_scroll=1290, max_y_scroll=1300, maybe_commanded_y_scroll=1250, last_step=40)
        assert not self.decide(**in_flight, maybe_target_y_scroll=900).follow
        assert self.decide(**in_flight, maybe_target_y_scroll=None).follow, \
            "without a target the position is at the end, so the target is what decided"

    def test_one_frame_of_our_own_animation_is_not_taken_for_the_reader(self):
        # The view reports one frame behind the value last written; early in the decay a frame is large.
        lagging = dict(y_scroll=900, max_y_scroll=1600, maybe_commanded_y_scroll=970, commanded_to_end=True,
                       maybe_target_y_scroll=1600)
        assert self.decide(**lagging, last_step=80).undisturbed
        assert not self.decide(**lagging, last_step=0.0).undisturbed, \
            "70 px is past the plain tolerance, so the slack is what accepted it"

    def test_content_shrinking_under_a_command_is_not_taken_for_the_reader(self):
        # The view clamps the position to the new maximum by itself, so the command is compared clamped.
        assert self.decide(y_scroll=800, max_y_scroll=800, maybe_commanded_y_scroll=1000,
                           commanded_to_end=True).undisturbed

    def test_with_nothing_commanded_only_the_position_decides(self):
        assert not self.decide(y_scroll=0, max_y_scroll=1300, commanded_to_end=True).follow

    def test_a_view_with_no_scrollbar_follows(self):
        assert self.decide(y_scroll=0, max_y_scroll=0).follow
