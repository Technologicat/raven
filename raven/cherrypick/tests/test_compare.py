"""Tests for `raven.cherrypick.compare`: the keys compare mode claims, and the way out of it.

These exist because the keys moved here. They used to live in `app.py`, which parses the command line at
import and so cannot be imported under pytest at all — a key handler there is untestable by construction,
which is why every other component in the constellation carries its own `handle_key`.

`CompareMode` reaches its surroundings through injected callables, so a test supplies stubs and reads what
was asked for. It owns no DPG items; the import of `dearpygui` is for the key constants.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.cherrypick import compare as compare_module  # noqa: E402 -- after importorskip by design
from raven.cherrypick import config  # noqa: E402 -- ditto


class StubGrid:
    def __init__(self):
        self._compare_badges = {}
        self._compare_active_idx = -1
        self._compare_active_alpha = 0.0
        self.cleared = []

    def clear_compare_badges(self):
        self.cleared.append("badges")

    def clear_compare_active(self):
        self.cleared.append("active")

    def set_compare_badges(self, *args, **kwargs):
        pass

    def set_compare_active(self, *args, **kwargs):
        pass


class StubPreload:
    def __init__(self):
        self.unpinned = 0

    def unpin_all(self):
        self.unpinned += 1

    def pin(self, *args, **kwargs):
        pass


@pytest.fixture
def mode():
    """A `CompareMode` with stubs for everything it reaches, and a record of what it asked for."""
    asked = {"loaded": [], "status": 0, "exited": 0}

    built = compare_module.CompareMode(
        get_image_view=lambda: None,
        get_grid=lambda: StubGrid(),
        get_preload=lambda: StubPreload(),
        get_triage=lambda: None,
        load_image_fn=lambda idx: asked["loaded"].append(idx),
        set_status_fn=lambda text: None,
        update_status_fn=lambda: asked.__setitem__("status", asked["status"] + 1),
        on_exit_fn=lambda: asked.__setitem__("exited", asked["exited"] + 1),
    )
    # Straight into the running state, rather than through `enter`, which wants a real grid and preloader.
    # What these tests are about is dispatch and the way out, both of which read only `active`.
    built.active = True
    built.frame_list = [10, 11, 12]
    built.saved_current = 7
    return built, asked


class TestItOwnsTheKeyboardWhileItRuns:
    def test_a_key_it_does_not_know_is_still_taken(self, mode):
        # The suppression that used to be a bare `return` in the app's dispatcher. Falling through would
        # let the grid move its cursor under a cycling comparison, which is the thing being compared
        # changing while you compare it.
        built, asked = mode
        assert built.handle_key(dpg.mvKey_Q) is True
        assert built.active, "an unrecognised key did something"

    def test_it_takes_nothing_while_it_is_not_running(self, mode):
        # The control: a handler that claimed keys unconditionally would satisfy the test above and would
        # swallow the whole keyboard for the rest of the session.
        built, asked = mode
        built.active = False
        assert built.handle_key(dpg.mvKey_Escape) is False
        assert built.handle_key(dpg.mvKey_Q) is False


class TestTheKeys:
    def test_escape_leaves_and_goes_back_to_where_you_were(self, mode):
        built, asked = mode
        assert built.handle_key(dpg.mvKey_Escape) is True
        assert not built.active
        assert asked["loaded"] == [7], "Escape did not restore the image that was current on the way in"

    def test_shift_and_a_digit_picks_that_frame(self, mode):
        built, asked = mode
        assert built.handle_key(dpg.mvKey_2, shift=True) is True
        assert not built.active
        assert asked["loaded"] == [11], "the second frame of [10, 11, 12] is 11"

    def test_a_bare_digit_no_longer_picks(self, mode):
        # The change of 2026-09-09, and the reason for it: `1` means zoom to 1:1 across the constellation,
        # and one key cannot also mean "pick the first frame and leave". A bare digit is taken, since this
        # mode takes everything, but it does nothing.
        built, asked = mode
        assert built.handle_key(dpg.mvKey_2) is True
        assert built.active, "a bare digit exited compare mode"
        assert asked["loaded"] == []

    def test_a_digit_past_the_end_does_nothing_at_all(self, mode):
        # And in particular does not report an exit. It used to: the app called its exit hook beside
        # `select_frame` rather than leaving it to `exit`, so an out-of-range digit put the toolbar back
        # while the mode went on running.
        built, asked = mode
        assert built.handle_key(dpg.mvKey_9, shift=True) is True
        assert built.active
        assert asked["exited"] == 0, "the toolbar was handed back while the mode was still running"

    def test_the_fps_keys(self, mode):
        built, asked = mode
        before = built.fps
        built.handle_key(dpg.mvKey_Period)
        assert built.fps == pytest.approx(before + config.COMPARE_FPS_STEP)
        built.handle_key(dpg.mvKey_Comma)
        assert built.fps == pytest.approx(before)
        built.handle_key(dpg.mvKey_Period)
        built.handle_key(dpg.mvKey_M)
        assert built.fps == pytest.approx(config.COMPARE_DEFAULT_FPS), "M did not reset the rate"

    def test_space_pauses_and_resumes(self, mode):
        built, asked = mode
        built.handle_key(dpg.mvKey_Spacebar)
        assert built.paused
        built.handle_key(dpg.mvKey_Spacebar)
        assert not built.paused


class TestLeavingIsAnnouncedOnce:
    """`on_exit` fires from `exit`, so every route out reports itself and none reports twice.

    It used to be a second call each caller made beside `exit`, which is the shape that goes wrong in both
    directions: `select_frame` on an out-of-range digit announced an exit that had not happened, and any
    new route out would have had to remember.
    """

    def test_escape_announces_it(self, mode):
        built, asked = mode
        built.handle_key(dpg.mvKey_Escape)
        assert asked["exited"] == 1

    def test_picking_a_frame_announces_it(self, mode):
        built, asked = mode
        built.handle_key(dpg.mvKey_3, shift=True)
        assert asked["exited"] == 1

    def test_leaving_twice_announces_once(self, mode):
        built, asked = mode
        built.exit(restore=True)
        built.exit(restore=True)
        assert asked["exited"] == 1, "`exit` on an inactive mode announced a second departure"

    def test_the_shutdown_path_stays_quiet(self, mode):
        # `redraw=False` is teardown: the widgets the hook would put back are already going, and touching
        # them there is what the flag exists to avoid.
        built, asked = mode
        built.exit(restore=False, redraw=False)
        assert not built.active
        assert asked["exited"] == 0
