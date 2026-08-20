"""Tests for `raven.common.gui.tooltip`.

What these can reach is the tooltip's *bookkeeping*: that it builds a hidden window and binds a hover
handler, that "on screen" and enrolment in the sweeper's set cannot disagree, that a target deleted under a
showing tooltip takes it down, and that `destroy` releases what it took.

What they cannot reach is the resize itself, and the reason is worth stating so nobody adds a test that
hangs. Settling a new size means waiting for the window to be laid out, and a wait needs frames; frames need
a mapped viewport, which these tests deliberately do not have (see `test_animation.py`). What
*is* reachable is the two-frame state machine that applies a change, because it advances on animator ticks
rather than by waiting: `animator.render_frame()` drives it here exactly as the render loop would.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import animation, tooltip as tooltip_module  # noqa: E402 -- after importorskip by design
from raven.common.gui.tooltip import Tooltip  # noqa: E402


@pytest.fixture(scope="module")
def dpg_context():
    """A DPG context with an unmapped viewport, torn down after the module."""
    dpg.create_context()
    dpg.create_viewport(width=400, height=300)  # never shown: these tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def target(dpg_context):
    """A button to hang a tooltip on, in its own window."""
    with dpg.window() as window:
        button = dpg.add_button(label="b")
    yield button
    animation.animator.clear()
    tooltip_module._visible.clear()
    tooltip_module._pending.clear()
    dpg.delete_item(window)


class TestConstruction:
    def test_builds_a_hidden_window_holding_the_text(self, target):
        tip = Tooltip(target, "hello")
        assert dpg.does_item_exist(tip.window)
        assert not dpg.is_item_shown(tip.window)
        assert tip.text == "hello"

    def test_binds_a_hover_handler_to_the_target(self, target):
        """Appearing is event-driven; without this the tooltip could never come up."""
        tip = Tooltip(target)
        assert dpg.get_item_info(target)["handlers"] == tip.handler_registry

    def test_a_missing_target_is_not_fatal(self, dpg_context):
        """A chat view rebuild can delete the widget between deciding to build a tooltip and building it."""
        with dpg.window() as window:
            doomed = dpg.add_button(label="b")
        dpg.delete_item(doomed)
        tip = Tooltip(doomed, "orphan")  # must not raise
        assert not tip.should_be_visible
        dpg.delete_item(window)


class TestText:
    def test_setting_the_same_text_does_not_move_the_window(self, target):
        """Nothing changes size, so there is nothing to settle and no reason to park anything offscreen."""
        tip = Tooltip(target, "same")
        tip.text = "same"
        animation.animator.render_frame()
        assert tip.text == "same"
        assert not dpg.is_item_shown(tip.window), "never shown, not even offscreen"

    def test_setting_text_from_the_render_thread_neither_waits_nor_raises(self, target):
        """A flash restores its message from `finish`, which the animator calls on the render thread.

        Anything the animator ticks runs *inside* the render loop, so it must never wait for a frame — the
        thread that would have to draw it is the one waiting. Assignment therefore queues rather than
        blocks, and must not raise either: this is an ordinary caller, not a misuse. pytest runs on the main
        thread, which is the render thread by Raven's convention, so getting here at all exercises it.
        """
        tip = Tooltip(target, "before")
        tip.text = "after"  # returns immediately; the sweeper carries it

    def test_a_queued_change_lands_over_the_next_two_frames(self, target):
        """One frame to apply it offscreen and let the window resize there, one to put the window back.

        The offscreen step is the whole trick, so it is what the first tick has to do: parking the window
        where the mis-sized frame cannot be seen, rather than skipping a frame that has to happen somewhere.
        """
        tip = Tooltip(target, "before")
        tip.text = "after"
        assert tip.text == "before", "queued, not yet applied"
        assert tip in tooltip_module._pending

        animation.animator.render_frame()
        assert tip.text == "after", "applied, and the window is resizing offscreen"
        assert tip in tooltip_module._pending, "still in flight until it has been put back"

        animation.animator.render_frame()
        assert tip not in tooltip_module._pending, "settled"
        assert not dpg.is_item_shown(tip.window), "and hidden again, since nothing is hovering the target"

    def test_a_change_to_an_unhovered_tooltip_is_still_settled(self, target):
        """Otherwise the resize is merely deferred to the next hover, which is the same glitch, later."""
        tip = Tooltip(target, "before")
        assert not tip._shown
        tip.text = "after"
        assert tip in tooltip_module._pending, "queued even though nothing is on screen"


class TestVisibility:
    def test_showing_enrols_and_hiding_removes(self, target):
        """The sweeper's set means exactly "currently on screen"; the two must not be able to disagree."""
        tip = Tooltip(target, "hi")
        tip._on_hover(None, None, None)
        assert tip._shown and tip in tooltip_module._visible

        tip._hide()
        assert not tip._shown and tip not in tooltip_module._visible
        assert not dpg.is_item_shown(tip.window)

    def test_the_sweeper_takes_down_a_tooltip_whose_target_vanished(self, target):
        """The case with no un-hover event to rely on: nothing will ever report the mouse leaving.

        A deleted target reports as not hovered, so the same sweep that handles an ordinary mouse-out
        handles this too — which is the reason the sweep is by state rather than by event.
        """
        tip = Tooltip(target, "hi")
        tip._on_hover(None, None, None)
        assert tip in tooltip_module._visible

        dpg.delete_item(target)
        animation.animator.render_frame()

        assert not tip._shown
        assert tip not in tooltip_module._visible

    def test_a_second_hover_while_already_up_does_not_re_enrol(self, target):
        """The hover handler fires every frame the mouse stays put, so it has to be idempotent."""
        tip = Tooltip(target, "hi")
        tip._on_hover(None, None, None)
        tip._on_hover(None, None, None)
        assert len([t for t in tooltip_module._visible if t is tip]) == 1


class TestTeardown:
    def test_destroy_releases_the_window_and_the_registry(self, target):
        tip = Tooltip(target, "bye")
        window, registry = tip.window, tip.handler_registry
        tip.destroy()
        assert not dpg.does_item_exist(window)
        assert not dpg.does_item_exist(registry)

    def test_destroy_while_showing_leaves_nothing_enrolled(self, target):
        """Otherwise the sweeper would keep a dead tooltip alive by holding the only reference to it."""
        tip = Tooltip(target, "bye")
        tip._on_hover(None, None, None)
        tip.destroy()
        assert tip not in tooltip_module._visible
