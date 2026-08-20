"""Tests for `raven.common.gui.tooltip`.

What these can reach is the tooltip's *bookkeeping*: that it builds a hidden window and binds a hover
handler, that "on screen" and enrolment in the sweeper's set cannot disagree, that a target deleted under a
showing tooltip takes it down, and that `destroy` releases what it took.

What they cannot reach is the resize itself, and the reason is worth stating so nobody adds a test that
hangs. Settling a new size means waiting for the window to be laid out, and a wait needs frames; frames need
a mapped viewport, which these tests deliberately do not have (see `test_animation.py`). Assignments made
here are handed to a worker that waits for a frame nobody will ever render — harmless, since it is a daemon
thread, but it does mean **no test may assert that assigned text has arrived**.
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
    def test_setting_the_same_text_is_a_no_op(self, target):
        """Which is what lets it skip the resize wait — and is why this one can be tested at all here."""
        tip = Tooltip(target, "same")
        tip.text = "same"  # no wait, so no RuntimeError even on the render thread
        assert tip.text == "same"

    def test_setting_text_from_the_render_thread_hands_off_instead_of_blocking(self, target):
        """A flash restores its message from `finish`, which the animator calls on the render thread.

        Settling the size means waiting for a frame, and that thread is the one that would have to draw it
        — so the assignment must return rather than wait. It must also not raise: this is an ordinary
        caller, not a misuse. pytest runs on the main thread, which is the render thread by Raven's
        convention, so simply getting here exercises the hand-off.
        """
        tip = Tooltip(target, "before")
        tip.text = "after"  # neither raises nor blocks; the worker settles it out of band


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
