"""Tests for `raven.common.gui.animation`'s `WidgetFlash` — the transient attention flash.

Most of this package is DPG glue and untested, but `WidgetFlash` carries two things worth
asserting: it *restores* what it borrowed (a widget's color or theme), and it has a de-duplication state
machine — at most one flash animates a given widget, and the losing instance ("ghost") must own nothing and
release nothing. Both are the kind of invariant that breaks silently: a wrong restore leaves a permanent mark
on a widget the flash was only supposed to point at, and a ghost that finalizes anyway evicts the instance
that is actually running.

DPG can run without a mapped window (`create_viewport` + `setup_dearpygui`, never `show_viewport`), so these
drive real widgets rather than mocks. The animation is stepped by calling `animator.render_frame()` directly,
the same way Raven's render loop does.
"""

import time

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import animation  # noqa: E402 -- after importorskip by design


TOOL_COLOR = (120, 200, 255)
FLASH_COLOR = (255, 255, 255)


@pytest.fixture(scope="module")
def dpg_context():
    """A DPG context with an unmapped viewport, torn down after the module."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: these tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def widgets(dpg_context):
    """A fresh text widget and button per test, in their own window."""
    with dpg.window() as window:
        text = dpg.add_text("gear", color=TOOL_COLOR)
        button = dpg.add_button(label="b")
    yield text, button
    animation.animator.clear()
    dpg.delete_item(window)


def _widget_color(item):
    """The widget's own color as 0-255 ints. DPG reports it normalized, hence the scaling."""
    return [round(255 * c) for c in dpg.get_item_configuration(item)["color"][:3]]


def _run_flash_to_completion(target, timeout=2.0):
    """Step the animator until `target`'s flash has finished (bounded, so a bug fails instead of hanging).

    Waits on this widget's own registration rather than on `animator.active_count`: the animator is a
    process-wide singleton, so an empty-animator condition would also wait on anything else running — and
    would report *this* flash as stuck if something ambient never ends. A reified flash removes itself from
    `WidgetFlash.instances` as it finishes, which is exactly the event of interest.
    """
    deadline = time.monotonic() + timeout
    while target in animation.WidgetFlash.instances and time.monotonic() < deadline:
        animation.animator.render_frame()
    assert target not in animation.WidgetFlash.instances, "flash did not finish within the timeout"


class TestTextTarget:
    def test_flash_brightens_then_restores_the_widgets_own_color(self, widgets):
        """A text widget has no background, so the flash rides its text color — and must hand it back intact."""
        text, _ = widgets
        assert _widget_color(text) == list(TOOL_COLOR)

        animation.highlight_widget(widget=text, duration=0.5, color=FLASH_COLOR)
        animation.animator.render_frame()
        assert _widget_color(text) == list(FLASH_COLOR)

        _run_flash_to_completion(text)
        assert _widget_color(text) == list(TOOL_COLOR)

    def test_deregisters_when_finished(self, widgets):
        """Otherwise the widget could never be flashed a second time."""
        text, _ = widgets
        animation.highlight_widget(widget=text, duration=0.3, color=FLASH_COLOR)
        assert text in animation.WidgetFlash.instances
        _run_flash_to_completion(text)
        assert text not in animation.WidgetFlash.instances

    def test_survives_the_widget_being_deleted_mid_flash(self, widgets):
        """A chat view rebuild deletes widgets under running animations; that must not raise."""
        text, _ = widgets
        animation.highlight_widget(widget=text, duration=5.0, color=FLASH_COLOR)
        animation.animator.render_frame()
        dpg.delete_item(text)
        _run_flash_to_completion(text)  # the flash notices the widget is gone and ends


class TestButtonTarget:
    def test_restores_the_theme_the_widget_actually_had(self, widgets):
        """Not a fixed theme: flashing a widget that had none must not leave one bound to it."""
        _, button = widgets
        assert dpg.get_item_theme(button) is None

        animation.flash_button(button=button, message=None, duration=0.3)
        animation.animator.render_frame()
        assert dpg.get_item_theme(button) is not None  # the flash theme is bound while it runs

        _run_flash_to_completion(button)
        assert dpg.get_item_theme(button) is None

    def test_restores_a_pre_existing_theme(self, widgets):
        """The other half of the same contract: a widget that had a theme gets that same theme back."""
        _, button = widgets
        with dpg.theme() as original:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (10, 20, 30))
        dpg.bind_item_theme(button, original)

        animation.flash_button(button=button, message=None, duration=0.3)
        _run_flash_to_completion(button)
        assert dpg.get_item_theme(button) == original


class TestDeduplication:
    def test_second_flash_on_the_same_widget_does_not_reify(self, widgets):
        """At most one flash owns a widget; the loser goes into ghost mode."""
        text, _ = widgets
        animation.highlight_widget(widget=text, duration=5.0, color=FLASH_COLOR)
        reified = animation.WidgetFlash.instances[text]

        ghost = animation.WidgetFlash(message=None, target=text, target_tooltip=None, target_text=None,
                                      original_theme=0, duration=5.0)
        assert not ghost.reified
        assert animation.WidgetFlash.instances[text] is reified

    def test_finalizing_a_ghost_leaves_the_running_instance_alone(self, widgets):
        """`Animator.clear` finalizes every registered animation, ghosts included — which must be a no-op.

        A ghost owns no resources, so acting as though it did would restore a widget its twin is still
        animating, and evict that twin from the registry (after which the next flash would bind over it).
        """
        text, _ = widgets
        animation.highlight_widget(widget=text, duration=5.0, color=FLASH_COLOR)
        reified = animation.WidgetFlash.instances[text]
        animation.animator.render_frame()
        color_while_running = _widget_color(text)

        ghost = animation.WidgetFlash(message=None, target=text, target_tooltip=None, target_text=None,
                                      original_theme=0, duration=5.0)
        ghost.finish()

        assert animation.WidgetFlash.instances.get(text) is reified
        assert _widget_color(text) == color_while_running  # the running flash was not undone
