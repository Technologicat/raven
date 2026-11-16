"""Tests for `raven.common.gui.pill`: what a pill builds, how each kind is coloured, and showing it."""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import animation  # noqa: E402 -- after importorskip by design
from raven.common.gui import pill  # noqa: E402
from raven.common.gui import tooltip as tooltip_module  # noqa: E402
from raven.common.gui import utils as guiutils  # noqa: E402


@pytest.fixture(scope="module")
def dpg_context():
    """A DPG context with an unmapped viewport, torn down after the module."""
    dpg.create_context()
    dpg.create_viewport(width=400, height=300)  # never shown: these tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def window(dpg_context):
    """A window to build pills in, emptied after each test."""
    with dpg.window() as win:
        yield win
    animation.animator.clear()
    tooltip_module._visible.clear()
    tooltip_module._pending.clear()
    dpg.delete_item(win)


def make_pill(window, tag="status", clicks=None):
    with dpg.group(parent=window):
        return pill.Pill(tag=tag, icon_font=0, callback=lambda: clicks.append(1) if clicks is not None else None)


def theme_of(widget):
    return dpg.get_item_theme(widget)


class TestWhatAPillBuilds:
    def test_its_widgets_are_named_after_its_tag(self, window):
        p = make_pill(window, tag="backend_status")
        assert (p.group, p.icon, p.button) == ("backend_status_pill", "backend_status_icon", "backend_status_button")
        for widget in (p.group, p.icon, p.button):
            assert dpg.does_item_exist(widget)

    def test_it_starts_hidden(self, window):
        assert not make_pill(window).shown

    def test_the_button_runs_the_callback(self, window):
        clicks = []
        p = make_pill(window, clicks=clicks)
        dpg.get_item_callback(p.button)()
        assert clicks == [1]


class TestSettingWhatItSays:
    def test_the_label_icon_and_tooltip_are_set(self, window):
        p = make_pill(window)
        p.set(label="LLM backend not connected", icon="X", tooltip="Nothing answers.", kind=pill.ALERT)
        assert dpg.get_item_label(p.button) == "LLM backend not connected"
        assert dpg.get_value(p.icon) == "X"
        assert dpg.get_item_configuration(p.icon)["show"]
        animation.animator.render_frame()  # a tooltip applies a new text on the animator's next tick
        assert p.tooltip.text == "Nothing answers."

    def test_without_an_icon_only_the_label_shows(self, window):
        p = make_pill(window)
        p.set(label="with", icon="X", kind=pill.OK)
        p.set(label="without", kind=pill.OK)
        assert not dpg.get_item_configuration(p.icon)["show"], "an icon left over from the previous set is hidden"

    def test_an_unknown_kind_is_refused_by_name(self, window):
        with pytest.raises(ValueError, match="sideways"):
            make_pill(window).set(label="x", kind="sideways")


class TestTheKindsColourIt:
    def test_an_alert_pulses_the_icon_and_keeps_the_label_steady(self, window):
        p = make_pill(window)
        p.set(label="x", icon="X", kind=pill.ALERT)
        assert theme_of(p.icon) in guiutils.item_identifiers(pill._PULSING_CAUTION_THEME)
        assert theme_of(p.button) in guiutils.item_identifiers(pill._CAUTION_THEME)

    def test_a_notice_is_the_caution_colour_and_steady(self, window):
        # The control is the alert above: same colour, and only the alert's icon pulses.
        p = make_pill(window)
        p.set(label="x", icon="X", kind=pill.NOTICE)
        assert theme_of(p.icon) in guiutils.item_identifiers(pill._CAUTION_THEME)
        assert theme_of(p.button) in guiutils.item_identifiers(pill._CAUTION_THEME)

    def test_ok_is_green_on_both(self, window):
        p = make_pill(window)
        p.set(label="x", icon="X", kind=pill.ALERT)
        p.set(label="x", icon="X", kind=pill.OK)
        assert theme_of(p.icon) in guiutils.item_identifiers(pill._OK_THEME)
        assert theme_of(p.button) in guiutils.item_identifiers(pill._OK_THEME)

    def test_re_setting_the_same_kind_leaves_a_flash_on_the_button(self, window):
        # A click flash binds its own theme to the button; a poll re-setting the pill must not end it.
        p = make_pill(window)
        p.set(label="x", icon="X", kind=pill.ALERT)
        with dpg.theme() as flash_theme:
            pass
        dpg.bind_item_theme(p.button, flash_theme)
        p.set(label="still down", icon="X", kind=pill.ALERT)
        assert theme_of(p.button) in guiutils.item_identifiers(flash_theme)
        p.set(label="up", icon="X", kind=pill.OK)
        assert theme_of(p.button) in guiutils.item_identifiers(pill._OK_THEME), "a change of kind does rebind"

    def test_every_pill_shares_one_set_of_themes(self, window):
        # So that two pills alerting side by side pulse in phase.
        first, second = make_pill(window, tag="first"), make_pill(window, tag="second")
        first.set(label="x", icon="X", kind=pill.ALERT)
        second.set(label="y", icon="Y", kind=pill.ALERT)
        assert theme_of(first.icon) == theme_of(second.icon)


class TestShowingIt:
    def test_shown_shows_and_hides_it(self, window):
        p = make_pill(window)
        p.shown = True
        assert p.shown and dpg.get_item_configuration(p.group)["show"]
        p.shown = False
        assert not p.shown and not dpg.get_item_configuration(p.group)["show"]
