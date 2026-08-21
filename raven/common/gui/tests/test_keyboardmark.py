"""Tests for `raven.common.gui.keyboardmark`.

Three things here break silently, and each is one the eye cannot check either:

- **Which style variable a kind sets.** `FrameBorderSize` on a panel borders every button inside it and
  `ChildBorderSize` on a row of buttons does nothing at all, so a swapped mapping shows up as a mark that
  is merely in the wrong place — or absent — rather than as anything that fails.
- **That marks lit together share one animation.** Two `PulsatingColor`s at the same period but different
  start times drift half a cycle apart and read as two things blinking at each other. Nothing errors.
- **That a mark gives back the theme it displaced.** DPG binds one theme per item, so a mark on a widget
  that already had one is a theft that has to be undone exactly.

These need no layout, so they run headless: a DPG context with an unmapped viewport, and the animation
stepped by calling `animator.render_frame()` the way Raven's render loop does.

Theme contents are read back through the item tree — a theme's components, their colour and style items,
each carrying the `target` it applies to — which is how these assert against what an app would see rather
than against the component's private attributes.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import animation as gui_animation  # noqa: E402 -- after importorskip by design
from raven.common.gui import keyboardmark  # noqa: E402 -- ditto


@pytest.fixture(scope="module")
def dpg_context():
    """A DPG context with an unmapped viewport, torn down after the module."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: these tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def quiet_pulse(dpg_context):
    """Leave the shared pulse stopped, whatever the test did to it.

    It is module state and the animator is a process-wide singleton, so a mark left lit would keep writing
    a colour for the rest of the session and make the next test's "is the pulse running" ambiguous.
    """
    yield
    if keyboardmark.pulse_is_running():
        gui_animation.animator.cancel(keyboardmark._pulse)
        keyboardmark._pulse = None


@pytest.fixture
def make_widget(dpg_context, request):
    """Build a uniquely tagged widget. A shared context means tags must be unique across the module."""
    created = []

    def factory(kind="button", suffix=""):
        tag = f"{request.node.name}_{kind}{suffix}"
        if not dpg.does_item_exist("kbmark_test_window"):
            dpg.add_window(tag="kbmark_test_window")
        if kind == "child_window":
            widget = dpg.add_child_window(tag=tag, parent="kbmark_test_window", width=50, height=50)
        elif kind == "text":
            widget = dpg.add_text("*", tag=tag, parent="kbmark_test_window")
        else:
            widget = dpg.add_button(tag=tag, parent="kbmark_test_window")
        created.append(widget)
        return widget

    yield factory
    for widget in created:
        dpg.delete_item(widget)


def theme_items(widget):
    """What the theme bound to `widget` sets: a list of `(item type, target, value)`.

    The route an app would take to ask the same question — `get_item_theme`, then the theme's components,
    then each component's colour and style items, each of which reports the style variable or colour slot
    it targets in its configuration.
    """
    theme = dpg.get_item_theme(widget)
    if theme is None:
        return []
    out = []
    for component in dpg.get_item_children(theme, slot=1):
        for item in dpg.get_item_children(component, slot=1):
            kind = dpg.get_item_info(item)["type"].split("::")[-1]
            out.append((kind, dpg.get_item_configuration(item)["target"], dpg.get_value(item)))
    return out


def colors(widget):
    """The colour values the theme bound to `widget` sets, in order."""
    return [value for kind, _target, value in theme_items(widget) if kind == "mvThemeColor"]


def targets(widget, kind):
    """The style variables (or colour slots) the theme bound to `widget` sets, in order."""
    return [target for k, target, _value in theme_items(widget) if k == kind]


# --------------------------------------------------------------------------------
# What a mark puts on a widget

class TestMarkAppearance:
    def test_a_new_mark_is_invisible_until_it_is_lit(self, make_widget, quiet_pulse):
        widget = make_widget()
        mark = keyboardmark.Mark(widget)
        assert mark.lit is False
        assert [color[3] for color in colors(widget)] == [0.0]
        assert not keyboardmark.pulse_is_running()

    def test_lighting_a_mark_gives_it_the_shared_colour(self, make_widget, quiet_pulse):
        widget = make_widget()
        mark = keyboardmark.Mark(widget)
        mark.lit = True
        assert mark.lit is True
        assert colors(widget)[0][:3] == list(keyboardmark.COLOR[:3])
        assert colors(widget)[0][3] > 0.0
        assert keyboardmark.pulse_is_running()

    def test_the_kind_decides_which_edge_gets_the_border(self, make_widget, quiet_pulse):
        """A swapped mapping here is silent: the wrong style variable borders the wrong thing, or nothing."""
        framed = keyboardmark.Mark(make_widget("button"), kind=keyboardmark.MarkKind.FRAME)
        panel = keyboardmark.Mark(make_widget("child_window"), kind=keyboardmark.MarkKind.PANEL)
        dot = keyboardmark.Mark(make_widget("text"), kind=keyboardmark.MarkKind.DOT)

        assert targets(framed.target, "mvThemeStyle") == [dpg.mvStyleVar_FrameBorderSize]
        assert targets(panel.target, "mvThemeStyle") == [dpg.mvStyleVar_ChildBorderSize]
        assert targets(dot.target, "mvThemeStyle") == []  # a glyph has no edge to widen

        assert targets(framed.target, "mvThemeColor") == [dpg.mvThemeCol_Border]
        assert targets(panel.target, "mvThemeColor") == [dpg.mvThemeCol_Border]
        assert targets(dot.target, "mvThemeColor") == [dpg.mvThemeCol_Text]

    def test_the_mark_can_be_narrowed_to_one_kind_of_descendant(self, make_widget, quiet_pulse):
        """What lets a widget be marked inside a row it shares with widgets that must not be."""
        widget = make_widget()
        keyboardmark.Mark(widget, item_type=dpg.mvInputText)
        theme = dpg.get_item_theme(widget)
        component = dpg.get_item_children(theme, slot=1)[0]
        assert dpg.get_item_configuration(component)["item_type"] == dpg.mvInputText

    def test_the_border_is_as_thick_as_it_was_asked_to_be(self, make_widget, quiet_pulse):
        widget = make_widget()
        keyboardmark.Mark(widget, thickness=5)
        style_values = [value for kind, _target, value in theme_items(widget) if kind == "mvThemeStyle"]
        assert style_values[0][0] == 5.0


# --------------------------------------------------------------------------------
# One rhythm

class TestSharedPulse:
    def test_marks_lit_together_breathe_as_one(self, make_widget, quiet_pulse):
        """Same animation, so same phase. Two of their own would drift apart and read as two signals."""
        first = keyboardmark.Mark(make_widget("button", "1"))
        second = keyboardmark.Mark(make_widget("button", "2"))

        before = gui_animation.animator.active_count
        first.lit = True
        second.lit = True
        assert gui_animation.animator.active_count == before + 1

        gui_animation.animator.render_frame()
        assert colors(first.target)[0] == colors(second.target)[0]

    def test_a_mark_lit_later_joins_the_pulse_already_running(self, make_widget, quiet_pulse):
        first = keyboardmark.Mark(make_widget("button", "1"))
        second = keyboardmark.Mark(make_widget("button", "2"))

        first.lit = True
        gui_animation.animator.render_frame()
        pulse = keyboardmark._pulse

        second.lit = True
        gui_animation.animator.render_frame()
        assert keyboardmark._pulse is pulse  # not restarted, so not re-phased
        assert colors(first.target)[0] == colors(second.target)[0]

    def test_a_mark_going_dark_leaves_the_others_breathing(self, make_widget, quiet_pulse):
        first = keyboardmark.Mark(make_widget("button", "1"))
        second = keyboardmark.Mark(make_widget("button", "2"))
        first.lit = True
        second.lit = True

        first.lit = False
        gui_animation.animator.render_frame()
        assert keyboardmark.pulse_is_running()
        assert colors(first.target)[0][3] == 0.0
        assert colors(second.target)[0][3] > 0.0

    def test_the_pulse_stops_once_the_last_mark_is_dark(self, make_widget, quiet_pulse):
        """An app with nothing marked must not be writing a colour nobody can see, once a frame, forever."""
        mark = keyboardmark.Mark(make_widget())
        mark.lit = True
        assert keyboardmark.pulse_is_running()

        mark.lit = False
        assert not keyboardmark.pulse_is_running()
        assert colors(mark.target)[0][3] == 0.0

    def test_a_dark_mark_is_no_longer_written_by_the_pulse(self, make_widget, quiet_pulse):
        """It has to *leave* the animation, not merely be painted over: the next frame would relight it.

        This says nothing about the order of the two steps inside `leave_pulse`. Nothing renders between
        them here, so a version that painted first and left afterwards passes this too — the race it would
        lose is against the render thread, and a single-threaded test cannot stage it.
        """
        first = keyboardmark.Mark(make_widget("button", "1"))
        second = keyboardmark.Mark(make_widget("button", "2"))
        first.lit = True
        second.lit = True

        first.lit = False
        assert first._color_widget not in keyboardmark._pulse.theme_color_widgets
        for _ in range(3):
            gui_animation.animator.render_frame()
        assert colors(first.target)[0][3] == 0.0

    def test_lighting_a_lit_mark_again_changes_nothing(self, make_widget, quiet_pulse):
        mark = keyboardmark.Mark(make_widget())
        mark.lit = True
        before = gui_animation.animator.active_count
        mark.lit = True
        assert gui_animation.animator.active_count == before
        assert keyboardmark._pulse.theme_color_widgets.count(mark._color_widget) == 1


# --------------------------------------------------------------------------------
# A mark that moves

class TestMovingMark:
    """One mark over many candidates — the Visualizer's current entry, Librarian's current message.

    How many candidates there are is the user's business and has no upper bound: a selection can run to
    thousands of entries, a chat to thousands of messages. A theme apiece would scale with that, so the
    mark moves instead, and the widget it leaves has to come back exactly as it was.
    """

    def test_a_mark_can_be_built_with_nothing_to_mark_yet(self, make_widget, quiet_pulse):
        mark = keyboardmark.Mark(None)
        assert mark.target is None
        assert mark.lit is False

    def test_moving_the_mark_takes_it_off_the_widget_it_leaves(self, make_widget, quiet_pulse):
        first = make_widget("button", "1")
        second = make_widget("button", "2")
        mark = keyboardmark.Mark(None)

        mark.target = first
        mark.lit = True
        assert colors(first)[0][3] > 0.0

        mark.target = second
        assert theme_items(first) == []  # unthemed again, so nothing of the mark is left on it
        assert colors(second)[0][3] > 0.0

    def test_the_widget_it_leaves_gets_its_own_theme_back(self, make_widget, quiet_pulse):
        """The shape a moving mark is meant for: every candidate wears a theme saying *not me*."""
        first = make_widget("button", "1")
        second = make_widget("button", "2")
        with dpg.theme() as not_me:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (*keyboardmark.COLOR[:3], 0))
        for widget in (first, second):
            dpg.bind_item_theme(widget, not_me)

        mark = keyboardmark.Mark(None, kind=keyboardmark.MarkKind.DOT)
        mark.target = first
        mark.target = second
        assert dpg.get_item_theme(first) == not_me
        assert dpg.get_item_theme(second) == mark._theme

    def test_clearing_the_target_darkens_the_mark(self, make_widget, quiet_pulse):
        """Otherwise the pulse would go on animating a colour no widget is wearing."""
        mark = keyboardmark.Mark(make_widget())
        mark.lit = True
        mark.target = None
        assert mark.lit is False
        assert not keyboardmark.pulse_is_running()

    def test_moving_a_mark_says_nothing_in_the_log(self, make_widget, quiet_pulse, caplog):
        """The warning is for a call site that displaced a theme by accident, which a move never is."""
        first = make_widget("button", "1")
        second = make_widget("button", "2")
        with dpg.theme() as not_me:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (*keyboardmark.COLOR[:3], 0))
        dpg.bind_item_theme(second, not_me)

        mark = keyboardmark.Mark(first)
        with caplog.at_level("WARNING"):
            mark.target = second
        assert caplog.records == []


# --------------------------------------------------------------------------------
# Giving the widget back

class TestDetach:
    def test_detaching_puts_back_the_theme_the_widget_had(self, make_widget, quiet_pulse):
        widget = make_widget()
        with dpg.theme() as own_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 60, 60, 255))
        dpg.bind_item_theme(widget, own_theme)

        mark = keyboardmark.Mark(widget)
        assert dpg.get_item_theme(widget) != own_theme  # displaced, which is why the warning exists

        mark.detach()
        assert dpg.get_item_theme(widget) == own_theme

    def test_detaching_leaves_an_unthemed_widget_unthemed(self, make_widget, quiet_pulse):
        widget = make_widget()
        mark = keyboardmark.Mark(widget)
        mark.detach()
        assert dpg.get_item_theme(widget) is None

    def test_detaching_a_lit_mark_stops_it(self, make_widget, quiet_pulse):
        mark = keyboardmark.Mark(make_widget())
        mark.lit = True
        mark.detach()
        assert mark.lit is False
        assert not keyboardmark.pulse_is_running()

    def test_detaching_twice_is_harmless(self, make_widget, quiet_pulse):
        mark = keyboardmark.Mark(make_widget())
        mark.detach()
        mark.detach()

    def test_marking_a_widget_that_already_has_a_theme_says_so(self, make_widget, quiet_pulse, caplog):
        """The displacement is otherwise silent, and shows up as a widget that quietly stops being styled."""
        widget = make_widget()
        with dpg.theme() as own_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 60, 60, 255))
        dpg.bind_item_theme(widget, own_theme)

        with caplog.at_level("WARNING"):
            keyboardmark.Mark(widget)
        assert any("already has a theme" in record.message for record in caplog.records)


# --------------------------------------------------------------------------------
# The one-call opt-in for combos

class TestFocusFollower:
    def test_the_follower_marks_every_widget_it_was_given(self, make_widget, quiet_pulse):
        widgets = [make_widget("button", "1"), make_widget("button", "2")]
        follower = keyboardmark.install_focus_follower(widgets)
        try:
            assert all(dpg.get_item_theme(widget) is not None for widget in widgets)
        finally:
            gui_animation.animator.cancel(follower)

    def test_cancelling_the_follower_takes_the_marks_off(self, make_widget, quiet_pulse):
        widgets = [make_widget("button", "1"), make_widget("button", "2")]
        follower = keyboardmark.install_focus_follower(widgets)
        gui_animation.animator.cancel(follower)
        assert all(dpg.get_item_theme(widget) is None for widget in widgets)

    def test_nothing_is_marked_while_nothing_is_focused(self, make_widget, quiet_pulse):
        """`get_focused_item` answers 0 here, and 0 must not match a widget by accident."""
        widgets = [make_widget("button", "1"), make_widget("button", "2")]
        follower = keyboardmark.install_focus_follower(widgets)
        try:
            gui_animation.animator.render_frame()
            assert not keyboardmark.pulse_is_running()
            assert all(colors(widget)[0][3] == 0.0 for widget in widgets)
        finally:
            gui_animation.animator.cancel(follower)
