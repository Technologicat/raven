"""Tests for `raven.common.gui.animation`'s `WidgetFlash` and `SmoothScrolling`.

Most of this package is DPG glue and untested, but `WidgetFlash` carries two things worth
asserting: it *restores* what it borrowed (a widget's color or theme), and it has a de-duplication state
machine — at most one flash animates a given widget, and the losing instance ("ghost") must own nothing and
release nothing. Both are the kind of invariant that breaks silently: a wrong restore leaves a permanent mark
on a widget the flash was only supposed to point at, and a ghost that finalizes anyway evicts the instance
that is actually running.

`SmoothScrolling` carries a second state machine, and a subtler one: an existing animation is *retargeted*
rather than replaced, so one long-lived object serves many scroll requests. What it adopts on the way is the
whole content of these tests. Adopting only the destination is the bug that looks correct — the surviving
instance would keep the *first* request's flasher and rate forever, which strobes the scroll-end overlay once
per streamed chunk for the length of a reply.

DPG can run without a mapped window (`create_viewport` + `setup_dearpygui`, never `show_viewport`), so these
drive real widgets rather than mocks. The animation is stepped by calling `animator.render_frame()` directly,
the same way Raven's render loop does.

What that does *not* buy is layout: `dpg.render_dearpygui_frame()` aborts the process on a GLFW assertion
when the viewport was never shown, so there are no real scroll extents here and nothing asserts against
`get_y_scroll_max`. The retarget logic is pure state manipulation and needs none.
"""

import time

import pytest

from unpythonic import box, unbox

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

    def test_each_flashed_widget_gets_back_its_own_theme(self, dpg_context):
        """Button, tooltip and text are three independent widgets, so one shared snapshot is not enough.

        The flash binds its animated theme to all three. Restoring a single captured theme to all three then
        hands two of them a theme belonging to the third — which is the same "silently gives a widget a theme
        it never had" fault as leaving a fixed theme behind, just distributed. Here the text widget is the one
        that would visibly acquire the tooltip's theme.
        """
        with dpg.theme() as text_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (1, 2, 3))
        with dpg.theme() as tooltip_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (4, 5, 6))

        with dpg.window():
            button = dpg.add_button(label="b")
            with dpg.tooltip(button) as tooltip:
                notification = dpg.add_text("ready")
        dpg.bind_item_theme(tooltip, tooltip_theme)
        dpg.bind_item_theme(notification, text_theme)

        animation.flash_button(button=button, message="working", duration=0.3,
                               tooltip=tooltip, text=notification)
        _run_flash_to_completion(button)

        assert dpg.get_item_theme(button) is None, "the button had no theme and must be left with none"
        assert dpg.get_item_theme(tooltip) == tooltip_theme
        assert dpg.get_item_theme(notification) == text_theme
        assert dpg.get_value(notification) == "ready", "the message must be restored too"


class TestDeduplication:
    def test_second_flash_on_the_same_widget_does_not_reify(self, widgets):
        """At most one flash owns a widget; the loser goes into ghost mode."""
        text, _ = widgets
        animation.highlight_widget(widget=text, duration=5.0, color=FLASH_COLOR)
        reified = animation.WidgetFlash.instances[text]

        ghost = animation.WidgetFlash(message=None, target=text, target_tooltip=None, target_text=None,
                                      duration=5.0)
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
                                      duration=5.0)
        ghost.finish()

        assert animation.WidgetFlash.instances.get(text) is reified
        assert _widget_color(text) == color_while_running  # the running flash was not undone


# ---------------------------------------------------------------------------
# SmoothScrolling: what a retarget adopts
# ---------------------------------------------------------------------------

@pytest.fixture
def scroll_target(dpg_context):
    """A child window to scroll, plus a clean `SmoothScrolling.instances` before and after."""
    with dpg.window() as window:
        with dpg.child_window(width=80, height=40) as child:
            for i in range(20):
                dpg.add_text(f"line {i}")
    animation.SmoothScrolling.instances.clear()
    yield child
    animation.SmoothScrolling.instances.clear()
    dpg.delete_item(window)


def _scroll(target, **kwargs):
    """A `SmoothScrolling` with the boilerplate defaulted; not registered with the animator."""
    kwargs.setdefault("target_y_scroll", 100)
    return animation.SmoothScrolling(target_child_window=target, **kwargs)


class _FakeFlasher:
    """Stands in for `ScrollEndFlasher`; identity is all these tests need."""


class TestSmoothScrollingRetarget:
    def test_the_second_request_does_not_start_a_second_animation(self, scroll_target):
        first = _scroll(scroll_target)
        second = _scroll(scroll_target, target_y_scroll=200)

        assert animation.SmoothScrolling.instances[scroll_target] is first
        assert first.reified is True
        assert second.reified is False  # a ghost: owns nothing, animates nothing

    def test_the_destination_is_adopted_by_the_running_instance(self, scroll_target):
        first = _scroll(scroll_target, target_y_scroll=100)
        _scroll(scroll_target, target_y_scroll=200)

        assert first.target_y_scroll == 200
        assert first._sv.target == 200  # the interpolator's copy moves with the field, or it animates to the old one

    def test_a_follow_scroll_clears_a_flasher_left_by_a_user_scroll(self, scroll_target):
        """The strobe. A user scroll attaches a flasher; every later tail-follow must take it back off.

        The flasher asserts "you tried to go further and could not", which is about a user's thwarted
        intent. Tail-following has none — reaching the end is its purpose — so an inherited flasher fires
        once per arriving chunk for the length of the reply.
        """
        first = _scroll(scroll_target, flasher=_FakeFlasher())
        _scroll(scroll_target, flasher=None)  # a follow scroll

        assert first.flasher is None

    def test_a_user_scroll_attaches_its_flasher_to_an_in_flight_follow(self, scroll_target):
        """The mirror case: clicking jump-to-latest mid-stream must still get its confirming flash."""
        flasher = _FakeFlasher()
        first = _scroll(scroll_target, flasher=None)  # a follow scroll already running
        _scroll(scroll_target, flasher=flasher)  # user clicks jump-to-latest

        assert first.flasher is flasher

    def test_the_rate_reaches_the_interpolator_and_not_only_the_field(self, scroll_target):
        """`smooth_step` is stored twice — on the animation and inside `SmoothInt` — and both must move.

        Only the interpolator's copy is read per frame, so updating the field alone changes nothing
        visible. Worth its own test because the failure is silent: the scroll still works, at the old rate.
        """
        first = _scroll(scroll_target, smooth_step=0.8)
        _scroll(scroll_target, smooth_step=0.2)

        assert first.smooth_step == 0.2
        assert first._sv.rate == 0.2

    def test_smoothness_itself_is_adopted(self, scroll_target):
        first = _scroll(scroll_target, smooth=True)
        _scroll(scroll_target, smooth=False)

        assert first.smooth is False

    def test_the_commanded_position_box_is_adopted(self, scroll_target):
        first = _scroll(scroll_target, commanded_y_scroll=box(0))
        newer = box(0)
        _scroll(scroll_target, commanded_y_scroll=newer)

        assert first.commanded_y_scroll is newer

    def test_the_running_object_survives_a_retarget(self, scroll_target):
        """Identity is the point: keeping the object keeps its subpixel position, so the movement bends
        toward the new target instead of restarting from an integer position and jumping."""
        first = _scroll(scroll_target)
        for target in (200, 300, 400):
            _scroll(scroll_target, target_y_scroll=target)

        assert animation.SmoothScrolling.instances[scroll_target] is first
        assert first.target_y_scroll == 400


class TestSmoothScrollingFinishCallbacks:
    def test_callbacks_chain_rather_than_replace(self, scroll_target):
        calls = []
        first = _scroll(scroll_target, finish_callback=lambda: calls.append("first"))
        _scroll(scroll_target, finish_callback=lambda: calls.append("second"))

        first.finish()

        assert calls == ["first", "second"]  # registration order

    def test_the_outgoing_callback_does_not_fire_at_handover(self, scroll_target):
        """A retarget ends nothing — the instance persists — so nobody may be told it ended.

        Visualizer keeps a reference to the reified instance so it can stop *that* animation before
        swapping the info panel's content, and its callback exists to null that reference. Firing it here
        would null a live reference, and the later stop would silently stop nothing.
        """
        calls = []
        _scroll(scroll_target, finish_callback=lambda: calls.append("first"))
        _scroll(scroll_target, target_y_scroll=200)

        assert calls == []

    def test_the_same_callback_is_not_registered_twice(self, scroll_target):
        """Bounds the chain by distinct callers rather than by retargets — a streaming reply retargets
        once per arriving chunk. Also correct on its own terms: "it ended" should fire once."""
        calls = []

        def callback():
            calls.append(1)

        first = _scroll(scroll_target, finish_callback=callback)
        for _ in range(5):
            _scroll(scroll_target, finish_callback=callback)

        first.finish()

        assert calls == [1]

    def test_a_raising_callback_stops_neither_the_others_nor_deregistration(self, scroll_target):
        """Teardown must complete. Leaving the instance in `instances` would make this GUI element
        permanently unanimatable: every later request would retarget a dead object."""
        calls = []

        def boom():
            raise RuntimeError("callback failed")

        first = _scroll(scroll_target, finish_callback=boom)
        _scroll(scroll_target, finish_callback=lambda: calls.append("ran anyway"))

        first.finish()

        assert calls == ["ran anyway"]
        assert scroll_target not in animation.SmoothScrolling.instances


class TestCommandedScrollBox:
    def test_every_written_position_reaches_the_box(self, scroll_target):
        """The box exists so a caller can tell our writes from the user's. A write we do not record
        reads as a user scroll, which is what stops the chat view following a streaming reply."""
        commanded = box(0)
        animation_ = _scroll(scroll_target, commanded_y_scroll=commanded)

        animation_._set_y_scroll(42)
        assert unbox(commanded) == 42

        animation_._set_y_scroll(77)
        assert unbox(commanded) == 77

    def test_no_box_is_a_supported_configuration(self, scroll_target):
        """Visualizer passes none; the setter must not require one."""
        animation_ = _scroll(scroll_target, commanded_y_scroll=None)
        animation_._set_y_scroll(42)  # must not raise
