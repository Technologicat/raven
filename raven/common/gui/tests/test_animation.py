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

from raven.common.gui import animation, tooltip  # noqa: E402 -- after importorskip by design


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

    def test_a_widget_with_no_colour_of_its_own_fades_back_to_the_default(self, dpg_context):
        """Most text declares no colour — Raven declares one only where the text departs from normal.

        DPG reports an undeclared colour as a sentinel rather than as the colour in effect, so a fade that
        aims at what it read runs to black. The flash colour here is red rather than white so that the two
        behaviours can disagree: fading to the default sends green and blue *up* from zero, and fading to
        the sentinel leaves them there.
        """
        with dpg.window() as window:
            bare = dpg.add_text("plain")  # no `color=`: the common case
        try:
            assert dpg.get_item_configuration(bare)["color"][0] == -1.0, "DPG's undeclared-colour sentinel"

            animation.highlight_widget(widget=bare, duration=0.4, color=(255, 0, 0))
            time.sleep(0.3)  # most of the way through the fade
            animation.animator.render_frame()

            _, g, b = _widget_color(bare)
            assert g > 100 and b > 100, "the fade is heading for the default text colour, not for black"

            _run_flash_to_completion(bare)
            assert dpg.get_item_configuration(bare)["color"][0] == -1.0, "and it is handed back undeclared"
        finally:
            animation.animator.clear()
            dpg.delete_item(window)

    def test_a_text_target_can_carry_a_message_too(self, widgets):
        """Which is how a status line says something in the color that says it: one widget, both channels.

        Painting and saying are independent, so a widget can be asked for both at once — and naming no
        `message_target` puts the message on `target`, which is the whole shape of this case.
        """
        text, _ = widgets
        dpg.set_value(text, "")

        animation.animator.add(animation.WidgetFlash(target=text, duration=0.5,
                                                     message="cannot open that",
                                                     text_color=FLASH_COLOR))
        animation.animator.render_frame()
        assert dpg.get_value(text) == "cannot open that"
        assert _widget_color(text) == list(FLASH_COLOR)

        _run_flash_to_completion(text)
        assert dpg.get_value(text) == "", "the line goes back to what it was showing"
        assert _widget_color(text) == list(TOOL_COLOR)

    def test_writing_under_a_flash_changes_what_it_restores(self, widgets):
        """A derived status line moves on while a message is being read; what comes back must be current.

        The flash captured the line before any of that, so restoring what it found puts back a value that
        has since stopped being true — and nothing corrects it, the next write only happening when the
        state moves again.
        """
        text, _ = widgets
        dpg.set_value(text, "before")

        animation.animator.add(animation.WidgetFlash(target=text, duration=0.3,
                                                     message="something went wrong",
                                                     text_color=FLASH_COLOR))
        animation.set_text_under_flash(text, "after")
        assert dpg.get_value(text) == "something went wrong", "the message keeps the line while it runs"

        _run_flash_to_completion(text)
        assert dpg.get_value(text) == "after"

    def test_writing_with_no_flash_is_an_ordinary_write(self, widgets):
        """The caller has one call for both cases, so it cannot pick the wrong one."""
        text, _ = widgets
        animation.set_text_under_flash(text, "plain")
        assert dpg.get_value(text) == "plain"

    def test_writing_under_a_color_only_flash_reaches_the_widget(self, widgets):
        """`highlight_widget` borrows the color and not the text, so there is nothing to write through to."""
        text, _ = widgets
        dpg.set_value(text, "before")
        animation.highlight_widget(widget=text, duration=0.3, color=FLASH_COLOR)

        animation.set_text_under_flash(text, "after")
        assert dpg.get_value(text) == "after"

    def test_a_message_can_outstay_the_fade(self, widgets):
        """A report wants the two apart: the fade catches the eye and must be quick, the message is read.

        A fade slow enough to be a comfortable reading time does not register as a flash at all, which is
        how one number for both ends up serving neither.
        """
        text, _ = widgets
        dpg.set_value(text, "")

        flash = animation.WidgetFlash(target=text, duration=0.05, message="cannot open that",
                                      message_duration=5.0, text_color=FLASH_COLOR)
        animation.animator.add(flash)
        time.sleep(0.1)  # past the fade, nowhere near the dwell
        animation.animator.render_frame()

        assert text in animation.WidgetFlash.instances, "the flash runs until the message has had its time"
        assert dpg.get_value(text) == "cannot open that"
        assert _widget_color(text) == list(TOOL_COLOR), "the color holds where the fade left it"

        flash.duration = flash.message_duration = 0.05  # let it end, rather than sleeping out the dwell
        _run_flash_to_completion(text)
        assert dpg.get_value(text) == ""


class TestButtonTarget:
    def test_restores_the_theme_the_widget_actually_had(self, widgets):
        """Not a fixed theme: flashing a widget that had none must not leave one bound to it."""
        _, button = widgets
        assert dpg.get_item_theme(button) is None

        animation.flash_button(button=button, duration=0.3)
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

        animation.flash_button(button=button, duration=0.3)
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

    def test_a_message_of_none_leaves_the_text_alone(self, widgets):
        """A message of `None` means "don't change", so a flash may ask for the color and nothing else.

        The failure it guards against is writing the `None` through onto the line — which reads as the
        word "None", not as an empty line, so it is loud but only where somebody is looking.
        """
        text, button = widgets
        dpg.set_value(text, "ready")

        animation.flash_button(button=button, message=None, duration=0.3, text=text)
        animation.animator.render_frame()
        assert dpg.get_value(text) == "ready"

        _run_flash_to_completion(button)
        assert dpg.get_value(text) == "ready"


class TestPaintList:
    def test_each_widget_is_painted_by_the_channel_it_has(self, dpg_context):
        """One list, two kinds of widget, and the widget decides — not the argument it arrived in.

        A caption alongside a button is a text item, so its own colour carries the flash and fades. Taking
        the button's animated theme instead would give it that theme's text colour, which is held constant
        for readability — so the caption would snap to the highlight and snap back, where a text `target`
        in the same role fades.
        """
        with dpg.window() as window:
            button = dpg.add_button(label="b")
            caption = dpg.add_text("ready", color=TOOL_COLOR)
        try:
            animation.animator.add(animation.WidgetFlash(target=button, duration=5.0,
                                                         also_flash=(caption,),
                                                         flash_color=(255, 0, 0), text_color=(255, 0, 0)))
            animation.animator.render_frame()

            assert dpg.get_item_theme(button) is not None, "the background channel: an animated theme"
            assert dpg.get_item_theme(caption) is None, "the foreground channel: no theme involved"
            assert _widget_color(caption) != list(TOOL_COLOR), "the caption is being painted"

            animation.WidgetFlash.instances[button].duration = 0.05  # let it end
            _run_flash_to_completion(button)
            assert dpg.get_item_theme(button) is None
            assert _widget_color(caption) == list(TOOL_COLOR), "and handed back what it had"
        finally:
            animation.animator.clear()
            dpg.delete_item(window)

    def test_the_label_fades_as_well_as_the_background(self, widgets):
        """And to a different destination, which is what keeps it legible the whole way.

        Holding the label at `text_color` for the duration leaves it to snap when the theme unbinds. Fading
        it toward the background's destination instead would run the two together and hide it mid-flash;
        toward the resting *text* colour they move apart, the label brightening as the background dims.
        """
        _, button = widgets
        animation.animator.add(animation.WidgetFlash(target=button, duration=10.0,
                                                     flash_color=(96, 128, 96), text_color=(255, 0, 0)))
        flash = animation.WidgetFlash.instances[button]

        def sample_at(r):
            """The label and background colours at fraction `r` of the fade, without waiting for a clock."""
            flash.t0 = time.monotonic_ns() - int(r * flash.duration * 10**9)
            animation.animator.render_frame()
            return (list(dpg.get_value(flash.animated_theme_text_color))[:3],
                    list(dpg.get_value(flash.animated_theme_colors[0]))[:3])

        label_start, background_start = sample_at(0.0)
        label_end, background_end = sample_at(0.99)

        assert label_start != label_end, "the label fades rather than being held"
        assert label_end[1] > label_start[1] and label_end[2] > label_start[2], "toward the resting text colour"
        assert sum(background_end) < sum(background_start), "while the background darkens beneath it"

    def test_the_message_can_land_in_something_that_stages_its_own_change(self, widgets):
        """A `Tooltip` resizes itself over two frames, which a raw `dpg.set_value` would drive straight past.

        So the message channel asks the target to take the text rather than writing it in — the difference
        between a tooltip that fits what it says and one that is briefly drawn at the size of what it said
        before.
        """
        _, button = widgets
        tip = tooltip.Tooltip(button, "resting caption")
        try:
            animation.animator.add(animation.WidgetFlash(target=button, duration=5.0,
                                                         message="Copied!", message_target=tip))
            assert tip.text == "resting caption", "queued, not written through"
            animation.animator.render_frame()
            assert tip.text == "Copied!"

            animation.WidgetFlash.instances[button].duration = 0.05
            _run_flash_to_completion(button)
            animation.animator.render_frame()
            assert tip.text == "resting caption", "and the restore is staged the same way"
        finally:
            tip.destroy()

    def test_a_message_with_nowhere_to_go_says_so(self, widgets, caplog):
        """Silently dropping it is how two flashes carried an inert `message=""` for months.

        A widget with a background cannot show text, so a message needs a `message_target` naming one that
        can. Getting nothing on screen and nothing in the log leaves a call site that looks like it says
        something and does not.
        """
        _, button = widgets
        with caplog.at_level("WARNING"):
            animation.animator.add(animation.WidgetFlash(target=button, duration=0.3, message="lost"))
        assert any("message" in record.message for record in caplog.records)


class TestDeduplication:
    def test_second_flash_on_the_same_widget_does_not_reify(self, widgets):
        """At most one flash owns a widget; the loser goes into ghost mode."""
        text, _ = widgets
        animation.highlight_widget(widget=text, duration=5.0, color=FLASH_COLOR)
        reified = animation.WidgetFlash.instances[text]

        ghost = animation.WidgetFlash(target=text, duration=5.0)
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

        ghost = animation.WidgetFlash(target=text, duration=5.0)
        ghost.finish()

        assert animation.WidgetFlash.instances.get(text) is reified
        assert _widget_color(text) == color_while_running  # the running flash was not undone

    def test_a_ghost_with_no_message_leaves_the_words_where_they_are(self, widgets):
        """A highlight landing on a widget that is mid-report restarts the fade and says nothing itself.

        The ghost path took the joining instance's `message` unconditionally, so a `highlight_widget` —
        which passes `None` — wiped the report it arrived on top of.
        """
        text, _ = widgets
        dpg.set_value(text, "before")
        reported = animation.WidgetFlash(target=text, duration=5.0, message="something went wrong",
                                         text_color=FLASH_COLOR)
        animation.animator.add(reported)

        animation.highlight_widget(widget=text, duration=5.0, color=FLASH_COLOR)
        assert dpg.get_value(text) == "something went wrong"

        reported.duration = 0.05  # let it end, rather than sleeping out the five seconds
        _run_flash_to_completion(text)
        assert dpg.get_value(text) == "before"

    def test_a_ghosts_message_is_put_back_by_the_flash_it_joins(self, widgets):
        """The instance being joined restores what it captured, so it has to capture before it is written to.

        A color-only flash captured nothing, having nothing to put back — and then a message written through
        it had no way home either, leaving it on the line after the animation had visibly ended.
        """
        text, _ = widgets
        dpg.set_value(text, "before")
        highlight = animation.WidgetFlash(target=text, duration=5.0, text_color=FLASH_COLOR)
        animation.animator.add(highlight)
        assert dpg.get_value(text) == "before", "a color-only flash does not touch the words"

        animation.WidgetFlash(target=text, duration=5.0,
                              message="something went wrong")  # ghost: updates `highlight` and exits
        assert dpg.get_value(text) == "something went wrong"

        highlight.duration = 0.05
        _run_flash_to_completion(text)
        assert dpg.get_value(text) == "before"


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
    animation.animator.clear()  # `scroll` registers with the animator; the by-hand `_scroll` does not
    animation.SmoothScrolling.instances.clear()
    dpg.delete_item(window)


def _scroll(target, **kwargs):
    """A started `SmoothScrolling`, boilerplate defaulted; not registered with the animator.

    Constructed and started by hand rather than through `SmoothScrolling.scroll`, because that classmethod
    returns the *running* instance and never a ghost — and the ghost is half of what these tests are about.
    """
    kwargs.setdefault("target_y_scroll", 100)
    scrolling = animation.SmoothScrolling(target_child_window=target, **kwargs)
    scrolling.start()
    return scrolling


class _FakeFlasher:
    """Stands in for `ScrollEndFlasher`; identity is all these tests need."""


class TestSmoothScrollingScrollEntryPoint:
    """`scroll` is the supported way in, and exists so that callers never handle a ghost themselves."""

    def test_returns_the_running_instance_rather_than_the_ghost(self, scroll_target):
        """The point of the classmethod: a caller keeps the returned object in order to stop that scroll
        later, and stopping a ghost would stop nothing while the view kept moving.
        """
        first = animation.SmoothScrolling.scroll(target_child_window=scroll_target, target_y_scroll=100)
        second = animation.SmoothScrolling.scroll(target_child_window=scroll_target, target_y_scroll=200)

        assert second is first
        assert second.reified is True
        assert second.target_y_scroll == 200  # the running instance took the new request

    def test_registers_one_animation_per_window_however_many_requests(self, scroll_target):
        """A retarget re-aims the running animation, so a second request must not add a second one.

        Counted as a delta rather than against zero: the animator is a process-wide singleton, so an
        absolute count would also be measuring anything else that happens to be running.
        """
        before = animation.animator.active_count

        animation.SmoothScrolling.scroll(target_child_window=scroll_target, target_y_scroll=100)
        animation.SmoothScrolling.scroll(target_child_window=scroll_target, target_y_scroll=200)
        animation.SmoothScrolling.scroll(target_child_window=scroll_target, target_y_scroll=300)

        assert animation.animator.active_count - before == 1

    def test_constructing_one_does_not_start_it(self, scroll_target):
        """Construction packages a request and does nothing with it.

        This is what makes the object safe to build in order to inspect or hand around — the property the
        old constructor-calls-`start` shape did not have.
        """
        scrolling = animation.SmoothScrolling(target_child_window=scroll_target, target_y_scroll=100)

        assert scrolling.reified is False
        assert scroll_target not in animation.SmoothScrolling.instances


class TestSmoothScrollingStop:
    def test_stop_deregisters_from_the_animator_so_the_scroll_really_ends(self, scroll_target):
        """The distinction `stop` exists for: leaving it in the animator means it keeps being rendered,
        which means it keeps scrolling — a "stop" that does not stop.
        """
        scrolling = animation.SmoothScrolling.scroll(target_child_window=scroll_target, target_y_scroll=400)
        animation.animator.render_frame()

        animation.SmoothScrolling.stop(scroll_target)

        assert scrolling not in animation.animator._animations
        assert scroll_target not in animation.SmoothScrolling.instances

    def test_stop_runs_the_finish_callbacks(self, scroll_target):
        """A caller holding the instance has to be told it died, or its reference dangles at a corpse."""
        calls = []
        animation.SmoothScrolling.scroll(target_child_window=scroll_target, target_y_scroll=400,
                                         finish_callback=lambda: calls.append("told"))

        animation.SmoothScrolling.stop(scroll_target)

        assert calls == ["told"]

    def test_stopping_an_idle_window_is_harmless(self, scroll_target):
        """Idempotent, so a teardown path need not first ask whether anything is running."""
        animation.SmoothScrolling.stop(scroll_target)
        animation.SmoothScrolling.stop(scroll_target)


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


class TestSmoothScrollingTeardown:
    def test_a_ghost_finishing_leaves_the_running_instance_alone(self, scroll_target):
        """A ghost owns nothing — its request was handed to the running instance — so tearing down as
        though it did would evict the instance that is actually animating.

        `SmoothScrolling.scroll` registers only the reified instance, so a ghost no longer reaches the
        animator by that route. The no-op is what lets the ghost stay an internal detail rather than
        something every caller has to know about, and `finish` is public regardless.
        """
        calls = []
        first = _scroll(scroll_target, finish_callback=lambda: calls.append("first"))
        ghost = _scroll(scroll_target, target_y_scroll=200)

        ghost.finish()

        assert animation.SmoothScrolling.instances[scroll_target] is first
        assert calls == []  # nobody has been told anything ended, because nothing has

    def test_a_ghost_does_not_run_the_chained_callbacks_a_second_time(self, scroll_target):
        """The ghost still holds its own callback in its own list, having also chained it onto the
        running instance. Only the instance that owns the registration may fire them."""
        calls = []
        first = _scroll(scroll_target, finish_callback=lambda: calls.append("first"))
        ghost = _scroll(scroll_target, finish_callback=lambda: calls.append("second"))

        ghost.finish()
        assert calls == []

        first.finish()
        assert calls == ["first", "second"]  # exactly once each, from the instance that owned them

    def test_finishing_twice_is_harmless(self, scroll_target):
        """`instances.pop` has no default, so a second teardown used to raise `KeyError` — reachable
        through a caller holding a stale reference (Visualizer keeps one to stop the scroll on demand)."""
        calls = []
        first = _scroll(scroll_target, finish_callback=lambda: calls.append(1))

        first.finish()
        first.finish()  # must not raise

        assert calls == [1]

    def test_deregistration_happens_before_the_callbacks_run(self, scroll_target):
        """A callback may start a new scroll animation — the class docstring permits it.

        With the pop afterwards, that new request would find this dying instance still registered,
        retarget it, and then be thrown away by the pop. The scroll would silently never happen.
        """
        started = []

        def start_another():
            started.append(_scroll(scroll_target, target_y_scroll=999))

        first = _scroll(scroll_target, finish_callback=start_another)
        first.finish()

        replacement = started[0]
        assert replacement.reified is True, "the new scroll reified instead of retargeting a corpse"
        assert animation.SmoothScrolling.instances[scroll_target] is replacement
        assert replacement.target_y_scroll == 999


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

    def test_giving_up_records_where_the_panel_actually_is(self, scroll_target):
        """A request past the end is clamped by DPG, so the position never reaches what we wrote.

        The animation waits for it, times out, and stops. What it must not do is leave the box holding a
        value the panel never took: `should_follow_tail` compares the two and reads the difference as the
        reader having scrolled — and nothing is left running to correct it, because the timeout *is* the
        animation giving up. That is what made the chat view stop following a streaming reply for the rest of
        the message; see `investigations/follow-tail-drift/`.
        """
        commanded = box(0)
        animation_ = animation.animator.add(_scroll(scroll_target,
                                                    target_y_scroll=99999,  # far past the end: DPG will clamp
                                                    commanded_y_scroll=commanded))

        for _ in range(40):  # the timeout is a handful of frames; this is slack, not a wait
            animation.animator.render_frame()
            if animation_ not in animation.animator._animations:
                break
        else:
            pytest.fail("the animation never gave up, so the timeout branch under test did not run")

        actual = dpg.get_y_scroll(scroll_target)
        assert unbox(commanded) == actual, "the box still holds a position the panel never reached"
        assert unbox(commanded) != 99999, "precondition: DPG must have clamped the request, or this proves nothing"


class TestPulsatingColor:
    """One animation may drive several theme colors, which is what a mark drawn by several themes needs."""

    @staticmethod
    def _a_theme_color():
        with dpg.theme():
            with dpg.theme_component(dpg.mvAll):
                return dpg.add_theme_color(dpg.mvThemeCol_Text, (80, 160, 255))

    def test_several_colors_pulsate_as_one(self, dpg_context):
        """Same alpha on every widget, from one animation — a cursor split across theme variants is still
        one cursor, and two animations over it could drift out of phase."""
        colors = [self._a_theme_color() for _ in range(3)]
        pulse = animation.animator.add(animation.PulsatingColor(cycle_duration=2.0, theme_color_widget=colors))
        try:
            animation.animator.render_frame()
            alphas = {tuple(dpg.get_value(color)) for color in colors}
            assert len(alphas) == 1, f"the widgets disagree: {alphas}"
        finally:
            animation.animator.cancel(pulse)

    def test_assigning_the_color_recolors_every_widget(self, dpg_context):
        """`raven-conference-timer` recolors its pause glow while it runs, and that has to keep working."""
        colors = [self._a_theme_color() for _ in range(2)]
        pulse = animation.animator.add(animation.PulsatingColor(cycle_duration=2.0, theme_color_widget=colors))
        try:
            pulse.rgb = (255, 0, 0)
            animation.animator.render_frame()
            assert all(tuple(dpg.get_value(color))[:3] == (255, 0, 0) for color in colors)
        finally:
            animation.animator.cancel(pulse)

    def test_one_widget_is_taken_as_itself_rather_than_as_a_collection(self, dpg_context, request):
        """The single-widget spelling is what every existing caller uses, and a *tag* is the trap in it.

        A DPG widget is a tag or an ID, so "not a collection" has to be decided by type. A string tag is
        iterable, so a normalization that merely tried `list(...)` would take it apart into characters and
        then write colors to widgets named "e", "x", "_" — silently, since `set_value` on a name nothing
        answers to raises nothing.
        """
        tag = f"pulsating_color_{request.node.name}"  # tag
        with dpg.theme():
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (80, 160, 255), tag=tag)

        pulse = animation.PulsatingColor(cycle_duration=2.0, theme_color_widget=tag)

        assert pulse.theme_color_widgets == [tag]


class TestAmbientAndTransient:
    """What the idle-framerate throttle asks, and why `active_count` is the wrong question for it.

    An ambient animation runs for as long as its widget exists, so an app that treated it as a sign of
    activity would never idle again. `transient_count` is what excludes it; the flag is per instance
    because ambience belongs to the use rather than to the class.
    """

    def test_an_animation_is_transient_unless_it_says_otherwise(self):
        """The safe default: forgetting the flag costs frame rate, and hides nothing."""
        assert animation.Animation().ambient is False

    def test_an_ambient_animation_is_not_counted_as_activity(self, dpg_context):
        with dpg.theme():
            with dpg.theme_component(dpg.mvAll):
                color = dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))

        before_active = animation.animator.active_count
        before_transient = animation.animator.transient_count

        ambient = animation.animator.add(animation.PulsatingColor(cycle_duration=2.0,
                                                                  theme_color_widget=color))
        try:
            assert animation.animator.active_count - before_active == 1
            assert animation.animator.transient_count - before_transient == 0
        finally:
            animation.animator.cancel(ambient)

    def test_the_same_animation_counts_as_activity_when_it_is_not_ambient(self, dpg_context):
        """Same class, same widget, opposite answer — which is what makes this a property of the use.

        `PulsatingColor` defaults the other way from the base class, so this is also what pins that the
        default is a default and not a hard-wired answer.
        """
        with dpg.theme():
            with dpg.theme_component(dpg.mvAll):
                color = dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))

        before = animation.animator.transient_count
        transient = animation.animator.add(animation.PulsatingColor(cycle_duration=2.0,
                                                                    theme_color_widget=color,
                                                                    ambient=False))
        try:
            assert animation.animator.transient_count - before == 1
        finally:
            animation.animator.cancel(transient)
