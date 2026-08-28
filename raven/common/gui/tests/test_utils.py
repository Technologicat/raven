"""Tests for `raven.common.gui.utils`' frame-waiting guard.

`dpg.split_frame()` waits for the render loop to complete a frame. Called *from* that loop — or from app
startup, which runs on the same thread before the loop begins — the wait can never be satisfied, and the app
hangs with no traceback, no log line, and nothing to bisect. `guiutils.split_frame` converts that into either
a `RuntimeError` naming the operation (when waiting is the whole job) or a warning and a stale-geometry
fallback (when it is only an improvement).

**The successful wait is deliberately not tested**, and that is not an oversight: with no render loop running,
a real `dpg.split_frame()` would hang the test suite in exactly the way this guard exists to prevent. What is
asserted is the guard, and that the two library functions with opposite policies are really wired to it —
`wait_for_resize` raising and `recenter_window` degrading. pytest runs on the main thread, so every test here
is already standing on the hazardous thread.
"""

import logging
import threading

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import utils as guiutils  # noqa: E402 -- after importorskip by design


@pytest.fixture
def dpg_context():
    """A DPG context with an unmapped viewport, fresh per test so the item registry starts empty."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def mapped_dpg_context(mapped_gui_context):
    """The session's mapped viewport, with whatever this test builds in it taken down afterwards.

    For the tests that need frames rendered into a window somebody could look at, as opposed to the
    unmapped `dpg_context` above.
    """
    # The viewport is the session-wide one rather than a context of this fixture's own, because creating and
    # destroying a context per test is what segfaulted the whole `gui` group: DPG keeps its context in
    # process-global state, and a focus event arriving after `destroy_context` reaches a freed backend.
    #
    # Its 1280x800 is load-bearing, and the reason this fixture cannot simply borrow the unmapped one's
    # 100x100: the windows these tests build are several times that, so every one of them would be larger
    # than the screen it is drawn on. What that breaks is measurement rather than drawing. ImGui only pulls
    # an offscreen window back inside the viewport when there is an inside to pull it to, so at 100 wide a
    # parked window simply stays parked — and a test asking whether a park was clamped gets the same answer
    # either way, which is no answer.
    #
    # Nothing here needs per-test tags, the two tests using this fixture building disjoint trees that are
    # each created once in a session. Cleaning up regardless is what keeps that true: a leftover window goes
    # on rendering into the shared viewport, where the next test's measurements have to live.
    before = set(dpg.get_all_items())
    yield
    for item in dpg.get_all_items():
        if item not in before and dpg.does_item_exist(item):  # a deleted container takes its children
            dpg.delete_item(item)


def test_the_test_runner_itself_is_on_the_render_thread():
    """Establishes the premise the rest of the module depends on."""
    assert guiutils.is_render_thread() is True


def test_a_worker_thread_is_not_the_render_thread():
    """The distinction has to be real, or the guard would refuse every caller including the legitimate ones."""
    answers = []
    worker = threading.Thread(target=lambda: answers.append(guiutils.is_render_thread()))
    worker.start()
    worker.join()
    assert answers == [False]


def test_a_required_wait_raises_instead_of_hanging():
    with pytest.raises(RuntimeError):
        guiutils.split_frame(operation="unit test: a wait that cannot be skipped")


def test_the_error_names_the_operation_so_the_call_site_is_findable():
    """A hang tells you nothing; the point of raising is that the message has to be worth reading."""
    with pytest.raises(RuntimeError, match="a distinctive operation name"):
        guiutils.split_frame(operation="a distinctive operation name")


def test_an_optional_wait_reports_that_it_did_not_happen(caplog):
    """`False` is what lets a caller adapt, and the warning is what makes the skip discoverable."""
    with caplog.at_level("WARNING", logger="raven.common.gui.utils"):
        waited = guiutils.split_frame(operation="unit test: a wait we can live without", required=False)
    assert waited is False
    assert "unit test: a wait we can live without" in caplog.text


def test_wait_for_resize_raises_rather_than_hanging(dpg_context):
    """Waiting *is* the operation, so there is nothing to degrade to."""
    with dpg.window() as window:
        dpg.add_text("x")
    with pytest.raises(RuntimeError):
        guiutils.wait_for_resize(window)


def test_recenter_window_degrades_instead_of_raising(dpg_context):
    """The opposite policy: an off-center window beats both a hang and an exception."""
    with dpg.window(width=100, height=100) as reference_window:
        dpg.add_text("reference")
    with dpg.window(autosize=True) as thewindow:
        dpg.add_text("centered on the reference")
    guiutils.recenter_window(thewindow, reference_window=reference_window)  # must not raise


# --------------------------------------------------------------------------------
# Recognizing a widget DPG handed back

def test_a_widget_is_recognized_by_either_name_dpg_may_answer_with(dpg_context, request):
    """A getter returns an alias for a tagged item and an ID for an untagged one.

    Testing against one name only is right for half the widgets and silently wrong for the other half —
    and "never matches" is indistinguishable from "this widget is never the one", so whatever was gated on
    it simply never happens and nothing is logged.
    """
    tag = f"{request.node.name}_button"
    with dpg.window():
        widget = dpg.add_button(tag=tag)
    identifiers = guiutils.item_identifiers(widget)
    assert tag in identifiers
    assert dpg.get_alias_id(tag) in identifiers


def test_an_untagged_widget_is_recognized_by_its_id_alone(dpg_context):
    """And nothing else creeps in — an empty alias must not become a name that matches other unnamed items."""
    with dpg.window():
        widget = dpg.add_button()
    assert guiutils.item_identifiers(widget) == {widget}


# --------------------------------------------------------------------------------
# Widget position, in viewport coordinates

@pytest.mark.gui
def test_get_widget_pos_is_viewport_coordinates_however_deeply_nested(mapped_dpg_context):
    """A child window's position must be where it is on screen, not where it is inside its parent.

    Windows and child windows have no `rect_min`, so this cannot be read off DPG directly — and
    `get_item_pos`, the obvious substitute, answers a different question. One level below a window at the
    origin the two agree, which is what let the difference go unnoticed until a modal dialog three levels
    deep reported `(0, 0)` for a widget that was nowhere near it, and its thumbnail grid stopped responding
    to clicks.

    A *button* does have `rect_min`, and that is a true viewport position — so it serves as the reference
    the nesting is checked against.

    Carries the `gui` marker: rendered frames are needed for any of these to have a position at all, and
    DPG aborts the process if asked to render without a mapped viewport. So this one maps a window and
    takes keyboard focus.
    """
    dpg.show_viewport()
    try:
        # The nesting matters, and each part of it earns its place. A title bar, so the window's own offset
        # is not simply its position. A *horizontal* group holding a sibling before the branch we measure,
        # so the x offset is real. And a group that is **not the first item in its parent** — that is the
        # case a simpler tree misses, because a first-position group's own offset is zero and a
        # double-counting bug then adds nothing.
        with dpg.window(tag="probe_window", pos=(50, 50), width=700, height=500):
            with dpg.group(horizontal=True):
                dpg.add_child_window(tag="probe_side", width=120, height=400)
                with dpg.child_window(tag="probe_outer", width=520, height=400):
                    with dpg.group(tag="probe_wrapper"):
                        dpg.add_input_text(tag="probe_row", width=-1)
                        with dpg.group(horizontal=True, tag="probe_sortrow"):  # a non-first group
                            dpg.add_button(tag="probe_sortbutton", label="Name")
                        with dpg.child_window(tag="probe_inner", width=-1, height=-1, border=False):
                            with dpg.group(tag="probe_group"):
                                with dpg.child_window(tag="probe_deep", width=300, height=150):
                                    dpg.add_button(tag="probe_button", label="X")
        for _ in range(10):
            dpg.render_dearpygui_frame()

        deep_x, deep_y = guiutils.get_widget_pos("probe_deep")  # tag
        button_x, button_y = dpg.get_item_rect_min("probe_button")  # tag  # a real item: viewport coords
        pad_x, pad_y = dpg.get_item_pos("probe_button")  # tag  # the child window's own content padding

        assert (deep_x, deep_y) == (button_x - pad_x, button_y - pad_y)
        # Past the side panel and below the header rows, so a sum that lost or doubled a link would show.
        assert deep_x > 170 and deep_y > 100

        # And it has to follow a *scrolled* ancestor, since a layout position knows nothing of scroll.
        # `probe_inner` is shorter than what it holds, so it can be scrolled; the widget's own scroll is
        # deliberately not corrected for, because that moves its contents rather than the widget.
        dpg.configure_item("probe_inner", height=100)  # tag  # now shorter than `probe_deep`
        seen = []
        for scroll in (30, 80):
            dpg.set_y_scroll("probe_inner", scroll)  # tag
            for _ in range(5):
                dpg.render_dearpygui_frame()
            deep_y = guiutils.get_widget_pos("probe_deep")[1]  # tag
            button_y = dpg.get_item_rect_min("probe_button")[1]  # tag
            pad_y = dpg.get_item_pos("probe_button")[1]  # tag
            assert deep_y == button_y - pad_y

            # The other path through the function, and the reason it needs no scroll correction of its
            # own: an item that *has* `rect_min` is answered from it directly, and `rect_min` is a
            # rendered position that already moved with the scroll — where `get_item_pos`, being layout,
            # did not. Asserted rather than inferred, since the whole correction rests on the difference.
            assert guiutils.get_widget_pos("probe_button") == tuple(dpg.get_item_rect_min("probe_button"))  # tag
            assert dpg.get_item_pos("probe_button") == [pad_x, pad_y]  # tag  # unmoved by the scroll
            seen.append(button_y)
        assert seen[0] != seen[1]  # ...and the rendered position really did move
    finally:
        dpg.delete_item("probe_window")  # tag


@pytest.mark.gui
def test_a_park_has_to_be_renewed_every_frame_to_hold(mapped_dpg_context):
    """ImGui pulls a parked window back inside the viewport on the frame after it was positioned.

    Parking a window offscreen is how a caller measures one before placing it, and the pattern is only
    sound for a single frame: ImGui clamps a window whose position did not come through the API that
    frame, so a settle spanning several frames draws the window on screen for all but the first of them.
    `park_offscreen` says so, and this is what holds it — the failure it guards is a tooltip or a help
    card flashing in the corner, which no assertion elsewhere would notice.

    The reference is read off a *child* item's `rect_min`, a true viewport position: a window has no
    `rect_min`, and its `get_item_pos` reports the position that was set rather than the one drawn, so it
    cannot tell the two behaviours apart at all.

    Carries the `gui` marker: nothing has a drawn position until frames are rendered, and DPG aborts the
    process if asked to render without a mapped viewport.
    """
    dpg.show_viewport()
    try:
        with dpg.window(tag="park_probe", width=600, height=400):
            dpg.add_text("content", tag="park_probe_text")  # tag
        for _ in range(3):
            dpg.render_dearpygui_frame()

        def drawn_x_over(frames, renew):
            seen = []
            guiutils.park_offscreen("park_probe")  # tag
            for _ in range(frames):
                if renew:
                    guiutils.park_offscreen("park_probe")  # tag
                dpg.render_dearpygui_frame()
                seen.append(dpg.get_item_rect_min("park_probe_text")[0])  # tag
            return seen

        edge = dpg.get_viewport_client_width()
        parked_once = drawn_x_over(3, renew=False)
        renewed = drawn_x_over(3, renew=True)

        assert parked_once[0] >= edge, "the frame right after positioning is the one park that holds"
        assert min(parked_once[1:]) < edge, ("nothing was clamped, so this fixture cannot tell a renewed "
                                             "park from an abandoned one")
        assert min(renewed) >= edge, f"a renewed park was still pulled on screen: {renewed} against {edge}"
    finally:
        dpg.delete_item("park_probe")  # tag


class TestNonexistentOkAndWhatDPGSaysAboutDeadItems:
    """A dead item is reported two different ways, and only one of them says "not found".

    Which matters because Raven's GUI code deletes widgets from background threads all the time — a view
    rebuild ending a streaming message is the ordinary case — so "the thing I am drawing into just went
    away" is expected rather than exceptional, and `nonexistent_ok` is what makes it survivable.

    The first two tests characterize DPG rather than Raven. They exist because the distinction is invisible
    until it costs you an afternoon: an `add_*` handed a dead parent does *not* say "Item not found", so a
    guard matching that string sails straight past it, and the exception surfaces somewhere that has
    nothing to do with the widget that died. If a future DPG unifies the two, these fail, and that failure
    is the signal to go simplify the guard.
    """

    @staticmethod
    def _dead_item():
        """A group that existed and does not any more."""
        item = dpg.add_group(parent="nonexistent_ok_probe_window")  # tag
        dpg.delete_item(item)
        return item

    @pytest.fixture
    def probe_window(self, dpg_context):
        with dpg.window(tag="nonexistent_ok_probe_window"):  # tag
            pass
        yield
        dpg.delete_item("nonexistent_ok_probe_window")  # tag

    def test_operating_on_a_dead_item_says_item_not_found(self, probe_window):
        dead = self._dead_item()
        with pytest.raises(Exception) as excinfo:
            dpg.set_value(dead, 1)
        assert guiutils._is_dpg_item_not_found(excinfo.value)

    def test_adding_under_a_dead_parent_says_something_else_entirely(self, probe_window):
        dead = self._dead_item()
        with pytest.raises(Exception) as excinfo:
            dpg.add_text("into the void", parent=dead)
        assert not guiutils._is_dpg_item_not_found(excinfo.value), "DPG has unified the two; the guard can be simplified"
        assert guiutils._is_dpg_parent_gone(excinfo.value)

    def test_the_plain_guard_does_not_swallow_a_dead_parent(self, probe_window):
        """Deliberate: "parent could not be deduced" is also what a missing parent argument produces, and
        that is a mistake in the calling code rather than a widget that went away."""
        dead = self._dead_item()
        with pytest.raises(Exception):
            with guiutils.nonexistent_ok():
                dpg.add_text("into the void", parent=dead)

    def test_asking_for_it_swallows_a_dead_parent(self, probe_window):
        dead = self._dead_item()
        with guiutils.nonexistent_ok(parent_gone_ok=True) as nok:
            dpg.add_text("into the void", parent=dead)
        assert nok.errored, "the fixture's parent was alive, so this proves nothing about the suppression"

    def test_what_is_swallowed_is_still_written_down(self, probe_window, caplog):
        """A suppressed exception that leaves no trace is a debugging session nobody can start.

        DPG's own message names the *new* item and never the parent, so the log line has to carry the call
        site instead — that is the half that says whose parent went away.
        """
        dead = self._dead_item()
        with caplog.at_level(logging.DEBUG, logger="raven.common.gui.utils"):
            with guiutils.nonexistent_ok(parent_gone_ok=True):
                dpg.add_text("into the void", parent=dead)
        assert "test_utils.py" in caplog.text, "the log names DPG's own wrapper instead of the code that called it"
        assert "parent gone" in caplog.text

    def test_the_log_names_the_parent_that_went_away(self, probe_window, caplog):
        """`[1011]` names the item being created, never the parent — so the value comes from the frame."""
        parent = self._dead_item()
        with caplog.at_level(logging.DEBUG, logger="raven.common.gui.utils"):
            with guiutils.nonexistent_ok(parent_gone_ok=True):
                dpg.add_text("into the void", parent=parent)
        assert f"parent=id {parent}" in caplog.text
        assert "[deleted]" in caplog.text


class TestDescribeItem:
    """Both spellings of a widget's identity in one string, for a log line."""

    def test_a_tagged_widget_gives_tag_and_id(self, dpg_context):
        with dpg.window(tag="describe_probe_window"):  # tag
            widget = dpg.add_group(tag="describe_probe_group")  # tag
        try:
            described = guiutils.describe_item(widget)
            assert "describe_probe_group" in described
            assert str(dpg.get_alias_id("describe_probe_group")) in described  # tag
        finally:
            dpg.delete_item("describe_probe_window")  # tag

    def test_an_untagged_widget_gives_its_number(self, dpg_context):
        with dpg.window(tag="describe_probe_window2"):  # tag
            widget = dpg.add_group()
        try:
            assert guiutils.describe_item(widget) == f"id {widget}"
        finally:
            dpg.delete_item("describe_probe_window2")  # tag

    def test_a_deleted_widget_says_so(self, dpg_context):
        with dpg.window(tag="describe_probe_window3"):  # tag
            widget = dpg.add_group()
        dpg.delete_item("describe_probe_window3")  # tag
        described = guiutils.describe_item(widget)
        assert described.endswith("[deleted]"), described

    def test_a_deleted_widget_is_still_named_by_its_tag(self, dpg_context):
        """The reason this helper is worth having: a dead widget can still say who it was.

        DPG frees items lazily, and the alias outlives the item — so the tag is still answerable after a
        delete, which is exactly when someone is reading the log. The liveness marker is what says the
        widget is gone; the tag's presence is no evidence either way.
        """
        with dpg.window(tag="describe_probe_window4"):  # tag
            widget = dpg.add_group(tag="describe_probe_doomed")  # tag
        assert "[deleted]" not in guiutils.describe_item(widget), "the fixture was dead before the delete"
        dpg.delete_item("describe_probe_window4")  # tag

        described = guiutils.describe_item(widget)
        assert "describe_probe_doomed" in described, described
        assert "[deleted]" in described, described


class TestAddSectionSeparator:
    def test_it_puts_space_on_both_sides_of_the_line(self, dpg_context):
        # A bare `dpg.add_separator` draws flush against its neighbours, which reads as a line attached
        # to one of them rather than as a break between the two.
        with dpg.window() as window:
            dpg.add_text("above")
            group = guiutils.add_section_separator(spacing=6)
            dpg.add_text("below")
        kinds = [dpg.get_item_type(item) for item in dpg.get_item_children(group, slot=1)]
        assert kinds == ["mvAppItemType::mvSpacer",
                         "mvAppItemType::mvSeparator",
                         "mvAppItemType::mvSpacer"], kinds
        dpg.delete_item(window)

    def test_it_is_one_item_among_what_it_divides(self, dpg_context):
        # Grouped so the item registry inspector shows one node rather than three loose items sitting
        # between the widgets they separate.
        with dpg.window() as window:
            above = dpg.add_text("above")
            group = guiutils.add_section_separator()
            below = dpg.add_text("below")
        assert dpg.get_item_children(window, slot=1) == [above, group, below]
        assert dpg.get_item_type(group) == "mvAppItemType::mvGroup"
        dpg.delete_item(window)

    def test_a_tag_names_the_group(self, dpg_context):
        with dpg.window() as window:
            guiutils.add_section_separator(tag="a_named_separator")  # tag
        assert dpg.does_item_exist("a_named_separator")  # tag
        assert dpg.get_item_type("a_named_separator") == "mvAppItemType::mvGroup"  # tag
        dpg.delete_item(window)

    def test_the_spacing_is_what_was_asked_for(self, dpg_context):
        with dpg.window() as window:
            group = guiutils.add_section_separator(spacing=13)
        spacers = [item for item in dpg.get_item_children(group, slot=1)
                   if dpg.get_item_type(item) == "mvAppItemType::mvSpacer"]
        assert [dpg.get_item_configuration(item)["height"] for item in spacers] == [13, 13]
        dpg.delete_item(window)

    def test_an_explicit_parent_is_honoured(self, dpg_context):
        # The panel builds inside a `with dpg.window(...)`, but a caller adding to an existing container
        # after the fact has only the parent to go on.
        with dpg.window() as window:
            container = dpg.add_group()
        separator = guiutils.add_section_separator(parent=container)
        assert dpg.get_item_children(container, slot=1) == [separator]
        dpg.delete_item(window)


class TestSnapSlider:
    """`snap_slider` exists because ImGui's float slider has no step: `format` decides how the number
    is drawn and not what it is, so a drag stores more digits than the control ever offered.
    """

    @pytest.fixture
    def slider(self, dpg_context):
        with dpg.window() as window:
            yield dpg.add_slider_float(min_value=0.0, max_value=10.0, default_value=1.0, parent=window)
            dpg.delete_item(window)

    def test_it_rounds_to_the_asked_precision(self, slider):
        assert guiutils.snap_slider(slider, 2.5327194213867188) == 2.5

    def test_it_snaps_the_handle_too(self, slider):
        # Without this the widget keeps the unrounded number and the next drag starts from a value
        # nothing ever stored — invisible, since the display rounds either way.
        guiutils.snap_slider(slider, 2.5327194213867188)
        assert dpg.get_value(slider) == pytest.approx(2.5)

    def test_more_decimals_keep_more(self, slider):
        assert guiutils.snap_slider(slider, 2.5327194213867188, decimals=3) == 2.533

    def test_zero_decimals_give_a_whole_number(self, slider):
        assert guiutils.snap_slider(slider, 2.5327194213867188, decimals=0) == 3.0

    def test_it_cannot_be_a_dpg_callback_as_it_stands(self):
        # DPG passes a callback as many positional arguments as `len(signature.parameters)` — which
        # counts `decimals`, keyword-only and defaulted though it is. So a call site must wrap this in
        # a two-parameter lambda, and this asserts the reason rather than leaving the wrapper looking
        # like something a later tidy-up could remove.
        import inspect
        parameters = inspect.signature(guiutils.snap_slider).parameters
        assert len(parameters) == 3, "if this is 2, the wrappers at the call sites are no longer needed"
        assert parameters["decimals"].kind is inspect.Parameter.KEYWORD_ONLY
