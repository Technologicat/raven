"""Tests for `helpcard`: who gets the keys, what a card too young to exist reports, and what pages build.

The first two are about the seam between a card and whatever put it on the screen. The dispatch rule needs
no DPG at all — one module-level function, one global — so it is tested against a stand-in rather than a
rendered window. The `show` contract needs a real `HelpWindow`, but not a rendered one: declining to build
is precisely what it does before the GUI has settled, which a test suite renders no frames to change.

The page tests are structural for the same reason: which widgets exist, which are shown, what the toolbar
says. The one thing about pages that needs *layout* is the height fit, which measures each page and keeps
the tallest — that needs frames, and has none here.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from unpythonic.env import env  # noqa: E402 -- after importorskip by design

from raven.common.gui import helpcard  # noqa: E402 -- after importorskip by design
from raven.common.gui import utils as guiutils  # noqa: E402 -- after importorskip by design


class FakeCard:
    """Enough of a `HelpWindow` for the module-level key handler to decide about."""
    def __init__(self, handle_own_hotkeys, page_count=1):
        self.handle_own_hotkeys = handle_own_hotkeys
        self.page_count = page_count
        self.hidden = False
        self.turned = []

    def hide(self):
        self.hidden = True

    def first_page(self):
        self.turned.append("first")

    def previous_page(self):
        self.turned.append("previous")

    def next_page(self):
        self.turned.append("next")

    def last_page(self):
        self.turned.append("last")


@pytest.fixture
def visible_card(monkeypatch):
    """Put a stand-in card on the screen, and take it off again however the test leaves things."""
    def install(handle_own_hotkeys, page_count=1):
        card = FakeCard(handle_own_hotkeys, page_count=page_count)
        monkeypatch.setattr(helpcard, "visible_help_window_instance", card)
        return card
    return install


def test_a_card_closes_itself_on_escape(visible_card):
    card = visible_card(handle_own_hotkeys=True)
    helpcard.helpcard_hotkeys_callback(None, dpg.mvKey_Escape)
    assert card.hidden


def test_a_card_whose_owner_routes_keys_leaves_escape_alone(visible_card):
    """The opt-out exists so that a card belonging to another modal has exactly one handler.

    Its owner — a file dialog, say — reads Escape as "close the card" and must not also read it as its own
    cancel; that is only decidable if the shared handler keeps its hands off.
    """
    card = visible_card(handle_own_hotkeys=False)
    helpcard.helpcard_hotkeys_callback(None, dpg.mvKey_Escape)
    assert not card.hidden


def test_a_key_that_is_not_escape_closes_nothing(visible_card):
    card = visible_card(handle_own_hotkeys=True)
    helpcard.helpcard_hotkeys_callback(None, dpg.mvKey_A)
    assert not card.hidden


def test_showing_a_card_before_the_gui_has_settled_reports_that_it_did_not(request):
    """`_render` waits for ten frames, and this suite renders none — the same condition as app startup.

    The answer is what a caller needs when it has hidden something *behind* the card: a file dialog takes
    itself off the screen to make room, and a card that silently failed to appear would leave nothing there.
    """
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    try:
        with dpg.window(tag=f"reference_{request.node.name}"):  # tag
            pass
        card = helpcard.HelpWindow(hotkey_info=[env(key_indent=0, key="F1", action_indent=0, action="Help", notes="")],
                                   width=400, height=200,
                                   reference_window=f"reference_{request.node.name}",  # tag
                                   themes_and_fonts=env(font_size=20))
        assert card.show() is False
        assert not card.is_visible()
    finally:
        dpg.destroy_context()


def test_the_two_show_hooks_fire_at_the_moments_they_promise(request, monkeypatch):
    """`on_parked` while the card is parked and unplaced; `on_show` once it is placed and up.

    The order is a contract two callers depend on from opposite ends, and getting it wrong is invisible
    in a still image. A file dialog measures its card from `on_parked`, which needs the card drawn but
    not yet placed. `raven.visualizer`'s `enter_modal_mode` runs from `on_show` and asks the GUI what is
    currently visible — *and spends a frame doing it*, so running it while the card is parked draws a
    frame ImGui clamps back inside the viewport, and the card appears at the bottom right before jumping
    to the middle. That is what this pins.

    Needs no rendered frames: what is asserted is where the card *is* at each callback, and the position
    is set by the code under test rather than by layout. `_render` refuses before frame 10, which is the
    one thing that has to be faked.
    """
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    try:
        monkeypatch.setattr(dpg, "get_frame_count", lambda: 100)  # past `_render`'s ten-frame threshold
        monkeypatch.setattr(guiutils, "split_frame", lambda **kwargs: None)  # this suite renders none
        # The card's header is its one piece of Markdown, and that renderer measures text by asking DPG
        # until it gets an answer — a `while 1` that never returns where no frame is drawn. It only renders
        # inline once its `STARTUP_DONE` is set, and queues the work otherwise, so this suite is safe from
        # it today. But that flag is a *class* attribute and so lives as long as the process: a run in which
        # some earlier module rendered frames would set it, and this test would then hang rather than fail.
        # Stub the header out instead of depending on collection order; what is under test is when the two
        # hooks fire.
        monkeypatch.setattr(helpcard.dpg_markdown, "add_text", lambda *args, **kwargs: None)
        with dpg.window(tag=f"reference_{request.node.name}", width=800, height=600):  # tag
            pass

        seen = {}

        def note(which):
            def record():
                seen[which] = (tuple(dpg.get_item_pos(card._window)),
                               dpg.get_item_configuration(card._window)["show"])
            return record

        card = helpcard.HelpWindow(hotkey_info=[env(key_indent=0, key="F1", action_indent=0, action="Help", notes="")],
                                   width=400, height=200,
                                   reference_window=f"reference_{request.node.name}",  # tag
                                   themes_and_fonts=env(font_size=20),
                                   on_parked=note("parked"),
                                   on_show=note("shown"))
        assert card.show() is True

        park = (dpg.get_viewport_client_width(), dpg.get_viewport_client_height())
        assert seen["parked"] == (park, True), "on_parked must run with the card drawn, and parked"
        assert seen["shown"][1] is True
        assert seen["shown"][0] != park, ("on_show must run with the card placed — a handler that spends a "
                                          "frame here would otherwise have it drawn at the park, which "
                                          "ImGui clamps back into view")
    finally:
        dpg.destroy_context()


def test_a_card_that_was_never_built_has_no_measurable_content(request):
    """What a caller would size the window to is layout's answer, and layout needs a rendered frame.

    The same condition as above, and the answer has to be `None` rather than a number: a caller asks this
    in order to resize a window, so an invented figure would be acted on. The measured case needs frames
    and so lives with the `gui` tests, in `raven/vendor/file_dialog/tests/test_fdialog.py`.
    """
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    try:
        with dpg.window(tag=f"reference_{request.node.name}"):  # tag
            pass
        card = helpcard.HelpWindow(hotkey_info=[env(key_indent=0, key="F1", action_indent=0, action="Help", notes="")],
                                   width=400, height=200,
                                   reference_window=f"reference_{request.node.name}",  # tag
                                   themes_and_fonts=env(font_size=20))
        assert card.measure_content_height() is None
    finally:
        dpg.destroy_context()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@pytest.fixture
def dpg_context():
    """A DPG context with an unmapped viewport, for tests that build a real `HelpWindow`.

    Per test, and torn down after each, matching what the tests above do by hand: constructing a card
    registers a process-wide key handler, and that registry belongs to the context that created it.
    """
    dpg.create_context()
    dpg.create_viewport(width=400, height=300)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


class TestDeclaringPages:
    """What `helpcard.page` and the constructor accept, and what they refuse.

    The refusals are the point of testing this at all: a card is built once at app start and shown much
    later, so a mistake here surfaces as a blank or wrong card the first time somebody presses F1, which
    may be days after the code that caused it was written.
    """

    def test_a_page_needs_a_name(self):
        # The toolbar shows it, and a toolbar that cannot say where you are is most of the way to no
        # toolbar. Enforced here rather than left as a convention each caller has to have read.
        with pytest.raises(ValueError, match="needs a name"):
            helpcard.page("", hotkey_info=[])

    def test_a_page_needs_something_on_it(self):
        with pytest.raises(ValueError, match="would be blank"):
            helpcard.page("Empty")

    def test_a_page_takes_a_table_or_extras_or_both(self):
        assert helpcard.page("Keys", hotkey_info=[]).name == "Keys"
        assert helpcard.page("About", on_render_extras=lambda card, parent: None).name == "About"
        both = helpcard.page("Both", hotkey_info=[], on_render_extras=lambda card, parent: None)
        assert both.hotkey_info is not None and both.on_render_extras is not None

    def _card(self, **kwargs):
        return helpcard.HelpWindow(width=400, height=200, reference_window="nonexistent",
                                   themes_and_fonts=env(font_size=20), **kwargs)

    def test_the_two_forms_are_mutually_exclusive(self):
        # Both given, there would be no answer to what goes on the first page, and picking one silently
        # would drop the other's content off the card.
        with pytest.raises(ValueError, match="not both"):
            self._card(hotkey_info=[], pages=[helpcard.page("Keys", hotkey_info=[])])

    def test_a_card_with_nothing_at_all_is_refused(self):
        with pytest.raises(ValueError, match="nothing to show"):
            self._card()

    def test_an_empty_page_list_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            self._card(pages=[])

    def test_a_single_screen_card_is_one_page(self, dpg_context):
        card = self._card(hotkey_info=[])
        assert card.page_count == 1


class TestTurningPages:
    """Where the page index goes, in the absence of any GUI. Clamped at the ends rather than wrapped."""

    def _card(self, count=3):
        pages = [helpcard.page(f"Page {i}", hotkey_info=[]) for i in range(count)]
        return helpcard.HelpWindow(width=400, height=200, reference_window="nonexistent",
                                   themes_and_fonts=env(font_size=20), pages=pages)

    def test_it_starts_on_the_first_page(self, dpg_context):
        assert self._card().page_index == 0

    def test_next_and_previous_step_one(self, dpg_context):
        card = self._card()
        card.next_page()
        assert card.page_index == 1
        card.previous_page()
        assert card.page_index == 0

    def test_the_ends_clamp_rather_than_wrap(self, dpg_context):
        # Wrapping would put the reader at the far end of the run with nothing having said they arrived —
        # the same call the chat graph's sibling steps make.
        card = self._card()
        card.previous_page()
        assert card.page_index == 0, "stepping back from the first page went somewhere"
        card.last_page()
        assert card.page_index == 2, "End did not reach the last page"
        card.next_page()
        assert card.page_index == 2, "stepping past the last page went somewhere"
        card.first_page()
        assert card.page_index == 0

    def test_setting_the_index_out_of_range_clamps(self, dpg_context):
        card = self._card()
        card.page_index = 99
        assert card.page_index == 2
        card.page_index = -5
        assert card.page_index == 0

    def test_a_single_screen_card_has_nowhere_to_turn(self, dpg_context):
        card = helpcard.HelpWindow(width=400, height=200, reference_window="nonexistent",
                                   themes_and_fonts=env(font_size=20), hotkey_info=[])
        card.next_page()
        card.last_page()
        assert card.page_index == 0


class TestThePagingKeys:
    """`Left`, `Right`, `Home` and `End` reach the card, and only when there is more than one page."""

    def test_the_four_keys_turn_pages(self, visible_card):
        card = visible_card(handle_own_hotkeys=True, page_count=3)
        for key in (dpg.mvKey_Right, dpg.mvKey_Left, dpg.mvKey_Home, dpg.mvKey_End):
            helpcard.helpcard_hotkeys_callback(None, key)
        assert card.turned == ["next", "previous", "first", "last"]

    def test_a_single_screen_card_ignores_them(self, visible_card):
        # The control: a handler that turned pages regardless would satisfy the test above just as well,
        # and would be taking four keys from a card that has no use for them.
        card = visible_card(handle_own_hotkeys=True, page_count=1)
        helpcard.helpcard_hotkeys_callback(None, dpg.mvKey_Right)
        assert card.turned == []

    def test_paging_works_even_when_the_owner_handles_escape(self, visible_card):
        # `handle_own_hotkeys` is about Escape, which an owning modal wants for itself. These four collide
        # with nothing: a card is up with its owner hidden behind it, and the card is modal besides.
        card = visible_card(handle_own_hotkeys=False, page_count=2)
        helpcard.helpcard_hotkeys_callback(None, dpg.mvKey_Right)
        assert card.turned == ["next"]
        helpcard.helpcard_hotkeys_callback(None, dpg.mvKey_Escape)
        assert not card.hidden, "the owner's Escape was taken after all"


class TestWhatAPagedCardBuilds:
    """The widgets, in a context that renders no frames — so this is about structure, not about layout."""

    @staticmethod
    def _build(request, monkeypatch, pages):
        monkeypatch.setattr(dpg, "get_frame_count", lambda: 100)  # past `_render`'s ten-frame threshold
        monkeypatch.setattr(guiutils, "split_frame", lambda **kwargs: None)  # this suite renders none
        # The header is Markdown, and that renderer measures text by asking DPG until it answers, which
        # never happens where no frame is drawn. See the note in the show-hooks test above.
        monkeypatch.setattr(helpcard.dpg_markdown, "add_text", lambda *args, **kwargs: None)
        with dpg.window(tag=f"reference_{request.node.name}", width=800, height=600):  # tag
            pass
        card = helpcard.HelpWindow(width=400, height=200,
                                   reference_window=f"reference_{request.node.name}",  # tag
                                   themes_and_fonts=env(font_size=20),
                                   pages=pages)
        assert card.show() is True
        return card

    @staticmethod
    def _entry(key):
        return env(key_indent=0, key=key, action_indent=0, action="does a thing", notes="")

    def test_every_page_is_built_and_only_one_is_shown(self, request, monkeypatch, dpg_context):
        # All of them up front: the extras renderers are documented to run once, and a page appearing only
        # when first turned to would have to be measured then too — a resize under the reader's eye.
        seen = []
        card = self._build(request, monkeypatch,
                           [helpcard.page("Keys", hotkey_info=[self._entry("F1")]),
                            helpcard.page("About", on_render_extras=lambda c, parent: seen.append(parent))])
        assert len(card._page_groups) == 2
        assert seen, "the second page's extras renderer never ran"
        shown = [dpg.get_item_configuration(group)["show"] for group in card._page_groups]
        assert shown == [True, False]

        card.next_page()
        assert [dpg.get_item_configuration(group)["show"] for group in card._page_groups] == [False, True]

    def test_the_toolbar_says_where_you_are(self, request, monkeypatch, dpg_context):
        card = self._build(request, monkeypatch,
                           [helpcard.page("Keys", hotkey_info=[self._entry("F1")]),
                            helpcard.page("About", hotkey_info=[self._entry("F2")]),
                            helpcard.page("Notes", hotkey_info=[self._entry("F3")])])
        assert dpg.get_value(card._toolbar.name_widget) == "Keys"
        assert dpg.get_value(card._toolbar.counter_widget) == "1 / 3"
        card.last_page()
        assert dpg.get_value(card._toolbar.name_widget) == "Notes"
        assert dpg.get_value(card._toolbar.counter_widget) == "3 / 3"

    def test_a_button_is_enabled_exactly_when_pressing_it_would_do_something(self, request, monkeypatch, dpg_context):
        card = self._build(request, monkeypatch,
                           [helpcard.page("Keys", hotkey_info=[self._entry("F1")]),
                            helpcard.page("About", hotkey_info=[self._entry("F2")]),
                            helpcard.page("Notes", hotkey_info=[self._entry("F3")])])

        def enabled():
            return [dpg.get_item_configuration(widget)["enabled"]
                    for widget in (card._toolbar.first_button, card._toolbar.previous_button,
                                   card._toolbar.next_button, card._toolbar.last_button)]

        assert enabled() == [False, False, True, True], "on the first page, going back does nothing"
        card.next_page()
        assert enabled() == [True, True, True, True], "in the middle, all four go somewhere"
        card.next_page()
        assert enabled() == [True, True, False, False], "on the last page, going on does nothing"

    def test_a_single_screen_card_grows_no_toolbar(self, request, monkeypatch, dpg_context):
        # The control, and the reason the toolbar is conditional at all: four dead buttons and a `1 / 1`
        # counter are chrome that answers nothing, which is the shape of a box that invites a click and
        # does nothing when clicked.
        monkeypatch.setattr(dpg, "get_frame_count", lambda: 100)
        monkeypatch.setattr(guiutils, "split_frame", lambda **kwargs: None)
        monkeypatch.setattr(helpcard.dpg_markdown, "add_text", lambda *args, **kwargs: None)
        with dpg.window(tag=f"reference_{request.node.name}", width=800, height=600):  # tag
            pass
        card = helpcard.HelpWindow(width=400, height=200,
                                   reference_window=f"reference_{request.node.name}",  # tag
                                   themes_and_fonts=env(font_size=20),
                                   hotkey_info=[self._entry("F1")])
        assert card.show() is True
        assert card._toolbar is None
        assert len(card._page_groups) == 1
