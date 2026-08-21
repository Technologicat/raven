"""Tests for `helpcard`: who gets the Escape key, and what a card too young to exist reports.

Both are about the seam between a card and whatever put it on the screen. The dispatch rule needs no DPG
at all — one module-level function, one global — so it is tested against a stand-in rather than a rendered
window. The `show` contract needs a real `HelpWindow`, but not a rendered one: declining to build is
precisely what it does before the GUI has settled, which a test suite renders no frames to change.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from unpythonic.env import env  # noqa: E402 -- after importorskip by design

from raven.common.gui import helpcard  # noqa: E402 -- after importorskip by design


class FakeCard:
    """Enough of a `HelpWindow` for the module-level key handler to decide about."""
    def __init__(self, handle_own_hotkeys):
        self.handle_own_hotkeys = handle_own_hotkeys
        self.hidden = False

    def hide(self):
        self.hidden = True


@pytest.fixture
def visible_card(monkeypatch):
    """Put a stand-in card on the screen, and take it off again however the test leaves things."""
    def install(handle_own_hotkeys):
        card = FakeCard(handle_own_hotkeys)
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
