"""`raven.common.gui.keyboardmark`'s followers under real focus changes, in a mapped window.

`test_keyboardmark.py` stands in for DPG's focus by monkeypatching what the followers ask, because a headless
context cannot move the focus. These tests let DPG move it, so they check the other half: that what
`guiutils.focus_item` and `gui_animation.give_caret` actually do to the focus agrees with what the followers
were told to expect.

Mapping a window takes keyboard focus from whatever the developer is typing into, so these are marked `gui`
and skipped unless `--run-gui` is passed. The window is the session-wide one from `mapped_gui_context`.
No key presses are synthesized.
"""

import time

import pytest

from unpythonic.env import env

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import animation as gui_animation  # noqa: E402 -- after importorskip by design
from raven.common.gui import keyboardmark  # noqa: E402 -- ditto
from raven.common.gui import utils as guiutils  # noqa: E402 -- ditto

pytestmark = pytest.mark.gui

# Comfortably past a focus change landing (the frame after the request) and a text field activating (the one
# after that), and still imperceptible.
_MAX_FRAMES = 20


@pytest.fixture
def widgets(mapped_gui_context, request):
    """A window holding a text field, which a caret follower marks, and a button to park the focus on."""
    name = request.node.name
    tags = env(main=f"kbmark_main_{name}",
               field=f"kbmark_field_{name}",
               button=f"kbmark_button_{name}")

    with dpg.window(tag=tags.main):
        dpg.add_input_text(tag=tags.field, width=300)
        dpg.add_button(tag=tags.button, label="a button")
    dpg.set_primary_window(tags.main, True)
    tags.follower = keyboardmark.install_caret_follower([tags.field])

    yield tags

    # The animator and the pulse are process-wide, and outlive the widgets they point at.
    gui_animation.animator.clear()
    keyboardmark._pulse = None
    guiutils._expected_focus = None
    dpg.set_primary_window(tags.main, False)
    dpg.delete_item(tags.main)


def frame() -> None:
    """One frame, in the order a Raven app's render loop takes them."""
    gui_animation.animator.render_frame()
    dpg.render_dearpygui_frame()


def lit(widget) -> bool:
    """Whether the keyboard mark on `widget` is showing: a lit mark's colour has a nonzero alpha."""
    theme = dpg.get_item_theme(widget)
    for component in dpg.get_item_children(theme, slot=1):
        for item in dpg.get_item_children(component, slot=1):
            if dpg.get_item_info(item)["type"].endswith("mvThemeColor"):
                return dpg.get_value(item)[3] > 0.0
    raise AssertionError(f"no mark colour on '{widget}'")


def frames_until_active(field, start=None) -> list[bool]:
    """Render until `field` holds the caret. Return whether its mark was lit on each frame it held it for.

    Each entry is read after the frame's animator tick, which is when the follower decides. Stops a few frames
    after the caret arrives.
    """
    lit_while_active = []
    for _ in range(_MAX_FRAMES):
        frame()
        if dpg.is_item_active(field):
            gui_animation.animator.render_frame()  # the follower's verdict on the state this frame produced
            lit_while_active.append(lit(field))
            if len(lit_while_active) >= 3:
                break
    return lit_while_active


class TestCaretFollowerUnderRealFocus:
    def test_give_caret_lights_the_field(self, widgets):
        gui_animation.give_caret(widgets.field)
        states = frames_until_active(widgets.field)
        assert states, "the field never took the caret"
        assert all(states), f"the field held the caret and its mark was dark: {states}"

    def test_a_park_takes_the_caret_and_the_mark_with_it(self, widgets):
        gui_animation.give_caret(widgets.field)
        assert frames_until_active(widgets.field), "the field never took the caret"

        guiutils.focus_item(widgets.button)
        for _ in range(_MAX_FRAMES):
            frame()
        gui_animation.animator.render_frame()
        assert not dpg.is_item_active(widgets.field), "parking on a button left the field holding the caret"
        assert not lit(widgets.field)

    def test_give_caret_right_after_a_park_lights_the_field_the_frame_it_arrives(self, widgets, monkeypatch):
        """Tab back into a field just after a park: the park's expectation must not hold the field dark."""
        # The control first: the same sequence with a `give_caret` whose moves onto the field say nothing
        # about where they are going. The park's expectation is then still in force, so the field is dark
        # while it holds the caret — which is what shows this fixture can tell the two apart. It relies on the
        # frames rendering well inside the park's 150 ms window.
        recording_focus_item = guiutils.focus_item
        with monkeypatch.context() as m:
            m.setattr(guiutils, "focus_item",
                      lambda w: dpg.focus_item(w) if w == widgets.field else recording_focus_item(w))
            t0 = time.monotonic()
            guiutils.focus_item(widgets.button)
            request = gui_animation.give_caret(widgets.field)
            control = frames_until_active(widgets.field)
            elapsed_ms = 1000 * (time.monotonic() - t0)
            if request is not None:
                gui_animation.animator.cancel(request)
        assert control, "the field never took the caret from the control's `give_caret`"
        assert not control[0], (f"the park's expectation did not hold the field dark ({elapsed_ms:.0f} ms to "
                                f"the caret), so this fixture cannot tell a `give_caret` that records its "
                                f"target from one that does not")

        guiutils.focus_item(widgets.button)
        for _ in range(_MAX_FRAMES):  # the park lands, and the field lets go of the caret
            frame()
        assert not dpg.is_item_active(widgets.field)

        guiutils.focus_item(widgets.button)
        gui_animation.give_caret(widgets.field)
        states = frames_until_active(widgets.field)
        assert states, "the field never took the caret"
        assert all(states), f"the park held the field's mark dark after `give_caret`: {states}"
