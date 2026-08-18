"""Pins the DearPyGui keyboard-focus behaviour that Raven's global hotkey handlers are built on.

Every app in the constellation has a text field living in the same window as the content its bare-key
shortcuts drive, so each one needs to answer "does this key belong to the field or to the app?" — and the
answer that looks obvious is wrong twice over. Both apps shipped the wrong version of it:

- `is_item_focused` is true of a text field nobody has touched, because ImGui hands nav focus to the first
  navigable item of a newly focused window on its own. A bare-key branch gated on it is dead from app start
  until the user clicks something else. `is_item_active` is the state that means "this field holds the
  caret", which is the real question.
- `dpg.focus_item` cannot focus a child window. Asked to, it does not merely fail: focus lands on the
  enclosing window's first navigable item and is *activated*. So "park focus on the scrollable panel", the
  natural way to express "the reader is reading now", is instead a reliable way to hand a text field the
  caret.

Neither is documented, and neither produces an error — the failure is a hotkey that silently does nothing.
So these assertions exist to notice a change, in either direction: if a future DPG fixes the child-window
case, `test_focus_item_on_a_child_window_activates_a_text_field_instead` starts failing, and that failure is
the signal to go simplify the handlers rather than a defect to work around.

These tests map a real window, because focus is only meaningful once frames are rendering, and
`dpg.render_dearpygui_frame()` aborts the process when there is nothing to render into. Mapping a window
takes keyboard focus from whatever the developer is typing into, so they are marked `gui` and skipped unless
`--run-gui` is passed. The window is on screen for well under a second.

Most of these only read focus state back. Two synthesize key presses, and say so in their names: that a
focused button ignores Space and Enter — DPG leaves ImGui's keyboard-nav activation off — is what makes a
button a safe place to park focus, and `FileDialog` now parks there whenever Tab hands the arrow keys to
its listing. An assumption load-bearing enough to be worth the intrusion: it is an upstream default nothing
in DPG's API exposes for inspection, so behaviour is the only place it can be read.

Should it ever flip, the key that breaks is **Enter**, not the arrows the parking was done for. Enter means
*descend into the folder under the cursor* there, and a focused OK button would answer it too — so the
dialog would commit and close instead of stepping into the directory, which in save mode means writing a
file. The arrow keys would be unaffected; a button has nothing to do with them either way.
"""

import shutil
import subprocess
import time

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

pytestmark = pytest.mark.gui

# Focus changes land on the frame *after* the request, and ImGui's own auto-focus needs a few frames to
# settle. Ten is comfortably past both and still imperceptible.
_SETTLE_FRAMES = 10


@pytest.fixture
def mapped_viewport():
    """A DPG context whose viewport is actually shown, so frames can be rendered.

    Contrast `dpg_context` in the sibling test modules, which deliberately never shows its viewport. Here
    the mapping is the point: an unmapped viewport cannot render, and focus does not exist until it does.
    """
    dpg.create_context()
    dpg.create_viewport(title="raven focus semantics test", width=320, height=240)
    dpg.setup_dearpygui()

    with dpg.window(tag="main"):
        dpg.add_child_window(tag="panel", width=300, height=90)
        dpg.add_input_text(tag="field", multiline=True, width=300, height=60)
        dpg.add_button(tag="button", label="a button")
    dpg.set_primary_window("main", True)

    dpg.show_viewport()
    yield
    dpg.destroy_context()


def render(n_frames: int = _SETTLE_FRAMES) -> None:
    """Render `n_frames` frames, letting DPG apply anything queued for a later frame."""
    for _ in range(n_frames):
        dpg.render_dearpygui_frame()


def test_a_text_field_reports_focused_without_anyone_touching_it(mapped_viewport):
    """ImGui's auto-focus, and the reason `is_item_focused` is the wrong thing to gate a hotkey on."""
    render()
    assert dpg.is_item_focused("field") is True  # tag  # nobody has clicked or typed


def test_an_untouched_text_field_is_not_active(mapped_viewport):
    """The other half: *active* distinguishes the auto-focused field from one holding the caret.

    Without this, the previous test would merely say focus is unreliable. Together they say which of the
    two predicates a hotkey handler should ask.
    """
    render()
    assert dpg.is_item_active("field") is False  # tag


def test_focus_item_moves_focus_between_ordinary_items(mapped_viewport):
    """The baseline: `focus_item` does work, so the child-window result below is about child windows."""
    render()
    dpg.focus_item("button")  # tag
    render()
    assert dpg.is_item_focused("button") is True  # tag
    assert dpg.is_item_focused("field") is False  # tag


def test_focus_item_does_not_focus_a_child_window(mapped_viewport):
    """`focus_item` cannot give a child window keyboard focus, however plainly one asks.

    Scoped to the API on purpose: a child window is not unfocusable in general. Clicking one — including
    grabbing its scrollbar — does focus it, which is measurable but needs synthetic input, so it lives in
    `investigations/dpg-focus/` rather than here. What has no working spelling is *asking*.
    """
    render()
    dpg.focus_item("button")  # tag  # somewhere definite first, so the result cannot be the initial state
    render()
    dpg.focus_item("panel")  # tag
    render()
    assert dpg.is_item_focused("panel") is False  # tag


def test_focus_item_on_a_child_window_activates_a_text_field_instead(mapped_viewport):
    """And the request does not fail quietly — it hands the caret to the field it was meant to leave.

    This is the one that makes the child-window case a hazard rather than a no-op, so it is asserted
    separately from the "does not focus" result above. A failure here most likely means DPG fixed it, in
    which case the handlers that work around it can be simplified.
    """
    render()
    dpg.focus_item("button")  # tag
    render()
    assert dpg.is_item_active("field") is False, "precondition: the field starts without the caret"  # tag

    dpg.focus_item("panel")  # tag
    render()
    assert dpg.is_item_focused("field") is True  # tag
    assert dpg.is_item_active("field") is True  # tag


def _press(keysym: str, window_title: str) -> bool:
    """Send one key press to the window named `window_title`. False if the desktop tools are missing.

    The window is really activated first, rather than addressed with `xdotool key --window`: GLFW reads
    input from the X server directly and ignores the `XSendEvent` that the windowed form produces, so a
    targeted press is silently discarded. Which means this genuinely takes the keyboard for a moment.
    """
    if not all(shutil.which(tool) for tool in ("xdotool", "wmctrl")):
        return False
    found = subprocess.run(["xdotool", "search", "--name", window_title],
                           capture_output=True, text=True)
    window_ids = found.stdout.split()
    if not window_ids:
        return False
    subprocess.run(["xdotool", "windowactivate", "--sync", window_ids[-1]], check=False)
    time.sleep(0.2)
    subprocess.run(["xdotool", "key", keysym], check=False)
    time.sleep(0.2)
    return True


def test_synthesized_keys_reach_the_app_at_all(mapped_viewport):
    """The control for the test below, and the reason its silence can be read as an answer.

    A test asserting that a key did *nothing* passes just as well when the key never arrived — so one
    keystroke has to be shown landing somewhere before the absence of another means anything. A text field
    holding the caret is the cheapest witness: type into it and its value changes.
    """
    dpg.set_value("field", "")  # tag
    render()
    dpg.focus_item("field")  # tag
    render()

    if not _press("x", "raven focus semantics test"):
        pytest.skip("xdotool/wmctrl not available, cannot synthesize a key press")
    render()

    assert dpg.get_value("field") == "x", "synthesized keys are not reaching the app"  # tag


@pytest.mark.parametrize("keysym", ["Return", "space"])
def test_a_focused_button_ignores_the_keys_that_would_press_it(mapped_viewport, keysym):
    """The property that makes a button a safe place to park focus.

    Parking focus somewhere is how a panel says "the keyboard is mine now" — `FileDialog` does it on every
    Tab — and the target has to be an item that will not act on the keys the panel then wants to use.
    A button is that, because DPG does not enable ImGui's keyboard navigation, so the focus a button holds
    is inert. Nothing in the API reports this, hence a behavioural test.

    Synthesizes real key presses, and therefore really takes the keyboard for about half a second. Skipped
    where `xdotool` is absent, which is every CI runner.
    """
    pressed = []
    dpg.configure_item("button", callback=lambda: pressed.append(keysym))  # tag
    render()
    dpg.focus_item("button")  # tag
    render()
    assert dpg.is_item_focused("button") is True, "precondition: the button holds focus"  # tag

    if not _press(keysym, "raven focus semantics test"):
        pytest.skip("xdotool/wmctrl not available, cannot synthesize a key press")
    render()

    assert pressed == [], (f"a focused button acted on {keysym}: ImGui keyboard navigation appears to be "
                          f"enabled, and parking focus on a button is no longer safe")
