"""Why does an inline-code background vanish on a help card?

The renderer decorates inline code with a quad drawn behind the text, sized from
`dpg.get_item_rect_size` of the text group and placed from `dpg.get_item_pos`. That work is deferred to a
worker thread (`CallInNextFrame`), so it happens a frame or more after the text was built.

It draws correctly in a chat message and drew correctly on Raven-avatar-pose-editor's card while that was a
single page. It draws nothing at all on every multi-page card, all of which also use
`HelpWindow.prose_columns`. Those two differences are confounded in every card we have, so this varies them
independently:

    nesting          direct parent  vs  the prose_columns shape (horizontal group -> vertical group)
    visible when     shown          vs  hidden at build time and revealed afterwards
    the worker runs                     (which is what a card's non-current pages are)

The suspicion is the second: a widget DPG has never laid out has no metrics, so the quad comes out 0x0 and
there is nothing to see — where a misplaced quad would at least be findable somewhere on screen, and we
looked for one and did not find it.

    python investigations/dpg-markdown-decorations/probe_nesting.py

Needs a display: sizes and positions are meaningless until frames have been rendered. Prints to stdout, so
do not pipe it — and it leaves the process with `os._exit`, the renderer's worker thread not participating
in DPG teardown (see `TODO_DEFERRED.md`, the fleet shutdown item).
"""

import os
import sys

import dearpygui.dearpygui as dpg

from raven.common.gui import utils as guiutils
from raven.vendor import DearPyGui_Markdown as dpg_markdown
from raven.vendor.DearPyGui_Markdown import text_attributes

PARAGRAPH = "A path such as `raven/avatar/assets/emotions/` sits in the middle of a sentence."
WRAP = 600

observations = {}
cases = {}


def instrument() -> None:
    """Wrap `Code.render` so each call reports what it read and what it built."""
    original = text_attributes.Code.render.__func__  # a classmethod

    def traced(cls, dpg_text_group):
        before = set(dpg.get_all_items())
        record = {"get_item_pos": _safe(dpg.get_item_pos, dpg_text_group),
                  "rect_size": _safe(dpg.get_item_rect_size, dpg_text_group),
                  "rect_min": _safe(dpg.get_item_rect_min, dpg_text_group),
                  "get_widget_pos": _safe(guiutils.get_widget_pos, dpg_text_group),
                  "visible": _safe(dpg.is_item_visible, dpg_text_group)}
        result = original(cls, dpg_text_group)
        record["created"] = sorted(set(dpg.get_all_items()) - before)
        observations.setdefault(dpg_text_group, []).append(record)
        return result

    text_attributes.Code.render = classmethod(traced)


def _safe(fn, *args):
    """Call `fn`, returning the exception instead of raising — these getters raise for some item types."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 -- a probe reports the failure rather than dying of it
        return f"{type(exc).__name__}: {exc}"


def build_case(name: str, *, nested: bool, hidden: bool) -> dict:
    """Build one paragraph under the named combination of nesting and initial visibility."""
    container = dpg.add_group(parent="win", show=not hidden)
    dpg.add_text(f"[{name}]", parent=container)
    if nested:
        outer = dpg.add_group(horizontal=True, parent=container)
        inner = dpg.add_group(parent=outer)
        dpg.add_spacer(width=WRAP + 8, parent=inner)
        parent = inner
    else:
        parent = container
    widget = dpg_markdown.add_text(PARAGRAPH, parent=parent, wrap=WRAP, color=(180, 180, 190))
    cases[name] = env = {"widget": widget, "container": container, "hidden": hidden}
    return env


def render(n: int) -> None:
    for _ in range(n):
        dpg.render_dearpygui_frame()


def main() -> None:
    dpg.create_context()
    guiutils.bootup(font_size=20)
    dpg.create_viewport(title="markdown decoration probe", width=1400, height=950)
    dpg.setup_dearpygui()
    instrument()

    dpg.add_window(label="cases", tag="win", width=1380, height=930, pos=(0, 0))

    # The cases are built *after* frames are running, not in the window above. Markdown cannot be measured
    # until the font it wants is in the atlas, and a font reaches the atlas only between frames — so
    # building it before the first frame raises, naming that rule.
    dpg.show_viewport()
    render(10)

    build_case("flat/shown", nested=False, hidden=False)
    build_case("nested/shown", nested=True, hidden=False)
    build_case("flat/hidden", nested=False, hidden=True)
    build_case("nested/hidden", nested=True, hidden=True)

    render(90)                      # let the worker decorate everything it is going to decorate

    # Now reveal the two that were hidden, exactly as turning to a card's second page does, and give the
    # renderer as many frames again. Nothing re-triggers the decoration: that is the point.
    for name, case in cases.items():
        if case["hidden"]:
            dpg.show_item(case["container"])
    render(90)

    print("\n=== after building hidden, then revealing ===")
    for name, case in cases.items():
        widget = case["widget"]
        print(f"\n{name}: markdown group {widget}")
        print(f"    rect_size now   {_safe(dpg.get_item_rect_size, widget)}")
        print(f"    get_item_pos    {_safe(dpg.get_item_pos, widget)}")
        print(f"    get_widget_pos  {_safe(guiutils.get_widget_pos, widget)}")

    print(f"\n=== Code.render fired for {len(observations)} text groups ===")
    for text_group, records in observations.items():
        for record in records:
            print(f"\n  text_group {text_group}  visible at decoration time: {record['visible']}")
            print(f"    rect_size read  {record['rect_size']}   <- the quad's size")
            print(f"    get_item_pos    {record['get_item_pos']}")
            print(f"    rect_min        {record['rect_min']}")
            print(f"    get_widget_pos  {record['get_widget_pos']}")
            print(f"    created         {record['created']}")

    sys.stdout.flush()
    os._exit(0)   # the renderer's worker keeps calling DPG through teardown; do not give it the chance


if __name__ == "__main__":
    main()
