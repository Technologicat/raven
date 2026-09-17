"""Does a decoration that waits while its text is hidden get drawn when the text is shown, and what does waiting cost?

The renderer decorates text a frame after building it, sized from the laid-out text, and a hidden widget is
not laid out. Decorations now wait, a frame at a time, for as long as their text is hidden. This checks the
four things that change could get wrong:

  1. a code span built hidden and revealed long afterwards comes out the size of one built shown;
  2. a paragraph deleted while hidden leaves the wait queue, rather than waiting forever;
  3. a fenced code block (whose border is a second deferred step, depending on the first) survives the same;
  4. the per-frame cost of many hidden decorated paragraphs waiting at once.

    python investigations/dpg-markdown-decorations/probe_waiting.py

**Maps a window**, so it takes keyboard focus while it runs. Prints to stdout; do not pipe it. Leaves with
`os._exit`, like `probe_nesting.py` and for its reason.
"""

import os
import statistics
import sys
import time

import dearpygui.dearpygui as dpg

from raven.common.gui import utils as guiutils
from raven.vendor import DearPyGui_Markdown as dpg_markdown

CODE_SPAN = "`raven/avatar/assets/emotions/`"
PARAGRAPH = f"A path such as {CODE_SPAN} sits in the middle of a sentence."
FENCED = "```\nsome code\nmore code\n```"
WRAP = 600
N_WAITING = 300  # hidden paragraphs for the cost measurement, each with three code spans
NESTING = 6  # containers between the window and each of them, roughly a help card's depth


def render(n: int) -> None:
    for _ in range(n):
        dpg.render_dearpygui_frame()


def code_quads(markdown_group):
    """`(width, height)` of every drawlist under `markdown_group` — the code backgrounds and block borders."""
    found = []

    def walk(item):
        for slot in dpg.get_item_children(item).values():
            for child in slot:
                if dpg.get_item_type(child).endswith("mvDrawlist"):
                    found.append(tuple(round(v) for v in dpg.get_item_rect_size(child)))
                walk(child)
    walk(markdown_group)
    return found


def waiting() -> tuple[int, int]:
    """`(entries, distinct blockers)` in the waiting room.

    Not the worker's queue: the worker empties that into a local batch at the start of every pass, so sampling
    it reads 0 whether or not anything is pending.
    """
    by_blocker = dpg_markdown.WaitUntilShown._by_blocker
    return sum(len(entries) for entries in by_blocker.values()), len(by_blocker)


def main() -> None:
    dpg.create_context()
    guiutils.bootup(font_size=20)
    dpg.create_viewport(title="markdown decoration wait probe", width=900, height=700)
    dpg.setup_dearpygui()
    dpg.add_window(tag="win", width=880, height=680, pos=(0, 0))
    dpg.show_viewport()
    render(10)

    # 1 and 3: shown, versus hidden and revealed long afterwards.
    shown = dpg_markdown.add_text(PARAGRAPH + "\n\n" + FENCED, parent="win", wrap=WRAP)
    hidden_container = dpg.add_group(parent="win", show=False)
    revealed = dpg_markdown.add_text(PARAGRAPH + "\n\n" + FENCED, parent=hidden_container, wrap=WRAP)
    render(300)  # far longer than any fixed bound on waiting would allow
    print(f"1/3: waiting while hidden (entries, blockers): {waiting()}")
    print(f"     built shown:            {code_quads(shown)}")
    print(f"     hidden, before reveal:  {code_quads(revealed)}")
    dpg.show_item(hidden_container)
    render(10)
    print(f"     hidden, after reveal:   {code_quads(revealed)}")
    print(f"     waiting after reveal: {waiting()}")

    # 2: deleted while hidden. The count before the delete is the control: zero there would mean nothing had
    # been waiting, and the zero after it would say nothing.
    doomed_container = dpg.add_group(parent="win", show=False)
    dpg_markdown.add_text(PARAGRAPH + "\n\n" + FENCED, parent=doomed_container, wrap=WRAP)
    render(10)
    before = waiting()
    dpg.delete_item(doomed_container)
    render(10)
    print(f"2:   waiting before delete {before}, after {waiting()}")

    # 4: many waiting at once. Timed around the one call each waiting decoration makes per frame.
    outer = dpg.add_group(parent="win", show=False)
    parent = outer
    for _ in range(NESTING):
        parent = dpg.add_group(parent=parent)
    source = " ".join([PARAGRAPH] * 3)
    t_build = time.perf_counter()
    for _ in range(N_WAITING):
        dpg_markdown.add_text(source, parent=parent, wrap=WRAP)
    t_build = time.perf_counter() - t_build
    render(5)

    render(30)  # let every decoration reach the waiting room

    calls = {"n": 0, "t": 0.0}
    original = dpg_markdown.WaitUntilShown.release_shown.__func__

    def timed(cls):
        t0 = time.perf_counter()
        try:
            return original(cls)
        finally:
            calls["t"] += time.perf_counter() - t0
            calls["n"] += 1
    dpg_markdown.WaitUntilShown.release_shown = classmethod(timed)

    per_pass_ms, frame_ms = [], []
    for _ in range(60):
        calls["n"], calls["t"] = 0, 0.0
        t0 = time.perf_counter()
        dpg.render_dearpygui_frame()
        frame_ms.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.005)  # let the worker take its turn inside this frame's window
        if calls["n"]:
            per_pass_ms.append(calls["t"] * 1000 / calls["n"])
    dpg_markdown.WaitUntilShown.release_shown = classmethod(original)
    print(f"4:   {N_WAITING} hidden paragraphs x 3 code spans, nesting {NESTING}; built in {t_build:.2f} s; "
          f"waiting (entries, blockers) {waiting()}")
    print(f"     release_shown per pass: median {statistics.median(per_pass_ms):.3f} ms, max {max(per_pass_ms):.3f} ms, "
          f"over {len(per_pass_ms)} passes in 60 frames")
    print(f"     render_dearpygui_frame: median {statistics.median(frame_ms):.2f} ms")

    sys.stdout.flush()
    os._exit(0)  # the renderer's worker keeps calling DPG through teardown; do not give it the chance


if __name__ == "__main__":
    main()
