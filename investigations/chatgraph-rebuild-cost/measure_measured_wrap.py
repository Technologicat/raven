"""What measuring the text costs a rebuild, against estimating it from an average glyph advance.

`chatgraph` wraps labels to a width. It can ask DPG how wide a string actually is, or estimate from an
average advance -- and the estimate is visibly wrong in the direction that matters: the figure that keeps
capitals inside the box cuts ordinary lowercase prose short of the edge, so boxes come out with unused room
on the right and messages wrap that would have fitted on one line.

Measuring fixes that and costs DPG calls, one per candidate line per word. This says how many milliseconds
that is, against the same 16.7 ms frame budget `measure_rebuild.py` uses.

**It maps a window**, unlike its sibling here, and that is not incidental: `dpg.get_text_size` answers only
once a font atlas exists, and an atlas is built by rendering a frame. A headless run would measure the
fallback twice and report no difference at all.

    python investigations/chatgraph-rebuild-cost/measure_measured_wrap.py
"""
import logging
import time

import dearpygui.dearpygui as dpg

from raven.common.gui import utils as guiutils
from raven.librarian import chatgraph
from raven.librarian.chattree import Forest

# The same ladder `chatgraph_panel` loads, so the nearest-atlas pick costs what it costs there.
FONT_SIZES = (10, 14, 20, 28, 40)


_serial = 0


def payload(role, text):
    """A node payload, with a timestamp.

    The timestamp is not filler. `chatutil.descend_to_latest` orders siblings by it; with none the builder
    cannot say which child is latest, logs a warning per drawn box, and draws the branch only as far as
    the focus rather than on to its tip. That is a different picture from the one the app renders, so a run
    without timestamps times a shape nobody sees.
    """
    global _serial
    _serial += 1
    return {"message": {"role": role, "content": [{"type": "text", "text": text}]},
            "general_metadata": {"persona": None, "timestamp": _serial}}


def make_forest(n_chats, depth_per_chat, head_depth):
    forest = Forest()
    root = forest.create_node(payload("system", "the card"), parent_id=None)
    greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
    head = None
    for c in range(n_chats):
        node = forest.create_node(payload("user", f"chat {c} opening message"), parent_id=greeting)
        for d in range(depth_per_chat):
            node = forest.create_node(payload("assistant" if d % 2 == 0 else "user",
                                              f"message {d} of a fairly ordinary length, as these go"),
                                      parent_id=node)
            if c == n_chats // 2 and d == head_depth:
                head = node
    return forest, head


def timeit(forest, head, each_side, measure_text, repeats=20):
    config = chatgraph.LayoutConfig(siblings_each_side=each_side)
    state = chatgraph.ViewState(head_node_id=head)
    built = chatgraph.build(forest, state, config, measure_text=measure_text)
    t0 = time.perf_counter()
    for _ in range(repeats):
        chatgraph.build(forest, state, config, measure_text=measure_text)
    return (time.perf_counter() - t0) / repeats * 1e3, len(built.graph.nodes)


logging.getLogger('raven.librarian.chatgraph').setLevel(logging.ERROR)

dpg.create_context()
dpg.create_viewport(title="chatgraph wrap cost", width=200, height=120, x_pos=40, y_pos=40)
dpg.setup_dearpygui()
themes_and_fonts = guiutils.bootup(font_size=20)
fonts = [(size, guiutils.load_extra_font(themes_and_fonts, size, "OpenSans", "Regular")[1])
         for size in FONT_SIZES]
dpg.show_viewport()
for _ in range(5):  # the atlas is built by rendering; before this `get_text_size` returns nothing
    dpg.render_dearpygui_frame()

calls = [0]


def measure_text(text, font_size):
    """What `DPGChatGraphPanel._measure_text` does, plus a counter."""
    calls[0] += 1
    atlas_size, font_id = min(fonts, key=lambda pair: abs(pair[0] - font_size))
    measured = dpg.get_text_size(text, font=font_id)
    if not measured:
        return None
    return measured[0] * (font_size / atlas_size)


assert measure_text("probe", 20.0) is not None, \
    "the atlas is not up, so this would time the fallback twice and report no difference"

print("Budget: one frame at 60 fps is 16.7 ms. Same forests and placement as `measure_rebuild.py`.\n")
print(f"{'forest':>22} | {'each_side':>9} | {'estimated':>9} | {'measured':>8} | {'boxes':>5} | {'calls':>6}")
for n_chats, depth in [(50, 10), (200, 20), (1000, 20)]:
    forest, head = make_forest(n_chats, depth, head_depth=4)
    for each_side in (2, 5, 20):
        estimated_ms, boxes = timeit(forest, head, each_side, None)
        calls[0] = 0
        measured_ms, _ = timeit(forest, head, each_side, measure_text)
        per_build = calls[0] // 21  # the warm-up build plus the timed repeats
        label = f"{n_chats} chats, {len(forest.nodes)} nodes" if each_side == 2 else ""
        print(f"{label:>22} | {each_side:>9} | {estimated_ms:>8.2f} | {measured_ms:>7.2f} | "
              f"{boxes:>5} | {per_build:>6}")
    print()

dpg.destroy_context()
