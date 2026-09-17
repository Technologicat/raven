"""What in-place search highlighting in the chat view costs, and how a re-rendered paragraph can be swapped in calmly.

Three questions, one mapped window:

  1. **Cost.** How long `dpg_markdown` takes per real chat paragraph — parse and build separately — plain,
     highlighted in colour, and highlighted in bold colour, for a one-letter and a three-letter fragment.
  2. **Size.** Whether a highlight changes a paragraph's rendered size, in colour and in bold, against a
     negative control that builds the same paragraph plain twice. This decides whether off-screen
     paragraphs can be updated without scroll compensation.
  3. **Swap.** A paragraph in a scrolled view is replaced by its highlighted version, several ways, while a
     per-frame recorder on the render thread watches a neighbouring paragraph's screen position and the
     scroll offset. A transient position — one that is neither the baseline nor where things settle — is a
     visible jump. The same cases report whether the replacement's decorations (an inline-code background,
     a link underline) came out sized and placed like those of a paragraph built in the ordinary way.

The highlight is prototyped as the proposed renderer feature rather than as markup in the source: `parser.parse`
is wrapped so that, after parsing, a red `MessageEntityFont` (and a `MessageEntityBold`, for the bold variant)
is appended over each match in the *visible* text. Offsets are Python string offsets, which disagree with the
parser's own wherever the text holds a character outside the BMP — good enough for timing, and something the
real implementation has to get right.

    python investigations/chat-search-highlight/probe_highlight.py [--datastore PATH] [--n-paragraphs N]

Reads paragraphs from the Librarian chat datastore (default: the configured one) and prints only numbers and
lengths, never text. **Maps a window, so it takes keyboard focus while it runs.** Needs a display, prints to
stdout — do not pipe it — and leaves with `os._exit`, the renderer's worker thread not participating in DPG
teardown.
"""

import argparse
import collections
import json
import os
import pathlib
import random
import re
import statistics
import threading
import time
import traceback

import dearpygui.dearpygui as dpg

from raven.common.gui import utils as guiutils
from raven.vendor import DearPyGui_Markdown as dpg_markdown
from raven.vendor.DearPyGui_Markdown import parser as md_parser

WRAP_W = 850  # Librarian's chat text width at a ~1000 px panel: panel width minus `chat_text_right_margin_w`
TEXT_COLOR = "#e0e0e0"
RED = [255, 0, 0, 255]
FRAGMENTS = ("e", "the")

# The paragraph swapped in part 3. It carries both decorations whose placement is measured, and a match for
# the fragment below, twice.
TARGET_TEXT = ("To rebuild the search index, run `raven-indexer` on the documents directory, as "
               "[the indexer docs](https://example.org/docs) describe; the index is then searched on every turn.")
TARGET_FRAGMENT = "rebuild"  # must not occur inside the code or link runs, or it splits them and they cannot be found
CODE_RUN = "raven-indexer"
LINK_RUN = "the indexer docs"


# --------------------------------------------------------------------------------
# The highlight, prototyped as a post-parse entity injection

_highlight = {"regex": None, "bold": False}
_original_parse = md_parser.parse


def _parse_with_highlight(markdown_text):
    clear_text, entities = _original_parse(markdown_text)
    maybe_regex = _highlight["regex"]
    if maybe_regex is not None and clear_text:
        for match in maybe_regex.finditer(clear_text):
            start, end = match.span()
            if end <= start:
                continue
            entities.append(md_parser.MessageEntityFont(offset=start, length=end - start, color=list(RED)))
            if _highlight["bold"]:
                entities.append(md_parser.MessageEntityBold(offset=start, length=end - start))
    return clear_text, entities


md_parser.parse = _parse_with_highlight  # `MarkdownText` looks it up on the module at call time


def set_variant(variant, fragment):
    """`variant`: "plain", "color" or "bold"."""
    if variant == "plain":
        _highlight["regex"] = None
    else:
        _highlight["regex"] = re.compile(re.escape(fragment), re.IGNORECASE)
    _highlight["bold"] = (variant == "bold")


def build(text, parent, variant, fragment):
    """Render `text` into `parent` under a highlight variant. Returns `(group, t_parse_s, t_add_s)`."""
    set_variant(variant, fragment)
    t0 = time.perf_counter()
    markdown = dpg_markdown.MarkdownText(text, color=TEXT_COLOR)
    t1 = time.perf_counter()
    group = markdown.add(wrap=WRAP_W, parent=parent)
    t2 = time.perf_counter()
    set_variant("plain", None)
    return group, t1 - t0, t2 - t1


# --------------------------------------------------------------------------------
# Render-thread plumbing: a queue drained between frames, and per-frame recorders

_render_thread_queue = collections.deque()
_frame_hooks = []


def run_on_render_thread(fn):
    """Run `fn` on the render thread, between two frames, and wait for it."""
    done = threading.Event()
    box = {}

    def wrapper():
        try:
            box["result"] = fn()
        except Exception as exc:  # noqa: BLE001 -- reported to the waiting thread
            box["exc"] = exc
        finally:
            done.set()
    _render_thread_queue.append(wrapper)
    done.wait()
    if "exc" in box:
        raise box["exc"]
    return box.get("result")


def wait_frames(n, why):
    for _ in range(n):
        guiutils.split_frame(operation=f"probe: {why}")


class Recorder:
    """Sample some readings after every frame, on the render thread."""
    def __init__(self, readings):
        self.readings = readings  # {name: zero-argument callable}
        self.rows = []

    def __call__(self):
        row = {}
        for name, read in self.readings.items():
            try:
                row[name] = read()
            except Exception as exc:  # noqa: BLE001 -- an item deleted mid-recording is a reading too
                row[name] = f"{type(exc).__name__}"
        self.rows.append(row)

    def start(self):
        _frame_hooks.append(self)

    def stop(self):
        _frame_hooks.remove(self)


# --------------------------------------------------------------------------------
# Widget-tree inspection

def descendants(item):
    out = []
    for slot in dpg.get_item_children(item).values():
        for child in slot:
            out.append(child)
            out.extend(descendants(child))
    return out


def item_type(item):
    return dpg.get_item_type(item).rsplit(":", 1)[-1]


def is_red(color):
    if not color or len(color) < 3:
        return False
    scale = 1.0 if max(color[:3]) <= 1.0 else 255.0
    r, g, b = (c / scale for c in color[:3])
    return r > 0.9 and g < 0.1 and b < 0.1


def red_runs(group):
    """The text runs inside `group` that are drawn red."""
    return [dpg.get_value(item) for item in descendants(group)
            if item_type(item) == "mvText" and is_red(dpg.get_item_configuration(item).get("color"))]


def run_fonts(group):
    """`(fonts of red runs, fonts of other runs)`, as sets of DPG font ids."""
    red, other = set(), set()
    for item in descendants(group):
        if item_type(item) != "mvText":
            continue
        font = dpg.get_item_font(item)
        (red if is_red(dpg.get_item_configuration(item).get("color")) else other).add(font)
    return red, other


def decoration_geometry(group):
    """Where the code background and the link underline sit, relative to the text runs they decorate.

    Returns `{"code": (dx, dy, w, h, text_w, text_h) | None, "underline": (dx, dy, w, h, text_w, text_h) | None}`,
    offsets in pixels from the decorated run's text group. The code quad is a drawlist inserted into the run's
    own text group; the underline is a drawlist in the paragraph's attributes group.
    """
    items = descendants(group)
    texts = {dpg.get_value(item): item for item in items if item_type(item) == "mvText"}
    drawlists = [item for item in items if item_type(item) == "mvDrawlist"]
    result = {"code": None, "underline": None}
    for key, run in (("code", CODE_RUN), ("underline", LINK_RUN)):
        maybe_text = texts.get(run)
        if maybe_text is None:
            continue
        text_group = dpg.get_item_parent(maybe_text)
        tx, ty = dpg.get_item_rect_min(text_group)
        tw, th = dpg.get_item_rect_size(text_group)
        best = None
        for drawlist in drawlists:
            inside = dpg.get_item_parent(dpg.get_item_parent(drawlist)) == text_group
            if (key == "code") != inside:
                continue
            dx_, dy_ = dpg.get_item_rect_min(drawlist)
            dw, dh = dpg.get_item_rect_size(drawlist)
            candidate = (round(dx_ - tx), round(dy_ - ty), round(dw), round(dh), round(tw), round(th))
            if best is None or abs(candidate[0]) + abs(candidate[1]) < abs(best[0]) + abs(best[1]):
                best = candidate
        result[key] = best
    return result


# --------------------------------------------------------------------------------
# Data

def load_paragraphs(datastore_path):
    """Every non-blank line of every text part, active revisions only — what `_render_text` renders one widget per."""
    data = json.loads(pathlib.Path(datastore_path).read_text(encoding="utf-8"))
    paragraphs = []
    for node in data.values():
        payload = node["data"][str(node["active_revision"])]
        content = payload.get("message", {}).get("content") or []
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for part in content:
            if part.get("type") != "text":
                continue
            for line in part.get("text", "").split("\n"):
                if line.strip():
                    paragraphs.append(line.strip())
    return paragraphs


def summarize_ms(values):
    values = sorted(v * 1000 for v in values)
    if not values:
        return "n/a"
    p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
    return (f"median {statistics.median(values):6.2f}  p90 {p90:6.2f}  max {values[-1]:7.2f}  "
            f"sum {sum(values):8.1f} ms")


# --------------------------------------------------------------------------------
# Part 1 and 2: cost and height

def part_cost_and_height(paragraphs):
    print("\n=== Part 1: cost per paragraph, and Part 2: height stability ===")
    lengths = sorted(len(p) for p in paragraphs)
    print(f"paragraphs sampled: {len(paragraphs)}; length median {statistics.median(lengths)}, "
          f"p90 {lengths[int(0.9 * len(lengths))]}, max {lengths[-1]} characters")

    # Warm the font atlas for every face the variants use, so first-use waits are not timed.
    for text in paragraphs[:5]:
        for variant in ("plain", "color", "bold"):
            group, _, _ = build(text, "bench", variant, "e")
            wait_frames(2, "warm-up")
            dpg.delete_item(group)

    timings = collections.defaultdict(lambda: {"parse": [], "add": []})
    height_changes = collections.Counter()
    width_changes = collections.Counter()
    matched = collections.Counter()
    red_ok = collections.Counter()
    bold_rewrap_examples = []

    width_deltas = collections.defaultdict(list)
    fonts_seen = collections.defaultdict(lambda: [set(), set()])

    def measure(text, variant, fragment):
        """Build alone at the top of the bench, so nothing is clipped when it is measured."""
        group, t_parse, t_add = build(text, "bench", variant, fragment)
        wait_frames(2, "lay out the bench paragraph")
        size = [round(v) for v in dpg.get_item_rect_size(group)]
        n_red = len(red_runs(group))
        red_fonts, other_fonts = run_fonts(group)
        fonts_seen[variant][0].update(red_fonts)
        fonts_seen[variant][1].update(other_fonts)
        dpg.delete_item(group)
        return size, t_parse, t_add, n_red

    for index, text in enumerate(paragraphs):
        plain_size, t_parse, t_add, _ = measure(text, "plain", None)
        timings["plain"]["parse"].append(t_parse)
        timings["plain"]["add"].append(t_add)
        # Negative control: the same paragraph built plain a second time must measure identically,
        # or the size comparisons below say nothing.
        control_size, _, _, _ = measure(text, "plain", None)
        if control_size[1] != plain_size[1]:
            height_changes["control"] += 1
        if control_size[0] != plain_size[0]:
            width_changes["control"] += 1
        for fragment in FRAGMENTS:
            if fragment.lower() not in text.lower():
                continue
            matched[fragment] += 1
            for variant in ("color", "bold"):
                label = f"{variant} '{fragment}'"
                size, t_parse, t_add, n_red = measure(text, variant, fragment)
                timings[label]["parse"].append(t_parse)
                timings[label]["add"].append(t_add)
                if size[1] != plain_size[1]:
                    height_changes[label] += 1
                    if (variant, fragment) == ("bold", "e") and len(bold_rewrap_examples) < 3:
                        bold_rewrap_examples.append(index)
                if size[0] != plain_size[0]:
                    width_changes[label] += 1
                if size[1] == plain_size[1]:  # same line count, so the widths are comparable
                    width_deltas[label].append(size[0] - plain_size[0])
                if n_red:
                    red_ok[label] += 1

    print("\ntiming (per paragraph):")
    for label, parts in timings.items():
        print(f"  {label:12s} parse  {summarize_ms(parts['parse'])}")
        print(f"  {'':12s} build  {summarize_ms(parts['add'])}")
    print("\nheight / width changes against plain, and whether any red run was produced:")
    print(f"  control      built {len(paragraphs):4d}   height changed {height_changes['control']:4d}   "
          f"width changed {width_changes['control']:4d}")
    for fragment in FRAGMENTS:
        for variant in ("color", "bold"):
            label = f"{variant} '{fragment}'"
            deltas = collections.Counter(width_deltas[label])
            print(f"  {label:12s} matched {matched[fragment]:4d}   height changed {height_changes[label]:4d}   "
                  f"width changed {width_changes[label]:4d}   red run present {red_ok[label]:4d}   "
                  f"width delta px (same height) {dict(sorted(deltas.items()))}")
    print("\nfonts bound (red runs / other runs):")
    for variant, (red, other) in fonts_seen.items():
        print(f"  {variant:6s} red {sorted(red, key=repr)}   other {sorted(other, key=repr)}")

    # Is the width growth rounding per text item? Compare measuring a string whole against measuring it in
    # the pieces a highlight on "e" would split it into. Plain text only, so the markup cannot interfere.
    whole_minus_split = collections.Counter()
    for text in paragraphs[:100]:
        sample = text[:120]
        pieces = [piece for piece in re.split(r"(e)", sample) if piece]
        if len(pieces) < 2:
            continue
        whole = dpg_markdown.get_text_size(sample)[0]
        split = sum(dpg_markdown.get_text_size(piece)[0] for piece in pieces)
        whole_minus_split[(round(split - whole), len(pieces) - 1)] += 1
    by_splits = collections.defaultdict(list)
    for (extra, n_splits), count in whole_minus_split.items():
        by_splits[n_splits].extend([extra] * count)
    ratios = [extra / n_splits for n_splits, extras in by_splits.items() for extra in extras]
    print(f"\nmeasuring text in pieces vs whole (default font, 100 samples): extra px per split, "
          f"median {statistics.median(ratios):.2f}, min {min(ratios):.2f}, max {max(ratios):.2f}")
    return [paragraphs[i] for i in bold_rewrap_examples]


# --------------------------------------------------------------------------------
# Part 3: swapping a paragraph in a scrolled view

def build_chat(fillers, target_text, target_index):
    """Fill the "chat" child window: fillers with the target paragraph at `target_index`. Returns the group ids."""
    dpg.delete_item("chat", children_only=True)
    groups = []
    texts = list(fillers)
    texts.insert(target_index, target_text)
    for text in texts:
        # Each paragraph sits in a wrapper group of its own, the same shape the swapped-in replacement has,
        # so that the swap does not change the item count and add an item spacing of its own.
        wrapper = dpg.add_group(parent="chat")
        build(text, wrapper, "plain", None)
        groups.append(wrapper)
    wait_frames(6, "lay out the chat and its decorations")
    return groups


def screen_y(item):
    return round(dpg.get_item_rect_min(item)[1])


def swap_case(name, *, strategy, variant, where, fillers, target_text, fragment, compensate=False, delay_frames=0):
    """Replace the target paragraph by its highlighted version and record what the view does.

    `strategy`:
        "bg"      -- build hidden in place, then show it and delete the old one, from this (background) thread.
        "rt"      -- build hidden in place here, then show + delete in one render-thread slot.
        "staged"  -- build visible in a one-pixel clipped staging child window, then `move_item` into place
                     and delete the old one in one render-thread slot.
    `where`: "visible" (target mid-viewport) or "above" (target scrolled out above the viewport).
    `delay_frames`: frames to wait between building and swapping, as a batch of several builds would.
    `compensate`: "staged" only -- adjust the scroll by the measured height change. `"same"` sets it in the
                  swap's own render-thread slot, `"before"` one frame earlier, `False` not at all.
    """
    target_index = 20 if where == "visible" else 3
    groups = build_chat(fillers, target_text, target_index)
    old = groups[target_index]
    content_top = screen_y(groups[0])
    target_y = screen_y(old) - content_top
    if where == "visible":
        y_scroll = max(0, target_y - 250)
    else:
        y_scroll = target_y + dpg.get_item_rect_size(old)[1] + 150
    dpg.set_y_scroll("chat", y_scroll)
    wait_frames(4, "settle the scroll")
    chat_top = guiutils.get_widget_pos("chat")[1]  # a child window has no `rect_min`
    # The watched neighbour: the first paragraph after the target whose top is on screen.
    watched = next((g for g in groups[target_index + 1:] if screen_y(g) >= chat_top), groups[-1])
    # And one before the target that is on screen, which nothing should ever move (visible case only).
    above = groups[target_index - 1] if where == "visible" else None

    readings = {"watched_y": lambda: screen_y(watched),
                "y_scroll": lambda: round(dpg.get_y_scroll("chat"))}
    if above is not None:
        readings["above_y"] = lambda: screen_y(above)
    recorder = Recorder(readings)
    recorder.start()
    wait_frames(3, "baseline")

    old_h = dpg.get_item_rect_size(old)[1]
    if strategy in ("bg", "rt"):
        holder = dpg.add_group(parent="chat", before=old, show=False)
        build(target_text, holder, variant, fragment)
        wait_frames(delay_frames, "batch delay") if delay_frames else None
        if strategy == "bg":
            dpg.show_item(holder)
            dpg.delete_item(old)
        else:
            def swap():
                dpg.show_item(holder)
                dpg.delete_item(old)
            run_on_render_thread(swap)
    elif strategy == "staged":
        holder = dpg.add_group(parent="staging")
        build(target_text, holder, variant, fragment)
        wait_frames(max(delay_frames, 4), "lay out in staging")
        new_h = dpg.get_item_rect_size(holder)[1]
        staged_code = decoration_geometry(holder)["code"]

        if compensate == "before":
            before_scroll = run_on_render_thread(lambda: dpg.get_y_scroll("chat"))
            run_on_render_thread(lambda: dpg.set_y_scroll("chat", before_scroll + (new_h - old_h)))

        def swap():
            before_scroll = dpg.get_y_scroll("chat")
            dpg.move_item(holder, parent="chat", before=old)
            dpg.delete_item(old)
            if compensate == "same":
                dpg.set_y_scroll("chat", before_scroll + (new_h - old_h))
        run_on_render_thread(swap)
    else:
        raise ValueError(strategy)

    wait_frames(20, "watch the swap")
    recorder.stop()
    geometry = decoration_geometry(holder)
    new_h = dpg.get_item_rect_size(holder)[1]

    baseline = recorder.rows[0]
    final = recorder.rows[-1]
    transients = collections.Counter()
    for key in readings:
        for row in recorder.rows:
            if row[key] not in (baseline[key], final[key]):
                transients[key] += 1
    trail = {key: [] for key in readings}
    for row in recorder.rows:
        for key in readings:
            if not trail[key] or trail[key][-1] != row[key]:
                trail[key].append(row[key])
    extra = f"  staged code (in staging) {staged_code}" if strategy == "staged" else ""
    print(f"\n[{name}] height {round(old_h)} -> {round(new_h)}; red runs {len(red_runs(holder))}")
    for key in readings:
        print(f"    {key:10s} sequence {trail[key]}   transient frames {transients[key]}")
    print(f"    decorations: code {geometry['code']}   underline {geometry['underline']}{extra}")
    return {"trail": trail, "transients": dict(transients), "geometry": geometry}


def part_swap(fillers, bold_rewrap_texts):
    print("\n=== Part 3: swapping a paragraph in a scrolled view ===")
    print("decorations are (dx, dy, w, h, text_w, text_h) relative to the decorated run's text group")

    # Reference: the target built visible, in the ordinary way.
    groups = build_chat(fillers, TARGET_TEXT, 20)
    wait_frames(4, "reference decorations")
    print(f"\n[reference, built visible] decorations {decoration_geometry(groups[20])}")

    common = dict(fillers=fillers, target_text=TARGET_TEXT, fragment=TARGET_FRAGMENT)
    for rep in range(3):
        swap_case(f"bg color visible #{rep}", strategy="bg", variant="color", where="visible", **common)
    for rep in range(3):
        swap_case(f"rt color visible #{rep}", strategy="rt", variant="color", where="visible", **common)
    swap_case("rt color visible, 3-frame batch delay", strategy="rt", variant="color", where="visible",
              delay_frames=3, **common)
    swap_case("staged color visible", strategy="staged", variant="color", where="visible", **common)
    swap_case("rt bold visible", strategy="rt", variant="bold", where="visible", **common)

    if bold_rewrap_texts:
        rewrap = dict(fillers=fillers, target_text=bold_rewrap_texts[0], fragment="e")
        swap_case("rt color above (re-wrapping paragraph)", strategy="rt", variant="color", where="above", **rewrap)
        swap_case("rt bold above, uncompensated", strategy="rt", variant="bold", where="above", **rewrap)
        swap_case("staged bold above, uncompensated", strategy="staged", variant="bold", where="above", **rewrap)
        swap_case("staged bold above, compensated in the swap's slot", strategy="staged", variant="bold",
                  where="above", compensate="same", **rewrap)
        swap_case("staged bold above, compensated one frame before", strategy="staged", variant="bold",
                  where="above", compensate="before", **rewrap)
    else:
        print("\n(no sampled paragraph re-wrapped under bold; the above-viewport cases were skipped)")


# --------------------------------------------------------------------------------

def main():
    from raven.librarian import config as librarian_config

    argparser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    argparser.add_argument("--datastore", default=str(librarian_config.llm_datastore_file))
    argparser.add_argument("--n-paragraphs", type=int, default=200)
    args = argparser.parse_args()

    all_paragraphs = load_paragraphs(args.datastore)
    total_chars = sum(len(p) for p in all_paragraphs)
    print(f"datastore: {len(all_paragraphs)} paragraphs, {total_chars} characters")
    for fragment in FRAGMENTS + (TARGET_FRAGMENT, "search"):
        n = sum(fragment in p.lower() for p in all_paragraphs)
        print(f"  paragraphs containing '{fragment}': {n} ({100 * n / max(1, len(all_paragraphs)):.0f}%)")
    rng = random.Random(0)
    paragraphs = rng.sample(all_paragraphs, min(args.n_paragraphs, len(all_paragraphs)))
    fillers = [p for p in paragraphs if len(p) < 600][:40]

    dpg.create_context()
    guiutils.bootup(font_size=20)
    dpg.create_viewport(title="probe: chat search highlight", width=1000, height=1000)
    dpg.setup_dearpygui()
    with dpg.window(tag="main", width=990, height=990, no_title_bar=True, no_move=True):
        dpg.add_child_window(tag="bench", width=-1, height=180)
        dpg.add_child_window(tag="chat", width=-1, height=700)
        dpg.add_child_window(tag="staging", width=-1, height=1, no_scrollbar=True)
    dpg.show_viewport()

    def driver():
        try:
            wait_frames(3, "startup")
            # `MarkdownText.add` renders synchronously only once this flag is up, and it goes up only after
            # something has been queued.
            dpg_markdown.add_text("warm-up", parent="bench")
            while not dpg_markdown.CallWhenDPGStarted.STARTUP_DONE:
                wait_frames(1, "markdown startup")
            dpg.delete_item("bench", children_only=True)
            bold_rewrap_texts = part_cost_and_height(paragraphs)
            part_swap(fillers, bold_rewrap_texts)
        except Exception:
            traceback.print_exc()
        finally:
            print("\ndone.", flush=True)
            os._exit(0)

    threading.Thread(target=driver, daemon=True).start()
    while dpg.is_dearpygui_running():
        while _render_thread_queue:
            _render_thread_queue.popleft()()
        dpg.render_dearpygui_frame()
        for hook in list(_frame_hooks):
            hook()
    os._exit(0)


if __name__ == "__main__":
    main()
