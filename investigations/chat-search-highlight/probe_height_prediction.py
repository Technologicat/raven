"""Can a paragraph's rendered height be known before it is laid out?

A search swap that changes a paragraph's height above the viewport has to correct the scroll a frame
*before* the swap (see `probe_highlight.py` and this bundle's README), so it needs the height change before
the replacement is on screen — where a hidden build measures `[0, 0]`.

The renderer settles line breaks itself, before layout: `wrap_text_entity` returns the lines, and
`LineEntity.render` gives each one a row as tall as `line.get_height()`. So the candidate prediction is the
sum of those heights. This compares it against the laid-out height, for real chat paragraphs plain and
highlighted, and reports every mismatch by the kind of line it was (heading, list item, quote, other).

    python investigations/chat-search-highlight/probe_height_prediction.py [--datastore PATH] [--n-paragraphs N]

Uses `probe_highlight.py`'s highlight prototype. **Maps a window, so it takes keyboard focus while it runs.**
Prints only numbers; do not pipe it.
"""

import argparse
import collections
import os
import pathlib
import random
import sys
import threading
import time
import traceback

import dearpygui.dearpygui as dpg

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import probe_highlight as shared  # noqa: E402 -- the shared instrument; importing it installs the highlight prototype

from raven.common.gui import utils as guiutils  # noqa: E402
from raven.vendor import DearPyGui_Markdown as dpg_markdown  # noqa: E402

BATCH = 12  # paragraphs built per frame wait; items clipped out of the bench are still laid out (see README)


def kind(text):
    stripped = text.lstrip()
    if stripped.startswith("#"):
        return "heading"
    if stripped[:2] in ("- ", "* ", "+ ") or (stripped[:1].isdigit() and ". " in stripped[:5]):
        return "list"
    if stripped.startswith(">"):
        return "quote"
    if stripped.startswith("```") or stripped.startswith("|") or set(stripped) <= set("-*_ "):
        return "fence/table/rule"
    return "other"


ROW_OFFSET_PX = 6  # measured: every laid-out row came out this much taller than `line.get_height()`


def predict(text, variant, fragment):
    """Sum of the wrapped lines' row heights, computed the way `MarkdownText.add` wraps."""
    shared.set_variant(variant, fragment)
    try:
        markdown = dpg_markdown.MarkdownText(text, color=shared.TEXT_COLOR)
        t0 = time.perf_counter()
        lines = dpg_markdown.wrap_text_entity(markdown.text_entity, width=shared.WRAP_W)
        t_wrap = time.perf_counter() - t0
        return sum(line.get_height() for line in lines), len(lines), t_wrap
    finally:
        shared.set_variant("plain", None)


def run(paragraphs):
    jobs = []  # (paragraph index, variant, fragment)
    for index, text in enumerate(paragraphs):
        jobs.append((index, "plain", None))
        jobs.append((index, "control", None))
        for fragment in shared.FRAGMENTS:
            if fragment in text.lower():
                jobs.append((index, "color", fragment))
                jobs.append((index, "bold", fragment))

    exact = collections.Counter()
    total = collections.Counter()
    mismatches = collections.defaultdict(list)  # (variant, kind) -> [(measured - predicted)]
    delta_exact = collections.Counter()
    delta_total = collections.Counter()
    wrap_times = []
    control_disagreements = []
    residuals_per_line = collections.defaultdict(collections.Counter)
    delta_misses = collections.defaultdict(list)
    delta_nonzero = collections.Counter()
    heading_font_checks = collections.Counter()  # True: red run shares a font with the rest of its heading
    measured_plain = {}
    predicted_plain = {}  # already corrected by the per-row offset
    for start in range(0, len(jobs), BATCH):
        batch = jobs[start:start + BATCH]
        built = []
        for index, variant, fragment in batch:
            text = paragraphs[index]
            build_variant = "plain" if variant == "control" else variant
            predicted, n_lines, t_wrap = predict(text, build_variant, fragment)
            wrap_times.append(t_wrap)
            group, _, _ = shared.build(text, "bench", build_variant, fragment)
            built.append((index, variant, fragment, group, predicted, n_lines))
        shared.wait_frames(3, "lay out the batch")
        for index, variant, fragment, group, predicted, n_lines in built:
            measured = dpg.get_item_rect_size(group)[1]
            label = variant if fragment is None else f"{variant} '{fragment}'"
            total[label] += 1
            residual = round(measured - predicted)
            residuals_per_line[label][residual / n_lines if n_lines else None] += 1
            corrected = predicted + ROW_OFFSET_PX * n_lines
            if round(measured) == round(corrected):
                exact[label] += 1
            else:
                mismatches[(label, kind(paragraphs[index]))].append(round(measured - corrected))
            if variant == "color" and kind(paragraphs[index]) == "heading":
                # A highlight is a `Font` entity with no size. If that resets the run to body size, the red
                # run's font will be one no unhighlighted run in the same heading uses.
                red_fonts, other_fonts = shared.run_fonts(group)
                heading_font_checks[not red_fonts or red_fonts <= other_fonts] += 1
            if variant == "plain":
                measured_plain[index] = measured
                predicted_plain[index] = corrected
            elif variant == "control":
                # The same paragraph built plain a second time, possibly further down the batch. If the two
                # disagree, the measurement itself is unreliable (clipping, an unsettled layout), and so is
                # everything else this prints.
                if round(measured) != round(measured_plain[index]):
                    control_disagreements.append(round(measured - measured_plain[index]))
            elif index in measured_plain:
                delta_total[label] += 1
                measured_delta = round(measured - measured_plain[index])
                if measured_delta == round(corrected - predicted_plain[index]):
                    delta_exact[label] += 1
                else:
                    delta_misses[(label, kind(paragraphs[index]))].append(measured_delta)
                if measured_delta != 0:
                    delta_nonzero[label] += 1
            dpg.delete_item(group)

    print(f"\nnegative control: plain built twice disagreed {len(control_disagreements)} times "
          f"of {total['control']} {control_disagreements[:10]}")
    print("\nuncorrected residual per line (measured minus sum of line heights, divided by line count):")
    for label, counter in residuals_per_line.items():
        print(f"  {label:12s} {dict(sorted(counter.items(), key=repr))}")
    print(f"\nabsolute height, predicted from the wrap plus {ROW_OFFSET_PX} px per row, vs laid out:")
    for label in total:
        print(f"  {label:12s} exact {exact[label]:4d} of {total[label]:4d}")
    print("\nheight *change* against plain, predicted vs laid out (and how many changes were non-zero):")
    for label in delta_total:
        print(f"  {label:12s} exact {delta_exact[label]:4d} of {delta_total[label]:4d}   "
              f"non-zero measured {delta_nonzero[label]:4d}")
    print("\nabsolute mismatches after the correction, measured minus predicted, px, by line kind:")
    for (label, line_kind), diffs in sorted(mismatches.items()):
        print(f"  {label:12s} {line_kind:16s} n={len(diffs):3d}  {dict(sorted(collections.Counter(diffs).items()))}")
    print("\nheight-change misses, the measured change, px, by line kind:")
    for (label, line_kind), diffs in sorted(delta_misses.items()):
        print(f"  {label:12s} {line_kind:16s} n={len(diffs):3d}  {dict(sorted(collections.Counter(diffs).items()))}")
    print(f"\nhighlighted headings (colour): red run shares its heading's font {heading_font_checks[True]}, "
          f"does not {heading_font_checks[False]}")
    kinds = collections.Counter(kind(p) for p in paragraphs)
    print(f"\nline kinds in the sample: {dict(kinds)}")
    print(f"wrap alone (the part a prediction repeats): {shared.summarize_ms(wrap_times)}")


def main():
    from raven.librarian import config as librarian_config

    argparser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    argparser.add_argument("--datastore", default=str(librarian_config.llm_datastore_file))
    argparser.add_argument("--n-paragraphs", type=int, default=400)
    args = argparser.parse_args()

    all_paragraphs = shared.load_paragraphs(args.datastore)
    rng = random.Random(1)
    sample = rng.sample(all_paragraphs, min(args.n_paragraphs, len(all_paragraphs)))
    # Every structured line in the datastore too, since a random sample holds few and they are the likely misses.
    structured = [p for p in all_paragraphs if kind(p) != "other" and p not in sample]
    paragraphs = sample + structured
    print(f"paragraphs: {len(sample)} sampled + {len(structured)} structured")

    dpg.create_context()
    guiutils.bootup(font_size=20)
    dpg.create_viewport(title="probe: height prediction", width=1000, height=900)
    dpg.setup_dearpygui()
    with dpg.window(tag="main", width=990, height=890, no_title_bar=True, no_move=True):
        dpg.add_child_window(tag="bench", width=-1, height=-1)
    dpg.show_viewport()

    def driver():
        try:
            shared.wait_frames(3, "startup")
            dpg_markdown.add_text("warm-up", parent="bench")
            while not dpg_markdown.CallWhenDPGStarted.STARTUP_DONE:
                shared.wait_frames(1, "markdown startup")
            dpg.delete_item("bench", children_only=True)
            run(paragraphs)
        except Exception:
            traceback.print_exc()
        finally:
            print("\ndone.", flush=True)
            os._exit(0)

    threading.Thread(target=driver, daemon=True).start()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    os._exit(0)


if __name__ == "__main__":
    main()
