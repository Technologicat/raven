"""`MarkdownText(..., highlight=...)`: search matches marked over the text as displayed.

Nothing here maps a window or renders a frame. What a run *looks like* is read from its attributes, which is
what rendering reads too; fonts are not loaded, so font identities are not checked, only which font
attributes a run carries.
"""

import re

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.vendor import DearPyGui_Markdown as dpg_markdown  # noqa: E402 -- after importorskip by design
from raven.vendor.DearPyGui_Markdown import font_attributes  # noqa: E402 -- after importorskip by design
from raven.vendor.DearPyGui_Markdown import text_attributes  # noqa: E402 -- after importorskip by design
from raven.vendor.DearPyGui_Markdown import text_entities  # noqa: E402 -- after importorskip by design

RED = [255, 0, 0, 255]
DOCUMENT = [200, 200, 200, 255]


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


def runs(entity) -> list:
    """Every non-empty text run in a built entity tree, in document order."""
    if isinstance(entity, text_entities.StrEntity):
        return [entity] if str(entity) else []
    return [run for item in entity for run in runs(item)]


def highlighted(source, *patterns, bold=True):
    """`(text, is red, is bold)` per run."""
    markdown = dpg_markdown.MarkdownText(source, color=DOCUMENT,
                                         highlight=[re.compile(p, re.IGNORECASE) if isinstance(p, str) else p
                                                    for p in patterns],
                                         highlight_bold=bold)
    result = []
    for run in runs(markdown.text_entity):
        attributes = run.attributes
        is_bold = any(a in attributes for a in (font_attributes.Bold, font_attributes.BoldItalic))
        result.append((str(run), attributes.get_color() == RED, is_bold))
    return result


def red_text(source, *patterns, **kwargs):
    return "|".join(text for text, is_red, _ in highlighted(source, *patterns, **kwargs) if is_red)


def test_without_a_highlight_nothing_is_red(dpg_context):
    """The control for everything below: the document colour is not red to begin with."""
    assert red_text("find the needle here") == ""
    assert red_text("find the needle here", None) == ""


def test_a_match_is_red_and_bold_and_its_surroundings_are_not(dpg_context):
    runs_ = highlighted("find the needle here", "needle")
    assert ("needle", True, True) in runs_
    assert all(not is_red and not is_bold for text, is_red, is_bold in runs_ if text != "needle")


def test_highlight_bold_false_leaves_the_weight_alone(dpg_context):
    assert ("needle", True, False) in highlighted("find the needle here", "needle", bold=False)


def test_every_match_of_every_pattern_is_marked_and_none_entries_are_skipped(dpg_context):
    assert red_text("Ab ab cd AB", None, "ab", re.compile("cd")) == "Ab|ab|cd|AB"


def test_a_match_spanning_styling_is_matched_on_the_displayed_text(dpg_context):
    """`**las**er` displays as "laser", so searching "laser" finds it."""
    assert red_text("a **las**er beam", "laser").replace("|", "") == "laser"


def test_markup_and_link_targets_are_not_searched(dpg_context):
    assert red_text("see [the docs](https://example.org/needle) now", "needle") == ""
    assert red_text("**bold** text", r"\*\*") == ""


def test_a_match_wins_over_a_link_colour(dpg_context):
    runs_ = highlighted("see [the needle docs](https://example.org) now", "needle")
    assert ("needle", True, True) in runs_
    link_runs = [text for text, is_red, _ in runs_ if text.strip() in ("the", "docs")]
    assert link_runs, "the fixture has no unhighlighted link text to compare against"
    assert all(not is_red for text, is_red, _ in runs_ if text.strip() in ("the", "docs"))


def test_a_match_inside_a_heading_keeps_the_heading(dpg_context):
    """A highlight is colour only. A font span would carry a size, and an unset size is body size."""
    markdown = dpg_markdown.MarkdownText("### a needle heading", highlight=[re.compile("needle")])
    needle = next(run for run in runs(markdown.text_entity) if str(run) == "needle")
    assert font_attributes.H3 in needle.attributes
    assert font_attributes.Font not in needle.attributes


def test_a_match_after_an_emoji_lands_on_its_own_text(dpg_context):
    assert red_text("😀 🦜 find the needle", "needle") == "needle"


def test_a_highlighted_link_run_keeps_its_colour_when_the_link_renders(dpg_context):
    """`Url.render` recolours its runs on the worker and again on hover; a highlighted run must be left out."""
    with dpg.window():
        plain_run = dpg.add_text("the", color=DOCUMENT)
        matched_run = dpg.add_text("needle", color=RED)
    url = text_attributes.Url("https://example.org", attribute_connector=None)
    url.render(plain_run)
    url.render(matched_run, highlighted=True)
    colors = {item: [round(c * 255) if c <= 1.0 else round(c) for c in dpg.get_item_configuration(item)["color"]]
              for item in (plain_run, matched_run)}
    assert colors[plain_run] == list(text_attributes.Url.color), "the control: an unhighlighted run takes the link colour"
    assert colors[matched_run] == RED
    assert matched_run not in url.dpg_text_objects
    assert plain_run in url.dpg_text_objects


def test_before_inserts_among_the_parent_s_children(dpg_context, monkeypatch):
    # Before a render loop has run, `add` queues the render with `CallWhenDPGStarted`, whose first use starts
    # a worker thread polling DPG — which outlives this module's context and segfaults the process when the
    # context goes. Placement is decided before that queue is reached, so the queue is stubbed out.
    monkeypatch.setattr(dpg_markdown.CallWhenDPGStarted, "append", classmethod(lambda cls, *args, **kwargs: None))
    with dpg.window() as window:
        first = dpg.add_text("first")
        last = dpg.add_text("last")
    group = dpg_markdown.add_text("middle", parent=window, before=last)
    assert dpg.get_item_children(window, 1) == [first, group, last]
