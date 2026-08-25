"""The renderer's document colour, and the block-construct barrier it exists to remove.

Colour used to reach the renderer as a `<font color=...>` tag wrapped around the source by the caller.
That works for inline styling and silently destroys every block construct: an open tag on the same line
as the content makes the whole thing one paragraph as far as CommonMark is concerned, and a heading
cannot occur inside a paragraph. `MarkdownText(..., color=...)` carries it out of band instead.

Nothing here maps a window or renders a frame — colour resolution and parsing need neither.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.vendor import DearPyGui_Markdown as dpg_markdown  # noqa: E402 -- after importorskip by design
from raven.vendor.DearPyGui_Markdown import parser  # noqa: E402 -- after importorskip by design
from raven.vendor.DearPyGui_Markdown import text_entities  # noqa: E402 -- after importorskip by design

SOURCE = "### A heading\nplain text"
ORANGE = [255, 136, 0, 255]
DEFAULT_WHITE = [255, 255, 255, 255]


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport.

    `AttributeController` builds a theme when the first one is constructed, so a context has to exist
    before any `MarkdownText`. Module-scoped, which is the house pattern for every DPG test here.
    """
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


def runs(entity) -> list:
    """Every non-empty text run in a built entity tree, in document order.

    Dispatched on `StrEntity` rather than on carrying an `attributes` member: the containers have one
    too, so a `hasattr` test matches the whole tree and never descends — which reads back exactly the
    colour the caller set and asserts nothing.
    """
    if isinstance(entity, text_entities.StrEntity):
        return [entity] if str(entity) else []
    return [run for item in entity for run in runs(item)]


def test_a_heading_parses_as_a_heading():
    entities = {type(entity).__name__ for entity in parser.parse(SOURCE)[1]}
    assert "MessageEntityH3" in entities


def test_wrapping_the_source_in_a_font_tag_destroys_the_heading():
    """The negative control, and the whole reason the `color` argument exists.

    Without this, `test_a_heading_parses_as_a_heading` above passes just as well against the code that
    shipped the bug — it says a heading parses, not that Raven's own call site lets one through. This is
    what the caller used to hand over.
    """
    clear_text, attributes = parser.parse(f"<font color='#ff8800'>{SOURCE}</font>")
    entities = {type(entity).__name__ for entity in attributes}
    assert "MessageEntityH3" not in entities
    assert clear_text.startswith("### "), ("the markers should survive verbatim into the text, which is "
                                           f"what the reader saw on screen: {clear_text!r}")


def test_document_color_reaches_text_the_markdown_does_not_color(dpg_context):
    colors = [run.attributes.get_color() for run in runs(dpg_markdown.MarkdownText(SOURCE, color="#ff8800").text_entity)]
    assert colors, "no text runs at all, so this asserts nothing about their colour"
    assert all(color == ORANGE for color in colors), colors


def test_document_color_survives_the_attribute_rebuild_that_wrapping_triggers(dpg_context):
    """`LineEntity.append` calls `recreate_attributes` on every line it takes, so the whole wrapped
    path (`wrap >= 0`, which is every chat message) goes through a controller built from scratch.

    Without the carry-over this asserts, the colour survives an unwrapped render and is lost by a
    wrapped one — a split that would show up as "works in the help card, not in the chat log".
    """
    built = dpg_markdown.MarkdownText(SOURCE, color="#ff8800")
    for run in runs(built.text_entity):
        run.recreate_attributes()
    colors = [run.attributes.get_color() for run in runs(built.text_entity)]
    assert colors, "no text runs at all, so this asserts nothing about their colour"
    assert all(color == ORANGE for color in colors), colors


def test_omitting_the_color_leaves_the_renderer_default_alone(dpg_context):
    """Callers that pass no colour must be unaffected, the argument being optional."""
    colors = [run.attributes.get_color() for run in runs(dpg_markdown.MarkdownText(SOURCE).text_entity)]
    assert colors, "no text runs at all, so this asserts nothing about their colour"
    assert all(color == DEFAULT_WHITE for color in colors), colors


def test_an_empty_prefix_does_not_impose_its_own_default_on_the_run_it_joins(dpg_context):
    """`wrap_text_entity` seeds every paragraph with an empty `StrEntity` and adds words onto it, so this
    merge is on the path of every wrapped render — which is every chat message.

    The seed is built outside any document and carries the class-default colour. Attribute *lists* are
    what the merge compares, and two controllers differing only in their default compare equal, so the
    result used to take the seed's colour whenever the word carried no attributes of its own. Plain
    paragraphs came out in the default while headings and list items kept the document's, which is what
    it looked like on screen.
    """
    built = dpg_markdown.MarkdownText("plain text here", color="#ff8800")
    run = runs(built.text_entity)[0]
    assert run.attributes.get_color() == ORANGE, "the run itself should carry the document colour"

    seed = text_entities.StrEntity("")
    assert seed.attributes.get_color() == DEFAULT_WHITE, ("the seed should carry the class default, or this "
                                                          "fixture cannot tell the two colours apart")
    assert (seed + run).attributes.get_color() == ORANGE


def test_a_font_span_still_beats_the_document_color(dpg_context):
    """Out-of-band colour is a default, not an override: markup in the source still wins."""
    built = dpg_markdown.MarkdownText("plain <font color='(0, 255, 0)'>green</font>", color="#ff8800")
    colors = [run.attributes.get_color() for run in runs(built.text_entity)]
    assert ORANGE in colors, f"the unstyled run should take the document colour: {colors}"
    assert [0, 255, 0, 255] in colors, f"the `<font>` span should keep its own: {colors}"
