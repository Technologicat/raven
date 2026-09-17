"""Entity offsets from `parser.parse`, which everything downstream slices the returned text with.

Pure parsing: no DPG context needed.
"""

import pytest

pytest.importorskip("mistletoe", reason="mistletoe not installed")

from raven.vendor.DearPyGui_Markdown import parser  # noqa: E402 -- after importorskip by design


def styled_spans(source: str, entity_type: type) -> list[str]:
    text, entities = parser.parse(source)
    return [text[e.offset:e.offset + e.length] for e in entities if type(e) is entity_type]


@pytest.mark.parametrize("prefix", ["", "😀 ", "😀😀 ", "🦜 ä ", "𝔸𝔹ℂ "])
def test_a_styled_span_covers_its_own_text_after_characters_outside_the_bmp(prefix):
    assert styled_spans(f"{prefix}plain **bold** end", parser.MessageEntityBold) == ["bold"]


def test_an_emoji_inside_the_span_is_covered_whole():
    assert styled_spans("x **a😀b** y 😀 *it*", parser.MessageEntityBold) == ["a😀b"]
    assert styled_spans("x **a😀b** y 😀 *it*", parser.MessageEntityItalic) == ["it"]


def test_a_link_after_an_emoji_covers_the_link_text():
    assert styled_spans("🔗 see [the docs](https://example.org) now", parser.MessageEntityTextUrl) == ["the docs"]
