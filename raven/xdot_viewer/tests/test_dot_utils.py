"""Tests for _strip_xdot_layout_attrs."""

import shutil
import subprocess
from pathlib import Path

import pytest

from ..dot_utils import _strip_xdot_layout_attrs


# ---------------------------------------------------------------------------
# Test fixtures: DOT / xdot snippets
# ---------------------------------------------------------------------------

# Minimal xdot output: graph-level bb, node pos and draw attrs.
XDOT_SIMPLE = (
    'digraph G {\n'
    '    graph [bb="0,0,200,150"];\n'
    '    a [pos="50,100", label="hello", shape=ellipse,\n'
    '       _draw_="c 7 -#000000 e 50 100 27 18 ",\n'
    '       _ldraw_="F 14 11 -Times-Roman c 7 -#000000 T 50 100 0 5 1 -a "];\n'
    '    b [pos="150,100", label="world"];\n'
    '    a -> b [pos="77,100 95,100 113,100 123,100",\n'
    '           _draw_="c 7 -#000000 B 4 77 100 95 100 113 100 123 100 ",\n'
    '           _hdraw_="S 5 -solid c 7 -#000000 C 7 -#000000 P 3 123 104 131 100 123 96 "];\n'
    '}\n'
)

# Clean DOT with no layout attrs at all.
CLEAN_DOT = (
    'digraph G {\n'
    '    a [label="hello", shape=ellipse];\n'
    '    b [label="world"];\n'
    '    a -> b;\n'
    '}\n'
)

# All attrs inside brackets are layout attrs.
ALL_LAYOUT = (
    'digraph G {\n'
    '    graph [bb="0,0,200,150"];\n'
    '    a [pos="50,100", _draw_="c 7 -#000000 e 50 100 27 18 "];\n'
    '}\n'
)

# Mixed attrs: label and pos in the same bracket, comma-separated.
MIXED_COMMA = 'digraph G { a [label="hello", pos="50,100", shape=box]; }'

# Mixed attrs with semicolons.
MIXED_SEMI = 'digraph G { a [label="hello"; pos="50,100"; shape=box]; }'

# Quoted attr name: "pos" = "50,100"
QUOTED_NAME = 'digraph G { a ["pos"="50,100", label="ok"]; }'

# HTML attr value: label=<bold text>
HTML_VALUE = 'digraph G { a [_draw_=<some html content>, label="ok"]; }'

# Multiple bracket groups at different levels.
MULTI_BRACKET = (
    'digraph G {\n'
    '    graph [bb="0,0,200,150"];\n'
    '    node [pos="50,100", label="default"];\n'
    '    a -> b [pos="77,100 95,100", _draw_="B 4 77 100 95 100 113 100 123 100"];\n'
    '}\n'
)

# Node named "pos" outside brackets — must not be stripped.
NAME_OUTSIDE = 'digraph G { pos [label="position node"]; bb -> pos; }'

# Layout attr name appearing as a value — must not be stripped.
NAME_AS_VALUE = 'digraph G { a [label="pos"]; }'

# Partial name match — "position" should not be stripped.
PARTIAL_NAME = 'digraph G { a [position="top", _draw_extra="stuff"]; }'

# Trailing whitespace-only separator (no comma/semicolon).
WHITESPACE_SEP = 'digraph G { a [pos="50,100"  label="ok"]; }'

# Graph with comments (DotScanner SKIP tokens).
WITH_COMMENTS = (
    'digraph G {\n'
    '    // a comment\n'
    '    a [pos="50,100", /* inline */ label="ok"];\n'
    '}\n'
)

# lp attr on an edge.
LP_ATTR = 'digraph G { a -> b [lp="100,50", label="edge"]; }'

# _background attr on graph.
BACKGROUND_ATTR = 'digraph G { graph [_background="...", label="G"]; }'

# rects attr.
RECTS_ATTR = 'digraph G { a [rects="0,0,50,50", label="ok"]; }'

# _hdraw_ and _tdraw_ on an edge.
HEAD_TAIL_DRAW = (
    'digraph G { a -> b [_hdraw_="P 3 1 2 3 4 5 6",'
    ' _tdraw_="P 3 7 8 9 10 11 12", label="e"]; }'
)

# _hldraw_ and _tldraw_.
HL_TL_DRAW = 'digraph G { a -> b [_hldraw_="...", _tldraw_="...", label="e"]; }'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TESTDATA = Path(__file__).resolve().parent.parent / "testdata"


def _has_attr(dot_src: str, attr_name: str) -> bool:
    """Rough check: does `attr_name=` appear inside brackets in `dot_src`?

    Good enough for assertions — we're not re-parsing, just eyeballing.
    """
    import re
    # Match: attr_name (possibly quoted) followed by optional whitespace and =
    pattern = rf'(?:"{re.escape(attr_name)}"|{re.escape(attr_name)})\s*='
    return bool(re.search(pattern, dot_src))


# ---------------------------------------------------------------------------
# Tests: basic functionality
# ---------------------------------------------------------------------------

class TestBasicStripping:
    """Core behavior: strip layout attrs, preserve the rest."""

    def test_strips_known_layout_attrs(self):
        """pos, bb, _draw_, _ldraw_, _hdraw_ are removed."""
        result = _strip_xdot_layout_attrs(XDOT_SIMPLE)
        for attr in ("pos", "bb", "_draw_", "_ldraw_", "_hdraw_"):
            assert not _has_attr(result, attr), f"{attr} should be stripped"

    def test_preserves_non_layout_attrs(self):
        """label, shape survive untouched."""
        result = _strip_xdot_layout_attrs(XDOT_SIMPLE)
        assert _has_attr(result, "label")
        assert _has_attr(result, "shape")

    def test_identity_on_clean_dot(self):
        """Input without layout attrs is returned unchanged."""
        result = _strip_xdot_layout_attrs(CLEAN_DOT)
        assert result == CLEAN_DOT

    def test_preserves_formatting(self):
        """Output preserves original quoting and whitespace (deletion, not reconstruction)."""
        result = _strip_xdot_layout_attrs(XDOT_SIMPLE)
        # The label value should still be double-quoted exactly as in the input.
        assert 'label="hello"' in result
        assert 'label="world"' in result


# ---------------------------------------------------------------------------
# Tests: state machine edge cases
# ---------------------------------------------------------------------------

class TestStateMachineEdgeCases:
    """Edge cases in bracket tracking and name=value matching."""

    def test_quoted_attr_name(self):
        """Quoted name "pos"="..." is recognized and stripped."""
        result = _strip_xdot_layout_attrs(QUOTED_NAME)
        assert not _has_attr(result, "pos")
        assert _has_attr(result, "label")

    def test_html_attr_value(self):
        """HTML-valued attr like _draw_=<...> is consumed correctly."""
        result = _strip_xdot_layout_attrs(HTML_VALUE)
        assert not _has_attr(result, "_draw_")
        assert _has_attr(result, "label")

    def test_mixed_attrs_comma(self):
        """Only pos removed from [label, pos, shape]; comma handling correct."""
        result = _strip_xdot_layout_attrs(MIXED_COMMA)
        assert not _has_attr(result, "pos")
        assert _has_attr(result, "label")
        assert _has_attr(result, "shape")
        # Result should be syntactically valid — no double commas or leading commas.
        assert ",," not in result
        assert "[," not in result

    def test_mixed_attrs_semicolon(self):
        """Same as above but with semicolons."""
        result = _strip_xdot_layout_attrs(MIXED_SEMI)
        assert not _has_attr(result, "pos")
        assert _has_attr(result, "label")
        assert _has_attr(result, "shape")
        assert ";;" not in result

    def test_all_attrs_stripped(self):
        """When all attrs are layout attrs, brackets end up empty-ish."""
        result = _strip_xdot_layout_attrs(ALL_LAYOUT)
        assert not _has_attr(result, "bb")
        assert not _has_attr(result, "pos")
        assert not _has_attr(result, "_draw_")
        # No trailing comma/semicolon inside brackets.
        # Find bracket contents and check.
        import re
        for m in re.finditer(r'\[([^\]]*)\]', result):
            inside = m.group(1).strip()
            assert not inside.startswith(",")
            assert not inside.startswith(";")
            assert not inside.endswith(",")
            assert not inside.endswith(";")

    def test_whitespace_only_separator(self):
        """Whitespace-only separator between attrs doesn't break parsing."""
        result = _strip_xdot_layout_attrs(WHITESPACE_SEP)
        assert not _has_attr(result, "pos")
        assert _has_attr(result, "label")

    def test_multiple_attribute_lists(self):
        """Multiple bracket groups are each processed independently."""
        result = _strip_xdot_layout_attrs(MULTI_BRACKET)
        assert not _has_attr(result, "bb")
        assert not _has_attr(result, "pos")
        assert not _has_attr(result, "_draw_")
        assert _has_attr(result, "label")

    def test_comments_preserved(self):
        """Comments (SKIP tokens) don't break the state machine."""
        result = _strip_xdot_layout_attrs(WITH_COMMENTS)
        assert not _has_attr(result, "pos")
        assert _has_attr(result, "label")
        assert "// a comment" in result
        assert "/* inline */" in result


# ---------------------------------------------------------------------------
# Tests: all layout attr names
# ---------------------------------------------------------------------------

class TestAllLayoutAttrs:
    """Verify each layout attr name in the frozenset is stripped."""

    def test_bb(self):
        result = _strip_xdot_layout_attrs('digraph G { graph [bb="0,0,1,1"]; }')
        assert not _has_attr(result, "bb")

    def test_pos(self):
        result = _strip_xdot_layout_attrs('digraph G { a [pos="50,100"]; }')
        assert not _has_attr(result, "pos")

    def test_lp(self):
        result = _strip_xdot_layout_attrs(LP_ATTR)
        assert not _has_attr(result, "lp")
        assert _has_attr(result, "label")

    def test_rects(self):
        result = _strip_xdot_layout_attrs(RECTS_ATTR)
        assert not _has_attr(result, "rects")
        assert _has_attr(result, "label")

    def test_background(self):
        result = _strip_xdot_layout_attrs(BACKGROUND_ATTR)
        assert not _has_attr(result, "_background")
        assert _has_attr(result, "label")

    def test_draw(self):
        result = _strip_xdot_layout_attrs('digraph G { a [_draw_="e 1 2 3 4"]; }')
        assert not _has_attr(result, "_draw_")

    def test_ldraw(self):
        result = _strip_xdot_layout_attrs('digraph G { a [_ldraw_="T 1 2 3 4 1 -x"]; }')
        assert not _has_attr(result, "_ldraw_")

    def test_hdraw_tdraw(self):
        result = _strip_xdot_layout_attrs(HEAD_TAIL_DRAW)
        assert not _has_attr(result, "_hdraw_")
        assert not _has_attr(result, "_tdraw_")
        assert _has_attr(result, "label")

    def test_hldraw_tldraw(self):
        result = _strip_xdot_layout_attrs(HL_TL_DRAW)
        assert not _has_attr(result, "_hldraw_")
        assert not _has_attr(result, "_tldraw_")
        assert _has_attr(result, "label")


# ---------------------------------------------------------------------------
# Tests: attrs that should NOT be stripped
# ---------------------------------------------------------------------------

class TestPreservation:
    """Things that look like layout attrs but aren't."""

    def test_layout_attr_name_outside_brackets(self):
        """A node named 'pos' (graph element, not inside [...]) is kept."""
        result = _strip_xdot_layout_attrs(NAME_OUTSIDE)
        # The node name 'pos' and edge 'bb -> pos' must survive.
        assert "pos" in result
        assert "bb" in result

    def test_layout_attr_name_as_value(self):
        """label="pos" — 'pos' is a value, not a name. Must not be stripped."""
        result = _strip_xdot_layout_attrs(NAME_AS_VALUE)
        assert 'label="pos"' in result

    def test_partial_name_no_match(self):
        """'position' and '_draw_extra' are not in the frozenset, must survive."""
        result = _strip_xdot_layout_attrs(PARTIAL_NAME)
        assert _has_attr(result, "position")
        assert _has_attr(result, "_draw_extra")

    def test_common_non_layout_attrs_preserved(self):
        """style, fillcolor, fontname, color, penwidth all survive."""
        dot = ('digraph G { a [style=filled, fillcolor="#aabbcc",'
               ' fontname="Helvetica", color=red, penwidth=2.0]; }')
        result = _strip_xdot_layout_attrs(dot)
        for attr in ("style", "fillcolor", "fontname", "color", "penwidth"):
            assert _has_attr(result, attr), f"{attr} should be preserved"


# ---------------------------------------------------------------------------
# Tests: real-world round-trip through GraphViz
# ---------------------------------------------------------------------------

_dot_available = shutil.which("dot") is not None


@pytest.mark.skipif(not _dot_available, reason="GraphViz not installed")
class TestGraphVizRoundTrip:
    """Run real DOT files through `dot -Txdot`, strip, re-layout.

    This verifies that the stripped output is valid DOT that GraphViz
    can re-process without errors.
    """

    @staticmethod
    def _dot_to_xdot(dot_source: str, engine: str = "dot") -> str:
        result = subprocess.run(
            [engine, "-Txdot"],
            input=dot_source, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"dot failed: {result.stderr}"
        return result.stdout

    def test_test_graph_round_trip(self):
        """test_graph.dot -> xdot -> strip -> xdot (re-layout)."""
        dot_src = (TESTDATA / "test_graph.dot").read_text()
        xdot_out = self._dot_to_xdot(dot_src)

        # The xdot output should have layout attrs.
        assert _has_attr(xdot_out, "pos")
        assert _has_attr(xdot_out, "bb")

        # Strip them.
        stripped = _strip_xdot_layout_attrs(xdot_out)
        assert not _has_attr(stripped, "pos")
        assert not _has_attr(stripped, "bb")
        assert not _has_attr(stripped, "_draw_")

        # Re-layout: should succeed without errors.
        re_xdot = self._dot_to_xdot(stripped)
        assert _has_attr(re_xdot, "pos")
        assert _has_attr(re_xdot, "bb")

    def test_test_callgraph_round_trip(self):
        """test_callgraph.dot -> xdot -> strip -> xdot (re-layout)."""
        dot_src = (TESTDATA / "test_callgraph.dot").read_text()
        xdot_out = self._dot_to_xdot(dot_src)

        stripped = _strip_xdot_layout_attrs(xdot_out)
        assert not _has_attr(stripped, "pos")
        assert not _has_attr(stripped, "_draw_")

        # The callgraph has rankdir=LR — verify it survives stripping.
        assert "rankdir=LR" in stripped or "rankdir" in stripped

        # Re-layout succeeds.
        re_xdot = self._dot_to_xdot(stripped)
        assert _has_attr(re_xdot, "pos")

    def test_re_layout_with_different_engine(self):
        """Strip then re-layout with a different engine (neato)."""
        dot_src = (TESTDATA / "test_graph.dot").read_text()
        xdot_out = self._dot_to_xdot(dot_src, engine="dot")
        stripped = _strip_xdot_layout_attrs(xdot_out)

        # neato should be able to process the stripped output.
        if shutil.which("neato"):
            re_xdot = self._dot_to_xdot(stripped, engine="neato")
            assert _has_attr(re_xdot, "pos")
