"""Tests for shared bibliography utilities."""

import bibtexparser
from bibtexparser.model import Entry, Field
from bibtexparser import Library

from raven.papers.utils import bibtex_escape, bibtex_unescape


class TestBibtexEscape:
    """Verify that bibtex_escape produces valid BibTeX field values."""

    def _roundtrip(self, text: str) -> str:
        """Write text as a BibTeX field, parse it back, return the parsed value."""
        lib = Library()
        escaped = bibtex_escape(text)
        lib.add(Entry("article", "test", fields=[
            Field("abstract", f"{{{escaped}}}"),
        ]))
        bib_str = bibtexparser.write_string(lib)
        parsed = bibtexparser.parse_string(bib_str)
        assert not parsed.failed_blocks, f"BibTeX parse failed for input {text!r}"
        return parsed.entries[0].fields_dict["abstract"].value

    def test_plain_text_unchanged(self):
        assert bibtex_escape("hello world") == "hello world"

    def test_backslash(self):
        assert bibtex_escape("a \\ b") == "a \\\\ b"

    def test_braces(self):
        assert bibtex_escape("{text}") == r"\{text\}"

    def test_percent(self):
        assert bibtex_escape("20% more") == r"20\% more"

    def test_ampersand(self):
        assert bibtex_escape("A & B") == r"A \& B"

    def test_hash(self):
        assert bibtex_escape("sample #1") == r"sample \#1"

    def test_dollar(self):
        assert bibtex_escape("costs $5") == r"costs \$5"

    def test_brackets(self):
        assert bibtex_escape("[note]") == "{[}note{]}"

    # -- Round-trip tests: write to BibTeX, parse back -----------------------

    def test_roundtrip_plain(self):
        val = self._roundtrip("plain text")
        assert "plain text" in val

    def test_roundtrip_lone_opening_brace(self):
        """The original bug: a lone { in source text broke bibtexparser parsing."""
        val = self._roundtrip("text { more")
        assert val  # parsed without error

    def test_roundtrip_lone_closing_brace(self):
        val = self._roundtrip("text } more")
        assert val

    def test_roundtrip_matched_braces(self):
        val = self._roundtrip("hydrogen {H2} storage")
        assert val

    def test_roundtrip_hash_in_text(self):
        val = self._roundtrip("sample #1 result")
        assert val

    def test_roundtrip_percent(self):
        val = self._roundtrip("20% increase in yield")
        assert val

    def test_roundtrip_multiple_specials(self):
        val = self._roundtrip("H{2} costs $5 & is 20% of #1")
        assert val


class TestBibtexUnescape:
    """Verify that bibtex_unescape reverses bibtex_escape."""

    def test_roundtrip_plain(self):
        assert bibtex_unescape(bibtex_escape("hello")) == "hello"

    def test_roundtrip_all_specials(self):
        original = r"a \ b { c } d & e % f # g $ h [ i"
        assert bibtex_unescape(bibtex_escape(original)) == original

    def test_roundtrip_backslash_then_brace(self):
        """Tricky: \\{ in source — backslash escapes first, then brace."""
        original = r"\{"
        assert bibtex_unescape(bibtex_escape(original)) == original

    def test_individual_unescapes(self):
        assert bibtex_unescape(r"\%") == "%"
        assert bibtex_unescape(r"\$") == "$"
        assert bibtex_unescape(r"\#") == "#"
        assert bibtex_unescape(r"\&") == "&"
        assert bibtex_unescape(r"\{") == "{"
        assert bibtex_unescape(r"\}") == "}"
        assert bibtex_unescape("\\\\") == "\\"
        assert bibtex_unescape("{[}") == "["
        assert bibtex_unescape("{]}") == "]"
