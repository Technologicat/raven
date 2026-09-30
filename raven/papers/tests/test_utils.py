"""Tests for shared bibliography utilities."""

import bibtexparser
from bibtexparser.model import Entry, Field
from bibtexparser import Library

import csv

from raven.papers.utils import bibtex_escape, bibtex_unescape, write_tsv


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


class TestWriteTsv:
    """A report opens in a spreadsheet with one row per record and every cell intact."""

    def _read(self, path):
        """The data rows, parsed the way a spreadsheet's import does: `"` is a quote character."""
        lines = [line for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
                 if not line.startswith("#")]
        return list(csv.reader(lines, delimiter="\t"))

    def test_a_cell_opening_with_a_quote_survives_the_round_trip(self, tmp_path):
        path = tmp_path / "report.tsv"
        rows = [("a_2024", '"Hey ChatGPT": students ask'), ("b_2024", "plain title")]
        write_tsv(path, ["tool 1.0"], ("key", "title"), rows)
        parsed = self._read(path)
        assert parsed[0] == ["key", "title"]
        assert [tuple(row) for row in parsed[1:]] == rows

    def test_a_quote_mid_cell_is_quoted_too(self, tmp_path):
        # Asserted on the text rather than by parsing it back: Python's reader tolerates a bare `"` inside a
        # cell, so a round trip passes whether or not it was quoted, while a spreadsheet import does not.
        path = tmp_path / "report.tsv"
        write_tsv(path, [], ("key", "title"), [("a_2024", 'The "learning assistant" in STEM')])
        assert path.read_text(encoding="utf-8").splitlines()[1] == 'a_2024\t"The ""learning assistant"" in STEM"'

    def test_tabs_and_newlines_in_a_cell_do_not_split_it(self, tmp_path):
        path = tmp_path / "report.tsv"
        write_tsv(path, [], ("key", "title"), [("a_2024", "two\tpart\ntitle ")])
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
        assert lines == ["key\ttitle", "a_2024\ttwo part title"]

    def test_comments_come_first_as_hash_lines(self, tmp_path):
        path = tmp_path / "report.tsv"
        write_tsv(path, ["tool 1.0", "input: corpus.bib"], ("key",), [("a_2024",)])
        assert path.read_text(encoding="utf-8").splitlines() == ["# tool 1.0", "# input: corpus.bib",
                                                                 "key", "a_2024"]
