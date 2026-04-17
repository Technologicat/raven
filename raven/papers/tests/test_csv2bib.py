"""Tests for CSV → BibTeX conversion."""

import itertools
import textwrap

import bibtexparser

from raven.common import readcsv
from raven.papers.csv2bib import rows_to_library


# ---- Synthetic CSV fixture (tab-separated, one row per paper) ---------------
#
# Mirrors the format documented in raven/papers/csv2bib.py's module docstring.
SAMPLE_CSV_TSV = textwrap.dedent("""\
    Author\tYear\tTitle\tAbstract
    Smith, Alice\t2021\tA Study of Widgets\tWidgets are interesting.
    Jones, Bob and Zhang, Cai\t2022\tOn Sprockets\tSprockets predate widgets.
""")


def _counter_key_fn():
    """Produce deterministic slugs: ``key1``, ``key2``, ..."""
    counter = itertools.count(1)
    return lambda: f"key{next(counter)}"


class TestRowsToLibrary:
    """Verify the CSV-rows → bibtexparser.Library transform."""

    def test_empty_rows(self):
        library = rows_to_library([])
        assert len(library.entries) == 0

    def test_single_row(self):
        rows = [{"Author": "Smith, Alice", "Year": "2021", "Title": "X"}]
        library = rows_to_library(rows, key_fn=_counter_key_fn())
        assert len(library.entries) == 1
        entry = library.entries[0]
        assert entry.entry_type == "article"
        assert entry.key == "key1"
        # BibTeX field values come back brace-wrapped (bibtexparser convention).
        assert entry["Author"] == "{Smith, Alice}"
        assert entry["Year"] == "{2021}"
        assert entry["Title"] == "{X}"

    def test_multiple_rows_keep_order(self):
        rows = [{"Author": "A", "Year": "2020"},
                {"Author": "B", "Year": "2021"},
                {"Author": "C", "Year": "2022"}]
        library = rows_to_library(rows, key_fn=_counter_key_fn())
        assert [e.key for e in library.entries] == ["key1", "key2", "key3"]
        assert [e["Author"] for e in library.entries] == ["{A}", "{B}", "{C}"]

    def test_default_keys_are_unique(self):
        """The default UUID-based key function produces unique entry keys."""
        rows = [{"Author": f"Person {i}"} for i in range(5)]
        library = rows_to_library(rows)
        keys = [e.key for e in library.entries]
        assert len(set(keys)) == 5

    def test_bibtex_escape_applied(self):
        """Values with BibTeX specials come out escaped."""
        rows = [{"Title": "50% off & cheap"}]
        library = rows_to_library(rows, key_fn=_counter_key_fn())
        value = library.entries[0]["Title"]
        # `%`, `&` are BibTeX specials; they should appear escaped.
        assert "\\%" in value
        assert "\\&" in value

    def test_writer_roundtrip(self):
        """The resulting library serializes to BibTeX without errors."""
        rows = [{"Author": "Smith, Alice", "Year": "2021", "Title": "X"}]
        library = rows_to_library(rows, key_fn=_counter_key_fn())
        output = bibtexparser.writer.write(library)
        assert "@article{key1," in output
        assert "Author" in output
        assert "Smith, Alice" in output


class TestCsvToLibraryEndToEnd:
    """Parse a synthetic CSV file, feed it through rows_to_library."""

    def test_tab_separated(self, tmp_path):
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(SAMPLE_CSV_TSV)

        rows = readcsv.parse_csv(str(csv_path), has_header=True)
        assert len(rows) == 2
        assert rows[0]["Author"] == "Smith, Alice"
        assert rows[1]["Title"] == "On Sprockets"

        library = rows_to_library(rows, key_fn=_counter_key_fn())
        assert len(library.entries) == 2
        output = bibtexparser.writer.write(library)
        assert "Smith, Alice" in output
        assert "Zhang, Cai" in output
        assert "Sprockets predate widgets." in output

    def test_semicolon_separated(self, tmp_path):
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text("Author;Year;Title\nSmith, Alice;2021;A Paper\n")
        rows = readcsv.parse_csv(str(csv_path), has_header=True)
        library = rows_to_library(rows, key_fn=_counter_key_fn())
        assert len(library.entries) == 1
        assert library.entries[0]["Author"] == "{Smith, Alice}"
