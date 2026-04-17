"""Unit tests for raven.common.readcsv."""

import pytest

from raven.common.readcsv import parse_csv


@pytest.fixture
def write_csv(tmp_path):
    """Return a helper that writes `content` to a temp file and returns its path."""
    def _write(content: str, name: str = "data.csv"):
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p
    return _write


class TestDelimiterAutodetect:
    def test_tab(self, write_csv):
        p = write_csv("name\tage\nAlice\t30\nBob\t25\n")
        rows = parse_csv(p)
        assert rows == [{"name": "Alice", "age": "30"},
                        {"name": "Bob", "age": "25"}]

    def test_semicolon(self, write_csv):
        p = write_csv("name;age\nAlice;30\nBob;25\n")
        rows = parse_csv(p)
        assert rows == [{"name": "Alice", "age": "30"},
                        {"name": "Bob", "age": "25"}]

    def test_tab_takes_precedence_over_semicolon(self, write_csv):
        # Both present → tab wins (first autodetect branch).
        p = write_csv("a\tb;c\n1\t2;3\n")
        rows = parse_csv(p)
        # With tab as delimiter, the "b;c" field becomes a single column header, and "2;3" is its value.
        assert rows == [{"a": "1", "b;c": "2;3"}]

    def test_no_recognized_delimiter_raises(self, write_csv):
        p = write_csv("name,age\nAlice,30\n")  # comma-delimited — not auto-detected
        with pytest.raises(ValueError, match="Could not determine delimiter"):
            parse_csv(p)


class TestExplicitDelimiter:
    def test_comma(self, write_csv):
        p = write_csv("name,age\nAlice,30\n")
        rows = parse_csv(p, delimiter=",")
        assert rows == [{"name": "Alice", "age": "30"}]

    def test_pipe(self, write_csv):
        p = write_csv("x|y\n1|2\n3|4\n")
        rows = parse_csv(p, delimiter="|")
        assert rows == [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}]


class TestHeaderHandling:
    def test_autodetect_with_alphabetic_header(self, write_csv):
        p = write_csv("name\tage\nAlice\t30\n")
        rows = parse_csv(p)
        assert rows[0] == {"name": "Alice", "age": "30"}

    def test_autodetect_all_numeric_treated_as_headerless(self, write_csv):
        # Heuristic: header row must contain at least one alphabetic char somewhere.
        p = write_csv("1\t2\n3\t4\n")
        rows = parse_csv(p)
        # No header detected → generic "column N" names, first row included as data.
        assert rows == [{"column 1": "1", "column 2": "2"},
                        {"column 1": "3", "column 2": "4"}]

    def test_explicit_has_header_true(self, write_csv):
        p = write_csv("h1\th2\nv1\tv2\n")
        rows = parse_csv(p, has_header=True)
        assert rows == [{"h1": "v1", "h2": "v2"}]

    def test_explicit_has_header_false(self, write_csv):
        p = write_csv("h1\th2\nv1\tv2\n")
        rows = parse_csv(p, has_header=False)
        # Both lines treated as data; generic header names assigned.
        assert rows == [{"column 1": "h1", "column 2": "h2"},
                        {"column 1": "v1", "column 2": "v2"}]


class TestEdgeCases:
    def test_empty_file(self, write_csv):
        p = write_csv("")
        assert parse_csv(p) == []

    def test_empty_rows_skipped(self, write_csv):
        p = write_csv("name\tage\n\nAlice\t30\n\nBob\t25\n")
        rows = parse_csv(p)
        assert rows == [{"name": "Alice", "age": "30"},
                        {"name": "Bob", "age": "25"}]

    def test_missing_file_raises_valueerror(self, tmp_path):
        # The implementation wraps IO errors into ValueError.
        with pytest.raises(ValueError):
            parse_csv(tmp_path / "does-not-exist.csv")

    def test_path_as_string(self, write_csv):
        p = write_csv("a\tb\n1\t2\n")
        # Pass the path as a string, not pathlib.Path — both must work.
        rows = parse_csv(str(p))
        assert rows == [{"a": "1", "b": "2"}]

    def test_unicode_content(self, write_csv):
        p = write_csv("city\tcountry\nJyväskylä\tFinland\n")
        rows = parse_csv(p)
        assert rows == [{"city": "Jyväskylä", "country": "Finland"}]
