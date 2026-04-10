"""Tests for the arXiv feed → BibTeX converter."""

from unittest.mock import MagicMock

from raven.papers.bibtex import _clean_whitespace, _make_key, entries_to_bibtex


def _fake_entry(
    arxiv_id="2103.12345v2",
    title="Some Title",
    authors=None,
    published="2021-03-23T00:00:00Z",
    summary="An abstract.",
    primary_category="quant-ph",
    doi=None,
    journal_ref=None,
):
    """Build a dict mimicking a feedparser entry for an arXiv result."""
    if authors is None:
        authors = [{"name": "Alice Smith"}, {"name": "Bob Jones"}]
    entry = MagicMock()
    entry.id = f"http://arxiv.org/abs/{arxiv_id}"
    entry.published = published
    entry.get = lambda key, default=None: {
        "title": title,
        "summary": summary,
        "authors": authors,
        "arxiv_primary_category": {"term": primary_category},
        "arxiv_doi": doi,
        "arxiv_journal_ref": journal_ref,
    }.get(key, default)
    return entry


class TestCleanWhitespace:
    def test_collapses_newlines(self):
        assert _clean_whitespace("a\n  b\n  c") == "a b c"

    def test_collapses_tabs(self):
        assert _clean_whitespace("a\t\tb") == "a b"

    def test_strips_leading_trailing(self):
        assert _clean_whitespace("  hello  ") == "hello"


class TestMakeKey:
    def test_basic(self):
        entry = _fake_entry()
        assert _make_key(entry) == "Smith_2021_2103.12345"

    def test_strips_version(self):
        entry = _fake_entry(arxiv_id="2103.12345v3")
        key = _make_key(entry)
        assert "v3" not in key
        assert "2103.12345" in key

    def test_old_style_id(self):
        entry = _fake_entry(arxiv_id="hep-ex/0307015v1")
        key = _make_key(entry)
        assert key == "Smith_2021_hep-ex_0307015"

    def test_no_authors(self):
        entry = _fake_entry(authors=[])
        key = _make_key(entry)
        assert key.startswith("Unknown_")


class TestEntriesToBibtex:
    def test_basic_output(self):
        entry = _fake_entry(
            title="Quantum Error Correction",
            doi="10.1234/test",
            journal_ref="Nature 605, 669 (2022)",
        )
        bib = entries_to_bibtex([entry])

        assert "@article{Smith_2021_2103.12345" in bib
        assert "Quantum Error Correction" in bib
        assert "Alice Smith and Bob Jones" in bib
        assert "2021" in bib
        assert "arXiv" in bib
        assert "10.1234/test" in bib
        assert "Nature 605, 669 (2022)" in bib

    def test_multiple_entries(self):
        e1 = _fake_entry(arxiv_id="2101.00001v1")
        e2 = _fake_entry(arxiv_id="2102.00002v1")
        bib = entries_to_bibtex([e1, e2])
        assert bib.count("@article{") == 2

    def test_no_doi_or_journal(self):
        entry = _fake_entry()
        bib = entries_to_bibtex([entry])
        assert "doi" not in bib.lower().split("archiveprefix")[0]  # no doi field
        assert "journal" not in bib.lower()
