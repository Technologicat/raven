"""Tests for the BibTeX reader and the arXiv feed → BibTeX converter."""

from unittest.mock import MagicMock

from raven.papers.bibtex import _clean_whitespace, _make_key, entries_to_bibtex, parse_file, parse_string


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


# ---------------------------------------------------------------------------
# The reader side: `parse_string` / `parse_file` and the middleware they carry
# ---------------------------------------------------------------------------

# These pin the middleware chain rather than `bibtexparser` itself. The chain is the reason the readers
# exist: two call sites (the Visualizer importer and Librarian's paste sniffer) had assembled it
# independently, and a silent divergence between them would mean the same `.bib` parsing differently
# depending on which door it came in by.

def _last_names(entry):
    """The `last` name parts of every author, as joined strings — what `SplitNameParts` should produce."""
    return [" ".join(name.last) for name in entry["author"]]


class TestParseString:
    def test_field_keys_are_normalized_to_lowercase(self):
        # A Web of Science export writes `Title = {...}`; the BibTeX literature writes `title = {...}`.
        library = parse_string("@article{k, Title={A Study}, YEAR={2024}}")
        assert sorted(f.key for f in library.entries[0].fields) == ["title", "year"]

    def test_coauthors_are_separated(self):
        library = parse_string("@article{k, author={Alice Smith and Bob Jones}, year={2024}}")
        assert _last_names(library.entries[0]) == ["Smith", "Jones"]

    def test_name_parts_survive_a_von_particle(self):
        library = parse_string("@article{k, author={Ludwig van Beethoven}, year={2024}}")
        name = library.entries[0]["author"][0]
        assert name.von == ["van"]
        assert name.last == ["Beethoven"]

    def test_name_parts_survive_a_compound_last_name(self):
        # "Brinch Hansen, Per" — the comma is what marks the whole of "Brinch Hansen" as the surname.
        library = parse_string("@article{k, author={Brinch Hansen, Per}, year={2024}}")
        name = library.entries[0]["author"][0]
        assert name.last == ["Brinch", "Hansen"]
        assert name.first == ["Per"]

    def test_name_parts_survive_a_suffix(self):
        library = parse_string("@article{k, author={Beeblebrox, IV, Zaphod}, year={2024}}")
        name = library.entries[0]["author"][0]
        assert name.last == ["Beeblebrox"]
        assert name.jr == ["IV"]
        assert name.first == ["Zaphod"]

    def test_an_unreadable_record_lands_in_failed_blocks_rather_than_raising(self):
        # The documented contract: a successful return is not a promise that every record was understood.
        # Librarian's paste sniffer relies on partial success, and the importer reports the failures.
        library = parse_string("@article{good, title={Fine}, year={2024}}\n@article{good, title={Dupe}}")
        assert len(library.entries) == 1
        assert len(library.failed_blocks) == 1


class TestParseFile:
    def test_reads_a_file_with_the_same_middleware_as_parse_string(self, tmp_path):
        path = tmp_path / "refs.bib"
        path.write_text("@article{k, Title={A Study}, author={Alice Smith and Bob Jones}, year={2024}}",
                        encoding="utf-8")

        from_file = parse_file(path)
        from_string = parse_string(path.read_text(encoding="utf-8"))

        assert [f.key for f in from_file.entries[0].fields] == [f.key for f in from_string.entries[0].fields]
        assert _last_names(from_file.entries[0]) == _last_names(from_string.entries[0]) == ["Smith", "Jones"]

    def test_accepts_a_path_object_and_a_string(self, tmp_path):
        # The importer passes a str, Librarian-side callers are likelier to hold a Path. Both must work.
        path = tmp_path / "refs.bib"
        path.write_text("@article{k, title={A Study}, year={2024}}", encoding="utf-8")

        assert len(parse_file(path).entries) == 1
        assert len(parse_file(str(path)).entries) == 1
