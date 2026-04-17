"""Tests for Web of Science → BibTeX conversion."""

import bibtexparser

from raven.papers.wos2bib import (
    _format_author_addresses,
    record_to_bibtex_entry,
    records_to_library,
)


# ---- Synthetic WOS record ---------------------------------------------------
#
# ``wosfile.Record`` is a dict-like object with an ``.author_address`` attribute
# populated from the undocumented ``C1`` field.  ``FakeRecord`` mimics just
# enough of that interface for the conversion layer to exercise.
class FakeRecord(dict):
    """Minimal stand-in for ``wosfile.Record``."""

    def __init__(self, data, author_address=None):
        super().__init__(data)
        self.author_address = author_address


def _full_journal_record(**overrides):
    """A journal article with every optional field populated."""
    base = {
        "UT": "WOS:000123456789",
        "PT": "J",
        "AU": ["Smith, Alice", "Jones, Bob"],
        "PY": "2021",
        "TI": "A Study of Widgets",
        "SO": "Journal of Widget Studies",
        "VL": "42",
        "IS": "7",
        "BP": "100",
        "EP": "120",
        "DI": "10.1234/jws.2021.042",
        "WC": ["Engineering", "Materials Science"],
        "AB": "Widgets are interesting.",
        "CR": ["Prior, A. 2020", "Other, B. 2019"],
        "NR": 2,
    }
    base.update(overrides)
    return FakeRecord(base, author_address={"Smith, Alice": ["MIT"],
                                            "Jones, Bob": ["Stanford"]})


# ---- _format_author_addresses ----------------------------------------------

class TestFormatAuthorAddresses:
    """Verify the C1 (author_address) normalizer."""

    def test_none_empty_string(self):
        assert _format_author_addresses(None) == ""

    def test_empty_dict(self):
        assert _format_author_addresses({}) == ""

    def test_dict_with_authors(self):
        result = _format_author_addresses({"Smith": ["MIT", "MIT"]})
        # Duplicates within one author are collapsed.
        assert result == "Smith, MIT"

    def test_dict_multiple_authors_newline_separated(self):
        result = _format_author_addresses({"Smith": ["MIT"], "Jones": ["Stanford"]})
        # Order is dict-iteration order (insertion order since 3.7).
        assert result == "Smith, MIT\nJones, Stanford"

    def test_list_of_addresses(self):
        result = _format_author_addresses(["MIT", "Stanford", "MIT"])
        # Set-dedup, then period-joined (order not guaranteed).
        assert set(result.split(". ")) == {"MIT", "Stanford"}

    def test_unrecognized_format_empty(self):
        """Any other shape (e.g. a bare string) → empty string."""
        assert _format_author_addresses("unexpected") == ""


# ---- record_to_bibtex_entry: required-field skips ---------------------------

class TestRecordToEntrySkips:
    """Verify each required-field check produces the right skip reason."""

    def test_missing_ut(self):
        rec = FakeRecord({"PT": "J", "AU": ["X"], "PY": "2021", "TI": "Y"})
        entry, reason = record_to_bibtex_entry(rec)
        assert entry is None
        assert "unique identifier" in reason

    def test_missing_pt(self):
        rec = FakeRecord({"UT": "ID1", "AU": ["X"], "PY": "2021", "TI": "Y"})
        entry, reason = record_to_bibtex_entry(rec)
        assert entry is None
        assert "publication type" in reason
        assert "ID1" in reason

    def test_missing_au(self):
        rec = FakeRecord({"UT": "ID1", "PT": "J", "PY": "2021", "TI": "Y"})
        entry, reason = record_to_bibtex_entry(rec)
        assert entry is None
        assert "authors" in reason

    def test_missing_py(self):
        rec = FakeRecord({"UT": "ID1", "PT": "J", "AU": ["X"], "TI": "Y"})
        entry, reason = record_to_bibtex_entry(rec)
        assert entry is None
        assert "year" in reason

    def test_missing_ti(self):
        rec = FakeRecord({"UT": "ID1", "PT": "J", "AU": ["X"], "PY": "2021"})
        entry, reason = record_to_bibtex_entry(rec)
        assert entry is None
        assert "title" in reason

    def test_empty_authors_list_skipped(self):
        """An explicitly empty AU list is still a skip."""
        rec = FakeRecord({"UT": "ID1", "PT": "J", "AU": [], "PY": "2021", "TI": "Y"})
        entry, reason = record_to_bibtex_entry(rec)
        assert entry is None
        assert "authors" in reason


# ---- record_to_bibtex_entry: successful conversion --------------------------

class TestRecordToEntrySuccess:
    """Verify a complete WOS record converts to a populated BibTeX entry."""

    def test_full_journal_record(self):
        entry, reason = record_to_bibtex_entry(_full_journal_record())
        assert reason is None
        assert entry.entry_type == "article"  # PT "J" → article
        assert entry.key == "WOS:000123456789"
        # Required fields
        assert entry["Author"] == "{Smith, Alice and Jones, Bob}"
        assert entry["Year"] == "{2021}"
        assert entry["Title"] == "{A Study of Widgets}"
        # Optional fields
        assert "Widget Studies" in entry["Journal"]
        assert entry["Volume"] == "{42}"
        assert entry["Number"] == "{7}"
        assert entry["Pages"] == "{100-120}"
        assert entry["DOI"] == "{10.1234/jws.2021.042}"
        assert "Engineering" in entry["Web-Of-Science-Categories"]
        assert "Widgets are interesting" in entry["Abstract"]
        assert "Prior, A. 2020" in entry["Cited-References"]
        assert entry["Number-Of-Cited-References"] == "{2}"
        assert "MIT" in entry["Affiliation"]

    def test_book_type(self):
        entry, _ = record_to_bibtex_entry(_full_journal_record(PT="B"))
        assert entry.entry_type == "book"

    def test_missing_abstract_still_converts(self):
        """Missing AB is a soft skip — the entry is still emitted."""
        entry, reason = record_to_bibtex_entry(_full_journal_record(AB=None))
        assert reason is None
        with_fields = {f.key for f in entry.fields}
        assert "Abstract" not in with_fields

    def test_only_one_page_means_no_pages_field(self):
        """Pages field is only emitted when both BP and EP are present."""
        entry, _ = record_to_bibtex_entry(_full_journal_record(BP="100", EP=None))
        with_fields = {f.key for f in entry.fields}
        assert "Pages" not in with_fields

    def test_zero_references_no_nr_field(self):
        """``NR = 0`` is not rendered (the field is ""there are this many references"")."""
        entry, _ = record_to_bibtex_entry(_full_journal_record(NR=0))
        with_fields = {f.key for f in entry.fields}
        assert "Number-Of-Cited-References" not in with_fields

    def test_minimal_record(self):
        """Just the required fields — optional ones are absent."""
        rec = FakeRecord({
            "UT": "ID1", "PT": "J", "AU": ["Smith"], "PY": "2021", "TI": "Y",
        })
        entry, reason = record_to_bibtex_entry(rec)
        assert reason is None
        with_fields = {f.key for f in entry.fields}
        assert with_fields == {"Author", "Year", "Title"}


# ---- records_to_library -----------------------------------------------------

class TestRecordsToLibrary:
    """Verify batch conversion and skip accounting."""

    def test_empty_input(self):
        library, skipped = records_to_library([])
        assert len(library.entries) == 0
        assert skipped == []

    def test_all_valid(self):
        records = [_full_journal_record(UT=f"ID{i}") for i in range(3)]
        library, skipped = records_to_library(records)
        assert len(library.entries) == 3
        assert skipped == []

    def test_mixed_valid_and_invalid(self):
        bad = FakeRecord({"PT": "J"})  # missing UT, AU, PY, TI
        good = _full_journal_record()
        library, skipped = records_to_library([bad, good])
        assert len(library.entries) == 1
        assert len(skipped) == 1
        assert "unique identifier" in skipped[0]

    def test_library_writable(self):
        """The final library serializes to BibTeX."""
        library, _ = records_to_library([_full_journal_record()])
        output = bibtexparser.writer.write(library)
        assert "@article{WOS:000123456789," in output
        assert "Smith, Alice and Jones, Bob" in output
