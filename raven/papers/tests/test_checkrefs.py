"""Tests for `raven.papers.checkrefs`: matching a bibliography against a written reference list."""

import pytest

from raven.common import tabular
from raven.papers import bibtex, checkrefs

BIB = """
@article{bydoi,
  title = {Ravens in the city},
  doi = {10.1234/ravens.2020},
  year = {2020},
}

@article{https://doi.org/10.1234/keyonly.7,
  title = {A record whose DOI is only in its key},
  year = {2021},
}

@article{bytitle,
  title = {Corvid tool use in the wild},
  year = {2019},
}

@article{respelled,
  title = {Healthy children and a healthy planet: a role for education},
  year = {2011},
}

@article{absent,
  title = {Migration routes of the common raven},
  year = {2018},
}
"""

DOCUMENT = """References

1. Smith, A. (2020). Ravens in the City. Journal of Birds, 3, 1-2. https://doi.org/10.1234/RAVENS.2020
2. Jones, B. (2021). Something else entirely. Press. DOI: 10.1234/keyonly.7
3. Lee, C. (2019). Corvid tool use in the wild.
   Animal Cognition, 22, 5-9.
4. Cooke, S. M. (2011). Healthy children, healthy planet: A role for education. In Climate Change and
   Rural Child Health (pp. 71-84).
5. Brown, D. (2022). A study nobody put in the bibliography. Elsewhere.
"""


@pytest.fixture
def references(tmp_path):
    path = tmp_path / "corpus.bib"
    path.write_text(BIB, encoding="utf-8")
    return checkrefs.references_of(bibtex.parse_file(path, split_names=False))


@pytest.fixture
def entries():
    texts, mode = checkrefs.split_entries(DOCUMENT)
    assert mode == "numbered"
    return checkrefs.entries_of(texts)


def by_key(rows):
    return {row.maybe_reference.key: row for row in rows if row.maybe_reference}


class TestSplitting:
    def test_a_numbered_list_drops_its_heading_and_joins_wrapped_lines(self):
        texts, mode = checkrefs.split_entries(DOCUMENT)
        assert (mode, len(texts)) == ("numbered", 5)
        assert texts[2] == "Lee, C. (2019). Corvid tool use in the wild. Animal Cognition, 22, 5-9."

    @pytest.mark.parametrize("marker", ["{}.", "{})", "[{}]"])
    def test_every_numbering_style(self, marker):
        text = "\n".join(f"{marker.format(i)} Entry {i}" for i in (1, 2, 3))
        assert checkrefs.split_entries(text) == (["Entry 1", "Entry 2", "Entry 3"], "numbered")

    def test_an_unnumbered_list_is_one_entry_per_line(self):
        text = "Smith, A. (2020). One.\n\nJones, B. (2021). Two.\n"
        assert checkrefs.split_entries(text) == (["Smith, A. (2020). One.", "Jones, B. (2021). Two."], "lines")

    def test_a_wrapped_line_starting_with_a_number_is_not_a_list_item(self):
        """An article number or a page range wrapped onto a line of its own, in an author-year list."""
        text = ("Smith, A. (2020). One. Journal, 5,\n100436. https://doi.org/10.1/x.\n"
                "Jones, B. (2021). Two. Conference, paper\n04025. https://doi.org/10.1/y.\n")
        texts, mode = checkrefs.split_entries(text)
        assert mode == "lines" and len(texts) == 4

    def test_a_stray_number_inside_a_numbered_list_joins_its_entry(self):
        texts, mode = checkrefs.split_entries("1. A, 2020. Journal, 5,\n100436. doi.\n2. B, 2021. Two.\n")
        assert (mode, texts) == ("numbered", ["A, 2020. Journal, 5, 100436. doi.", "B, 2021. Two."])

    def test_a_number_inside_an_entry_does_not_split_it(self):
        texts, _ = checkrefs.split_entries("1. A, 2020. Vol 3.\n2. B, 2021. Pages 4-5.\n")
        assert len(texts) == 2


class TestMatching:
    def test_each_record_lands_where_it_should(self, references, entries):
        rows = by_key(checkrefs.check(references, entries))
        assert {key: (row.status, row.maybe_entry.number if row.maybe_entry else None)
                for key, row in rows.items()} == {
            "bydoi": (checkrefs.MATCHED_BY_DOI, 1),
            "https://doi.org/10.1234/keyonly.7": (checkrefs.MATCHED_BY_DOI, 2),
            "bytitle": (checkrefs.MATCHED_BY_TITLE, 3),
            "respelled": (checkrefs.PROPOSED, 4),
            "absent": (checkrefs.NOT_IN_DOCUMENT, None)}

    def test_an_entry_with_no_record_is_reported(self, references, entries):
        orphans = [row for row in checkrefs.check(references, entries) if row.status == checkrefs.NO_RECORD]
        assert [row.maybe_entry.number for row in orphans] == [5]

    def test_a_doi_match_needs_no_title(self, references, entries):
        # Entry 2's title has nothing in common with the record's; the DOI decides alone.
        row = by_key(checkrefs.check(references, entries))["https://doi.org/10.1234/keyonly.7"]
        assert row.status == checkrefs.MATCHED_BY_DOI

    def test_a_proposal_carries_its_margin(self, references, entries):
        row = by_key(checkrefs.check(references, entries))["respelled"]
        assert row.score > 0.85 and row.score - row.runner_up > 0.3, (row.score, row.runner_up)

    def test_a_record_with_no_clear_partner_is_not_proposed(self, references, entries):
        # Negative control: the absent record has a free entry to be paired with (5), and stays unpaired
        # only because it shares nothing with it. Without that entry the test would pass for the wrong reason.
        free = [row for row in checkrefs.check(references, entries) if row.status == checkrefs.NO_RECORD]
        assert free, "no free entry, so this fixture cannot tell a refused pairing from an impossible one"
        assert by_key(checkrefs.check(references, entries))["absent"].status == checkrefs.NOT_IN_DOCUMENT

    def test_a_record_whose_best_entry_is_taken_is_not_proposed_elsewhere(self, references, entries, tmp_path):
        # A second, respelled copy of an already-matched record. Its best entry is the one the first copy
        # took, so it is reported missing rather than paired with a free entry it merely overlaps.
        path = tmp_path / "dup.bib"
        path.write_text(BIB + "@article{copy, title = {Ravens in the City: a field study}}\n", encoding="utf-8")
        refs = checkrefs.references_of(bibtex.parse_file(path, split_names=False))
        rows = by_key(checkrefs.check(refs, entries))
        assert rows["copy"].status == checkrefs.NOT_IN_DOCUMENT
        assert checkrefs.title_coverage(refs[-1].maybe_title_key, entries[4].maybe_key) > 0.0, \
            "the copy shares nothing with the free entry, so this fixture cannot tell the rule from no candidate"
        assert rows["absent"].status == checkrefs.NOT_IN_DOCUMENT, "the copy must not crowd out another record"

    def test_the_rows_put_what_needs_a_person_first(self, references, entries):
        statuses = [row.status for row in checkrefs.check(references, entries)]
        assert statuses[:3] == [checkrefs.PROPOSED, checkrefs.NOT_IN_DOCUMENT, checkrefs.NO_RECORD]
        assert set(statuses[3:]) == {checkrefs.MATCHED_BY_DOI, checkrefs.MATCHED_BY_TITLE}


class TestCoverage:
    def test_a_title_found_whole_is_one(self):
        assert checkrefs.title_coverage("ravensinthecity", "smitha2020ravensinthecityjournal") == 1.0

    def test_short_coincidences_do_not_count(self):
        # Every letter of the title occurs in the entry, but never in a run of four.
        assert checkrefs.title_coverage("abcabc", "axbxcxaxbxcx") == 0.0

    def test_empty_is_zero(self):
        assert checkrefs.title_coverage("", "anything") == 0.0


class TestAudit:
    @pytest.mark.parametrize("table_format", tabular.FORMATS)
    def test_every_row_is_written_with_an_empty_decision(self, references, entries, tmp_path, table_format):
        rows = checkrefs.check(references, entries)
        path = tmp_path / f"audit.{table_format}"
        checkrefs.write_audit(path, rows, bib_path="corpus.bib", document_path="doc.odt", split_mode="numbered")
        table = tabular.read_table(path)
        assert len(table) == len(rows) == 6
        assert table[0]["status"] == checkrefs.PROPOSED and table[0]["score"]
        assert all(row["decision"] == "" for row in table)
