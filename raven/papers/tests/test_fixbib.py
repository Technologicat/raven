"""Tests for `raven.papers.fixbib` — repairing BibTeX records a parser refuses, and reporting the rest."""

import logging

from ..fixbib import (repair_bibtex, RepairReport,
                      KIND_UNBALANCED_BRACES, KIND_DUPLICATE_FIELD_KEYS, KIND_UNREADABLE)
from .. import bibtex

# `bibtexparser` warns about every record it cannot read, and these tests feed it unreadable records on
# purpose. The warnings are the expected outcome, not a finding.
logging.getLogger("bibtexparser").setLevel(logging.ERROR)

GOOD = "@article{good,\n    title = {A Sound Record},\n    year = {2024},\n}\n"

# A stray opening brace, as a PDF extractor produces when it drops the closing one of `{0 <= rho <= 1}`.
# The value then has no terminator, so the parser reads on and eventually gives up on the record.
BROKEN = ("@article{broken,\n"
          "    title = {A Record With Mathematics},\n"
          "    abstract = {We require {0 <= rho <= 1 throughout the domain,\n"
          "and report the results below.},\n"
          "}\n")

# A field named three times, as a ProQuest export writes it — one `annote` per kind of note. BibTeX has
# no way to say that, so the parser rejects the entry whole, title and all.
REPEATED = ("@article{repeated,\n"
            "    title = {A Record From A Database Export},\n"
            "    annote = {Copyright - Copyright Some University 2022},\n"
            "    year = {2022},\n"
            "    annote = {Last updated - 2024-04-23},\n"
            "    annote = {SubjectsTermNotLitGenreText - Learning Analytics; Pedagogy},\n"
            "}\n")


class TestRepairBibtex:
    def test_a_sound_file_is_returned_unchanged(self):
        repaired, recovered, unrecovered = repair_bibtex(GOOD)
        assert repaired == GOOD
        assert recovered == [] and unrecovered == []

    def test_a_broken_record_is_recovered_whole(self):
        repaired, recovered, unrecovered = repair_bibtex(BROKEN)
        assert [r.key for r in recovered] == ["broken"] and unrecovered == []
        entries = bibtex.parse_string(repaired).entries
        assert len(entries) == 1
        fields = {f.key: f.value for f in entries[0].fields}
        # The title is the point: it is on a sound line, and an unbalanced brace elsewhere loses it anyway.
        assert fields["title"] == "A Record With Mathematics"
        assert "0 <= rho <= 1" in fields["abstract"]

    def test_only_braces_are_escaped_and_nothing_else_moves(self):
        repaired, _recovered, _unrecovered = repair_bibtex(GOOD + "\n" + BROKEN + "\n" + GOOD)
        assert repaired.replace("\\{", "{").replace("\\}", "}") == GOOD + "\n" + BROKEN + "\n" + GOOD

    def test_the_records_around_a_repair_are_untouched(self):
        source = GOOD + "\n" + BROKEN + "\n" + GOOD.replace("good", "alsogood")
        repaired, recovered, _unrecovered = repair_bibtex(source)
        assert [r.key for r in recovered] == ["broken"]
        assert [entry.key for entry in bibtex.parse_string(repaired).entries] == ["good", "broken", "alsogood"]

    def test_a_record_missing_its_terminator_is_reported_not_invented(self):
        # Nothing says where the absent brace belonged, so the honest outcome is to leave it and say so.
        source = "@article{hopeless,\n    abstract = {an abstract that simply never closes\n}\n"
        repaired, recovered, unrecovered = repair_bibtex(source)
        assert recovered == []
        assert len(unrecovered) == 1 and unrecovered[0].key == "hopeless"
        assert repaired == source

    def test_a_line_boundary_that_is_not_a_newline_does_not_misplace_the_repair(self):
        # `str.splitlines` breaks on `\x1c` and friends while a newline count does not, so a file carrying
        # one drifts the two apart. A bibliography extracted from PDFs carries them; ECCOMAS 2024 has six.
        source = GOOD.replace("2024", "2024\x1c") + "\n" + BROKEN
        repaired, recovered, _unrecovered = repair_bibtex(source)
        assert [r.key for r in recovered] == ["broken"]
        assert "\x1c" in repaired
        assert repaired.replace("\\{", "{").replace("\\}", "}") == source

    def test_the_reported_line_number_points_at_the_record(self):
        source = GOOD + "\n" + "@article{hopeless,\n    abstract = {never closes\n}\n"
        _repaired, _recovered, unrecovered = repair_bibtex(source)
        line = unrecovered[0].line
        assert source.splitlines()[line - 1].startswith("@article{hopeless")


class TestRepairDuplicateFieldKeys:
    def test_a_record_naming_a_field_three_times_is_recovered_whole(self):
        repaired, recovered, unrecovered = repair_bibtex(REPEATED)
        assert [r.key for r in recovered] == ["repeated"] and unrecovered == []
        entries = bibtex.parse_string(repaired).entries
        assert len(entries) == 1
        fields = {f.key: f.value for f in entries[0].fields}
        # The title is the point, as above: a duplicate `annote` loses the whole record, not just a note.
        assert fields["title"] == "A Record From A Database Export"
        assert fields["year"] == "2022"

    def test_every_repeated_value_survives_the_merge(self):
        # The values are deliberately all different. Were they identical, keeping one would pass this
        # test while discarding somebody's data, and the fixture could not tell the two apart.
        repaired, _recovered, _unrecovered = repair_bibtex(REPEATED)
        annote = {f.key: f.value for f in bibtex.parse_string(repaired).entries[0].fields}["annote"]
        assert "Copyright - Copyright Some University 2022" in annote
        assert "Last updated - 2024-04-23" in annote
        assert "SubjectsTermNotLitGenreText - Learning Analytics; Pedagogy" in annote

    def test_the_merged_field_keeps_the_first_occurrence_position(self):
        repaired, _recovered, _unrecovered = repair_bibtex(REPEATED)
        keys = [f.key for f in bibtex.parse_string(repaired).entries[0].fields]
        assert keys == ["title", "annote", "year"]

    def test_the_fault_is_named_and_so_are_the_fields_carrying_it(self):
        _repaired, recovered, _unrecovered = repair_bibtex(REPEATED)
        assert recovered == [RepairReport(key="repeated", line=1,
                                          kind=KIND_DUPLICATE_FIELD_KEYS, detail="repeats annote")]

    def test_the_records_around_a_merge_are_untouched(self):
        source = GOOD + "\n" + REPEATED + "\n" + GOOD.replace("good", "alsogood")
        repaired, recovered, _unrecovered = repair_bibtex(source)
        assert [r.key for r in recovered] == ["repeated"]
        assert [e.key for e in bibtex.parse_string(repaired).entries] == ["good", "repeated", "alsogood"]
        assert repaired.startswith(GOOD) and repaired.endswith(GOOD.replace("good", "alsogood"))

    def test_a_record_broken_twice_over_is_reported_rather_than_half_repaired(self):
        # A ProQuest record whose author carries degrees inline. Merging the notes is not enough: BibTeX
        # gives a name at most two commas (`von Last, Jr, First`) and this one uses three, so the record
        # still does not parse and the honest outcome is to leave it and say why.
        source = REPEATED.replace("    year = {2022},\n",
                                  "    author = {Bloggs, PhD, MSc, Joan},\n    year = {2022},\n")
        repaired, recovered, unrecovered = repair_bibtex(source)
        assert recovered == []
        assert len(unrecovered) == 1 and unrecovered[0].key == "repeated"
        assert repaired == source

    def test_the_two_faults_are_told_apart_in_one_file(self):
        repaired, recovered, unrecovered = repair_bibtex(BROKEN + "\n" + REPEATED)
        assert unrecovered == []
        assert {r.key: r.kind for r in recovered} == {"broken": KIND_UNBALANCED_BRACES,
                                                      "repeated": KIND_DUPLICATE_FIELD_KEYS}
        assert len(bibtex.parse_string(repaired).entries) == 2

    def test_a_fault_the_tool_cannot_name_still_carries_the_parsers_own_words(self):
        # No field opens more braces than it closes, so the brace heuristic has nothing to offer and the
        # report falls back to what the parser said. Nothing here should invent a diagnosis.
        source = "@article{odd,\n    author = {Bloggs, PhD, MSc, Joan},\n    year = {2022},\n}\n"
        _repaired, recovered, unrecovered = repair_bibtex(source)
        assert recovered == []
        assert len(unrecovered) == 1
        assert unrecovered[0].kind == KIND_UNREADABLE
        assert "Too many commas" in unrecovered[0].detail
