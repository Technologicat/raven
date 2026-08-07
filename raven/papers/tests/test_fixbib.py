"""Tests for `raven.papers.fixbib` — repairing BibTeX records whose braces do not balance."""

import logging

from ..fixbib import repair_bibtex
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


class TestRepairBibtex:
    def test_a_sound_file_is_returned_unchanged(self):
        repaired, recovered, unrecovered = repair_bibtex(GOOD)
        assert repaired == GOOD
        assert recovered == [] and unrecovered == []

    def test_a_broken_record_is_recovered_whole(self):
        repaired, recovered, unrecovered = repair_bibtex(BROKEN)
        assert recovered == ["broken"] and unrecovered == []
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
        assert recovered == ["broken"]
        assert [entry.key for entry in bibtex.parse_string(repaired).entries] == ["good", "broken", "alsogood"]

    def test_a_record_missing_its_terminator_is_reported_not_invented(self):
        # Nothing says where the absent brace belonged, so the honest outcome is to leave it and say so.
        source = "@article{hopeless,\n    abstract = {an abstract that simply never closes\n}\n"
        repaired, recovered, unrecovered = repair_bibtex(source)
        assert recovered == []
        assert len(unrecovered) == 1 and "hopeless" in unrecovered[0]
        assert repaired == source

    def test_a_line_boundary_that_is_not_a_newline_does_not_misplace_the_repair(self):
        # `str.splitlines` breaks on `\x1c` and friends while a newline count does not, so a file carrying
        # one drifts the two apart. A bibliography extracted from PDFs carries them; ECCOMAS 2024 has six.
        source = GOOD.replace("2024", "2024\x1c") + "\n" + BROKEN
        repaired, recovered, _unrecovered = repair_bibtex(source)
        assert recovered == ["broken"]
        assert "\x1c" in repaired
        assert repaired.replace("\\{", "{").replace("\\}", "}") == source

    def test_the_reported_line_number_points_at_the_record(self):
        source = GOOD + "\n" + "@article{hopeless,\n    abstract = {never closes\n}\n"
        _repaired, _recovered, unrecovered = repair_bibtex(source)
        line = int(unrecovered[0].split("at line ")[1].split(",")[0])
        assert source.splitlines()[line - 1].startswith("@article{hopeless")
