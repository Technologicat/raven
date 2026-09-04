"""Unit tests for raven.papers.siftbib (removing the records a review cannot screen)."""

import pathlib

import pytest

from raven.papers import bibtex, siftbib


def library(*entries: str):
    """Parse a `.bib` written inline, so each test's fixture is readable next to its assertions."""
    return bibtex.parse_string("\n\n".join(entries), split_names=False)


FULL = """@article{full_2024,
  title = {A study of something},
  author = {Doe, Jane},
  year = {2024},
  journal = {Journal of Studies},
  abstract = {We studied something, at length, and here is what we found about it.}
}"""

NO_ABSTRACT = """@article{bare_2024,
  title = {A study with no abstract},
  author = {Roe, Richard},
  year = {2024},
  journal = {Proceedings of the Learning Analytics Conference}
}"""

BLANK_ABSTRACT = """@article{blank_2024,
  title = {A study whose abstract is whitespace},
  author = {Poe, Pat},
  year = {2024},
  abstract = {   }
}"""

TEASER = """@article{teaser_2024,
  title = {A study the publisher cut off},
  author = {Moe, Morgan},
  year = {2024},
  abstract = {Given the circumstances ...}
}"""

NO_YEAR = """@article{noyear_2024,
  title = {A study with no year},
  author = {Loe, Lee},
  abstract = {An abstract long enough to be worth reading, which this one is not really.}
}"""


class TestCriteria:
    def test_a_missing_field_and_a_blank_one_are_the_same_answer(self):
        kept, dropped = siftbib.sift(library(FULL, NO_ABSTRACT, BLANK_ABSTRACT),
                                     [siftbib.require_field("abstract")])
        assert [entry.key for entry in kept.entries] == ["full_2024"]
        # A database that exported `abstract = {   }` has told us nothing, so it must not read as told.
        assert {record.key for record in dropped} == {"bare_2024", "blank_2024"}
        assert {record.reason for record in dropped} == {"no abstract"}

    def test_min_chars_catches_the_field_that_is_present_and_useless(self):
        criteria = [siftbib.require_field("abstract")]
        kept_by_presence, _ = siftbib.sift(library(FULL, TEASER), criteria)
        # The negative control: `require_field` alone cannot tell these two apart, so a fixture that only
        # ever ran the length test would pass for the wrong reason.
        assert [entry.key for entry in kept_by_presence.entries] == ["full_2024", "teaser_2024"], (
            "the teaser has no abstract at all, so this fixture cannot show that length is what separates them")

        kept, dropped = siftbib.sift(library(FULL, TEASER), criteria + [siftbib.min_chars("abstract", 40)])
        assert [entry.key for entry in kept.entries] == ["full_2024"]
        assert dropped[0].reason == "abstract shorter than 40 characters"

    def test_criteria_compose_and_a_record_is_dropped_on_the_first_it_fails(self):
        kept, dropped = siftbib.sift(library(FULL, NO_ABSTRACT, NO_YEAR),
                                     [siftbib.require_field("abstract"), siftbib.require_field("year")])
        assert [entry.key for entry in kept.entries] == ["full_2024"]
        by_key = {record.key: record.reason for record in dropped}
        assert by_key == {"bare_2024": "no abstract", "noyear_2024": "no year"}

    def test_no_criteria_keeps_everything(self):
        # The CLI refuses this case, where it can say why; the library answers it honestly instead of
        # inventing a default opinion about what a usable record is.
        kept, dropped = siftbib.sift(library(FULL, NO_ABSTRACT), [])
        assert len(kept.entries) == 2
        assert dropped == []

    def test_the_audit_carries_what_a_reviewer_needs_to_chase_the_record(self):
        _kept, dropped = siftbib.sift(library(NO_ABSTRACT), [siftbib.require_field("abstract")])
        record = dropped[0]
        assert record.key == "bare_2024"
        assert record.title == "A study with no abstract"
        # The venue is what tells a reviewer whether a dropped record is worth going after by hand.
        assert record.venue == "Proceedings of the Learning Analytics Conference"


class TestParseMinChars:
    @pytest.mark.parametrize("spec, field, ok_length, bad_length", [
        ("abstract=600", "abstract", 600, 599),
        ("title=5", "title", 5, 4),
    ])
    def test_it_reads_a_field_and_a_length(self, spec, field, ok_length, bad_length):
        criterion = siftbib.parse_min_chars(spec)
        entry = library(f"@article{{k, {field} = {{{'x' * ok_length}}} }}").entries[0]
        short = library(f"@article{{k, {field} = {{{'x' * bad_length}}} }}").entries[0]
        assert criterion.holds(entry)
        assert not criterion.holds(short)

    @pytest.mark.parametrize("spec", ["abstract", "abstract=", "=600", "abstract=lots", "abstract=-1"])
    def test_a_malformed_spec_is_refused_rather_than_guessed_at(self, spec):
        with pytest.raises(ValueError):
            siftbib.parse_min_chars(spec)


class TestWriteAudit:
    def test_the_header_says_which_tool_on_what_input_and_by_what_test(self, tmp_path: pathlib.Path):
        criteria = [siftbib.require_field("abstract")]
        _kept, dropped = siftbib.sift(library(NO_ABSTRACT), criteria)
        path = tmp_path / "removed.tsv"
        siftbib.write_audit(path, dropped, ["corpus.bib"], criteria)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("# raven-siftbib ")
        assert "corpus.bib" in lines[1]
        assert "has a non-empty `abstract`" in lines[2]
        assert "records removed: 1" in lines[3]
        assert lines[4].split("\t") == list(siftbib.AUDIT_COLUMNS)
        assert lines[5].split("\t")[:2] == ["bare_2024", "no abstract"]


LATEX_TITLE = r"""@article{latex_2024,
  title = {{\o}nly {Tr{\c e}bicki} and the {AutoPBL} caf\'e},
  author = {Zoe, Zed},
  year = {2024},
  journal = {Journal of {\"O}stberg Studies}
}"""


class TestTheAuditIsReadable:
    """What a person sees in the audit, which is the only thing the title and venue columns are for."""

    def test_latex_in_a_title_is_resolved_rather_than_stripped(self):
        _kept, dropped = siftbib.sift(library(LATEX_TITLE), [siftbib.require_field("abstract")])
        title = dropped[0].title

        # The negative control, and the reason this test exists: stripping the braces by hand — which is
        # what stood here — gets the ordinary case right and these wrong, because the braces are what
        # terminate a LaTeX command. A fixture carrying no ligature or accent cannot tell the two
        # treatments apart, and would pass against either.
        raw = siftbib._field_text(library(LATEX_TITLE).entries[0], "title")
        naive = " ".join(raw.replace("{", "").replace("}", "").split())
        assert naive != title, "this fixture carries no markup that the naive strip mangles"

        assert title == "ønly Trȩbicki and the AutoPBL café"
        assert "\\" not in title, f"a LaTeX command survived into the audit: {title!r}"

    def test_the_venue_is_resolved_too(self):
        _kept, dropped = siftbib.sift(library(LATEX_TITLE), [siftbib.require_field("abstract")])
        assert dropped[0].venue == "Journal of Östberg Studies"


class TestTheFileSurvivesTheSifting:
    def test_what_is_kept_is_written_back_as_readable_bibtex(self):
        kept, _dropped = siftbib.sift(library(FULL, NO_ABSTRACT), [siftbib.require_field("abstract")])
        reparsed = bibtex.parse_string(bibtex.write_string(kept), split_names=False)
        assert [entry.key for entry in reparsed.entries] == ["full_2024"]
        assert reparsed.entries[0].fields_dict["title"].value == "A study of something"

    def test_a_comment_between_records_is_not_collateral_damage(self):
        # Filtering is at block level rather than rebuilt from `library.entries`, so whatever else the
        # file carries stays in it. A caller's own `@comment` is theirs, and this tool was asked to
        # remove records.
        source = library(FULL, "@comment{ a note the owner left themselves }", NO_ABSTRACT)
        kept, _dropped = siftbib.sift(source, [siftbib.require_field("abstract")])
        assert "a note the owner left themselves" in bibtex.write_string(kept)
