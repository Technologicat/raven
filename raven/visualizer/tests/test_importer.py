"""Unit tests for raven.visualizer.importer.

The first tests in this package. `importer` is reachable from pytest because everything expensive in it
is loaded lazily -- the LLM connection is set up at import time only when the config asks for cluster
keywords or summaries, and the NLP and embedding models are loaded on first use -- so `_parse_input_files`
can be exercised against a `.bib` written into `tmp_path`, with the one remote service it touches
(dehyphenation) replaced.
"""

import itertools

import pytest

# `raven.visualizer.importer` reaches sklearn, torch and spaCy, none of which CI installs -- and a
# module-level import failure is a *collection* error rather than a skip, so it would turn the matrix
# red rather than quietly sitting out. Guarding on the module itself rather than on a list of packages
# keeps this correct as the import chain changes. `scripts/check_ci_imports.py` is what reports it.
importer = pytest.importorskip("raven.visualizer.importer")

pytestmark = pytest.mark.ml

from raven.visualizer import config as visualizer_config  # noqa: E402 -- must follow the guard above


TWO_RECORDS = """
@article{alpha2024,
  author = {Alpha, Anna},
  year = {2024},
  title = {A first paper about something},
  abstract = {An abstract that mentions auto-\nmatic hyphenation.}
}

@article{beta2024,
  author = {Beta, Bob},
  year = {2024},
  title = {A second paper about something else},
  abstract = {A second abstract, with no hyphenation in it at all.}
}
"""


class ExplodingDehyphenator:
    """Stands in for `mayberemote.Dehyphenator`, failing the way a real one has been seen to."""
    def __init__(self, *args, **kwargs):
        pass

    def dehyphenate(self, text):
        raise RuntimeError("dehyphenation exploded")


@pytest.fixture
def two_record_bib(tmp_path):
    path = tmp_path / "two_records.bib"
    path.write_text(TWO_RECORDS, encoding="utf-8")
    return path


def parsed_entries(input_data):
    """Flatten `_parse_input_files` output into one list, as `import_bibtex` does."""
    return list(itertools.chain.from_iterable(input_data.parsed_data_by_filename.values()))


def test_parse_input_files_reads_both_records(two_record_bib, monkeypatch):
    # Negative control for the test below: with dehyphenation off, nothing can throw, so a fixture that
    # only ever ran this way could not tell a surviving import from a lucky one.
    monkeypatch.setattr(visualizer_config, "dehyphenate", False)
    entries = parsed_entries(importer._parse_input_files(str(two_record_bib)))
    assert len(entries) == 2
    assert entries[0].title == "A first paper about something"
    assert entries[1].title == "A second paper about something else"


def test_parse_input_files_survives_a_failing_dehyphenator(two_record_bib, monkeypatch, caplog):
    # One malformed abstract used to abort the whole import, discarding every record already processed --
    # an hour's work on a large bibliography, with no way to skip the offending record. Dehyphenation is
    # cosmetic, so a failure now costs that one abstract its tidying and nothing else.
    monkeypatch.setattr(visualizer_config, "dehyphenate", True)
    monkeypatch.setattr(importer.mayberemote, "Dehyphenator", ExplodingDehyphenator)

    with caplog.at_level("WARNING"):
        entries = parsed_entries(importer._parse_input_files(str(two_record_bib)))

    # Both records survive: the run continues past the failure rather than stopping at it.
    assert len(entries) == 2, "a failing dehyphenator must not cost us any records"
    # The record that failed keeps its abstract, untidied rather than dropped.
    assert entries[0].abstract is not None
    assert "hyphenation" in entries[0].abstract
    # The failure is reported rather than swallowed, and names the record so it can be found.
    assert any("alpha2024" in record.message for record in caplog.records), \
        f"the warning should name the offending entry; got {[r.message for r in caplog.records]}"


def test_parse_input_files_skips_a_record_that_fails_anywhere_else(two_record_bib, monkeypatch, caplog):
    # The guard above is specific to dehyphenation, which is cosmetic. This one is the general case:
    # anything else that throws costs its own record and lets the run continue. Bibliographies arrive
    # from exporters we do not control, so the point is not this particular failure but that no single
    # record can end the import.
    monkeypatch.setattr(visualizer_config, "dehyphenate", False)

    real_format_authors = importer.common_utils.format_bibtex_authors

    def explode_on_alpha(author_field):
        # `fields["author"].value` is bibtexparser's parsed name list, not a string, so match on its
        # repr rather than on the value -- checking `"Alpha" in author_field` silently matches nothing
        # and the test then passes against the unfixed code, having exercised no failure at all.
        if "Alpha" in str(author_field):
            raise ValueError("author field exploded")
        return real_format_authors(author_field)

    monkeypatch.setattr(importer.common_utils, "format_bibtex_authors", explode_on_alpha)

    with caplog.at_level("WARNING"):
        entries = parsed_entries(importer._parse_input_files(str(two_record_bib)))

    # The good record survives, so the failure did not end the run. Note this is the assertion that
    # distinguishes "skipped it" from "crashed": both leave the bad record out.
    assert len(entries) == 1, "the record after the failing one must still be parsed"
    assert entries[0].title == "A second paper about something else"
    assert any("beta2024" not in record.message and "alpha2024" in record.message
               for record in caplog.records), \
        f"the warning should name the skipped entry; got {[r.message for r in caplog.records]}"
