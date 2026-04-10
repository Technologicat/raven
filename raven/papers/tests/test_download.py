"""Tests for the arXiv download tool — BibTeX ID extraction and deduplication."""

import textwrap

from raven.papers.download import extract_ids_from_bib
from raven.papers.utils import deduplicate_arxiv_ids


def test_extracts_eprint_fields(tmp_path):
    bib = tmp_path / "test.bib"
    bib.write_text(textwrap.dedent("""\
        @article{Smith_2021_2103.12345,
            author = {Alice Smith},
            title = {A Paper},
            eprint = {2103.12345},
            archiveprefix = {arXiv},
        }
        @article{Jones_2022_2205.67890,
            author = {Bob Jones},
            title = {Another Paper},
            eprint = {2205.67890},
            archiveprefix = {arXiv},
        }
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2103.12345", "2205.67890"]


def test_skips_entries_without_eprint(tmp_path):
    bib = tmp_path / "test.bib"
    bib.write_text(textwrap.dedent("""\
        @article{nopreprint,
            author = {Carol},
            title = {No Preprint},
            journal = {Some Journal},
        }
        @article{haspreprint,
            author = {Dave},
            title = {Has Preprint},
            eprint = {2301.00001},
            archiveprefix = {arXiv},
        }
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2301.00001"]


def test_skips_non_arxiv_eprints(tmp_path):
    bib = tmp_path / "test.bib"
    bib.write_text(textwrap.dedent("""\
        @article{ssrn_paper,
            author = {Eve},
            title = {SSRN Paper},
            eprint = {12345},
            archiveprefix = {SSRN},
        }
        @article{arxiv_paper,
            author = {Frank},
            title = {arXiv Paper},
            eprint = {2401.99999},
            archiveprefix = {arXiv},
        }
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2401.99999"]


def test_accepts_missing_archiveprefix(tmp_path):
    """An eprint without archiveprefix is assumed to be arXiv."""
    bib = tmp_path / "test.bib"
    bib.write_text(textwrap.dedent("""\
        @article{test,
            author = {Grace},
            title = {Implicit arXiv},
            eprint = {2501.11111},
        }
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2501.11111"]


def test_empty_bib_returns_empty(tmp_path):
    bib = tmp_path / "test.bib"
    bib.write_text("")
    ids = extract_ids_from_bib(str(bib))
    assert ids == []


def test_deduplicates_versions_in_bib(tmp_path):
    """When a .bib has the same paper in multiple versions, keep the highest."""
    bib = tmp_path / "test.bib"
    bib.write_text(textwrap.dedent("""\
        @article{old,
            eprint = {2103.12345v1},
            archiveprefix = {arXiv},
        }
        @article{new,
            eprint = {2103.12345v3},
            archiveprefix = {arXiv},
        }
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2103.12345v3"]


# ---------------------------------------------------------------------------
# deduplicate_arxiv_ids (unit tests for the utility)
# ---------------------------------------------------------------------------

class TestDeduplicateArxivIds:
    def test_keeps_highest_version(self):
        assert deduplicate_arxiv_ids(["2103.12345v1", "2103.12345v3", "2103.12345v2"]) == ["2103.12345v3"]

    def test_bare_id_treated_as_v1(self):
        assert deduplicate_arxiv_ids(["2103.12345", "2103.12345v2"]) == ["2103.12345v2"]

    def test_preserves_order_of_first_occurrence(self):
        result = deduplicate_arxiv_ids(["2205.00001", "2103.12345v2", "2205.00001v3"])
        assert result == ["2205.00001v3", "2103.12345v2"]

    def test_no_duplicates(self):
        ids = ["2103.12345", "2205.67890", "2301.00001"]
        assert deduplicate_arxiv_ids(ids) == ids

    def test_empty(self):
        assert deduplicate_arxiv_ids([]) == []

    def test_single(self):
        assert deduplicate_arxiv_ids(["2103.12345v2"]) == ["2103.12345v2"]
