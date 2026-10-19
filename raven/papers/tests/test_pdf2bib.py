"""Unit tests for raven.papers.pdf2bib (BibTeX from PDF conference abstracts, via an LLM)."""

import types

import pytest

# `pdf2bib` reaches the ML stack through `raven.client.api`, and CI does not install it.
pytest.importorskip("spacy")

from raven.papers import pdf2bib  # noqa: E402 -- after the importorskip guard


TEXT = """Proceedings of the Example Conference 2024

A Study of Something

Jane Doe, Richard Roe

We studied something, at length, and here is what we found about it.
"""


@pytest.fixture
def replies(monkeypatch):
    """Script the LLM: append the replies the next extraction steps should get, in order."""
    queue = []

    def fake_extract(llm_settings, instruction, progress_symbol):
        return types.SimpleNamespace(reply=queue.pop(0), messages=[])

    monkeypatch.setattr(pdf2bib, "extract", fake_extract)
    return queue


@pytest.fixture
def prompts():
    return pdf2bib.setup_prompts(llm_settings=None, n_retries=1, conference_year="2024")


def run_field(prompts, field):
    kind, function, _ = prompts[field]
    assert kind == "function"
    return function("example-1", TEXT)


class TestAuthors:
    def test_an_author_list_is_extracted(self, prompts, replies):
        replies.extend(["Jane Doe, Richard Roe", "Jane Doe, Richard Roe", "Jane Doe and Richard Roe"])
        status, _, authors = run_field(prompts, "author")
        assert status is pdf2bib.status_success
        assert authors == "Jane Doe and Richard Roe"

    @pytest.mark.parametrize("reply", ["NO AUTHORS LISTED", "No authors listed.", "**No authors listed**"])
    def test_no_authors_omits_the_field_and_flags_the_entry(self, prompts, replies, reply):
        replies.append(reply)
        status, error_info, authors = run_field(prompts, "author")
        assert status is pdf2bib.status_failed
        assert authors is None
        assert "no author list" in error_info
        assert not replies, "the later author steps should not run on the sentinel"


class TestTitle:
    @pytest.mark.parametrize("reply, expected", [("A Study of Something", "A Study of Something"),
                                                 ('"A Study of Something."', "A Study of Something"),
                                                 # A title that merely starts with the word is a title.
                                                 ("Abstract Interpretation of Something", "Abstract Interpretation of Something")])
    def test_a_title_is_extracted(self, prompts, replies, reply, expected):
        text = TEXT.replace("A Study of Something", expected)
        replies.append(reply)
        kind, function, _ = prompts["title"]
        status, _, title = function("example-1", text)
        assert status is pdf2bib.status_success
        assert title == expected

    @pytest.mark.parametrize("reply", ["NO TITLE", "No title.", "Abstract", "ABSTRACT.", "*Abstract*"])
    def test_no_title_omits_the_field_and_flags_the_entry(self, prompts, replies, reply):
        replies.append(reply)
        status, error_info, title = run_field(prompts, "title")
        assert status is pdf2bib.status_failed
        assert title is None
        assert "no title" in error_info


class TestPrompts:
    def test_the_prompts_name_the_sentinels(self, prompts, monkeypatch):
        """A sentinel the prompt never asks for is one the model will never send."""
        seen = []

        def recording_extract(llm_settings, instruction, progress_symbol):
            seen.append(instruction)
            return types.SimpleNamespace(reply="NO AUTHORS LISTED" if len(seen) == 1 else "NO TITLE", messages=[])

        monkeypatch.setattr(pdf2bib, "extract", recording_extract)
        run_field(prompts, "author")
        run_field(prompts, "title")
        assert len(seen) == 2
        assert "NO AUTHORS LISTED" in seen[0]
        assert "NO TITLE" in seen[1]
