"""Unit tests for raven.papers.pdf2bib (BibTeX from PDF conference abstracts, via an LLM)."""

import argparse
import types

import pytest
from unpythonic.env import env

# `pdf2bib` reaches the ML stack through `raven.client.api`, and CI does not install it.
pytest.importorskip("spacy")

from raven.librarian import sweep  # noqa: E402
from raven.papers import pdf2bib  # noqa: E402 -- after the importorskip guard


TEXT = """Proceedings of the Example Conference 2024

A Study of Something

Jane Doe, Richard Roe

We studied something, at length, and here is what we found about it.
"""


@pytest.fixture
def replies(monkeypatch):
    """Script the LLM: append the replies the next model calls should get, in order.

    Faked at `agent.ask_record`, below `extract` and `agent.ask_until`, so that the retries run for real.
    """
    queue = []

    def fake_ask_record(llm_settings, instruction, *args, **kwargs):
        return types.SimpleNamespace(reply=queue.pop(0), messages=[])

    monkeypatch.setattr(pdf2bib.agent, "ask_record", fake_ask_record)
    return queue


@pytest.fixture
def prompts():
    prompts, _material = pdf2bib.setup_prompts(llm_settings=None, n_retries=2, conference_year="2024",
                                               conference_booktitle="Example Conference")
    return prompts


def _context():
    return env(unique_id="example-1", maybe_abort=None, turns=[])


def run_field(prompts, field, text=TEXT):
    kind, function, _ = prompts[field]
    assert kind == "function"
    return function(_context(), "example-1", text)


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
        assert status is pdf2bib.status_flagged
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
        status, _, title = run_field(prompts, "title", text)
        assert status is pdf2bib.status_success
        assert title == expected

    @pytest.mark.parametrize("reply", ["NO TITLE", "No title.", "Abstract", "ABSTRACT.", "*Abstract*"])
    def test_no_title_omits_the_field_and_flags_the_entry(self, prompts, replies, reply):
        replies.append(reply)
        status, error_info, title = run_field(prompts, "title")
        assert status is pdf2bib.status_flagged
        assert title is None
        assert "no title" in error_info


class TestPrompts:
    def test_the_prompts_name_the_sentinels(self, prompts, monkeypatch):
        """A sentinel the prompt never asks for is one the model will never send."""
        seen = []

        def recording_ask_record(llm_settings, instruction, *args, **kwargs):
            seen.append(instruction)
            return types.SimpleNamespace(reply="NO AUTHORS LISTED" if len(seen) == 1 else "NO TITLE", messages=[])

        monkeypatch.setattr(pdf2bib.agent, "ask_record", recording_ask_record)
        run_field(prompts, "author")
        run_field(prompts, "title")
        assert len(seen) == 2
        assert "NO AUTHORS LISTED" in seen[0]
        assert "NO TITLE" in seen[1]


class TestNoAnswer:
    def test_a_blank_step_is_asked_again_and_then_given_up_on(self, prompts, replies):
        replies.extend(["", "   "])  # both attempts of EXTRACT AUTHORS
        with pytest.raises(pdf2bib._NoAnswer, match="EXTRACT AUTHORS"):
            run_field(prompts, "author")
        assert not replies, "the step was not asked again"

    def test_a_retry_that_answers_goes_on(self, prompts, replies):
        replies.extend(["", "Jane Doe, Richard Roe", "Jane Doe, Richard Roe", "Jane Doe and Richard Roe"])
        status, _, authors = run_field(prompts, "author")
        assert status is pdf2bib.status_success and authors == "Jane Doe and Richard Roe"


class TestExtractFields:
    @pytest.fixture(autouse=True)
    def no_dehyphenator(self, monkeypatch):
        # The real one is a Raven-server client; the abstract text here has no hyphenation to undo.
        monkeypatch.setattr(pdf2bib, "_get_dehyphenator", lambda: types.SimpleNamespace(dehyphenate=lambda text: text))

    def test_a_clean_document_is_answered_with_its_fields_and_no_literals(self, prompts, replies):
        replies.extend(["Jane Doe, Richard Roe", "Jane Doe, Richard Roe", "Jane Doe and Richard Roe",
                        "A Study of Something",
                        "We studied something, at length, and here is what we found about it."])
        answer, turns = pdf2bib.extract_fields(None, prompts, "example-1", TEXT)
        assert answer["status"] == "success" and answer["notes"] == ""
        assert answer["fields"] == {"author": "Jane Doe and Richard Roe", "title": "A Study of Something",
                                    "keywords": None,  # the text has no keywords line, so none is asked for
                                    "abstract": "We studied something, at length, and here is what we found about it."}
        assert len(turns) == 5, "every model call goes to the traces"

    def test_a_step_with_no_usable_reply_fails_the_document(self, prompts, replies):
        replies.extend(["", ""])
        answer, turns = pdf2bib.extract_fields(None, prompts, "example-1", TEXT)
        assert isinstance(answer, sweep.Failure) and "EXTRACT AUTHORS" in answer.reason
        assert len(turns) == 2, "the failed attempts are what explains the failure, so they are kept"

    def test_a_flagged_field_flags_the_document(self, prompts, replies):
        replies.extend(["NO AUTHORS LISTED", "A Study of Something", "We studied something."])
        answer, _ = pdf2bib.extract_fields(None, prompts, "example-1", TEXT)
        assert answer["status"] == "flagged" and "no author list" in answer["notes"]
        assert answer["fields"]["author"] is None


class TestOutputs:
    ANSWER = {"status": "success", "notes": "",
              "fields": {"author": "Jane Doe", "title": "A Study", "keywords": None, "abstract": "Text."}}

    def test_an_entry_takes_the_literals_and_leaves_out_what_is_none(self, prompts):
        entry = pdf2bib.format_entry(prompts, "EX2024", "example-1", self.ANSWER)
        assert entry.splitlines() == ["@incollection{EX2024-example-1,",
                                      "    author = {Jane Doe},",
                                      "    year = {2024},",
                                      "    title = {A Study},",
                                      "    booktitle = {Example Conference},",
                                      "    abstract = {Text.},",
                                      "}"]

    def _opts(self, tmp_path, failed=True):
        return argparse.Namespace(conference_slug="EX2024", output_dir=None, failed_output_dir=str(tmp_path),
                                  success_filename=str(tmp_path / "ok.bib"),
                                  failed_filename=str(tmp_path / "flagged.bib") if failed else None)

    def _ledger(self, tmp_path):
        ledger = sweep.Ledger(tmp_path, "pdf2bib-EX2024", "abcd1234")
        ledger.write_answer("b-doc", self.ANSWER)
        ledger.write_answer("a-doc", {**self.ANSWER, "status": "flagged", "notes": "looks wrong"})
        ledger.write_failure("c-doc", "EXTRACT AUTHORS: no usable reply in 2 attempts")
        return ledger

    def test_outputs_are_rewritten_from_the_whole_ledger(self, prompts, tmp_path):
        (tmp_path / "ok.bib").write_text("an older run's content, to be replaced\n")
        counts = pdf2bib.write_outputs(self._ledger(tmp_path), prompts, self._opts(tmp_path))
        assert (counts["success"], counts["flagged"]) == (1, 1)
        assert (tmp_path / "ok.bib").read_text().startswith("@incollection{EX2024-b-doc,")
        assert "older run" not in (tmp_path / "ok.bib").read_text()
        assert (tmp_path / "flagged.bib").read_text().startswith("@incollection{EX2024-a-doc,")
        assert (tmp_path / "a-doc_errors.txt").read_text() == "looks wrong"
        assert "c-doc" not in (tmp_path / "ok.bib").read_text() + (tmp_path / "flagged.bib").read_text()

    def test_without_a_flagged_file_flagged_entries_join_the_success_output(self, prompts, tmp_path):
        pdf2bib.write_outputs(self._ledger(tmp_path), prompts, self._opts(tmp_path, failed=False))
        text = (tmp_path / "ok.bib").read_text()
        assert "EX2024-a-doc" in text and "EX2024-b-doc" in text


class TestSorting:
    @staticmethod
    def _opts(tmp_path, done=True, flagged=True):
        dirs = {name: tmp_path / name for name in ("in", "done", "flagged")}
        for directory in dirs.values():
            directory.mkdir()
        return argparse.Namespace(input_dir=str(dirs["in"]),
                                  output_dir=str(dirs["done"]) if done else None,
                                  failed_output_dir=str(dirs["flagged"]) if flagged else None)

    def test_a_moved_pdf_is_still_found(self, tmp_path):
        opts = self._opts(tmp_path)
        (tmp_path / "in" / "a.pdf").write_bytes(b"")
        (tmp_path / "flagged" / "b.pdf").write_bytes(b"")
        items = dict(pdf2bib.collect_items([opts.input_dir], opts))
        assert items == {"a": str(tmp_path / "in" / "a.pdf"), "b": str(tmp_path / "flagged" / "b.pdf")}

    def test_the_same_name_twice_is_refused(self, tmp_path):
        opts = self._opts(tmp_path)
        (tmp_path / "in" / "a.pdf").write_bytes(b"")
        (tmp_path / "done" / "a.pdf").write_bytes(b"")
        with pytest.raises(ValueError, match="'a'"):
            pdf2bib.collect_items([opts.input_dir], opts)

    def test_a_cleared_document_moves_from_flagged_to_done(self, tmp_path):
        opts = self._opts(tmp_path)
        (tmp_path / "flagged" / "a.pdf").write_bytes(b"")
        where = pdf2bib.sort_pdf(str(tmp_path / "flagged" / "a.pdf"), "success", opts)
        assert where == str(tmp_path / "done" / "a.pdf") and (tmp_path / "done" / "a.pdf").exists()

    def test_without_a_done_directory_a_cleared_document_goes_back_to_the_input(self, tmp_path):
        opts = self._opts(tmp_path, done=False)
        (tmp_path / "flagged" / "a.pdf").write_bytes(b"")
        where = pdf2bib.sort_pdf(str(tmp_path / "flagged" / "a.pdf"), "success", opts)
        assert where == str(tmp_path / "in" / "a.pdf")

    def test_an_unsorted_pdf_with_nowhere_to_go_stays(self, tmp_path):
        opts = self._opts(tmp_path, done=False, flagged=False)
        (tmp_path / "in" / "a.pdf").write_bytes(b"")
        assert pdf2bib.sort_pdf(str(tmp_path / "in" / "a.pdf"), "flagged", opts) == str(tmp_path / "in" / "a.pdf")

    def test_a_cleared_document_s_old_notes_are_removed(self, prompts, tmp_path):
        opts = argparse.Namespace(conference_slug="EX2024", output_dir=None, failed_output_dir=str(tmp_path),
                                  success_filename=str(tmp_path / "ok.bib"), failed_filename=str(tmp_path / "flagged.bib"))
        (tmp_path / "a-doc_errors.txt").write_text("from when it was flagged")
        (tmp_path / "a-doc.bib").write_text("@incollection{old}")
        ledger = sweep.Ledger(tmp_path, "pdf2bib-EX2024", "abcd1234")
        ledger.write_answer("a-doc", TestOutputs.ANSWER)
        pdf2bib.write_outputs(ledger, prompts, opts)
        assert not (tmp_path / "a-doc_errors.txt").exists() and not (tmp_path / "a-doc.bib").exists()
