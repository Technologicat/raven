"""Tests for `raven.librarian.lorebook`: the lorebook of a running app or script, and the real glossary.

The core's behaviour — parsing, matching, choosing what fires, lookup — is tested by the conformance cases in
`lorebook_conformance.json`, through `test_lorebookcore.py`. What stays here is what cannot be data.
"""

import pathlib
import time

import pytest
from unpythonic.env import env

from raven.librarian import llmclient
from raven.librarian import llmtools
from raven.librarian import lorebook


def _by_id(entries) -> dict[str, lorebook.Entry]:
    return {entry.id: entry for entry in entries}


class TestLoad:
    def test_a_missing_directory_is_an_empty_lorebook(self, tmp_path):
        assert lorebook.load(tmp_path / "nope") == ({}, {})

    def test_a_path_may_be_str_or_path(self, tmp_path):
        (tmp_path / "a.md").write_text("## Jazz\nImprovised.\n", encoding="utf-8")
        assert lorebook.load(str(tmp_path)) == lorebook.load(tmp_path)


class TestLorebook:
    @pytest.fixture
    def settings(self):
        model_info = env(label="m", model_id="m", context_length=4096, is_vlm=False, loaded=True)
        return llmclient.configure(model_info=model_info, backend_flavor="generic", backend_url="http://x", quiet=True)

    @pytest.fixture
    def folder(self, tmp_path):
        folder = tmp_path / "lorebook"
        folder.mkdir()
        (folder / "glossary.md").write_text("## Spork\nA utensil.\n\n## Fup\nNot a typo.\n", encoding="utf-8")
        return folder

    @staticmethod
    def _wait_until(predicate, timeout=10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def test_the_entries_and_the_tool_are_in_the_settings(self, settings, folder):
        with lorebook.loaded(settings, directory=folder) as book:
            assert list(book.entries) == ["spork", "fup"]
            assert list(settings.lorebook_entries) == ["spork", "fup"]
            assert settings.tool_entrypoints[llmtools.LOREBOOK_TOOL_NAME](term="spork") == "Spork\n\nA utensil."
            assert llmtools.LOREBOOK_TOOL_NAME in {spec["function"]["name"] for spec in settings.tools}
        assert llmtools.LOREBOOK_TOOL_NAME not in settings.tool_entrypoints, "leaving the block removes the tool"
        assert settings.lorebook_entries == {}

    def test_an_empty_lorebook_offers_no_tool(self, settings, tmp_path):
        with lorebook.loaded(settings, directory=tmp_path / "nope") as book:
            assert book.entries == {} and llmtools.LOREBOOK_TOOL_NAME not in settings.tool_entrypoints

    def test_watching_creates_a_missing_folder_whose_readme_is_no_entry(self, settings, tmp_path):
        folder = tmp_path / "lorebook"
        with lorebook.loaded(settings, directory=folder):
            assert not folder.exists(), "a script neither creates nor follows the folder"
        book = lorebook.Lorebook(settings, directory=folder)
        book.start(watch=True)
        try:
            assert (folder / "README.md").read_text(encoding="utf-8").startswith("# Put your lorebook entries here")
            assert book.entries == {} and not book.problems
        finally:
            book.shutdown()

    def test_an_entry_added_while_watching_is_offered_and_announced(self, settings, folder):
        calls = []
        book = lorebook.Lorebook(settings, directory=folder, on_changed=lambda: calls.append(1))
        book.start(watch=True)
        try:
            assert llmtools.LOREBOOK_TOOL_NAME in settings.tool_entrypoints
            (folder / "jazz.md").write_text("# Jazz\nImprovised.\n", encoding="utf-8")
            assert self._wait_until(lambda: "jazz" in settings.lorebook_entries)
            assert self._wait_until(lambda: calls)
            assert settings.tool_entrypoints[llmtools.LOREBOOK_TOOL_NAME](term="jazz") == "Jazz\n\nImprovised."
        finally:
            book.shutdown()

    def test_an_edited_definition_is_read_again_without_an_announcement(self, settings, folder):
        calls = []
        book = lorebook.Lorebook(settings, directory=folder, on_changed=lambda: calls.append(1))
        book.start(watch=True)
        try:
            (folder / "glossary.md").write_text("## Spork\nA fork that is also a spoon.\n\n## Fup\nNot a typo.\n", encoding="utf-8")
            assert self._wait_until(lambda: settings.lorebook_entries["spork"].content == "A fork that is also a spoon.")
            assert calls == [], "the same entries, so nothing for a frontend to update"
        finally:
            book.shutdown()

    def test_a_script_does_not_follow_the_folder(self, settings, folder):
        with lorebook.loaded(settings, directory=folder):
            (folder / "jazz.md").write_text("# Jazz\nImprovised.\n", encoding="utf-8")
            time.sleep(0.5)
            assert "jazz" not in settings.lorebook_entries

    def test_the_gate_withholds_the_tool(self, settings, folder):
        with lorebook.loaded(settings, directory=folder):
            gated = dict(documents_available=True, internet_available=True, mcp_available=True, skills_available=True)
            assert llmtools.maybe_tool_names_for_turn(settings, **gated, lorebook_available=True) is None
            offered = llmtools.maybe_tool_names_for_turn(settings, **gated, lorebook_available=False)
            assert llmtools.LOREBOOK_TOOL_NAME not in offered and "websearch" in offered

    def test_the_tool_is_costed_in_its_own_group_while_it_is_offered(self, settings, folder):
        assert llmclient.tool_cost_by_group(settings)["lorebook"] == 0, "no lorebook, so nothing to cost"
        with lorebook.loaded(settings, directory=folder):
            assert llmclient.tool_cost_by_group(settings)["lorebook"] > 0

    def test_a_path_may_be_str_or_path(self, settings, folder):
        with lorebook.loaded(settings, directory=str(folder)) as book:
            assert list(book.entries) == ["spork", "fup"]


@pytest.mark.skipif(not (pathlib.Path.home() / "Documents/koodit/substrate-independent/glossary.md").exists(),
                    reason="the real glossary is on the maintainer's machines only")
def test_the_real_glossary_parses_and_its_links_resolve():
    path = pathlib.Path.home() / "Documents/koodit/substrate-independent/glossary.md"
    entries = lorebook.parse(path.read_text(encoding="utf-8"), name="glossary", source=path)
    assert len(entries) > 50, "the letter sections or the title were taken for entries, or the entries were missed"
    anchors = {anchor for entry in entries for anchor in entry.anchors}
    assert {crossref for entry in entries for crossref in entry.crossrefs} <= anchors


@pytest.mark.skipif(not (pathlib.Path.home() / "Documents/koodit/substrate-independent/glossary.md").exists(),
                    reason="the real glossary is on the maintainer's machines only")
def test_on_the_real_glossary_a_typo_finds_its_entry_and_a_search_by_meaning_finds_none():
    """The near-miss cutoff against real data: typos of every title, and the terms a model tried by meaning."""
    path = pathlib.Path.home() / "Documents/koodit/substrate-independent/glossary.md"
    entries = _by_id(lorebook.parse(path.read_text(encoding="utf-8"), name="glossary", source=path))
    wrong = []
    for entry in entries.values():
        if len(entry.title) >= 4:
            typo = entry.title[:len(entry.title) // 2] + entry.title[len(entry.title) // 2 + 1:]
            found = lorebook.lookup(entries, typo)
            if found.maybe_entry is not None and found.maybe_entry.id != entry.id:
                wrong.append((typo, found.maybe_entry.title))
            elif found.maybe_entry is None and entry.title not in found.candidates:
                wrong.append((typo, None))
    assert wrong == [], "a one-letter typo of a title missed its entry or found another"
    # Asked by Flash Next looking for "endognosis" by meaning, on 2026-11-19.
    for query in ["introspection", "interoception", "metacognition", "meta-attention", "proprioception"]:
        assert lorebook.lookup(entries, query).maybe_entry is None, f"{query!r} was taken for an entry it is not"
