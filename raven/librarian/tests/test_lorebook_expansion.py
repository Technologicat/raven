"""Tests for `raven.librarian.lorebook_expansion`: asking for the terms' forms, caching them, and keeping them current.

The model is faked at `agent.ask_record`; the sweep and its ledger are the real ones, writing under `tmp_path`.
"""

import json
import re
import threading
import time

import pytest

from raven.librarian import agent
from raven.librarian import llmclient
from raven.librarian import lorebook
from raven.librarian import lorebook_expansion


class _Turn:
    """Stands in for an `agent.TurnRecord`: the reply, and the `to_dict` the ledger's traces file wants."""
    def __init__(self, reply: str):
        self.reply = reply

    def to_dict(self):
        return {"reply": self.reply}


class FakeModel:
    """Answers the expansion prompt, in English: each word's forms are the word plus "s", and plus "ed".

    `stems`: also offer the word itself as a stem, which a model might do whatever it was asked.
    `garbled`: reply with something that is not JSON.
    `finnish`: also answer in Finnish, unasked or not: the word plus "issa", and the word as a stem.
    `fail_first`: reply with something that is not JSON to this many calls first.
    `truncate`: end each reply right after its first term, mid-way through the second, as a reply that ran on
                and stopped does.
    """
    def __init__(self, *, stems: bool = False, garbled: bool = False, finnish: bool = False, fail_first: int = 0,
                 truncate: bool = False):
        self.stems = stems
        self.garbled = garbled
        self.finnish = finnish
        self.fail_first = fail_first
        self.truncate = truncate
        self.prompts: list[str] = []

    def terms_asked(self) -> list[str]:
        return [term for prompt in self.prompts for term in re.findall(r"^\d+\. (.*?) — words:", prompt, re.MULTILINE)]

    def __call__(self, settings, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        if self.garbled or len(self.prompts) <= self.fail_first:
            return _Turn("I am sorry, I cannot do that.")
        reply = {}
        for number, words in re.findall(r"^(\d+)\. .*? — words: (.*)$", prompt, re.MULTILINE):
            reply[number] = {}
            for word in words.split(", "):
                by_language = {"English": {"forms": [word + "s", word + "ed"], "stems": [word] if self.stems else []}}
                if self.finnish:
                    by_language["Finnish"] = {"forms": [word + "issa"], "stems": [word]}
                reply[number][word] = by_language
        text = json.dumps(reply)
        if self.truncate and len(reply) > 1:
            second = text.index('"2":')
            text = text[:second + 20]  # into the second term, which is then never closed
        return _Turn(text)


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    directory = tmp_path / "expansion"
    monkeypatch.setattr(lorebook_expansion.librarian_config, "lorebook_expansion_dir", directory)
    return directory

@pytest.fixture
def model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(agent, "ask_record", fake)
    return fake

@pytest.fixture
def folder(tmp_path):
    folder = tmp_path / "lorebook"
    folder.mkdir()
    (folder / "glossary.md").write_text("## Hydra tax\nClosing one opens two.\n\n## Spork (sporks)\nA utensil.\n",
                                        encoding="utf-8")
    return folder

def _wait_until(predicate, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestExpansionsFrom:
    ENGLISH = {"English": "forms"}
    BOTH = {"English": "forms", "Finnish": "stems"}

    def test_a_word_in_several_terms_gets_what_all_of_them_gave_it(self):
        answers = {"Hydra tax": {"tax": {"English": {"forms": ["taxes"], "stems": []}}},
                   "Tax day": {"tax": {"English": {"forms": ["taxed"], "stems": []}}}}
        assert lorebook_expansion.expansions_from(answers, self.ENGLISH)["tax"].forms == ("taxes", "taxed")

    def test_every_configured_language_is_read_and_no_other(self):
        answers = {"Kaupunki": {"kaupunki": {"English": {"forms": ["kaupunkis"], "stems": []},
                                             "Finnish": {"forms": ["kaupungit"], "stems": ["kaupung"]}}}}
        assert lorebook_expansion.expansions_from(answers, self.ENGLISH)["kaupunki"].forms == ("kaupunkis",)
        both = lorebook_expansion.expansions_from(answers, self.BOTH)["kaupunki"]
        assert both.forms == ("kaupunkis", "kaupungit") and both.stems == ("kaupung",)

    def test_stems_come_only_from_a_language_that_asks_for_them(self):
        answers = {"Kaupunki": {"kaupunki": {"English": {"forms": [], "stems": ["kaupunk"]},
                                             "Finnish": {"forms": [], "stems": ["kaupung"]}}}}
        assert lorebook_expansion.expansions_from(answers, self.BOTH)["kaupunki"].stems == ("kaupung",)

    def test_a_word_left_with_nothing_is_left_out(self):
        answers = {"Fup": {"fup": {"Finnish": {"forms": ["fup"], "stems": ["f"]}}}}
        assert lorebook_expansion.expansions_from(answers, self.BOTH) == {}


class TestSalvage:
    def test_the_complete_entries_of_a_reply_that_ran_on_are_kept(self):
        text = '{"1": {"a": {"English": {"forms": ["as"]}}}, "2": {"b": {"English": {"forms": ["b", "b", "b'
        assert lorebook_expansion.salvage(text) == {"1": {"a": {"English": {"forms": ["as"]}}}}

    def test_a_fence_or_a_preamble_before_the_object_is_skipped(self):
        assert lorebook_expansion.salvage('Here:\n```json\n{"1": [1], "2": [2') == {"1": [1]}

    def test_nothing_complete_is_nothing(self):
        assert lorebook_expansion.salvage("I cannot do that.") == {}
        assert lorebook_expansion.salvage('{"1": {"a":') == {}


class TestMaterial:
    def test_the_languages_are_in_the_question(self, llm_settings):
        text = lorebook_expansion.material(llm_settings, {"English": "forms", "Finnish": "stems"})
        assert text.startswith(llm_settings.lorebook_expansion_instruction)
        assert "- English: forms\n- Finnish: forms and stems" in text

    def test_an_unknown_kind_is_refused(self, llm_settings):
        with pytest.raises(ValueError, match="Finnish"):
            lorebook_expansion.material(llm_settings, {"Finnish": "lemmas"})


class TestExpander:
    def _start(self, llm_settings, folder):
        book = lorebook.Lorebook(llm_settings, directory=folder)
        book.start()
        expander = lorebook_expansion.Expander(llm_settings, book)
        expander.start()
        return book, expander

    def test_the_forms_arrive_and_an_inflected_term_fires(self, llm_settings, folder, ledger_dir, model):
        book, expander = self._start(llm_settings, folder)
        try:
            assert _wait_until(lambda: "tax" in book.expansions)
            assert book.expansions["tax"].forms == ("taxs", "taxed")
            assert set(lorebook.find_matches(book.entries, "a hydra taxed twice",
                                             expansions=llm_settings.lorebook_expansions)) == {"hydra-tax"}
            assert sorted(model.terms_asked()) == ["Hydra tax", "Spork", "sporks"]
        finally:
            expander.shutdown()
            book.shutdown()

    def test_what_was_answered_is_not_asked_again(self, llm_settings, folder, ledger_dir, model):
        book, expander = self._start(llm_settings, folder)
        assert _wait_until(lambda: "tax" in book.expansions)
        expander.shutdown()
        book.shutdown()
        asked = len(model.prompts)

        book, expander = self._start(llm_settings, folder)
        try:
            assert "tax" in book.expansions, "the cached forms are given at once, before anything is asked"
            time.sleep(0.3)
            assert len(model.prompts) == asked
        finally:
            expander.shutdown()
            book.shutdown()

    def test_a_script_reads_the_cache_and_never_asks(self, llm_settings, folder, ledger_dir, model):
        book, expander = self._start(llm_settings, folder)
        assert _wait_until(lambda: "tax" in book.expansions)
        expander.shutdown()
        book.shutdown()
        asked = len(model.prompts)
        with lorebook.loaded(llm_settings, directory=folder) as book:
            assert "tax" in lorebook_expansion.cached(llm_settings, book.entries)
        assert len(model.prompts) == asked

    def test_an_edited_term_is_asked_again_and_only_it(self, llm_settings, folder, ledger_dir, model):
        book, expander = self._start(llm_settings, folder)
        assert _wait_until(lambda: "tax" in book.expansions)
        expander.shutdown()
        book.shutdown()
        model.prompts.clear()

        (folder / "glossary.md").write_text("## Hydra tax\nClosing one opens two.\n\n## Spork (sporks, spork tine)\n"
                                            "A utensil.\n", encoding="utf-8")
        book, expander = self._start(llm_settings, folder)
        try:
            assert _wait_until(lambda: "tine" in book.expansions)
            assert model.terms_asked() == ["spork tine"]
        finally:
            expander.shutdown()
            book.shutdown()

    def test_a_reply_that_is_not_json_is_asked_again_next_time(self, llm_settings, folder, ledger_dir, monkeypatch):
        garbled = FakeModel(garbled=True)
        monkeypatch.setattr(agent, "ask_record", garbled)
        book, expander = self._start(llm_settings, folder)
        assert _wait_until(lambda: garbled.prompts)
        time.sleep(0.3)
        expander.shutdown()
        book.shutdown()
        assert book.expansions == {}

        working = FakeModel()
        monkeypatch.setattr(agent, "ask_record", working)
        book, expander = self._start(llm_settings, folder)
        try:
            assert _wait_until(lambda: "tax" in book.expansions)
            assert sorted(working.terms_asked()) == ["Hydra tax", "Spork", "sporks"]
        finally:
            expander.shutdown()
            book.shutdown()

    def test_what_a_model_offers_unasked_is_dropped(self, llm_settings, folder, ledger_dir, monkeypatch):
        monkeypatch.setattr(agent, "ask_record", FakeModel(stems=True, finnish=True))
        book, expander = self._start(llm_settings, folder)
        try:
            assert _wait_until(lambda: "hydra" in book.expansions)
            assert all(expansion.stems == () for expansion in book.expansions.values()), "English asks for forms only"
            assert book.expansions["hydra"].forms == ("hydras", "hydraed"), "and Finnish was not asked for at all"
        finally:
            expander.shutdown()
            book.shutdown()

    def test_with_finnish_on_its_forms_and_stems_arrive(self, llm_settings, folder, ledger_dir, monkeypatch):
        monkeypatch.setattr(agent, "ask_record", FakeModel(finnish=True))
        book = lorebook.Lorebook(llm_settings, directory=folder)
        book.start()
        expander = lorebook_expansion.Expander(llm_settings, book, languages={"English": "forms", "Finnish": "stems"})
        expander.start()
        try:
            assert _wait_until(lambda: "hydra" in book.expansions)
            assert book.expansions["hydra"] == lorebook.Expansion(forms=("hydras", "hydraed", "hydraissa"), stems=("hydra",))
        finally:
            expander.shutdown()
            book.shutdown()

    def test_an_alias_added_while_watching_is_asked_for_at_the_next_poll(self, llm_settings, folder, ledger_dir, model):
        # The set of entries does not change, so the lorebook's `on_changed` does not fire; its generation does.
        book = lorebook.Lorebook(llm_settings, directory=folder)
        book.start(watch=True)
        expander = lorebook_expansion.Expander(llm_settings, book)
        expander.start()
        try:
            assert _wait_until(lambda: "tax" in book.expansions)
            (folder / "glossary.md").write_text("## Hydra tax\nClosing one opens two.\n\n## Spork (sporks, foon)\n"
                                                "A utensil.\n", encoding="utf-8")
            assert _wait_until(lambda: "foon" in lorebook.terms(book.entries))
            expander.poll()
            assert _wait_until(lambda: "foon" in book.expansions)
        finally:
            expander.shutdown()
            book.shutdown()

    def test_with_the_backend_down_it_waits_and_asks_once_it_is_back(self, llm_settings, folder, ledger_dir, model,
                                                                      monkeypatch):
        status = [llmclient.backend_unreachable]
        monkeypatch.setattr(llmclient, "backend_status", lambda settings: status[0])
        book, expander = self._start(llm_settings, folder)
        try:
            time.sleep(0.3)
            assert model.prompts == [] and not list(ledger_dir.glob("*.jsonl")), "nothing asked, nothing written"
            expander.poll()
            time.sleep(0.3)
            assert model.prompts == [], "still down"
            status[0] = llmclient.backend_ready
            expander.poll()
            assert _wait_until(lambda: "tax" in book.expansions)
        finally:
            expander.shutdown()
            book.shutdown()

    def test_progress_shows_while_asking_and_on_done_reports_after(self, llm_settings, folder, ledger_dir, monkeypatch):
        release = threading.Event()
        fake = FakeModel()
        def slow(*args, **kwargs):
            release.wait(timeout=10)
            return fake(*args, **kwargs)
        monkeypatch.setattr(agent, "ask_record", slow)
        reported = []
        book = lorebook.Lorebook(llm_settings, directory=folder)
        book.start()
        expander = lorebook_expansion.Expander(llm_settings, book, on_done=reported.append)
        try:
            expander.start()
            assert _wait_until(lambda: expander.progress is not None and expander.progress.total == 3)
            assert expander.progress.done == 0 and reported == []
            release.set()
            assert _wait_until(lambda: reported == [3]), "on_done gets the number of terms with forms"
            assert expander.progress is None, "nothing is being asked now"
        finally:
            expander.shutdown()
            book.shutdown()

    def _start_reporting(self, llm_settings, folder):
        reported = []
        book = lorebook.Lorebook(llm_settings, directory=folder)
        book.start()
        expander = lorebook_expansion.Expander(llm_settings, book, on_done=reported.append)
        expander.start()
        return book, expander, reported

    def test_failed_terms_are_asked_again_in_the_same_run(self, llm_settings, folder, ledger_dir, monkeypatch):
        flaky = FakeModel(fail_first=1)
        monkeypatch.setattr(agent, "ask_record", flaky)
        book, expander, reported = self._start_reporting(llm_settings, folder)
        try:
            assert _wait_until(lambda: reported == [3]), "the retry answered them, so the run is done"
            assert len(flaky.prompts) == 2
        finally:
            expander.shutdown()
            book.shutdown()

    def test_after_the_retries_it_waits_for_the_next_refresh(self, llm_settings, folder, ledger_dir, monkeypatch):
        garbled = FakeModel(garbled=True)
        monkeypatch.setattr(agent, "ask_record", garbled)
        book, expander, reported = self._start_reporting(llm_settings, folder)
        try:
            assert _wait_until(lambda: len(garbled.prompts) == 1 + lorebook_expansion._RETRIES)
            time.sleep(0.3)
            assert len(garbled.prompts) == 1 + lorebook_expansion._RETRIES and reported == []
            assert expander.progress is None, "nothing is being asked any more"
        finally:
            expander.shutdown()
            book.shutdown()

    def test_the_terms_complete_in_a_reply_that_ran_on_are_kept(self, llm_settings, folder, ledger_dir, monkeypatch):
        truncating = FakeModel(truncate=True)
        monkeypatch.setattr(agent, "ask_record", truncating)
        book, expander, reported = self._start_reporting(llm_settings, folder)
        try:
            # One term survives each reply, so the three take the first try and both retries.
            assert _wait_until(lambda: reported == [3])
            assert len(truncating.prompts) == 3
        finally:
            expander.shutdown()
            book.shutdown()

    def test_on_done_is_not_called_when_everything_was_cached(self, llm_settings, folder, ledger_dir, model):
        book, expander = self._start(llm_settings, folder)
        assert _wait_until(lambda: "tax" in book.expansions)
        expander.shutdown()
        book.shutdown()

        reported = []
        book = lorebook.Lorebook(llm_settings, directory=folder)
        book.start()
        expander = lorebook_expansion.Expander(llm_settings, book, on_done=reported.append)
        try:
            expander.start()
            time.sleep(0.3)
            assert "tax" in book.expansions and reported == [] and expander.progress is None
        finally:
            expander.shutdown()
            book.shutdown()

    def test_another_model_gets_a_ledger_of_its_own(self, llm_settings, folder, ledger_dir, model):
        book, expander = self._start(llm_settings, folder)
        try:
            assert _wait_until(lambda: "tax" in book.expansions)
            model.prompts.clear()
            llm_settings.model_record = llmclient._model_info(**(dict(llm_settings.model_record) | {"model_id": "other"}))
            expander.poll()
            assert _wait_until(lambda: len(model.terms_asked()) == 3), "every term is asked again for the new model"
            assert len(list(ledger_dir.glob("expansions-*[!s].jsonl"))) == 2
        finally:
            expander.shutdown()
            book.shutdown()
