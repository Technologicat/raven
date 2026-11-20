"""The inflected forms of the lorebook's terms, asked of the LLM, so that a term fires however it is written.

The lorebook matches a term's words as written. Text inflects them — "two hydra taxes", and in Finnish nearly
every use of a word — so the LLM is asked, once per term, for each word's other forms in the configured
languages (`librarian_config.lorebook_languages`): whole-word `forms` for every language, and `stems` for an
agglutinative one, whose suffixes stack without limit. `lorebook.clean_expansion` then drops what would match
too much, and the result goes to the `Lorebook` as its `expansions`.

The asking is a `sweep.Sweep` over the terms, and its `Ledger` is the cache: an interrupted run resumes, a
failed batch is asked again next time, and a term already answered is not asked again. A ledger belongs to
one model and one prompt, so switching models asks every term again, in the background; switching back finds
the earlier answers.

`Expander` does this for a frontend, in the background, following changes to the lorebook and to the loaded
model. A script reads what is cached for its model with `cached`, and never asks.
"""

__all__ = ["material", "expansions_from", "cached", "Expander"]

import logging
import threading
from collections.abc import Mapping
from typing import Any

from unpythonic.env import env

from . import agent
from . import config as librarian_config
from . import llmclient
from . import lorebook
from . import sweep

logger = logging.getLogger(__name__)

_LEDGER_NAME = "expansions"

def _languages(maybe_languages: Mapping[str, str] | None) -> dict[str, str]:
    languages = dict(maybe_languages if maybe_languages is not None else librarian_config.lorebook_languages)
    for language, kind in languages.items():
        if kind not in ("forms", "stems"):
            raise ValueError(f"lorebook_languages: '{language}' wants \"forms\" or \"stems\", got {kind!r}")
    return languages

def material(settings: env, languages: Mapping[str, str] | None = None) -> str:
    """The question the terms are asked with, without the terms: the instruction, and the languages.

    What a ledger's fingerprint is taken over, so that a change to either asks every term again.
    """
    described = {"forms": "forms", "stems": "forms and stems"}
    listing = "\n".join(f"- {language}: {described[kind]}" for language, kind in _languages(languages).items())
    return f"{settings.lorebook_expansion_instruction}\n\nLanguages:\n{listing}"

def _items(entries: Mapping[str, lorebook.Entry]) -> list[tuple[str, dict]]:
    return [(phrase, {"term": phrase, "words": list(words)}) for phrase, words in lorebook.terms(entries).items()]

def _render(item: dict) -> str:
    """A term as the prompt shows it."""
    return f"{item['term']} — words: {', '.join(item['words'])}"

def _digest(item: dict) -> str:
    """What says an answer is about this term as it is now: words added to it, or taken away, ask it again."""
    return sweep.digest(_render(item))

def _ask_batch_for(languages: Mapping[str, str]):
    """The `ask_batch` a `Sweep` calls, for these languages."""
    def ask_batch(settings: env, batch: list[tuple[str, dict]], maybe_abort) -> tuple[dict[str, Any], list]:
        listing = "\n".join(f"{number}. {_render(item)}" for number, (_, item) in enumerate(batch, start=1))
        prompt = f"{material(settings, languages)}\n\nTerms:\n{listing}"
        # Without thinking, to keep the background work short. Whether thinking would give better forms is
        # unmeasured.
        record = agent.ask_record(settings, prompt, maybe_abort=maybe_abort, thinking_enabled=False)
        try:
            reply = agent.parse_json_reply(record.reply)
        except ValueError:
            return {}, [record]  # every term in the batch is recorded as failed, and asked again next run
        answers: dict[str, Any] = {}
        for number, (item_id, item) in enumerate(batch, start=1):
            got = reply.get(str(number)) if isinstance(reply, dict) else None
            if not isinstance(got, dict):
                answers[item_id] = sweep.Failure("no answer for this term")
                continue
            # Kept per language, as the model gave it, so the ledger shows what each language contributed.
            answers[item_id] = {word: {language: _strings_of(by_language.get(language))
                                       for language in languages if isinstance(by_language.get(language), dict)}
                                for word, by_language in got.items()
                                if word in item["words"] and isinstance(by_language, dict)}
        return answers, [record]
    return ask_batch

def _strings_of(given: dict) -> dict[str, list[str]]:
    """A language's answer for one word, with anything that is not a list of strings left out."""
    def strings(value: Any) -> list[str]:
        return [entry for entry in value if isinstance(entry, str)] if isinstance(value, list) else []
    return {"forms": strings(given.get("forms")), "stems": strings(given.get("stems"))}

def expansions_from(answers: Mapping[str, Any], languages: Mapping[str, str]) -> dict[str, lorebook.Expansion]:
    """Expansions, by key word, from a ledger's answers: each word's forms and stems, cleaned.

    `answers`: term -> word -> language -> `{"forms": [...], "stems": [...]}`.
    `languages`: as `librarian_config.lorebook_languages`. Only these languages are read, and stems only from
                 one marked `"stems"`, so stems a model offered for another are dropped.

    A word in several terms gets what all of them gave it.
    """
    forms: dict[str, list[str]] = {}
    stems: dict[str, list[str]] = {}
    for answer in answers.values():
        for word, by_language in answer.items():
            for language, kind in languages.items():
                given = by_language.get(language, {})
                forms.setdefault(word, []).extend(given.get("forms", ()))
                if kind == "stems":
                    stems.setdefault(word, []).extend(given.get("stems", ()))
    expansions = {word: lorebook.clean_expansion(word, forms[word], stems.get(word, ())) for word in forms}
    return {word: expansion for word, expansion in expansions.items() if expansion.forms or expansion.stems}

def _ledger(settings: env, languages: Mapping[str, str]) -> sweep.Ledger:
    return sweep.Ledger(librarian_config.lorebook_expansion_dir, _LEDGER_NAME,
                        sweep.fingerprint(settings, material(settings, languages)))

def _current_answers(ledger: sweep.Ledger, items: list[tuple[str, dict]]) -> dict[str, Any]:
    """The ledger's answers about the terms as they are now; a term since edited, or gone, is left out."""
    digests = {item_id: _digest(item) for item_id, item in items}
    answers = ledger.answers(digests=digests)
    return {item_id: answers[item_id] for item_id in digests if item_id in answers}

def cached(settings: env,
           entries: Mapping[str, lorebook.Entry],
           *,
           languages: Mapping[str, str] | None = None) -> dict[str, lorebook.Expansion]:
    """The expansions already asked for these entries' terms with the loaded model, without asking for more.

    `languages`: as `librarian_config.lorebook_languages`, which is the default.

    For a script, which gets what an app has asked before and nothing new: give the result to a `Lorebook`
    as its `expansions`.
    """
    languages = _languages(languages)
    answers = _current_answers(_ledger(settings, languages), _items(entries))
    return expansions_from(answers, languages)


class Expander:
    """Keeps a `Lorebook`'s expansions current, asking the LLM in the background for terms it has not answered.

    `settings`: the frontend's LLM settings, as given to the `Lorebook`.
    `book`: the `Lorebook` whose `expansions` this sets.
    `languages`: as `librarian_config.lorebook_languages`, which is the default.
    `batch_size`: terms per request. Default `librarian_config.lorebook_expansion_batch_size`.

    `start` gives the lorebook what is cached at once, then asks for the rest. Then call `poll` as often as is
    convenient, a GUI's frame update included: it notices an edit to the lorebook, a model swap, which needs a
    ledger of its own, and a backend back up after a run stopped for want of one, by reading a few attributes,
    and asks nothing otherwise.
    """
    def __init__(self,
                 settings: env,
                 book: lorebook.Lorebook,
                 *,
                 languages: Mapping[str, str] | None = None,
                 batch_size: int | None = None) -> None:
        self.settings = settings
        self.book = book
        self.languages = _languages(languages)
        self.batch_size = batch_size if batch_size is not None else librarian_config.lorebook_expansion_batch_size
        self._lock = threading.Lock()  # the state below
        self._refresh_lock = threading.Lock()  # one refresh at a time: from the lorebook's watcher, and from `poll`
        self._maybe_sweep: sweep.Sweep | None = None
        self._maybe_thread: threading.Thread | None = None
        self._model: tuple | None = None  # the model the current expansions were made with
        self._generation: int | None = None  # the lorebook's, when they were made
        self._waiting_for_backend = False  # the last run stopped because the backend was not there
        self._refresh_pending = False  # a refresh started by `poll` that has not finished yet
        self._closed = False

    def _model_identity(self) -> tuple:
        record = self.settings.model_record
        return (record.model_id, record.quantization)

    def start(self) -> None:
        """Give the lorebook its cached expansions, and ask for the terms that have none, in the background."""
        self.refresh()

    def poll(self) -> None:
        """Start over if the lorebook or the model has changed, or the backend is back after a run stopped without it.

        Cheap, and never blocks: what it starts runs on a thread of its own.
        """
        # Without waiting for the lock, since a GUI calls this from its render loop: a refresh holding it is busy
        # with exactly what this would ask about, and the next poll asks again.
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._closed or self._refresh_pending or self._model is None:
                return
            edited = self.book.generation != self._generation
            model_changed = self._model_identity() != self._model
            backend_back = (self._waiting_for_backend and
                            llmclient.backend_status(self.settings) is llmclient.backend_ready)
            if not (edited or model_changed or backend_back):
                return
            self._refresh_pending = True
        finally:
            self._lock.release()
        reason = ("the lorebook has changed" if edited else
                  "the model has changed" if model_changed else
                  "the backend is back")
        logger.info(f"Expander.poll: {reason}; reading the expansions, and asking for any missing.")
        def refresh_then_clear() -> None:
            try:
                self.refresh()
            finally:
                with self._lock:
                    self._refresh_pending = False
        threading.Thread(target=refresh_then_clear, name="lorebook_expansion_refresh", daemon=True).start()

    def refresh(self) -> None:
        """Bring the expansions up to date with the entries and the model: cached ones at once, the rest in the background.

        Blocks while a run in progress is cancelled, which abandons its request in flight.
        """
        with self._refresh_lock:
            self._refresh()

    def _refresh(self) -> None:
        self._stop_sweep()
        with self._lock:
            if self._closed:
                return
            self._waiting_for_backend = False
            self._generation = self.book.generation  # read before the entries, so an edit in between is seen next poll
            entries = self.book.entries
            self._model = self._model_identity()
            ledger = _ledger(self.settings, self.languages)
            items = _items(entries)
            answered = _current_answers(ledger, items)
            self.book.expansions = expansions_from(answered, self.languages)
            missing = [(item_id, item) for item_id, item in items if item_id not in answered]
            if not missing:
                return
            if llmclient.backend_status(self.settings) is not llmclient.backend_ready:
                self._waiting_for_backend = True  # `poll` notices it coming back
                logger.info("Expander.refresh: the backend is not ready; asking for the inflected forms once it is.")
                return
            the_sweep = sweep.Sweep(self.settings, _ask_batch_for(self.languages), missing, ledger,
                                    batch_size=self.batch_size,
                                    header=sweep.instrument_header(self.settings),
                                    digest=_digest)
            self._maybe_sweep = the_sweep
            noun = "term" if len(missing) == 1 else "terms"  # plural
            logger.info(f"Expander.refresh: asking for the inflected forms of {len(missing)} {noun}, in the background.")
            thread = threading.Thread(target=self._run, args=(the_sweep, items, ledger),
                                      name="lorebook_expansion", daemon=True)
            self._maybe_thread = thread
            thread.start()

    def _run(self, the_sweep: sweep.Sweep, items: list[tuple[str, dict]], ledger: sweep.Ledger) -> None:
        """Run the sweep, then give the lorebook everything answered. On the expander's own thread."""
        try:
            result = the_sweep.run()
        except Exception:  # noqa: BLE001 -- the next refresh gets another try
            logger.exception("Expander: asking for the inflected forms failed")
            return
        with self._lock:
            if self._closed or self._maybe_sweep is not the_sweep:
                return  # superseded by a refresh, or shut down, while it ran
            self._waiting_for_backend = result.outcome is sweep.backend_unavailable
            answered = _current_answers(ledger, items)
            self.book.expansions = expansions_from(answered, self.languages)
        logger.info(f"Expander: {result.outcome} {result.message}; inflected forms for {len(answered)} of "
                    f"{len(items)} terms.")

    def _stop_sweep(self) -> None:
        with self._lock:
            maybe_sweep, maybe_thread = self._maybe_sweep, self._maybe_thread
            self._maybe_sweep = self._maybe_thread = None
        if maybe_sweep is not None:
            maybe_sweep.cancel()
        if maybe_thread is not None:
            maybe_thread.join()

    def shutdown(self) -> None:
        """Stop asking, abandoning the request in flight. Safe to call more than once."""
        with self._lock:
            self._closed = True
        self._stop_sweep()
