"""Ask the same question of every item in a set, and survive the afternoon.

One level above `agent`, which runs one turn: this runs one turn per batch of items, over thousands of
items, with what an unattended run needs around it.

  - **Resume.** Every answer goes to a JSONL ledger as it arrives, so an interrupted run is resumed by
    running the same thing again. Done items are skipped; failed ones are tried again.
  - **A fingerprint per instrument.** The ledger is named for a hash of whatever decides what an answer
    means: the question, the model's identity, and any setting the backend cannot report. A change to any
    of them starts a fresh ledger rather than mixing two measurements in one file.
  - **Telling a bad item from a dead backend.** After every batch the backend is asked whether it is still
    there and still serving the same model. A failure while it is fine is the batch's own, recorded and
    retried next run; a backend that has gone away, or swapped models, stops the run.
  - **Progress and cancellation**, for a console and for a GUI alike.

The question itself is the caller's: a function from a batch of `(id, item)` pairs to the answers, which
is all this module knows about what is being asked. `investigations/aokk-corpus-scope/extract_fields.py`
is a worked example.

    ledger = sweep.Ledger(out_dir, "extracted", sweep.fingerprint(llm_settings, prompt, declared="KV cache q4_0"))
    result = sweep.Sweep(llm_settings, ask_batch, items, ledger, batch_size=10).run()
"""

__all__ = ["fingerprint", "instrument_header",
           "Ledger",
           "completed", "cancelled", "backend_unavailable", "model_changed",
           "Sweep"]

import datetime
import hashlib
import json
import logging
import pathlib
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from unpythonic import ETAEstimator, sym
from unpythonic.env import env

from ..common import netutil

from . import agent
from . import llmclient

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# The instrument

def fingerprint(llm_settings: env, material: str, declared: str | None = None) -> str:
    """A short hash of everything that decides what an answer means. Eight hex digits.

    `llm_settings`: the settings the run asks with; the model's id and quantization are read from
                    `model_record`.

    `material`: the caller's half of the instrument — the prompt, together with anything else that
                changes what the model is asked, such as the vocabularies it is offered. Any change to it
                starts a fresh ledger.

    `declared`: settings the backend does not report and the operator does, as free text, e.g. the KV
                cache quantization. `None` when nothing is declared.

    The load configuration the backend reports (batch sizes, flash attention, speculative decoding, ...)
    is not hashed, so a change to it keeps the cache. It goes into the ledger's header instead
    (`instrument_header`), and a change *during* a run stops the run.
    """
    record = llm_settings.model_record
    parts = (material, f"model={record.model_id}", f"quantization={record.quantization}", f"declared={declared}")
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:8]

def instrument_header(llm_settings: env, declared: str | None = None) -> dict:
    """What the run measured with, for the ledger's header: the whole model record and the declarations.

    A field the backend did not report is listed under `not_reported` rather than left looking like an
    answer, and an absent declaration says so.
    """
    record = llm_settings.model_record
    reported = {name: record[name] for name in ("model_id", "label", "quantization", "bits_per_weight", "load_config")}
    return {"backend_url": llm_settings.backend_url,
            "backend_flavor": llm_settings.backend_flavor,
            "model_requested": llm_settings.model_id,
            **reported,
            "not_reported": sorted(name for name, value in reported.items() if value is None),
            "declared": declared if declared is not None else "not stated"}

# --------------------------------------------------------------------------------
# The ledger

class Ledger:
    """A JSONL file of answers, one line per item, a later line for an item superseding an earlier one.

    `directory`: where the files go. Created if missing.

    `name`: what the run is called; the files are `{name}-{fingerprint}.jsonl` and, beside it,
            `{name}-{fingerprint}-traces.jsonl` for the model calls themselves.

    `fingerprint`: the instrument's, from `fingerprint`, or one recorded elsewhere. Opening a ledger under
                   an older instrument's fingerprint and writing answers into it is how answers made
                   elsewhere are seeded; a run under a new instrument never sees them, its file being a
                   different one.

    A line is `{"id", "v", "status": "ok", "answer"}` or `{"id", "v", "status": "failed", "error"}`, and a
    run's header is `{"header", "v", "started"}`. Only lines stamped with this ledger's fingerprint are
    read, which is what catches a file that has been renamed or had another appended to it.
    """

    def __init__(self, directory: str | pathlib.Path, name: str, fingerprint: str):
        self.directory = pathlib.Path(directory)
        self.name = name
        self.fingerprint = fingerprint
        self.path = self.directory / f"{name}-{fingerprint}.jsonl"
        self.traces_path = self.directory / f"{name}-{fingerprint}-traces.jsonl"

    def _append(self, path: pathlib.Path, payload: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_header(self, header: dict) -> None:
        """Record what a run measured with, at the start of each run."""
        self._append(self.path, {"header": header, "v": self.fingerprint,
                                 "started": datetime.datetime.now().isoformat(timespec="seconds")})

    def write_answer(self, item_id: str, answer: Any) -> None:
        """Record an answer. Anything JSON can hold."""
        self._append(self.path, {"id": item_id, "v": self.fingerprint, "status": "ok", "answer": answer})

    def write_failure(self, item_id: str, error: str) -> None:
        """Record that an item was tried and failed, so a later run tries it again."""
        self._append(self.path, {"id": item_id, "v": self.fingerprint, "status": "failed", "error": error})

    def write_trace(self, item_ids: Sequence[str], turns: Sequence[agent.TurnRecord]) -> None:
        """Record the model calls of one batch, whole, naming the items that shared them.

        Whole rather than the reasoning alone: the reply is what explains a batch whose answers did not
        parse, and it is the part a trace of the thinking leaves out.
        """
        self._append(self.traces_path, {"ids": list(item_ids), "v": self.fingerprint,
                                        "turns": [turn.to_dict() for turn in turns]})

    def latest(self) -> dict[str, dict]:
        """The latest line for each item, answered or failed, keyed by id."""
        out = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("v") == self.fingerprint and "id" in entry:
                    out[entry["id"]] = entry
        return out

    def answers(self) -> dict[str, Any]:
        """The answer for each item whose latest line is one, keyed by id."""
        return {item_id: entry["answer"] for item_id, entry in self.latest().items() if entry["status"] == "ok"}

    def headers(self) -> list[dict]:
        """The header of each run that wrote to this ledger, oldest first."""
        if not self.path.exists():
            return []
        entries = (json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())
        return [entry for entry in entries if entry.get("v") == self.fingerprint and "header" in entry]

# --------------------------------------------------------------------------------
# The run

completed = sym("completed")  # every item was tried; some may have failed, and are retried next run
cancelled = sym("cancelled")
backend_unavailable = sym("backend_unavailable")  # unreachable, or no model loaded
model_changed = sym("model_changed")  # another model, or the same one loaded with other settings

class Sweep:
    """One run of `ask_batch` over the items not yet answered in `ledger`.

    `llm_settings`: as `agent.turn`. Kept current while the run goes, by `llmclient.follow_model_swap`.

    `ask_batch`: `ask_batch(llm_settings, batch, maybe_abort) -> (answers, turns)`. `batch` is a list of
                 `(id, item)` pairs, `answers` a dict from id to answer, anything JSON can hold, and `turns`
                 the `agent.TurnRecord` of each model call made, which go to the ledger's traces file whole.
                 Pass `maybe_abort` on to `agent.ask_record` or `agent.turn`, which is what makes `cancel`
                 reach a call in flight. An id the answers leave out is recorded as failed; an exception
                 fails the whole batch, and its turns are not kept, so return rather than raise where the
                 reply is worth reading afterwards.

    `items`: `(id, item)` pairs, ids unique strings. Those the ledger already answers are skipped.

    `ledger`: a `Ledger`.

    `batch_size`: items per call. 1, the default, is one call per item.

    `header`: what to record as this run's header, normally `instrument_header(...)`. `None` writes none.

    `on_item`: called after each batch with an `env` of `done`, `failed` and `total` (this run's items),
               `ids` (the batch just finished), `elapsed` (seconds) and `eta` (the `unpythonic.ETAEstimator`,
               counting batches). Called in the thread that runs the sweep.

    `log_interval`: least seconds between the progress lines logged at INFO; the last is always logged.
                    Every failed item is logged as it happens, at WARNING, however often.
    """

    def __init__(self,
                 llm_settings: env,
                 ask_batch: Callable[[env, list[tuple[str, Any]], netutil.Abort],
                                     tuple[dict[str, Any], Sequence[agent.TurnRecord]]],
                 items: Sequence[tuple[str, Any]],
                 ledger: Ledger,
                 batch_size: int = 1,
                 header: dict | None = None,
                 on_item: Callable[[env], None] | None = None,
                 log_interval: float = 10.0):
        if batch_size < 1:
            raise ValueError(f"Sweep: `batch_size` must be at least 1, got {batch_size}")
        self.llm_settings = llm_settings
        self.ask_batch = ask_batch
        self.items = list(items)
        self.ledger = ledger
        self.batch_size = batch_size
        self.header = header
        self.on_item = on_item
        self.log_interval = log_interval
        self._lock = threading.Lock()
        self._cancel_requested = False
        self._maybe_abort = None  # the handle of the call in flight
        self._progress = env(done=0, failed=0, total=0, elapsed=0.0, eta=None, status_line="not started")

    def cancel(self) -> None:
        """Stop the run, abandoning the call in flight, prompt processing included. Safe from any thread.

        The batch in flight is not recorded, so a resumed run starts with it.
        """
        with self._lock:
            self._cancel_requested = True
            if self._maybe_abort is not None:
                self._maybe_abort.abort()

    def _get_progress(self) -> env:
        with self._lock:
            return env(**self._progress)
    progress = property(fget=_get_progress, doc="The latest progress, as `on_item` reports it, plus `status_line`. "
                                                "Safe to read from any thread, e.g. a GUI's frame update.")

    def run(self) -> env:
        """Run until every item has been tried, or until the run cannot go on. Blocks.

        Returns an `env` with `outcome` (`completed`, `cancelled`, `backend_unavailable` or `model_changed`),
        `message` (what to tell the operator, empty when completed), and `done`, `failed` and `total` for
        this run's items.
        """
        answered = set(self.ledger.answers())
        todo = [(item_id, item) for item_id, item in self.items if item_id not in answered]
        batches = [todo[i:i + self.batch_size] for i in range(0, len(todo), self.batch_size)]
        logger.info(f"Sweep.run: ledger '{self.ledger.path.name}': {len(answered)} items already answered, "
                    f"{len(todo)} to do in {len(batches)} {'batch' if len(batches) == 1 else 'batches'}.")  # plural
        if self.header is not None and todo:
            self.ledger.write_header(self.header)
        if todo:
            # Said before the first call, so that a run started against the wrong model loaded is caught
            # before it has spent anything.
            logger.info(f"Sweep.run: asking '{self.llm_settings.model}'.")

        # Counted in batches, which is the unit the time goes into; and from this run's start, so that a
        # resumed run's estimate is not flattered by the items an earlier run did.
        eta = ETAEstimator(total=len(batches), keep_last=20)
        t0 = time.monotonic()
        done = failed = 0
        last_logged = None
        last_logged_count = None
        outcome, message = completed, ""

        def report(ids: list[str], final: bool = False) -> None:
            nonlocal last_logged, last_logged_count
            elapsed = time.monotonic() - t0
            status_line = f"{done + failed} / {len(todo)} items, {failed} failed; {eta.formatted_eta}"
            with self._lock:
                self._progress = env(done=done, failed=failed, total=len(todo), elapsed=elapsed, eta=eta,
                                     status_line=status_line)
            now = time.monotonic()
            # The final line is always written, unless the last one already said the same.
            if (last_logged is None or now - last_logged >= self.log_interval or
                    (final and done + failed != last_logged_count)):
                logger.info(f"Sweep.run: {status_line}")
                last_logged, last_logged_count = now, done + failed
            if self.on_item is not None and ids:
                self.on_item(env(done=done, failed=failed, total=len(todo), ids=ids, elapsed=elapsed, eta=eta))

        def fail(item_id: str, error: str) -> None:
            nonlocal failed
            self.ledger.write_failure(item_id, error)
            failed += 1
            logger.warning(f"Sweep.run: item '{item_id}' failed: {error}")

        report([])
        for batch in batches:
            with self._lock:
                if self._cancel_requested:
                    outcome = cancelled
                    break
                maybe_abort = self._maybe_abort = netutil.Abort()
            ids = [item_id for item_id, _ in batch]
            try:
                answers, turns = self.ask_batch(self.llm_settings, batch, maybe_abort)
            except netutil.Aborted:
                outcome = cancelled  # the batch is dropped unwritten; a resumed run starts with it
                break
            except Exception as exc:  # noqa: BLE001 -- whose fault it was is asked of the backend below
                for item_id in ids:
                    fail(item_id, f"{type(exc)}: {exc}")
            else:
                for item_id in ids:
                    if item_id in answers:
                        self.ledger.write_answer(item_id, answers[item_id])
                        done += 1
                    else:
                        fail(item_id, "the reply had no answer for this item")
                if turns:
                    self.ledger.write_trace(ids, turns)
            finally:
                with self._lock:
                    self._maybe_abort = None
            eta.tick()

            # One probe after every batch, failed or not: a backend that has gone away fails every batch
            # from here on, and one that has swapped models answers them with another instrument.
            model_before = self.llm_settings.model
            swapped = llmclient.follow_model_swap(self.llm_settings)
            status = llmclient.backend_status(self.llm_settings)
            report(ids)
            if status is not llmclient.backend_ready:
                headline, advice = llmclient.describe_backend_status(status, self.llm_settings.backend_url)
                outcome, message = backend_unavailable, f"{headline} {advice}"
                break
            if swapped:
                outcome = model_changed
                change = (f"from '{model_before}' to '{self.llm_settings.model}'"
                          if self.llm_settings.model != model_before else f"'{model_before}' was reloaded with other settings")
                message = (f"The backend's model changed during the run: {change}. Answers from another model are "
                           f"another measurement; run again to continue under it.")
                break

        report([], final=True)
        if message:
            logger.warning(f"Sweep.run: stopped: {message}")
        return env(outcome=outcome, message=message, done=done, failed=failed, total=len(todo))
