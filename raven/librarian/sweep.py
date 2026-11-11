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

__all__ = ["fingerprint", "instrument_header", "digest",
           "Ledger",
           "completed", "cancelled", "backend_unavailable", "model_changed",
           "Failure", "Sweep",
           "status", "format_status", "main"]

import datetime
import hashlib
import json
import logging
import os
import pathlib
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterable

from unpythonic import ETAEstimator, sym
from unpythonic.env import env

from ..common import netutil
from ..common import utils as common_utils

from . import agent
from . import llmclient

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# The instrument

def fingerprint(llm_settings: env,
                material: str,
                declared: str | None = None,
                *,
                tools: Iterable[dict] = ()) -> str:
    """A short hash of everything that decides what an answer means. Eight hex digits.

    `llm_settings`: the settings the run asks with; the model's id and quantization are read from
                    `model_record`.

    `material`: the caller's half of the instrument — the prompt, together with anything else that
                changes what the model is asked, such as the vocabularies it is offered. Any change to it
                starts a fresh ledger.

    `declared`: settings the backend does not report and the operator does, as free text, e.g. the KV
                cache quantization. `None` when nothing is declared.

    `tools`: the tool specifications each call offers the model, from `llmclient.offered_tools`, in full,
             since a tool's description is read as much as its name. Leave it empty for a run that asks
             without tools, as a one-shot `agent.ask` does.

    The load configuration the backend reports (batch sizes, flash attention, speculative decoding, ...)
    is not hashed, so a change to it keeps the cache. It goes into the ledger's header instead
    (`instrument_header`), and a change *during* a run stops the run.
    """
    record = llm_settings.model_record
    parts = (material, f"model={record.model_id}", f"quantization={record.quantization}", f"declared={declared}")
    tools = list(tools)
    if tools:  # only when there are some, so that a run without tools keeps the fingerprints it has always had
        parts += (f"tools={json.dumps(tools, sort_keys=True)}",)
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

def digest(text: str) -> str:
    """A short hash of what the model was shown for one item, for `Ledger.answers` to tell a current answer
    from one given about different text. Sixteen hex digits.

    `text`: the item as the prompt renders it, and only that — what the model would see if the item were
            asked again. A change anywhere else in a record, one the prompt does not show, is no reason to ask
            again, and asking costs a model call.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

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

    An answer may also carry `"d"`, the `digest` of the item as it was asked. The fingerprint says the
    question has not changed; the digest says the item has not either.
    """

    def __init__(self, directory: str | pathlib.Path, name: str, fingerprint: str):
        self.directory = pathlib.Path(directory)
        self.name = name
        self.fingerprint = fingerprint
        self.path = self.directory / f"{name}-{fingerprint}.jsonl"
        self.traces_path = self.directory / f"{name}-{fingerprint}-traces.jsonl"
        self.state_path = self.directory / f"{name}-{fingerprint}.state.json"

    def write_state(self, state: dict) -> None:
        """Replace the run's state file, one JSON object read by `status`, which says how far a run has got."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with common_utils.atomic_write(self.state_path) as f:
            f.write(json.dumps(state, ensure_ascii=False) + "\n")

    def _append(self, path: pathlib.Path, payload: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        # A run killed mid-write leaves a last line with no newline. Start a fresh line after it, so that
        # it stays the unfinished last line `_entries` skips, rather than having this one glued on.
        unfinished = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
        with path.open("a", encoding="utf-8") as f:
            f.write(("\n" if unfinished else "") + json.dumps(payload, ensure_ascii=False) + "\n")

    def write_header(self, header: dict) -> None:
        """Record what a run measured with, at the start of each run."""
        self._append(self.path, {"header": header, "v": self.fingerprint,
                                 "started": datetime.datetime.now().isoformat(timespec="seconds")})

    def write_answer(self, item_id: str, answer: Any, *, maybe_digest: str | None = None) -> None:
        """Record an answer. Anything JSON can hold.

        `maybe_digest`: the `digest` of the item as it was asked, or `None` to record none.
        """
        line = {"id": item_id, "v": self.fingerprint, "status": "ok", "answer": answer}
        if maybe_digest is not None:
            line["d"] = maybe_digest
        self._append(self.path, line)

    def write_failure(self, item_id: str, error: str) -> None:
        """Record that an item was tried and failed, so a later run tries it again."""
        self._append(self.path, {"id": item_id, "v": self.fingerprint, "status": "failed", "error": error})

    def reask(self, item_ids: Sequence[str], reason: str = "asked again on request") -> None:
        """Have the next run ask these items again, answered or not.

        An append, so nothing is lost: the earlier answers stay in the file, superseded, and can be read
        for comparison.
        """
        for item_id in item_ids:
            self.write_failure(item_id, reason)

    def write_trace(self, item_ids: Sequence[str], turns: Sequence[agent.TurnRecord]) -> None:
        """Record the model calls of one batch, whole, naming the items that shared them.

        Whole rather than the reasoning alone: the reply is what explains a batch whose answers did not
        parse, and it is the part a trace of the thinking leaves out.
        """
        self._append(self.traces_path, {"ids": list(item_ids), "v": self.fingerprint,
                                        "turns": [turn.to_dict() for turn in turns]})

    def _entries(self) -> list[dict]:
        """Every line of the ledger, parsed, this instrument's or not.

        A line that does not parse is skipped, with a warning naming it: a run killed mid-write leaves one, and
        the answer it held is simply asked again. Later runs append after it, so it need not be the last line.
        """
        if not self.path.exists():
            return []
        out = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Ledger: '{self.path.name}', line {number}: does not parse, skipped "
                               f"(a run stopped mid-write leaves one); its item, if any, is asked again")
        return out

    def latest(self) -> dict[str, dict]:
        """The latest line for each item, answered or failed, keyed by id."""
        out = {}
        for entry in self._entries():
            if entry.get("v") == self.fingerprint and "id" in entry:
                out[entry["id"]] = entry
        return out

    def answers(self, digests: Mapping[str, str] | None = None) -> dict[str, Any]:
        """The answer for each item whose latest line is one, keyed by id.

        `digests`: the items as they are now, as `digest` of each, by id. When given, an answer counts only
                   if it was given about the same text: one with another digest, or with none, is left out,
                   so the item reads as not yet answered and a run asks it again. An id absent from
                   `digests` is not checked. `None` checks nothing.
        """
        return {item_id: entry["answer"] for item_id, entry in self.latest().items()
                if entry["status"] == "ok"
                and (digests is None or item_id not in digests or entry.get("d") == digests[item_id])}

    def stamp(self, digests: Mapping[str, str]) -> int:
        """Record, for answers carrying no digest, the digest of the item they were given about.

        Not normally needed: a `Sweep` given `digest` writes one with every answer. This is a one-time
        migration, for a ledger written before answers carried them, or by a sweep given none.

        Each `digests` entry must be the item *as it was asked* — from the corpus the run read, not a later
        one — or this declares old answers current for text they never saw. Appends, as everything here does;
        an answer that already has a digest, or an id `digests` does not name, is left alone. Returns how
        many answers were stamped.
        """
        stamped = 0
        for item_id, entry in self.latest().items():
            if entry["status"] == "ok" and "d" not in entry and item_id in digests:
                self.write_answer(item_id, entry["answer"], maybe_digest=digests[item_id])
                stamped += 1
        return stamped

    def headers(self) -> list[dict]:
        """The header of each run that wrote to this ledger, oldest first."""
        return [entry for entry in self._entries() if entry.get("v") == self.fingerprint and "header" in entry]

# --------------------------------------------------------------------------------
# The run

completed = sym("completed")  # every item was tried; some may have failed, and are retried next run
cancelled = sym("cancelled")
backend_unavailable = sym("backend_unavailable")  # unreachable, or no model loaded
model_changed = sym("model_changed")  # another model, or the same one loaded with other settings

class Failure:
    """An answer saying the item could not be answered: recorded as failed, with `reason`, and tried again
    on the next run. For an `ask_batch` that knows why an item failed and has turns worth keeping, which
    raising would lose."""

    def __init__(self, reason: str):
        self.reason = reason

    def __repr__(self) -> str:
        return f"Failure({self.reason!r})"

class Sweep:
    """One run of `ask_batch` over the items not yet answered in `ledger`.

    `llm_settings`: as `agent.turn`. Kept current while the run goes, by `llmclient.follow_model_swap`.

    `ask_batch`: `ask_batch(llm_settings, batch, maybe_abort) -> (answers, turns)`. `batch` is a list of
                 `(id, item)` pairs, `answers` a dict from id to answer, anything JSON can hold, and `turns`
                 the `agent.TurnRecord` of each model call made, which go to the ledger's traces file whole.
                 Pass `maybe_abort` on to `agent.ask_record` or `agent.turn`, which is what makes `cancel`
                 reach a call in flight. An id the answers leave out, or answer with a `Failure`, is
                 recorded as failed; an exception fails the whole batch, and its turns are not kept, so
                 return rather than raise where the reply is worth reading afterwards.

    `items`: `(id, item)` pairs, ids unique strings. Those the ledger already answers are skipped.

    `digest`: `digest(item) -> str`, the item as its prompt renders it, passed through `sweep.digest`. When
              given, each answer records it, and an answer given about other text — the item changed since,
              or the answer predates digests — does not count, so the item is asked again. `None` records no
              digests and skips every answered item, whatever it was asked about.

    `ledger`: a `Ledger`.

    `batch_size`: items per call. 1, the default, is one call per item.

    `header`: what to record as this run's header, normally `instrument_header(...)`. `None` writes none.

    `on_item`: called after each batch with an `env` of `done`, `failed` and `total` (this run's items),
               `ids` (the batch just finished), `elapsed` (seconds) and `eta` (the `unpythonic.ETAEstimator`,
               counting batches). Called in the thread that runs the sweep.

    `log_interval`: least seconds between the progress lines logged at INFO; the last is always logged.
                    Every failed item is logged as it happens, at WARNING, however often.

    `reconnect_attempts`, `reconnect_delay`: how many times, and how many seconds apart, to ask again a backend
                                             that stopped answering after a batch, before the run stops. Only
                                             an unreachable backend is asked again; one with no model, or
                                             another model, stops the run at once. `0` attempts stops at once.
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
                 log_interval: float = 10.0,
                 digest: Callable[[Any], str] | None = None,
                 reconnect_attempts: int = 3,
                 reconnect_delay: float = 5.0):
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
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.digest = digest
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
        digests = ({item_id: self.digest(item) for item_id, item in self.items} if self.digest is not None
                   else None)
        answered = set(self.ledger.answers(digests))
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

        started = time.time()

        def save_state(state: str, elapsed: float) -> None:
            # For `status`, from another terminal. A run with nothing to do leaves the state of the last one.
            if todo:
                self.ledger.write_state({"state": state, "total": len(self.items), "answered_before": len(answered),
                                         "todo": len(todo), "done": done, "failed": failed, "started": started,
                                         "updated": time.time(), "elapsed": elapsed, "pid": os.getpid(),
                                         "model": self.llm_settings.model})

        def report(ids: list[str], final: bool = False) -> None:
            nonlocal last_logged, last_logged_count
            elapsed = time.monotonic() - t0
            if not final:
                save_state("running", elapsed)
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
                    if item_id not in answers:
                        fail(item_id, "the reply had no answer for this item")
                    elif isinstance(answers[item_id], Failure):
                        fail(item_id, answers[item_id].reason)
                    else:
                        self.ledger.write_answer(item_id, answers[item_id],
                                                 maybe_digest=digests[item_id] if digests is not None else None)
                        done += 1
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
            # A dropped connection is waited out a few times before the run stops, since one short network
            # blip otherwise ends it. Only for "unreachable": a backend that answers without a model, or with
            # another one, is a fact about the backend, and a wait would not change it.
            for attempt in range(1, self.reconnect_attempts + 1):
                if status is not llmclient.backend_unreachable:
                    break
                with self._lock:
                    if self._cancel_requested:
                        break
                logger.warning(f"Sweep.run: cannot reach the backend at {self.llm_settings.backend_url}; "
                               f"trying again in {self.reconnect_delay:g} s ({attempt}/{self.reconnect_attempts}).")
                time.sleep(self.reconnect_delay)
                swapped = llmclient.follow_model_swap(self.llm_settings, reprobe=True) or swapped
                status = llmclient.backend_status(self.llm_settings)
                if status is not llmclient.backend_unreachable:
                    logger.info(f"Sweep.run: the backend answers again, after {attempt} "
                                f"{'retry' if attempt == 1 else 'retries'}.")  # plural
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
        save_state(str(outcome), time.monotonic() - t0)
        if message:
            logger.warning(f"Sweep.run: stopped: {message}")
        return env(outcome=outcome, message=message, done=done, failed=failed, total=len(todo))


# --------------------------------------------------------------------------------
# How far a run has got, from another terminal

# A running state not updated for this long, from a process that is gone, is a run that died without saying so.
_STALE_SECONDS = 600


def _process_alive(pid: int) -> bool:
    """Whether process `pid` exists. Always `False` on Windows, where this cannot be asked this way: there, signal 0
    is `CTRL_C_EVENT`, and `os.kill` would interrupt the process instead. A running state is then judged gone by its
    age alone."""
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, as another user's
        return True
    except OSError:
        return False
    return True


def status(directory: str | pathlib.Path) -> list[env]:
    """How far each sweep in `directory`, a ledger directory, has got. One `env` per ledger, sorted by name.

    Each has `name` (the ledger's file name without `.jsonl`), and from the state file a run writes as it goes:
    `state` (`"running"`, an outcome such as `"completed"` or `"model_changed"`, or `"gone"` for a run whose process
    has died without saying so, or on Windows one not updated for ten minutes), `total`, `todo`, `done`, `failed`, `started`, `updated` (both epoch seconds),
    `elapsed` and `eta` (seconds, `None` when it cannot be estimated) and `model`. A ledger written before runs kept
    a state file has `state` `"unknown"` and only `answered`, counted from the ledger itself.

    Reads files only, so it can be asked of a run going on in another process, or on another machine's disk.
    """
    directory = pathlib.Path(directory)
    out = []
    for path in sorted(directory.glob("*.jsonl")):
        stem = path.name[:-len(".jsonl")]
        if stem.endswith("-traces") or "-" not in stem:
            continue
        name, fp = stem.rsplit("-", 1)
        ledger = Ledger(directory, name, fp)
        if not ledger.state_path.exists():
            out.append(env(name=stem, state="unknown", answered=len(ledger.answers())))
            continue
        state = json.loads(ledger.state_path.read_text(encoding="utf-8"))
        tried = state["done"] + state["failed"]
        remaining = state["todo"] - tried
        if state["state"] == "running" and not _process_alive(state["pid"]) and time.time() - state["updated"] > _STALE_SECONDS:
            state["state"] = "gone"
        eta = state["elapsed"] / tried * remaining if (state["state"] == "running" and tried and remaining) else None
        out.append(env(name=stem, eta=eta, **{k: state[k] for k in ("state", "total", "todo", "done", "failed",
                                                                     "started", "updated", "elapsed", "model")}))
    return out


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min" if seconds >= 3600 else f"{seconds // 60} min {seconds % 60:02d} s"


def format_status(entries: Sequence[env], *, labels: Mapping[str, str] | None = None) -> list[str]:
    """`status`'s answer as lines for a terminal. `labels` maps a ledger's name (without the fingerprint) to what
    to call it, e.g. a stage's name, with the fingerprint after it in brackets, since a replicate shares the name;
    a ledger it does not name is called by its file name."""
    lines = []
    for entry in entries:
        name, fp = entry.name.rsplit("-", 1)
        label = f"{labels[name]} [{fp}]" if labels and name in labels else entry.name
        if entry.state == "unknown":
            lines.append(f"{label}: {entry.answered} answered (no state recorded)")
            continue
        progress = f"{entry.done + entry.failed} of {entry.todo} this run ({entry.failed} failed), {entry.total} items in all"
        when = datetime.datetime.fromtimestamp(entry.updated).strftime("%Y-%m-%d %H:%M")
        if entry.state == "running":
            eta = f", about {_duration(entry.eta)} left" if entry.eta is not None else ""
            lines.append(f"{label}: running, {progress}; {_duration(entry.elapsed)} so far{eta} (last update {when})")
        elif entry.state == "gone":
            lines.append(f"{label}: stopped without saying so, at {progress} (last update {when}); run again to resume")
        else:
            lines.append(f"{label}: {entry.state.replace('_', ' ')}, {progress}, in {_duration(entry.elapsed)} (at {when})")
    return lines


def main() -> None:  # pragma: no cover -- `status` and `format_status` carry the logic, and are tested
    import argparse
    parser = argparse.ArgumentParser(prog="python -m raven.librarian.sweep",
                                     description="How far the sweeps in a ledger directory have got.")
    parser.add_argument("directories", nargs="+", metavar="DIR", help="A ledger directory.")
    opts = parser.parse_args()
    for directory in opts.directories:
        entries = status(directory)
        if len(opts.directories) > 1:
            print(f"{directory}:")
        print("\n".join(format_status(entries)) if entries else f"no ledgers in {directory}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
