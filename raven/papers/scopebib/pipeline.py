"""Running the stages in order: what is left to ask, whether a model is needed for it, and asking it.

`load_corpus` reads and sifts the input, draws a sample where one is asked for, and measures the titles.
`read_state` says what the ledgers already settle and what each stage still has to ask. `run` does the
rest:

  - **It decides before any work whether a backend is needed.** Where every stage's instrument is named
    and nothing named is left to ask, it never connects: a replay needs no backend running.
  - **It refuses before any work when a stage that has something to ask cannot ask it**: the stage wants a
    model other than the loaded one, or its named instrument is not the one the current definition and model
    make, which makes it read-only. Answers asked now must never be filed under another instrument's fingerprint.
  - **Then it runs the stages in order**, each a `raven.librarian.sweep` with its own ledger. The reviewer
    and the extractor wait until the judge has settled every record, since the reviewer's control is drawn
    from the judge's keeps and a partial judge would draw another one next time.

A stage's instrument is a fingerprint of its material (`materials`) and the loaded model, which is why a
stage whose instrument is not named needs a connection even to find its ledger.
"""

__all__ = ["TITLE_BATCH", "ABSTRACT_BATCH", "REVIEW_BATCH", "EXTRACT_BATCH", "PILOT_SIZE",
           "PipelineError", "Corpus", "load_corpus", "materials",
           "State", "read_state", "RunResult", "run",
           "stamp_ledgers"]

import dataclasses
import functools
import json
import logging
import pathlib
import random
import re
from collections.abc import Callable, Container, Sequence

from unpythonic.env import env

from ...librarian import sweep
from .. import bibtex
from .. import fillbib
from .. import siftbib
from .. import utils as papers_utils
from . import definition as scope_definition
from . import extract as scope_extract
from . import judge as scope_judge
from . import records as scope_records
from . import report as scope_report
from . import review as scope_review
from . import models as scope_models
from . import runfile

logger = logging.getLogger(__name__)

# Records per model call, as the prototype asked them. Outside the instrument, as they were there, so that
# answers converted from it are read under the fingerprints they were stamped with.
TITLE_BATCH = 40
ABSTRACT_BATCH = 10
REVIEW_BATCH = 10
EXTRACT_BATCH = 10

#: How many records `--pilot` draws when it is given no number.
PILOT_SIZE = 50


class PipelineError(RuntimeError):
    """A run that cannot go ahead, refused before any work, with what to do about it."""


@dataclasses.dataclass(frozen=True)
class Corpus:
    """The records a run works on.

    `library`: the parsed `.bib`, after the sift and the sample; the reports cut the kept `.bib` from it.
    `records`: its records, in file order.
    `maybe_sift`: what the sift removed, or `None` where none ran.
    `measures`: each title's measure, by key (`raven.papers.scopebib.records.title_measures`).
    `maybe_fill`: the fill's audit (`raven.papers.fillbib.fill`), or `None` where none ran.
    """
    library: object
    records: list[scope_records.Record]
    maybe_sift: scope_report.SiftResult | None
    measures: dict[str, scope_records.TitleMeasure]
    maybe_fill: list[fillbib.AuditRow] | None = None


def load_corpus(bib_path: str | pathlib.Path, *,
                nlp,
                require: Sequence[str] = (),
                min_chars: Sequence[str] = (),
                maybe_sample: int | None = None,
                seed: int = 42,
                stopwords: Container[str] | None = None,
                fill: bool = False,
                maybe_crossref_cache: str | pathlib.Path | None = None) -> Corpus:
    """Read the `.bib` at `bib_path`, fill its abstracts, sift it, draw a sample, and measure the titles.

    `nlp`, `stopwords`: as `raven.papers.scopebib.records.title_measures` takes them.
    `fill`: fill missing and truncated abstracts from Crossref first, as `raven-fillbib` does. This sends
            every DOI in the corpus to Crossref.
    `maybe_crossref_cache`: where to keep Crossref's answers, so that a run after the first asks nothing
                            (`raven.papers.fillbib.fetch_crossref`); `None` for no cache.

    Raises `PipelineError` where a lookup failed: a record left unfilled would read differently from the run
    before, and every stage that shows it would ask about it again, on the worse text.
    `require`, `min_chars`: the sift's criteria, as `raven-siftbib` takes them; with neither, no sift runs.
    `maybe_sample`: draw this many records, after the sift; `None` for all of them.
    `seed`: which records the sample draws. The draw is over the keys in sorted order, so reordering the
                file does not change it.
    """
    library = bibtex.parse_file(pathlib.Path(bib_path), split_names=False)
    maybe_fill = None
    if fill:
        dois = [doi for doi in (papers_utils.normalize_doi(bibtex.entry_field(entry, "doi"))
                                for entry in library.entries) if doi]
        abstracts, failed = fillbib.fetch_crossref(
            dois, maybe_cache_path=pathlib.Path(maybe_crossref_cache) if maybe_crossref_cache else None)
        if failed:
            shown = ", ".join(sorted(failed)[:5]) + (f", and {len(failed) - 5} more" if len(failed) > 5 else "")
            raise PipelineError(f"filling abstracts: Crossref could not be asked about {len(failed)} "
                                f"{'DOI' if len(failed) == 1 else 'DOIs'} ({shown}); run again when it answers. "  # plural
                                f"First failure: {next(iter(failed.values()))}")
        library, maybe_fill = fillbib.fill(library, abstracts)
    maybe_sift = None
    criteria = [siftbib.require_field(field) for field in require] + [siftbib.parse_min_chars(spec)
                                                                      for spec in min_chars]
    if criteria:
        library, dropped = siftbib.sift(library, criteria)
        maybe_sift = scope_report.SiftResult(dropped=dropped, criteria=criteria)
    if maybe_sample is not None:
        keys = sorted(entry.key for entry in library.entries)
        library = scope_report.select(library, set(random.Random(seed).sample(keys, min(maybe_sample, len(keys)))))
    records = scope_records.records_of(library)
    return Corpus(library, records, maybe_sift, scope_records.title_measures(records, nlp, stopwords=stopwords),
                  maybe_fill)


def materials(definition: scope_definition.Definition, *, replicate: int = 1) -> dict[str, str]:
    """Each stage's instrument material, by stage.

    A replicate past the first is part of the material, so it gets ledgers of its own and asks afresh; the
    first adds nothing, so its fingerprints are those of a run that never heard of replicates.
    """
    out = {"judge": scope_judge.instrument_material(definition),
           "review": scope_review.instrument_material(definition),
           "extract": scope_extract.instrument_material(definition)}
    if replicate > 1:
        out = {stage: json.dumps({"material": material, "replicate": replicate}, sort_keys=True)
               for stage, material in out.items()}
    return out


@dataclasses.dataclass(frozen=True)
class _Ledgers:
    titles: sweep.Ledger
    abstracts: sweep.Ledger
    review: sweep.Ledger
    extract: sweep.Ledger


def _open_ledgers(directory: pathlib.Path, fingerprints: dict[str, str | None]) -> _Ledgers:
    # A stage whose fingerprint is `None` cannot be found (`run`); a ledger under it names no file that exists,
    # so it reads as answering nothing, and nothing writes to it.
    titles, abstracts = scope_judge.ledgers(directory, fingerprints["judge"])
    return _Ledgers(titles, abstracts,
                    scope_review.ledger(directory, fingerprints["review"]),
                    scope_extract.ledger(directory, fingerprints["extract"]))


def _ledger_paths(directory: pathlib.Path, fingerprints: dict[str, str | None]) -> dict[str, pathlib.Path | None]:
    """The file each stage's answers are in, `None` where the stage has no fingerprint; the judge's by its
    title pass, which every judged record has an answer in."""
    ledgers = _open_ledgers(directory, {stage: fingerprint or "" for stage, fingerprint in fingerprints.items()})
    paths = {"judge": ledgers.titles.path, "review": ledgers.review.path, "extract": ledgers.extract.path}
    return {stage: path if fingerprints[stage] else None for stage, path in paths.items()}


@dataclasses.dataclass(frozen=True)
class State:
    """What the ledgers settle, and what each stage still has to ask.

    `final`, `reviews`, `extractions`: as `raven.papers.scopebib.report.write_reports` takes them.
    `control`: the reviewer's control group, drawn once the judge has settled every record; empty until then.
    `todo`: records left to ask, by stage — `"titles"` and `"abstracts"` for the judge's two passes, then
            `"review"` and `"extract"`, which are `None` while they wait on the judge.
    """
    final: dict[str, dict]
    reviews: dict[str, dict]
    extractions: dict[str, dict]
    control: list[str]
    todo: dict[str, list[scope_records.Record] | None]

    def _get_judge_settled(self) -> bool:
        return not self.todo["titles"] and not self.todo["abstracts"]
    judge_settled = property(fget=_get_judge_settled, doc="Whether the judge has answered every record.")


def read_state(corpus: Corpus, definition: scope_definition.Definition, *,
               ledger_dir: str | pathlib.Path,
               fingerprints: dict[str, str],
               control_size: int,
               seed: int) -> State:
    """What the ledgers under `fingerprints` (by stage) settle for `corpus`, and what is left.

    `control_size`, `seed`: the reviewer's control, as `raven.papers.scopebib.review.draw_control` takes them.
    """
    ledgers = _open_ledgers(pathlib.Path(ledger_dir), fingerprints)
    records, measures = corpus.records, corpus.measures
    todo = {"titles": scope_judge.todo_titles(records, titles=ledgers.titles),
            "abstracts": scope_judge.todo_abstracts(records, titles=ledgers.titles, abstracts=ledgers.abstracts,
                                                     measures=measures),
            "review": None, "extract": None}
    final = scope_judge.final_answers(records, titles=ledgers.titles, abstracts=ledgers.abstracts, measures=measures)
    in_corpus = {record.key for record in records}
    # Read as the records are now: an answer about text that has been edited since is not one, and the
    # record is asked again.
    reviews = {key: answer for key, answer in
               ledgers.review.answers({record.key: scope_review.digest(record) for record in records}).items()
               if key in in_corpus}
    extractions = {key: answer for key, answer in
                   ledgers.extract.answers({record.key: scope_extract.digest(record) for record in records}).items()
                   if key in in_corpus}
    control = []
    if not todo["titles"] and not todo["abstracts"]:
        control = scope_review.draw_control(final, size=control_size, seed=seed)
        wanted = {key for key, answer in final.items() if answer["verdict"] == "drop"} | set(control)
        todo["review"] = [record for record in records if record.key in wanted and record.key not in reviews]
        todo["extract"] = ([record for record in records
                            if final[record.key]["verdict"] == "keep" and record.key not in extractions]
                           if definition.fields else [])
    return State(final, reviews, extractions, control, todo)


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What a run did.

    `state`: what the ledgers settle afterwards, for the reports.
    `stages`: each stage's model and instrument, for the run file.
    `models`: each stage's model as a methods section names it — key, quantization and what was declared —
              from its ledger's header, or the bare key where the ledger has none.
    `messages`: what the person should be told: a stage that stopped, or one that waits on the judge.
    `connected`: whether a backend was asked anything at all.
    """
    state: State
    stages: dict[str, runfile.StagePlan]
    models: dict[str, str]
    messages: list[str]
    connected: bool


_TODO_KEYS = {"judge": ("titles", "abstracts"), "review": ("review",), "extract": ("extract",)}


def _model_in_ledger(ledger: sweep.Ledger) -> str | None:
    headers = ledger.headers()
    return headers[-1]["header"].get("model_id") if headers else None


def _describe_model(ledger: sweep.Ledger, maybe_model: str | None) -> str:
    """`qwen3.6-35b-a3b (IQ4_NL_XL; KV cache 4-bit)` from the ledger's latest header; `maybe_model` where it has
    none, and "not recorded" where neither says."""
    headers = ledger.headers()
    if not headers:
        return maybe_model or "not recorded"
    header = headers[-1]["header"]
    details = [detail for detail in (header.get("quantization"), header.get("declared"))
               if detail and detail != "not stated"]
    return f"{header.get('model_id')}" + (f" ({'; '.join(details)})" if details else "")


def run(corpus: Corpus, definition: scope_definition.Definition, *,
        stages: Sequence[str],
        plans: dict[str, runfile.StagePlan],
        ledger_dir: str | pathlib.Path,
        connect: Callable[[], env],
        control_size: int,
        seed: int,
        replicate: int = 1,
        on_progress: Callable[[str, env], None] | None = None,
        on_instruments: Callable[[dict[str, runfile.StagePlan]], None] | None = None,
        make_switcher: Callable[[env], "scope_models.Switcher | None"] | None = None) -> RunResult:
    """Run `stages`, a subset of `raven.papers.scopebib.runfile.STAGES`, over `corpus`.

    `plans`: each stage's model and named instrument (`raven.papers.scopebib.runfile.StagePlan`).
    `ledger_dir`: where every stage's ledgers are.
    `connect`: returns the LLM settings, connecting to the backend; called at most once, and only when
               something has to be asked or a fingerprint computed. It should raise where no backend answers.
    `control_size`, `seed`: the reviewer's control.
    `replicate`: which replicate this is (`materials`).
    `on_progress`: called with the stage and the sweep's progress `env` after each batch.
    `on_instruments`: called once each stage's model and instrument are known and before anything is asked,
                      with them as `RunResult.stages` will hold them; for writing the run file at once, so that
                      a run that dies partway leaves its recipe behind. Called again whenever a stage's
                      instrument becomes known later, when its model is loaded.
    `make_switcher`: given the connected settings, a `raven.papers.scopebib.models.Switcher` for loading each
                     stage's model in turn, or `None` where the backend cannot switch. Without one, a stage
                     wanting a model other than the loaded one is refused before any work. With one, every
                     model wanted must be on the backend, each stage that has something to ask runs with its
                     own, and the model loaded at the start is loaded again at the end.

    A stage naming no model uses the one loaded when the run starts.

    Raises `PipelineError` before any work when a stage that has something to ask cannot ask it.
    """
    ledger_dir = pathlib.Path(ledger_dir)
    stage_materials = materials(definition, replicate=replicate)
    named = {stage: plans.get(stage, runfile.StagePlan()).maybe_instrument for stage in runfile.STAGES}
    wanted_models = {stage: plans.get(stage, runfile.StagePlan()).maybe_model for stage in runfile.STAGES}
    connect = functools.cache(connect)

    def state_under(fingerprints: dict[str, str]) -> State:
        return read_state(corpus, definition, ledger_dir=ledger_dir, fingerprints=fingerprints,
                          control_size=control_size, seed=seed)

    def has_work(state: State, stage: str) -> bool:
        """Whether `stage` will have something to ask, a stage waiting on a judge that will run counting."""
        lists = [state.todo[key] for key in _TODO_KEYS[stage]]
        if any(items is None for items in lists):
            return "judge" in stages and not state.judge_settled
        return any(lists)

    # A named instrument with no ledger at all is a typo, or the wrong ledgers directory, far more often than
    # a stage never run; either way the person should hear where it was looked for.
    missing = []
    for stage, path in _ledger_paths(ledger_dir, named).items():
        if path is not None and not path.exists():
            missing.append(f"{stage}: no ledger for the named instrument {named[stage]}; looked for {path}")

    # Without a connection, where every instrument is named: a replay that finds nothing to ask stops here.
    if all(named.values()):
        state = state_under(named)
        for stage in runfile.STAGES:
            left = sum(len(state.todo[key] or ()) for key in _TODO_KEYS[stage])
            if left:
                logger.info(f"run: {stage}: {left} records left under the named instrument {named[stage]}")
        if not any(has_work(state, stage) for stage in stages):
            result = _finish(state, named, wanted_models, ledger_dir, stages, connected=False)
            return dataclasses.replace(result, messages=missing)

    settings = connect()
    initial = settings.model_record.model_id
    switcher = make_switcher(settings) if make_switcher is not None else None
    target = {stage: wanted_models[stage] or initial for stage in runfile.STAGES}
    # A stage's fingerprint needs its model's record, so it is known now only for a stage on the model loaded;
    # one switched to is fingerprinted when its model is loaded.
    live = {stage: sweep.fingerprint(settings, stage_materials[stage]) for stage in runfile.STAGES
            if target[stage] == initial}
    behind = {stage: initial for stage in live}  # the model behind each fingerprint known

    problems = {}
    if switcher is not None:
        wanted_elsewhere = sorted({target[stage] for stage in stages} - {initial})
        if wanted_elsewhere:
            maybe_missing = sorted(set(wanted_elsewhere) - switcher.available())
            if maybe_missing:
                raise PipelineError(f"the backend has no model {', '.join(map(repr, maybe_missing))}, which "
                                    f"{'a stage wants' if len(maybe_missing) == 1 else 'stages want'}; "
                                    f"check the name with `lms ls`, or in LM Studio")
    for stage in runfile.STAGES:
        if target[stage] != initial and switcher is None:
            problems[stage] = (f"wants the model '{target[stage]}', and '{initial}' is loaded, which this backend "
                               f"cannot change by itself; only LM Studio's can")
        elif stage in live and named[stage] is not None and named[stage] != live[stage]:
            problems[stage] = (f"its named instrument {named[stage]} is not the one the current definition and "
                               f"model make ({live[stage]}), so it is read-only")
    # A stage whose instrument is not named, on a model not loaded, has a ledger nobody can find until that
    # model is: its fingerprint needs the model's record. Without a switcher that is a refusal where the stage
    # is to run; otherwise, and for a stage not run, its answers are simply absent from the reports.
    unfindable = [stage for stage in runfile.STAGES if named[stage] is None and stage not in live]
    refused = [stage for stage in unfindable if stage in stages and stage in problems]
    if refused:
        reasons = "; ".join(f"{stage}, which {problems[stage]}" for stage in refused)
        raise PipelineError(f"cannot find the answers of {reasons}. Load that model, or name the stage's "
                            f"instrument with --instrument or a run file.")
    fingerprints = {stage: named[stage] or live.get(stage) for stage in runfile.STAGES}
    skipped = [f"{stage}'s answers are not in the reports: it wants the model '{target[stage]}', which this run "
               f"did not load, and its instrument is not named"
               for stage in unfindable if stage not in stages or switcher is None]
    state = state_under(fingerprints)

    blocked = [stage for stage in stages if stage in problems and has_work(state, stage)]
    if blocked:
        message = "; ".join(f"{stage} has records left to ask, but {problems[stage]}" for stage in blocked)
        first = runfile.STAGES.index(blocked[0])
        runnable = [stage for stage in stages if runfile.STAGES.index(stage) < first and has_work(state, stage)]
        if runnable:
            message += f". The stages before it can run: --until {runnable[-1]}"
        raise PipelineError(message)

    def plans_now() -> dict[str, runfile.StagePlan]:
        return _stage_plans(fingerprints, target, ledger_dir, behind=behind)

    if on_instruments is not None:
        on_instruments(plans_now())
    messages = missing + skipped
    current = initial

    def use_model_of(stage: str) -> None:
        """Load `stage`'s model where another is loaded, and learn the stage's instrument under it."""
        nonlocal settings, current, state
        if target[stage] == current:
            return
        settings = switcher.load(target[stage])
        current = target[stage]
        fingerprint = sweep.fingerprint(settings, stage_materials[stage])
        if named[stage] is not None and named[stage] != fingerprint:
            raise _Stopped(f"{stage}: its named instrument {named[stage]} is not the one the current definition "
                           f"and model make ({fingerprint}), so it is read-only")
        fingerprints[stage] = fingerprint
        behind[stage] = current
        state = state_under(fingerprints)
        if on_instruments is not None:
            on_instruments(plans_now())

    def sweep_over(stage: str, ask, records: list[scope_records.Record], ledger: sweep.Ledger, batch_size: int,
                   digest):
        callback = (lambda progress: on_progress(stage, progress)) if on_progress else None
        result = sweep.Sweep(settings, functools.partial(ask, definition=definition),
                             [(record.key, record) for record in records], ledger, batch_size=batch_size,
                             header=sweep.instrument_header(settings), on_item=callback,
                             digest=digest).run()
        if result.outcome is not sweep.completed:
            raise _Stopped(f"{stage} stopped: {result.message or result.outcome}")
        if result.failed:
            messages.append(f"{stage}: {result.failed} records failed and are asked again on the next run")

    def ledgers() -> _Ledgers:
        return _open_ledgers(ledger_dir, fingerprints)

    try:
        try:
            if "judge" in stages and has_work(state, "judge"):
                use_model_of("judge")
                if state.todo["titles"]:
                    sweep_over("judge", scope_judge.ask_titles, state.todo["titles"], ledgers().titles, TITLE_BATCH,
                               scope_judge.title_digest)
                    state = state_under(fingerprints)
                if state.todo["abstracts"]:
                    sweep_over("judge", scope_judge.ask_abstracts, state.todo["abstracts"], ledgers().abstracts,
                               ABSTRACT_BATCH, scope_judge.abstract_digest)
                    state = state_under(fingerprints)
            if not state.judge_settled and any(stage in stages for stage in ("review", "extract")):
                messages.append("the reviewer and the extractor wait until the judge has answered every record")
            if "review" in stages and has_work(state, "review") and state.judge_settled:
                use_model_of("review")
                if state.todo["review"]:
                    # Shuffled, so that the control is not asked about in a block of its own.
                    todo = sorted(state.todo["review"], key=lambda record: record.key)
                    random.Random(seed).shuffle(todo)
                    sweep_over("review", scope_review.ask_reviews, todo, ledgers().review, REVIEW_BATCH,
                               scope_review.digest)
                    state = state_under(fingerprints)
            if "extract" in stages and has_work(state, "extract") and state.judge_settled:
                use_model_of("extract")
                if state.todo["extract"]:
                    sweep_over("extract", scope_extract.ask_fields, state.todo["extract"], ledgers().extract,
                               EXTRACT_BATCH, scope_extract.digest)
                    state = state_under(fingerprints)
        finally:
            if current != initial:
                logger.info(f"run: loading '{initial}' again, as at the start")
                settings = switcher.load(initial)
                current = initial
    except _Stopped as stopped:
        messages.append(str(stopped))
        state = state_under(fingerprints)
    result = _finish(state, fingerprints, target, ledger_dir, stages, connected=True, behind=behind)
    return dataclasses.replace(result, messages=messages + result.messages)


class _Stopped(Exception):
    pass


def _stage_plans(fingerprints: dict[str, str | None], wanted_models: dict[str, str | None], ledger_dir: pathlib.Path,
                 *, behind: dict[str, str] | None = None) -> dict[str, runfile.StagePlan]:
    """Each stage's model and instrument as the run file should record them.

    `behind`: the model behind each stage's fingerprint, where this run computed it with that model loaded.
    Elsewhere the model is the one the stage's ledger names, or else the one wanted.
    """
    behind = behind or {}
    ledgers = _open_ledgers(ledger_dir, fingerprints)
    by_stage = {"judge": ledgers.titles, "review": ledgers.review, "extract": ledgers.extract}
    plans = {}
    for stage in runfile.STAGES:
        model = behind.get(stage) or _model_in_ledger(by_stage[stage]) or wanted_models[stage]
        plans[stage] = runfile.StagePlan(model, fingerprints[stage])
    return plans


def _finish(state: State, fingerprints: dict[str, str | None], wanted_models: dict[str, str | None],
            ledger_dir: pathlib.Path, stages: Sequence[str], *, connected: bool,
            behind: dict[str, str] | None = None) -> RunResult:
    """The result, with each stage's model and instrument as `_stage_plans` records them."""
    plans = _stage_plans(fingerprints, wanted_models, ledger_dir, behind=behind)
    ledgers = _open_ledgers(ledger_dir, fingerprints)
    by_stage = {"judge": ledgers.titles, "review": ledgers.review, "extract": ledgers.extract}
    models = {stage: _describe_model(by_stage[stage], plans[stage].maybe_model) for stage in runfile.STAGES}
    return RunResult(state, plans, models, [], connected)


# Each stage's ledger file is `{name}-{fingerprint}.jsonl`, a replicate's under a fingerprint of its own.
_LEDGER_FILE = re.compile(r"^(judge-titles|judge-abstracts|review|extract)-([0-9a-f]{8})\.jsonl$")
_DIGESTS = {"judge-titles": scope_judge.title_digest, "judge-abstracts": scope_judge.abstract_digest,
            "review": scope_review.digest, "extract": scope_extract.digest}


def stamp_ledgers(records: Sequence[scope_records.Record], ledger_dir: str | pathlib.Path) -> dict[str, int]:
    """Give the answers in every ledger under `ledger_dir` the digest of the record they were given about,
    where they have none (`raven.librarian.sweep.Ledger.stamp`).

    For ledgers written before answers carried digests, which a run would otherwise read as answering nothing
    and ask again whole. `records` must be the corpus **as the run that wrote them saw it** — a later one
    would declare old answers current for text they were never given — and a record it does not hold is
    left alone. Returns how many answers were stamped, by ledger file name.
    """
    directory = pathlib.Path(ledger_dir)
    out = {}
    for path in sorted(directory.glob("*.jsonl")):
        match = _LEDGER_FILE.match(path.name)
        if not match:
            continue  # a traces file, or something else
        name, fingerprint = match.groups()
        digest = _DIGESTS[name]
        out[path.name] = sweep.Ledger(directory, name, fingerprint).stamp({record.key: digest(record)
                                                                           for record in records})
    return out
