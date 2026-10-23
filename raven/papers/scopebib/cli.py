"""`raven-scopebib`: which records of a literature search are about what it searched for.

    raven-scopebib run corpus.bib --definition corpus.scope.toml --out-dir out --require abstract
    raven-scopebib run --run-file out/run.toml          # the same run again, from its own record
    raven-scopebib finish out/04-CHECK-BY-HAND-marked.xlsx --run-file out/run.toml

Subcommands: `run` does every stage in order (`--until STAGE` stops after one); `judge`, `review` and
`extract` run one stage; `report` writes the reports from what is saved; `finish` applies a person's decisions
on a marked copy of the worklist; `compare` says how far two runs agree. Every command but `finish` writes `run.toml` beside its reports, naming
everything the run used, so passing it back with `--run-file` repeats the run.
"""

__all__ = ["COMMANDS", "DEFAULTS", "Plan", "UsageError", "resolve", "execute", "compare_runs", "main"]

import argparse
import dataclasses
import datetime
import logging
import pathlib
import sys

from ... import __version__
from ...common import tabular

logger = logging.getLogger(__name__)

#: The subcommands, and the stages each runs.
COMMANDS = {"run": ("judge", "review", "extract"), "judge": ("judge",), "review": ("review",),
            "extract": ("extract",), "report": (), "finish": ()}

#: What a setting is when neither the command line nor the run file says.
DEFAULTS = {"maybe_seed": 42, "maybe_control": 40, "maybe_replicate": 1, "maybe_format": "xlsx",
            "maybe_server_nlp": False}

_PILOT_SIZE = 50  # as `pipeline.PILOT_SIZE`, which is not imported here, the parser coming before the heavy imports


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--definition", default=None, metavar="FILE",
                        help="The definition file: the scope, the judge's tests, the extractor's fields and the "
                             "rules. Default: <bib stem>.scope.toml beside the .bib.")
    common.add_argument("--run-file", dest="run_file", default=None, metavar="FILE",
                        help="A run file, as every run writes beside its reports: its settings, models and "
                             "instruments, which the options here override.")
    common.add_argument("--out-dir", dest="out_dir", default=None, metavar="DIR",
                        help="Where the reports go. Default: the run file's directory.")
    common.add_argument("--ledgers", default=None, metavar="DIR",
                        help="Where the saved answers are. Default: 'ledgers' in the output directory.")
    common.add_argument("--require", action="append", default=None, metavar="FIELD",
                        help="Sift first: keep only records carrying FIELD, as raven-siftbib does. Repeatable.")
    common.add_argument("--min-chars", dest="min_chars", action="append", default=None, metavar="FIELD=N",
                        help="Sift first: keep only records whose FIELD runs to N characters. Repeatable.")
    sample = common.add_mutually_exclusive_group()
    sample.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Work on a seeded random sample of N records.")
    sample.add_argument("--pilot", type=int, nargs="?", const=_PILOT_SIZE, default=None, metavar="N",
                        help=f"A calibration run over a sample of N records (default {_PILOT_SIZE}), its reports "
                             f"in 'pilot' under the output directory. Its answers count toward the full run.")
    common.add_argument("--seed", type=int, default=None,
                        help=f"Which records a sample and the reviewer's control draw "
                             f"(default {DEFAULTS['maybe_seed']}).")
    common.add_argument("--control", type=int, default=None, metavar="N",
                        help=f"How many kept records the reviewer reads, unlabelled, as its control "
                             f"(default {DEFAULTS['maybe_control']}).")
    common.add_argument("--replicate", type=int, default=None, metavar="K",
                        help="Ask again, into ledgers of its own, for test-retest reliability (default 1).")
    common.add_argument("--format", default=None, choices=tabular.FORMATS,
                        help=f"The reports' table format (default {DEFAULTS['maybe_format']}).")
    common.add_argument("--model", action="append", default=[], metavar="STAGE=KEY",
                        help="The model a stage asks, overriding the run file. Repeatable.")
    common.add_argument("--instrument", action="append", default=[], metavar="STAGE=FP",
                        help="Read a stage's answers from the ledger of this instrument, as a ledger's file "
                             "name shows it. Read-only unless the current definition and model make "
                             "the same one. Repeatable.")
    common.add_argument("--context-length", dest="context_length", type=int, default=None, metavar="N",
                        help="The context length to load each model with, where a step's model is loaded for it. "
                             "Default: the one LM Studio has saved for each model.")
    common.add_argument("--backend-url", dest="backend_url", default=None, metavar="URL",
                        help="The LLM backend. Default: Raven-librarian's.")
    common.add_argument("--server-nlp", dest="server_nlp", action=argparse.BooleanOptionalAction, default=None,
                        help="Measure titles with Raven-server's spaCy model, where it answers. Default: locally, "
                             "with the model the title bounds were calibrated with, so that a run repeats "
                             "exactly on any machine.")
    common.add_argument("--server-url", dest="server_url", default=None, metavar="URL",
                        help="Raven-server, for --server-nlp.")
    common.add_argument("--log", metavar="PATH", default=None, help="Write the log here instead of to stderr.")
    common.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        help="How much to log.")

    parser = argparse.ArgumentParser(prog="raven-scopebib", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    helps = {"run": "Every stage in order, then the reports.",
             **{stage: f"The {stage} stage alone, then the reports." for stage in ("judge", "review", "extract")},
             "report": "The reports, from the answers already saved."}
    for command, text in helps.items():
        subparser = commands.add_parser(command, parents=[common], help=text)
        subparser.add_argument("bib", nargs="?", default=None, metavar="corpus.bib",
                               help="The BibTeX file to filter. Optional where --run-file names one.")
        if command == "run":
            subparser.add_argument("--until", default=None, choices=COMMANDS["run"], help="Stop after this stage.")
    # `finish` takes the marked sheet where the others take the `.bib`, which comes from the run file or `--bib`.
    finish = commands.add_parser("finish", parents=[common],
                                 help="Apply a person's decisions on a marked copy of 04-CHECK-BY-HAND.")
    finish.add_argument("sheet", metavar="MARKED_SHEET", help="The marked copy, in any table format.")
    finish.add_argument("--bib", default=None, metavar="FILE",
                        help="The BibTeX file the run filtered. Optional where --run-file names one.")
    finish.add_argument("--require-every-decision", dest="require_every_decision", action="store_true",
                        help="Refuse a blank decision rather than reading it as a drop.")
    compare = commands.add_parser("compare", help="How far two runs agree: a replicate against the first, or one "
                                                  "model against another.")
    compare.add_argument("run_a", metavar="RUN_A", help="The first run's run.toml.")
    compare.add_argument("run_b", metavar="RUN_B", help="The second run's run.toml. The tables go beside it.")
    compare.add_argument("--format", default=None, choices=tabular.FORMATS,
                         help="The tables' format (default: the second run's).")
    compare.add_argument("--server-url", dest="server_url", default=None, metavar="URL",
                         help="Raven-server, where the first run measured its titles there.")
    compare.add_argument("--log", metavar="PATH", default=None, help="Write the log here instead of to stderr.")
    compare.add_argument("--log-level", default="INFO",
                         choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), help="How much to log.")
    return parser


@dataclasses.dataclass(frozen=True)
class Plan:
    """A command, resolved: what to run, on what, and where everything goes.

    `run_file`: the settings and each stage's plan, the command line over the run file over `DEFAULTS`.
    `stages`: the stages to run, in order.
    `reports_dir`: where the reports and `run.toml` go.
    `ledger_dir`: where the answers are.
    """
    command: str
    run_file: object
    stages: tuple[str, ...]
    reports_dir: pathlib.Path
    ledger_dir: pathlib.Path


class UsageError(ValueError):
    """A command line that cannot be acted on."""


def resolve(opts: argparse.Namespace) -> Plan:
    """The `Plan` for parsed command-line options. Raises `UsageError`, or `runfile.RunFileError`."""
    from . import runfile

    maybe_run_file = pathlib.Path(opts.run_file).expanduser().resolve() if opts.run_file else None
    loaded = runfile.load_run_file(maybe_run_file) if maybe_run_file else runfile.RunFile(runfile.Settings(), {})

    def path(value):
        return pathlib.Path(value).expanduser().resolve() if value else None
    given = {"maybe_bib": path(getattr(opts, "bib", None)), "maybe_definition": path(opts.definition),
             "maybe_ledgers": path(opts.ledgers),
             "require": tuple(opts.require) if opts.require else None,
             "min_chars": tuple(opts.min_chars) if opts.min_chars else None,
             "maybe_sample": opts.sample, "maybe_pilot": opts.pilot, "maybe_seed": opts.seed,
             "maybe_control": opts.control, "maybe_replicate": opts.replicate, "maybe_format": opts.format,
             "maybe_server_nlp": opts.server_nlp, "maybe_context_length": opts.context_length}
    # A sample and a pilot given on the command line replace each other, not only their own setting.
    if opts.sample is not None:
        given["maybe_pilot"] = None
        loaded = dataclasses.replace(loaded, settings=dataclasses.replace(loaded.settings, maybe_pilot=None))
    if opts.pilot is not None:
        loaded = dataclasses.replace(loaded, settings=dataclasses.replace(loaded.settings, maybe_sample=None))
    merged = runfile.with_overrides(
        loaded, settings=given,
        models=[runfile.parse_override(spec, option="--model") for spec in opts.model],
        instruments=[runfile.parse_override(spec, option="--instrument") for spec in opts.instrument])
    settings = dataclasses.replace(merged.settings, **{name: getattr(merged.settings, name)
                                                       if getattr(merged.settings, name) is not None else value
                                                       for name, value in DEFAULTS.items()})
    if settings.maybe_bib is None:
        raise UsageError("no .bib to work on: give it, or a run file naming one")
    if settings.maybe_definition is None:
        default = settings.maybe_bib.parent / f"{settings.maybe_bib.stem}.scope.toml"
        if not default.exists():
            raise UsageError(f"no definition file: give --definition, or a run file naming one, or put it beside "
                             f"the .bib as {default}")
        settings = dataclasses.replace(settings, maybe_definition=default)
    for name in ("maybe_seed", "maybe_control", "maybe_replicate", "maybe_sample", "maybe_pilot",
                 "maybe_context_length"):
        value = getattr(settings, name)
        least = {"maybe_seed": None, "maybe_control": 0}.get(name, 1)
        if value is not None and least is not None and value < least:
            raise UsageError(f"--{name.removeprefix('maybe_')} must be at least {least}, got {value}")

    if opts.out_dir:
        root = path(opts.out_dir)
        reports_dir = root / "pilot" if settings.maybe_pilot is not None else root
    elif maybe_run_file is not None:
        # The run file lives beside the reports it describes, so passing it back writes there.
        root = reports_dir = maybe_run_file.parent
    else:
        raise UsageError("no output directory: give --out-dir, or a run file to write beside")
    ledger_dir = settings.maybe_ledgers or root / "ledgers"
    settings = dataclasses.replace(settings, maybe_ledgers=ledger_dir)

    stages = COMMANDS[opts.command]
    if opts.command == "run" and getattr(opts, "until", None):
        stages = stages[:stages.index(opts.until) + 1]
    return Plan(opts.command, runfile.RunFile(settings, merged.stages), stages, reports_dir, ledger_dir)


def _provenance(plan: Plan, models: dict[str, str], stages: dict, nlp_description: str) -> list[str]:
    settings = plan.run_file.settings
    lines = [f"raven-scopebib {__version__}, {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"input: {settings.maybe_bib}", f"definition: {settings.maybe_definition}"]
    criteria = list(settings.require) + list(settings.min_chars)
    if criteria:
        lines.append(f"sifted on: {', '.join(criteria)}")
    if settings.maybe_pilot is not None:
        lines.append(f"pilot of {settings.maybe_pilot} records, seed {settings.maybe_seed}")
    elif settings.maybe_sample is not None:
        lines.append(f"sample of {settings.maybe_sample} records, seed {settings.maybe_seed}")
    lines.append(f"reviewer's control: {settings.maybe_control} records, seed {settings.maybe_seed}"
                 + (f"; replicate {settings.maybe_replicate}" if settings.maybe_replicate > 1 else ""))
    if nlp_description:
        lines.append(f"titles measured with {nlp_description}")
    for stage, stage_plan in stages.items():
        lines.append(f"{stage}: model {models[stage]}, instrument {stage_plan.maybe_instrument or 'none'}")
    return lines


def execute(plan: Plan, *, nlp, connect, nlp_description: str = "", sheet: str | None = None,
            require_every_decision: bool = False) -> int:
    """Carry out `plan`. Returns the exit status.

    `nlp`: for measuring titles (`raven.papers.scopebib.records.title_measures`).
    `nlp_description`: what `nlp` is, for the reports' provenance.
    `connect`: returns the LLM settings, connecting to the backend, as `raven.papers.scopebib.pipeline.run`
               takes it.
    `sheet`, `require_every_decision`: for `finish`.
    """
    from . import definition as scope_definition
    from . import models as scope_models
    from . import pipeline, report, runfile

    settings = plan.run_file.settings
    definition = scope_definition.load_definition(settings.maybe_definition)
    corpus = pipeline.load_corpus(settings.maybe_bib, nlp=nlp, require=settings.require,
                                  min_chars=settings.min_chars,
                                  maybe_sample=settings.maybe_pilot or settings.maybe_sample, seed=settings.maybe_seed)
    print(f"{len(corpus.records)} records"
          + (f", after the sift removed {len(corpus.maybe_sift.dropped)}" if corpus.maybe_sift else ""))
    run_path = plan.reports_dir / "run.toml"

    def write_run_file(stages) -> None:
        maybe_archive = runfile.write_run_file(run_path, runfile.RunFile(settings, stages),
                                               comments=[f"raven-scopebib {__version__}"])
        if maybe_archive is not None:
            print(f"the hand-written {run_path.name} was kept as {maybe_archive.name}")

    # `finish` asks nothing and records no run, so it writes no run file.
    result = pipeline.run(corpus, definition, stages=plan.stages, plans=plan.run_file.stages,
                          ledger_dir=plan.ledger_dir, connect=connect, control_size=settings.maybe_control,
                          seed=settings.maybe_seed, replicate=settings.maybe_replicate,
                          on_instruments=write_run_file if plan.command != "finish" else None,
                          make_switcher=lambda connected: scope_models.switcher_for(
                              connected, reconnect=connect, maybe_context_length=settings.maybe_context_length))
    print(report.title_bounds_note(corpus.records, corpus.measures))
    title_only = sum(1 for record in corpus.records if not record.abstract)
    print(f"{title_only} title-only {'record' if title_only == 1 else 'records'}"  # plural
          + (" in this sample" if settings.maybe_pilot or settings.maybe_sample else ""))
    for message in result.messages:
        print(f"note: {message}")
    if not result.connected:
        print("no backend was needed: every instrument is named, and nothing was left to ask in the stages run")
    state = result.state
    common = dict(library=corpus.library, records=corpus.records, definition=definition, final=state.final,
                  reviews=state.reviews, extractions=state.extractions,
                  provenance=_provenance(plan, result.models, result.stages, nlp_description),
                  table_format=settings.maybe_format)

    if plan.command == "finish":
        written, warnings = report.finish(plan.reports_dir, sheet=sheet, require_every_decision=require_every_decision,
                                          **common)
        for warning in warnings:
            print(f"warning: {warning}")
    else:
        if settings.maybe_pilot is not None:
            written = [report.write_pilot_report(plan.reports_dir, records=corpus.records, measures=corpus.measures,
                                                 definition=definition, final=state.final, reviews=state.reviews,
                                                 extractions=state.extractions, provenance=common["provenance"],
                                                 seed=settings.maybe_seed, table_format=settings.maybe_format)]
        else:
            written = report.write_reports(plan.reports_dir, control=state.control, maybe_sift=corpus.maybe_sift,
                                           **common)
        write_run_file(result.stages)
        written.append(run_path)
        if settings.maybe_pilot is None:
            print(f"{report.CHECK_BY_HAND} is rewritten on every run: mark a copy saved under a name of your own")
    outcomes = report.outcomes(corpus.records, definition=definition, final=state.final, reviews=state.reviews,
                               extractions=state.extractions)
    counts = {action: sum(1 for outcome in outcomes.values() if outcome.action == action)
              for action in (report.REMOVED, report.TO_A_PERSON, report.KEPT, report.PENDING)}
    print("; ".join(f"{action}: {count}" for action, count in counts.items()))
    print(f"wrote, in {plan.reports_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


def compare_runs(path_a: str | pathlib.Path, path_b: str | pathlib.Path, *, nlp, nlp_description: str = "",
                 table_format: str | None = None) -> list[pathlib.Path]:
    """Compare the runs whose run files are `path_a` and `path_b`, writing the tables beside `path_b`.

    The records are the first run's — its input, sift and sample — and each run's answers are read under its
    own definition and instruments, which must all be named, as they are in a run file a run wrote. Never
    connects to a backend. Returns the paths written.
    """
    from . import compare, pipeline, runfile
    from . import definition as scope_definition

    runs = []
    for path in (pathlib.Path(path_a).expanduser().resolve(), pathlib.Path(path_b).expanduser().resolve()):
        loaded = runfile.load_run_file(path)
        unnamed = [stage for stage, plan in loaded.stages.items() if not plan.maybe_instrument]
        if unnamed:
            raise UsageError(f"{path}: no instrument named for {', '.join(unnamed)}; compare a run file a run wrote")
        if loaded.settings.maybe_definition is None or loaded.settings.maybe_bib is None:
            raise UsageError(f"{path}: names no definition or no input")
        runs.append((path, loaded))
    (path_a, first), (path_b, second) = runs
    settings = first.settings
    corpus = pipeline.load_corpus(settings.maybe_bib, nlp=nlp, require=settings.require, min_chars=settings.min_chars,
                                  maybe_sample=settings.maybe_pilot or settings.maybe_sample,
                                  seed=settings.maybe_seed if settings.maybe_seed is not None else DEFAULTS["maybe_seed"])
    states, definitions = [], []
    for path, loaded in runs:
        definition = scope_definition.load_definition(loaded.settings.maybe_definition)
        seed = loaded.settings.maybe_seed if loaded.settings.maybe_seed is not None else DEFAULTS["maybe_seed"]
        control = loaded.settings.maybe_control if loaded.settings.maybe_control is not None else DEFAULTS["maybe_control"]
        state = pipeline.read_state(corpus, definition, ledger_dir=loaded.settings.maybe_ledgers or path.parent / "ledgers",
                                    fingerprints={stage: plan.maybe_instrument for stage, plan in loaded.stages.items()},
                                    control_size=control, seed=seed)
        states.append({"final": state.final, "reviews": state.reviews, "extractions": state.extractions})
        definitions.append(definition)
    compared = compare.measures(states[0], states[1], definition_a=definitions[0], definition_b=definitions[1])
    provenance = [f"raven-scopebib {__version__}, {datetime.datetime.now().isoformat(timespec='seconds')}",
                  f"first run: {path_a}", f"second run: {path_b}",
                  f"records: the first run's, from {settings.maybe_bib}"]
    if nlp_description:
        provenance.append(f"titles measured with {nlp_description}")
    for label, loaded in (("first", first), ("second", second)):
        for stage, plan in loaded.stages.items():
            provenance.append(f"{label} run, {stage}: model {plan.maybe_model or 'not recorded'}, "
                              f"instrument {plan.maybe_instrument}")
    fmt = table_format or second.settings.maybe_format or DEFAULTS["maybe_format"]
    return compare.write_comparison(path_b.parent / f"compare-with-{path_a.parent.name}.{fmt}", compared,
                                    records=corpus.records, provenance=provenance)


def _connector(backend_url: str | None):
    """A `connect` for `pipeline.run`: reach the backend, check that it serves a model, and say which."""
    def connect():
        from ...librarian import config as librarian_config, llmclient
        from . import pipeline
        url = backend_url or librarian_config.llm_backend_url
        if not llmclient.test_connection(url):
            raise pipeline.PipelineError(f"cannot reach an LLM backend at {url}")
        settings = llmclient.setup(backend_url=url, quiet=True)
        if (status := llmclient.backend_status(settings)) is not llmclient.backend_ready:
            headline, advice = llmclient.describe_backend_status(status, url)
            raise pipeline.PipelineError(f"{headline} {advice}")
        print(f"asking {settings.model} at {url}")
        return settings
    return connect


def _nlp(*, on_server: bool, server_url: str | None):
    """spaCy for measuring titles, and a line saying which, for the provenance.

    Locally by default, with the pipeline the title bounds were calibrated with. With `on_server`, Raven-server's
    own, where it answers; then the measure is only as comparable across runs as the server's configuration.
    """
    from ...client import api, mayberemote
    from ...client import config as client_config
    from . import records as scope_records
    url = server_url or client_config.raven_server_url
    if on_server:
        api.initialize(raven_server_url=url, raven_api_key_file=client_config.raven_api_key_file)
    nlp = mayberemote.NLP(allow_local=True, allow_remote=on_server, model_name=scope_records.SPACY_MODEL,
                          device_string=client_config.devices["nlp"]["device_string"])
    pipeline = f"{nlp.model_info['model']} {nlp.model_info['version']}"
    if not nlp.is_local():
        return nlp, f"{pipeline}, on Raven-server at {url}"
    if on_server:
        return nlp, f"{pipeline}, locally: Raven-server at {url} does not measure"
    return nlp, f"{pipeline}, locally"


def main() -> None:  # pragma: no cover -- `resolve` and `execute` carry the logic, and are tested
    opts = _parser().parse_args()

    from ...common import logsetup
    logsetup.configure(level=getattr(logging, opts.log_level), logfile=opts.log)
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)

    from . import definition as scope_definition
    from . import pipeline, report, runfile
    if opts.command == "compare":
        try:
            first = runfile.load_run_file(pathlib.Path(opts.run_a).expanduser())
            nlp, nlp_description = _nlp(on_server=bool(first.settings.maybe_server_nlp), server_url=opts.server_url)
            written = compare_runs(opts.run_a, opts.run_b, nlp=nlp, nlp_description=nlp_description,
                                   table_format=opts.format)
        except (UsageError, runfile.RunFileError, scope_definition.DefinitionError) as exc:
            print(f"raven-scopebib: {exc}", file=sys.stderr)
            sys.exit(1)
        for path in written:
            print(f"wrote {path}")
        sys.exit(0)
    try:
        plan = resolve(opts)
        nlp, nlp_description = _nlp(on_server=plan.run_file.settings.maybe_server_nlp, server_url=opts.server_url)
        status = execute(plan, nlp=nlp, nlp_description=nlp_description, connect=_connector(opts.backend_url),
                         sheet=getattr(opts, "sheet", None),
                         require_every_decision=getattr(opts, "require_every_decision", False))
    except (UsageError, runfile.RunFileError, scope_definition.DefinitionError, pipeline.PipelineError,
            report.FinishError) as exc:
        print(f"raven-scopebib: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(status)


if __name__ == "__main__":  # pragma: no cover
    main()
