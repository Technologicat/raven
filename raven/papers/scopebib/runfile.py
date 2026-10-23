"""The run file: everything that decides what a run produces, so that passing it back repeats the run.

A TOML file. Its `[run]` table holds the run's settings — the input, the definition, the sift, the sample
and the control — and a table per stage that asks a model names the model to load and, optionally, the
instrument whose ledger to read:

    model = "qwen3-30b-a3b"          # any stage not naming its own

    [run]
    bib = "corpus.bib"               # relative to this file
    definition = "corpus.scope.toml"
    require = ["abstract"]
    seed = 42
    control = 40

    [judge]
    model = "qwen3-235b-a22b"
    instrument = "0970b39f"

Every run writes one (`write_run_file`) naming what it actually used, so a finished run carries its own
recipe. A person writes one by hand to choose the models, and the command line overrides it: an option
replaces the setting it names, and `--model STAGE=KEY` and `--instrument STAGE=FP` replace one stage's
(`parse_override`, `with_overrides`). Every key is optional.

A stage named nowhere asks whatever model is loaded, under the instrument the definition and that model make.
"""

__all__ = ["STAGES", "GENERATED", "RunFileError", "StagePlan", "Settings", "RunFile",
           "load_run_file", "parse_override", "with_overrides", "write_run_file"]

import dataclasses
import os
import pathlib
import re
import tomllib
from collections.abc import Iterable, Mapping

from ...common import tabular
from .. import siftbib

#: The stages that ask a model, in the order a run takes them.
STAGES = ("judge", "review", "extract")

#: The first line of every run file the tool writes; a file without it was written by a person.
GENERATED = "# Written by raven-scopebib"

_FINGERPRINT = re.compile(r"[0-9a-f]{8}")
_KEYS = ("model", "instrument")


class RunFileError(ValueError):
    """A run file, or an override, that cannot be used; the message names the file and the key."""


@dataclasses.dataclass(frozen=True)
class StagePlan:
    """What a stage asks with.

    `maybe_model`: the model to load, by the backend's key, or `None` for whatever is loaded.
    `maybe_instrument`: the fingerprint of the ledger to read, or `None` for the one the definition and the
                        model make.
    """
    maybe_model: str | None = None
    maybe_instrument: str | None = None


@dataclasses.dataclass(frozen=True)
class Settings:
    """The run's settings, each `None` where the file does not say. What each means is the command line's.

    `maybe_bib`, `maybe_definition`, `maybe_ledgers`: the input, the definition file, and the directory
                                                    holding the ledgers, as absolute paths.
    `require`, `min_chars`: the sift's criteria, as `raven-siftbib` takes them: field names, and `FIELD=N`.
    `maybe_sample`, `maybe_pilot`: how many records a sampled or a pilot run draws.
    `maybe_seed`: the seed of every draw, the sample's and the control's.
    `maybe_control`: how many kept records the reviewer reads as its control.
    `maybe_replicate`: which replicate of the model stages this is.
    `maybe_format`: the table format of the reports, one of `raven.common.tabular.FORMATS`.
    `maybe_server_nlp`: measure the titles with Raven-server's spaCy model, rather than locally with the one the
                        title bounds were calibrated with.
    `maybe_context_length`: the context length every model the run loads is loaded with; `None` for the one the
                            backend has saved for each.
    """
    maybe_bib: pathlib.Path | None = None
    maybe_definition: pathlib.Path | None = None
    maybe_ledgers: pathlib.Path | None = None
    require: tuple[str, ...] = ()
    min_chars: tuple[str, ...] = ()
    maybe_sample: int | None = None
    maybe_pilot: int | None = None
    maybe_seed: int | None = None
    maybe_control: int | None = None
    maybe_replicate: int | None = None
    maybe_format: str | None = None
    maybe_server_nlp: bool | None = None
    maybe_context_length: int | None = None


@dataclasses.dataclass(frozen=True)
class RunFile:
    """A run file's contents: the settings, and the plan for every stage in `STAGES`."""
    settings: Settings
    stages: dict[str, StagePlan]


# The `[run]` table's keys: the `Settings` field each fills, and its smallest value where it is a count.
_PATHS = {"bib": "maybe_bib", "definition": "maybe_definition", "ledgers": "maybe_ledgers"}
_LISTS = {"require": "require", "min_chars": "min_chars"}
_INTS = {"sample": ("maybe_sample", 1), "pilot": ("maybe_pilot", 1), "seed": ("maybe_seed", None),
         "control": ("maybe_control", 0), "replicate": ("maybe_replicate", 1),
         "context_length": ("maybe_context_length", 1)}


def _check_instrument(value: str, where: str) -> str:
    if not _FINGERPRINT.fullmatch(value):
        raise RunFileError(f"{where}: an instrument is eight hex digits, as in a ledger's file name, "
                           f"got {value!r}")
    return value


def load_run_file(path: str | pathlib.Path) -> RunFile:
    """The run file at `path`. A stage the file leaves out gets an empty plan, and a setting it leaves out is
    `None`. A relative path in it is relative to the file.

    Raises `RunFileError` for a file that is not TOML, a table or key the file has no business naming, or a
    value of the wrong shape.
    """
    path = pathlib.Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RunFileError(f"{path}: not valid TOML: {exc}") from None

    def string(value: object, where: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RunFileError(f"{where}: expected a non-empty string, got {value!r}")
        return value.strip()

    maybe_default = None
    for key, value in data.items():
        if key == "model":
            maybe_default = string(value, f"{path}: model")
        elif key not in (*STAGES, "run"):
            raise RunFileError(f"{path}: unknown key {key!r}; a run file has a top-level 'model', a [run] "
                               f"table, and a table for each of {', '.join(STAGES)}")
        elif not isinstance(value, dict):
            raise RunFileError(f"{path}: {key!r} must be a table, [{key}]")

    plans = {}
    for stage in STAGES:
        table = data.get(stage, {})
        for key in table:
            if key not in _KEYS:
                raise RunFileError(f"{path}: [{stage}]: unknown key {key!r}; expected one of {', '.join(_KEYS)}")
        maybe_model = string(table["model"], f"{path}: [{stage}] model") if "model" in table else maybe_default
        maybe_instrument = None
        if "instrument" in table:
            where = f"{path}: [{stage}] instrument"
            maybe_instrument = _check_instrument(string(table["instrument"], where), where)
        plans[stage] = StagePlan(maybe_model, maybe_instrument)
    return RunFile(_load_settings(data.get("run", {}), path), plans)


def _load_settings(table: dict, path: pathlib.Path) -> Settings:
    fields = {}
    for key, value in table.items():
        where = f"{path}: [run] {key}"
        if key in _PATHS:
            if not isinstance(value, str) or not value.strip():
                raise RunFileError(f"{where}: expected a path, got {value!r}")
            fields[_PATHS[key]] = (path.parent / value.strip()).resolve()
        elif key in _LISTS:
            if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
                raise RunFileError(f"{where}: expected a list of strings, got {value!r}")
            if key == "min_chars":
                for item in value:
                    try:
                        siftbib.parse_min_chars(item)
                    except ValueError as exc:
                        raise RunFileError(f"{where}: {exc}") from None
            fields[_LISTS[key]] = tuple(item.strip() for item in value)
        elif key in _INTS:
            name, least = _INTS[key]
            # `bool` is an `int` in Python, and `true` is no count.
            if not isinstance(value, int) or isinstance(value, bool) or (least is not None and value < least):
                floor = f" of at least {least}" if least is not None else ""
                raise RunFileError(f"{where}: expected a whole number{floor}, got {value!r}")
            fields[name] = value
        elif key == "server_nlp":
            if not isinstance(value, bool):
                raise RunFileError(f"{where}: expected true or false, got {value!r}")
            fields["maybe_server_nlp"] = value
        elif key == "format":
            if value not in tabular.FORMATS:
                raise RunFileError(f"{where}: expected one of {', '.join(tabular.FORMATS)}, got {value!r}")
            fields["maybe_format"] = value
        else:
            raise RunFileError(f"{path}: [run]: unknown key {key!r}; expected one of "
                               f"{', '.join([*_PATHS, *_LISTS, *_INTS, 'format', 'server_nlp'])}")
    return Settings(**fields)


def parse_override(spec: str, *, option: str) -> tuple[str, str]:
    """`(stage, value)` from a `STAGE=VALUE` command-line argument to `option`, such as `--model`.

    Raises `RunFileError` for anything else, naming the option, so a mistyped stage is a bad argument rather
    than a stage silently left to its default. An `--instrument` value is checked as an instrument.
    """
    stage, separator, value = spec.partition("=")
    stage, value = stage.strip(), value.strip()
    if not separator or not value:
        raise RunFileError(f"{option}: expected STAGE=VALUE, got {spec!r}")
    if stage not in STAGES:
        raise RunFileError(f"{option}: unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
    if option == "--instrument":
        _check_instrument(value, option)
    return stage, value


def with_overrides(run_file: RunFile, *,
                   settings: Mapping[str, object] | None = None,
                   models: Iterable[tuple[str, str]] = (),
                   instruments: Iterable[tuple[str, str]] = ()) -> RunFile:
    """`run_file` with the command line's choices replacing what it said.

    `settings`: `Settings` fields by name, each replacing the file's; `None` values are skipped, being
                options the command line did not give, so a caller can pass every option it has.
    `models`, `instruments`: `(stage, value)` pairs from `parse_override`, each replacing that stage's.
    """
    given = {name: value for name, value in (settings or {}).items() if value is not None}
    stages = {stage: run_file.stages.get(stage, StagePlan()) for stage in STAGES}
    for stage, model in models:
        stages[stage] = dataclasses.replace(stages[stage], maybe_model=model)
    for stage, instrument in instruments:
        stages[stage] = dataclasses.replace(stages[stage], maybe_instrument=instrument)
    return RunFile(dataclasses.replace(run_file.settings, **given), stages)


def _toml_string(value: str) -> str:
    """`value` as a TOML basic string."""
    escapes = {"\\": "\\\\", '"': '\\"', "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
    return '"' + "".join(escapes.get(char, f"\\u{ord(char):04x}" if ord(char) < 0x20 or ord(char) == 0x7f else char)
                         for char in value) + '"'


def _relative(target: pathlib.Path, base: pathlib.Path) -> str:
    """`target` relative to `base`, with forward slashes, so that the file reads the same on any system.

    Absolute where the two share nothing below the filesystem's root, or are on different drives: moving
    them together is then not a thing anyone does, and a path climbing to the root says less than the
    absolute one.
    """
    try:
        shared = pathlib.Path(os.path.commonpath([target, base]))
    except ValueError:  # different drives, on Windows
        return target.as_posix()
    if shared == pathlib.Path(shared.anchor):
        return target.as_posix()
    return pathlib.Path(os.path.relpath(target, base)).as_posix()


def _archive_name(path: pathlib.Path) -> pathlib.Path:
    """The first of `run_orig.toml`, `run_orig-2.toml`, ... beside `path` that does not exist."""
    candidate, n = path.with_name(f"{path.stem}_orig{path.suffix}"), 1
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.stem}_orig-{n}{path.suffix}")
    return candidate


def write_run_file(path: str | pathlib.Path, run_file: RunFile, *,
                   comments: Iterable[str] = ()) -> pathlib.Path | None:
    """Write `run_file` to `path`, every stage in full, `GENERATED` and then `comments` as `#` lines at the top.

    A file already at `path` that a person wrote — one not starting with `GENERATED` — is first moved aside
    to `run_orig.toml`, or `run_orig-2.toml` and so on where that exists, and the new name is returned;
    otherwise `None`. A file the tool wrote is replaced, edits included, since a run's file records what the
    run used, and edits that took effect are in it.

    Paths are written relative to `path`'s directory, so that a run directory moved or handed over with its
    input still resolves. No top-level `model` is written, even where every stage used the same one, so
    that the file says what each stage did without a reader having to resolve a default.
    """
    path = pathlib.Path(path)
    lines = [GENERATED, *(f"# {comment}" for comment in comments)]
    settings = run_file.settings
    run_lines = []
    for key, name in _PATHS.items():
        if (value := getattr(settings, name)) is not None:
            run_lines.append(f"{key} = {_toml_string(_relative(pathlib.Path(value).resolve(), path.parent.resolve()))}")
    for key, name in _LISTS.items():
        if values := getattr(settings, name):
            run_lines.append(f"{key} = [{', '.join(_toml_string(value) for value in values)}]")
    for key, (name, _) in _INTS.items():
        if (value := getattr(settings, name)) is not None:
            run_lines.append(f"{key} = {value}")
    if settings.maybe_format is not None:
        run_lines.append(f"format = {_toml_string(settings.maybe_format)}")
    if settings.maybe_server_nlp is not None:
        run_lines.append(f"server_nlp = {'true' if settings.maybe_server_nlp else 'false'}")
    if run_lines:
        lines += ["", "[run]", *run_lines]
    for stage in STAGES:
        plan = run_file.stages.get(stage, StagePlan())
        lines += ["", f"[{stage}]"]
        if plan.maybe_model is not None:
            lines.append(f"model = {_toml_string(plan.maybe_model)}")
        if plan.maybe_instrument is not None:
            lines.append(f"instrument = {_toml_string(plan.maybe_instrument)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    maybe_archive = None
    if path.exists() and not path.read_text(encoding="utf-8").startswith(GENERATED):
        maybe_archive = _archive_name(path)
        path.rename(maybe_archive)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return maybe_archive
