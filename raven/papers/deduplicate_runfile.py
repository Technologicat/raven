"""The run file of `raven-deduplicate`: what a run was asked to do, and which model answered its judge.

Written beside the output as `<output>_run.toml` as a run starts, before any work, so that a run that does not
start, or stops partway, can be finished as it was configured: `raven-deduplicate --run-file <it>` takes
everything from it, and anything given on the command line overrides that. A run with no output writes none.

What is recorded is what the run is: its input files, its output and audit, the options that decide the
result, the judge's ledgers and backend. How one invocation logs is not.

Paths are written relative to the run file, so a folder holding the run can be moved whole.
"""

__all__ = ["SETTINGS", "QUESTIONS", "RunFileError", "RunFile",
           "default_path", "load", "write", "with_instrument"]

import dataclasses
import os
import pathlib
import tomllib
from typing import Any

from ..common import utils as common_utils

#: The settings a run file records, as `raven-deduplicate`'s options name them.
SETTINGS = ("filenames", "output", "audit", "audit_format", "no_audit", "judge", "ledgers", "backend_url")

#: The judge's questions, each with a table of its own: which records are one work, and which DOI a work keeps.
QUESTIONS = ("pairs", "dois")

_PATHS = ("output", "audit", "ledgers")
_FLAGS = ("no_audit", "judge")


class RunFileError(ValueError):
    """A run file that does not read, or holds a value of the wrong kind, named with the file."""


@dataclasses.dataclass(frozen=True)
class RunFile:
    """A run file's contents.

    `settings`: by option name, from `SETTINGS`, an absent one not recorded. Paths are absolute here.
    `instruments`: by question, from `QUESTIONS`, `{"model": ..., "instrument": ...}`: the model and the
                   ledger fingerprint that question was asked with.
    """
    settings: dict[str, Any] = dataclasses.field(default_factory=dict)
    instruments: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)


def default_path(output: str | pathlib.Path) -> pathlib.Path:
    """Where a run writing to `output` keeps its run file: `<output>_run.toml`, beside it."""
    output = pathlib.Path(output)
    return output.with_name(f"{output.stem}_run.toml")


def load(path: str | pathlib.Path) -> RunFile:
    """Read the run file at `path`, its paths made absolute. Raises `RunFileError`."""
    path = pathlib.Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RunFileError(f"'{path}' does not read: {type(exc)}: {exc}") from exc
    run = data.get("run", {})
    base = path.parent

    def absolute(value: str) -> pathlib.Path:
        return (base / pathlib.Path(value).expanduser()).resolve()
    settings = {}
    for key, value in run.items():
        if key not in SETTINGS:
            raise RunFileError(f"'{path}': [run] has '{key}', which is not a setting; the settings are "
                               f"{', '.join(SETTINGS)}")
        if key == "filenames":
            if not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
                raise RunFileError(f"'{path}': 'filenames' must be a list of paths")
            settings[key] = [absolute(item) for item in value]
        elif key in _FLAGS:
            if not isinstance(value, bool):
                raise RunFileError(f"'{path}': '{key}' must be true or false")
            settings[key] = value
        else:
            if not isinstance(value, str):
                raise RunFileError(f"'{path}': '{key}' must be a string")
            settings[key] = absolute(value) if key in _PATHS else value
    instruments = {}
    for question in QUESTIONS:
        table = data.get(question)
        if isinstance(table, dict) and isinstance(table.get("instrument"), str):
            instruments[question] = {"model": str(table.get("model", "")), "instrument": table["instrument"]}
    return RunFile(settings, instruments)


def _relative(target: pathlib.Path, base: pathlib.Path) -> str:
    """`target` relative to `base`, with forward slashes; absolute where no relative path exists (another drive)."""
    try:
        return pathlib.Path(os.path.relpath(target, base)).as_posix()
    except ValueError:
        return pathlib.Path(target).as_posix()


def write(path: str | pathlib.Path, run_file: RunFile) -> None:
    """Write `run_file` to `path`, its paths relative to it."""
    path = pathlib.Path(path)
    base = path.parent.resolve()
    lines = ["# Written by raven-deduplicate as the run started. `raven-deduplicate --run-file <this file>` runs",
             "# it again, finishing what it did not; options on the command line override what is here.",
             "", "[run]"]
    settings = run_file.settings
    for key in SETTINGS:
        if key not in settings or settings[key] is None:
            continue
        value = settings[key]
        if key == "filenames":
            lines.append(f"filenames = [{', '.join(common_utils.toml_string(_relative(pathlib.Path(item), base)) for item in value)}]")
        elif key in _FLAGS:
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif key in _PATHS:
            lines.append(f"{key} = {common_utils.toml_string(_relative(pathlib.Path(value), base))}")
        else:
            lines.append(f"{key} = {common_utils.toml_string(str(value))}")
    for question in QUESTIONS:
        if question in run_file.instruments:
            table = run_file.instruments[question]
            lines += ["", f"[{question}]",
                      f"model = {common_utils.toml_string(table.get('model', ''))}",
                      f"instrument = {common_utils.toml_string(table['instrument'])}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def with_instrument(run_file: RunFile, question: str, *, model: str, instrument: str) -> RunFile:
    """`run_file`, recording that `question` is asked with `instrument`, on `model`."""
    if question not in QUESTIONS:
        raise ValueError(f"with_instrument: unknown question '{question}'; expected one of {QUESTIONS}")
    return dataclasses.replace(run_file, instruments={**run_file.instruments,
                                                      question: {"model": model, "instrument": instrument}})
