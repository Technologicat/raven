"""The run file: which model each stage asks, and under which instrument.

A TOML file with a table per stage that asks a model, each naming the model to load and, optionally, the
instrument whose ledger to read:

    model = "qwen3-30b-a3b"          # any stage not naming its own

    [judge]
    model = "qwen3-235b-a22b"
    instrument = "0970b39f"

    [extract]
    instrument = "14036139"

Every run writes one (`write_run_file`), naming what each stage actually used, so a finished run carries its
own recipe: passing the file back repeats it. A person writes one by hand to choose the models, and
`--model STAGE=KEY` and `--instrument STAGE=FP` on the command line override it stage by stage
(`parse_override`, `with_overrides`).

A stage named nowhere asks whatever model is loaded, under the instrument the definition and that model make.
"""

__all__ = ["STAGES", "RunFileError", "StagePlan",
           "load_run_file", "parse_override", "with_overrides", "write_run_file"]

import dataclasses
import pathlib
import re
import tomllib
from collections.abc import Iterable, Mapping

#: The stages that ask a model, in the order a run takes them.
STAGES = ("judge", "review", "extract")

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


def _check_instrument(value: str, where: str) -> str:
    if not _FINGERPRINT.fullmatch(value):
        raise RunFileError(f"{where}: an instrument is eight hex digits, as in a ledger's file name, "
                           f"got {value!r}")
    return value


def load_run_file(path: str | pathlib.Path) -> dict[str, StagePlan]:
    """The plan for every stage, from the run file at `path`. A stage the file leaves out gets an empty plan.

    Raises `RunFileError` for a file that is not TOML, a stage or key the file has no business naming, or a
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
        elif key not in STAGES:
            raise RunFileError(f"{path}: unknown key {key!r}; a run file has a top-level 'model' and a table "
                               f"for each of {', '.join(STAGES)}")
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
    return plans


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


def with_overrides(plans: Mapping[str, StagePlan], *,
                   models: Iterable[tuple[str, str]] = (),
                   instruments: Iterable[tuple[str, str]] = ()) -> dict[str, StagePlan]:
    """`plans` with each `(stage, value)` of `models` and `instruments` replacing what it named."""
    out = {stage: plans.get(stage, StagePlan()) for stage in STAGES}
    for stage, model in models:
        out[stage] = dataclasses.replace(out[stage], maybe_model=model)
    for stage, instrument in instruments:
        out[stage] = dataclasses.replace(out[stage], maybe_instrument=instrument)
    return out


def _toml_string(value: str) -> str:
    """`value` as a TOML basic string."""
    escapes = {"\\": "\\\\", '"': '\\"', "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
    return '"' + "".join(escapes.get(char, f"\\u{ord(char):04x}" if ord(char) < 0x20 or ord(char) == 0x7f else char)
                         for char in value) + '"'


def write_run_file(path: str | pathlib.Path, plans: Mapping[str, StagePlan], *, comments: Iterable[str] = ()) -> None:
    """Write `plans` to `path`, every stage in full, with `comments` as `#` lines at the top.

    No top-level `model` is written, even where every stage used the same one, so that the file says what
    each stage did without a reader having to resolve a default.
    """
    lines = [f"# {comment}" for comment in comments]
    for stage in STAGES:
        plan = plans.get(stage, StagePlan())
        lines += ["", f"[{stage}]"]
        if plan.maybe_model is not None:
            lines.append(f"model = {_toml_string(plan.maybe_model)}")
        if plan.maybe_instrument is not None:
            lines.append(f"instrument = {_toml_string(plan.maybe_instrument)}")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).lstrip("\n") + "\n", encoding="utf-8")
