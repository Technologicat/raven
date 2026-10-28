"""Visualizer state kept between sessions: choices made in the GUI, as they were when the app last closed.

A small JSON file. `load` reads it at startup, filling in a default for anything missing or unusable, and
`save` writes it at shutdown. Currently one entry, `panel_side`.

Adding an entry means adding it to `DEFAULTS` and `ALLOWED`; `load`, `save` and the tests derive from those.
"""

__all__ = ["DEFAULTS", "ALLOWED", "load", "save"]

import logging
logger = logging.getLogger(__name__)

import json
import pathlib

from ..common import utils as common_utils

#: Each entry's value when the file does not say, or says something unusable.
DEFAULTS = {"panel_side": "left"}  # which side of the window the info panel is on

#: The values each entry may take. A saved value outside these is replaced by the default.
ALLOWED = {"panel_side": ("left", "right")}


def load(path: str | pathlib.Path) -> dict:
    """The saved state at `path`, with the default for every entry the file lacks or holds an unusable value for.

    A missing file is the first run, and gives the defaults silently. An unreadable or malformed one gives the
    defaults with a warning.
    """
    path = pathlib.Path(path)
    state = dict(DEFAULTS)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return state
    except OSError as exc:
        logger.warning(f"load: cannot read '{path}', using defaults: {type(exc)}: {exc}")
        return state
    try:
        saved = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(f"load: '{path}' is not valid JSON, using defaults: {type(exc)}: {exc}")
        return state
    if not isinstance(saved, dict):
        logger.warning(f"load: '{path}' holds a {type(saved).__name__}, not an object; using defaults.")
        return state
    for key, value in saved.items():
        if key not in DEFAULTS:
            logger.info(f"load: '{path}': ignoring unknown entry '{key}'.")
        elif value not in ALLOWED[key]:
            logger.warning(f"load: '{path}': '{key}' is {value!r}, not one of {ALLOWED[key]}; using {DEFAULTS[key]!r}.")
        else:
            state[key] = value
    return state


def save(path: str | pathlib.Path, state: dict) -> None:
    """Write the entries of `state` that `DEFAULTS` names to `path`, creating its directory if needed.

    Atomic: a reader sees either the previous file or the new one.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with common_utils.atomic_write(path) as state_file:
        json.dump({key: state[key] for key in DEFAULTS}, state_file, indent=2)
