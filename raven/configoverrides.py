"""Machine-local overrides for the constellation's configuration files.

Raven is configured by editing Python modules named `config.py`. That works, and it has one flaw: what
the project ships and what one machine happens to be are the same lines in the same file, told apart only
by which ones somebody remembers not to commit. A backend URL naming a personal host, an audio device
that exists on one box, a user's own name — each is a local answer living in a tracked file.

This module supplies the boundary. Shipped defaults stay in `config.py`; a machine's own answers go in a
JSON file outside the repository, where they cannot be committed by accident and survive a `git pull`
that touches the same lines.

The file is `~/.config/raven/overrides.json`, and its top-level keys are config modules::

    {
        "raven.librarian.config": {
            "llm_backend_url": "http://otherbox.local:1234",
            "llm_user_name": "Juha"
        },
        "raven.visualizer.config": {
            "clusters_keyword_method": "llm",
            "gui_config.word_cloud_w": 1024,
            "gui_config.word_cloud_background_color": "white"
        },
        "raven.config": {
            "GUI_IDLE_FRAMERATE": 30
        }
    }

A setting named with dots reaches inside whatever the first component holds — `gui_config` in the
Visualizer's and Librarian's configs is an `unpythonic` `env`, so its fields are addressed that way. One
rule covers both shapes a config module has, and needs no declaration of which is which.

Only settings that already exist can be overridden, and a name matching nothing is reported rather than
created. That is the whole of the typo protection available here: a `config.py` is Python and a setting
is just a name, so there is nothing to validate a *value* against beyond the type the default has.
"""

__all__ = ["OVERRIDES_PATH", "apply"]

import json
import logging
import pathlib

from unpythonic import sym

logger = logging.getLogger(__name__)

# Where a machine's own answers live. Deliberately **not** derived from `raven.config.toplevel_userdata_dir`,
# and not itself overridable: this is the file that would have to say so, and it has to be found before it
# can say anything. Same reason a shell finds `~/.bashrc` at a fixed path.
OVERRIDES_PATH = pathlib.Path("~/.config/raven/overrides.json").expanduser()

# Returned by `_coerce` for a value it will not fit to the setting's shipped shape.
_refused = sym("refused")

# Paths whose contents have been announced to the log, so that eight config modules reading one file
# produce one line about it rather than eight.
_announced: set[pathlib.Path] = set()


def _read(path: pathlib.Path) -> dict:
    """Read the override file at `path`. Return its mapping, or an empty one if there is nothing usable.

    A missing file is the ordinary case and says nothing. Anything else — unreadable, malformed, or not
    an object at the top level — is reported at ERROR and then treated as empty, so that a typo in this
    file cannot stop an app from starting. The app then runs on shipped defaults, which is a visible
    outcome rather than a silent one: the log line names the file and what is wrong with it.
    """
    if not path.exists():
        logger.debug(f"_read: no override file at {path}; shipped defaults apply.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error(f"_read: {path} is not valid JSON, so NO overrides are in effect: {type(exc)}: {exc}")
        return {}
    except OSError as exc:
        logger.error(f"_read: cannot read {path}, so NO overrides are in effect: {type(exc)}: {exc}")
        return {}
    if not isinstance(data, dict):
        logger.error(f"_read: {path} should hold an object keyed by config module name, not {type(data)}. NO overrides are in effect.")
        return {}

    # Announced once per file, at INFO, because the failure this catches is invisible otherwise: a
    # mistyped *module* name is claimed by nobody, so no module can report it, and the user sees only a
    # setting that did not take effect. Printing the names the file offers lets them spot it.
    if path not in _announced:
        _announced.add(path)
        logger.info(f"_read: {path} has overrides for: {', '.join(sorted(data)) if data else '(nothing)'}")
        for key in sorted(data):
            if not (key == "raven.config" or (key.startswith("raven.") and key.endswith(".config"))):
                logger.warning(f"_read: in {path}, '{key}' does not name a Raven config module (expected 'raven.config' or 'raven.<component>.config'); nothing will claim it.")
    return data


def _coerce(default, value, where: str):
    """Fit `value`, as JSON produced it, to the shape of the shipped `default`. Return it, or `_refused`.

    `where`: how to name this setting in a log message.

    JSON has fewer types than Python, so some shipped defaults cannot be spelled exactly, and two of
    those are worth fitting rather than reporting: a path arrives as a string, and anything tuple-shaped
    — a color, a size, a pair of thresholds — arrives as a list.

    A value that does not fit is refused rather than applied. Both choices produce the same warning, and
    refusing additionally leaves the app running on a default that works.
    """
    if default is None:  # nothing to fit to, and plenty of settings ship as "unset"
        return value
    if isinstance(default, pathlib.PurePath):
        return pathlib.Path(value).expanduser().resolve()
    if isinstance(default, bool) or isinstance(value, bool):
        # Asked before the numeric case below, because `bool` is a subclass of `int`: otherwise `true`
        # would quietly pass for a setting that wants a number, and `1` for one that wants a switch.
        if isinstance(default, bool) and isinstance(value, bool):
            return value
    elif isinstance(default, float) and isinstance(value, int):
        return float(value)  # JSON writes a whole number without its point; the setting still wants a float
    elif isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    elif isinstance(value, type(default)):
        return value
    logger.warning(f"_coerce: {where} ships as {type(default)} and the override is {type(value)}; ignored, so the shipped default applies.")
    return _refused


def _apply_one(namespace: dict, dotted_name: str, value, module_name: str, path: pathlib.Path) -> bool:
    """Bind one setting, walking `dotted_name` into whatever holds it. Return whether it was applied."""
    where = f"'{dotted_name}' of '{module_name}'"
    head, *rest = dotted_name.split(".")

    if head not in namespace:
        logger.warning(f"_apply_one: {where} names nothing in that module (from {path}); ignored.")
        return False

    # Everything but the last component is a container to reach through; the last is the setting itself.
    # `hasattr` is asked at every step, because an `env` accepts a brand-new key without complaint, so
    # nothing downstream would notice a misspelling — it would simply add a field nobody reads.
    target = namespace[head]
    for step in rest[:-1]:
        if not hasattr(target, step):
            logger.warning(f"_apply_one: {where} names nothing: '{step}' is not in {target!r} (from {path}); ignored.")
            return False
        target = getattr(target, step)

    if not rest:
        fitted = _coerce(namespace[head], value, where)
        if fitted is _refused:
            return False
        namespace[head] = fitted
        return True

    leaf = rest[-1]
    if not hasattr(target, leaf):
        logger.warning(f"_apply_one: {where} names nothing: '{leaf}' is not in {target!r} (from {path}); ignored.")
        return False
    fitted = _coerce(getattr(target, leaf), value, where)
    if fitted is _refused:
        return False
    setattr(target, leaf, fitted)
    return True


def apply(module_name: str, namespace: dict, *, path: pathlib.Path | None = None) -> list[str]:
    """Apply this machine's overrides for `module_name`. Return the names that were applied.

    `module_name`: the config module's dotted name. Pass `__name__`.
    `namespace`: that module's global namespace. Pass `globals()`.
    `path`: which override file to read. Defaults to `OVERRIDES_PATH`.

    Call this at the *foot* of a config module, below everything it defines.
    """
    # The foot, because overriding a name has to happen after that name exists, and a config module is
    # read top to bottom. The consequence worth knowing: a setting *derived* from another — a directory
    # built from `toplevel_userdata_dir`, a filename built from that — was computed on the way down and
    # is not recomputed here. Overriding the value it came from does not move it. Override the derived
    # setting in its own right, which is why every one of them is a plain module-level name.
    path = path if path is not None else OVERRIDES_PATH
    settings = _read(path).get(module_name, {})
    if not isinstance(settings, dict):
        logger.error(f"apply: in {path}, the entry for '{module_name}' should be an object of settings, not {type(settings)}; ignored.")
        return []

    applied = [name for name, value in settings.items()
               if _apply_one(namespace, name, value, module_name, path)]
    if applied:
        logger.info(f"apply: {module_name}: overridden from {path}: {', '.join(applied)}")
    return applied
