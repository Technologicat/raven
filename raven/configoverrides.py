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

The top-level key names the module; the keys inside it name settings in that module. The two are never
joined into one key.

A setting named with dots reaches inside whatever the first component holds, to any depth, and needs no
declaration of what that is:

  - **An object's attribute** — `gui_config.word_cloud_w`, `gui_config` in the Visualizer's and
    Librarian's configs being an `unpythonic` `env`. A `NamedTuple`'s field or a frozen dataclass's is
    named the same way.
  - **A mapping's entry**, by its key — `devices.tts.device_string`. JSON spells every key as a string,
    so a key that is not one is named by its digits: `PLACEHOLDER_POOL_SIZES.32`. A name matching two
    keys (`1` and `"1"`) is refused.
  - **A sequence's item**, by its index — `TILE_SIZES.2`.

Whatever holds the setting is updated in place where it can be. An immutable container — a `NamedTuple`,
a frozen dataclass, a tuple — is rebuilt with the new value and stored back into its own parent, and so on
up, so everything mutable on the way keeps its identity.

A value naming a whole mapping replaces it rather than merging into it: the keys are part of the value,
and replacing is the only way to drop one. Its keys are converted to the type the shipped keys have, and a
mapping whose key type cannot be told — shipped empty, or with keys of mixed types — is refused.

Only settings that already exist can be overridden, and a name matching nothing is reported rather than
created. That is the whole of the typo protection available here: a `config.py` is Python and a setting
is just a name, so there is nothing to validate a *value* against beyond the type the default has.

**A name beginning with `//` is a commented-out entry**, and is skipped in silence. JSON has no comments,
and the `config.py` this replaces had them — a settings file is somewhere people keep the alternative they
switch to occasionally, not only the answer they are using today, and losing that would make the file
worse than what it replaces. It works at either level::

    "raven.visualizer.config": {
        "gui_config.word_cloud_w": 768,
        "// gui_config.word_cloud_w": 1024
    },
    "// raven.server.config": {"...": "a whole component, switched off"}

Deleting the three characters turns one back on, which is what a comment is for. Nothing else can collide
with the marker, `//` being unspellable as a Python name.

Since *any* `//` key is skipped, one naming no setting at all is a free-form comment — `"// note": "the
larger white word cloud is for print articles"` — which is how a commented-out entry says why it is being
kept. Give each a distinct name, JSON having nothing to say about two keys spelled the same.
"""

# **What this costs, since the docstring above only says what it buys.** A value that is pure data — a
# window size, a colour, a device name — can now be spelled in two places and two syntaxes, and a reader
# asking "what is this set to" has two files to look in. That is a genuine cost and not a wart to hide.
#
# It was taken knowingly, and two things hold it down. The precedence is one-directional and absolute — the
# JSON always wins, so there is no puzzle about which, only about where — and every applied override is
# logged at INFO, naming the setting and the file it came from, so a running app can be asked rather than
# reasoned about.
#
# The alternative, moving *everything* to JSON, is what the cost would buy, and it is not available. A
# `config.py` is three things JSON is not, and each rules out a different part of the contents:
#
#   - **Prose.** The comments are user-facing documentation — the avatar's postprocessor chain is mostly
#     explanation — and JSON has nowhere to put a paragraph.
#   - **Code.** Half these files are computed: `librarian_userdata_dir` from the global one, `TOOLBAR_H`
#     from a font size and two paddings, a `User-Agent` from `__version__`. JSON cannot derive anything,
#     which is also why an override applies to the *derived* name rather than to what it came from.
#   - **Python values.** `torch.float16`, a `Timeout`, a `pathlib.Path` — some of what these files hold has
#     no JSON spelling at all. `_coerce` rebuilds what can be reconstructed without guessing: a
#     `pathlib.Path` from a string, a `NamedTuple` from a mapping or a list, and a mapping's non-string keys
#     from the type its shipped keys have. A `torch.dtype` is the other kind — "float16" could name any
#     number of things — and is refused, the shipped value standing.
#
# So the split runs along the line where each format is good at its half: `config.py` documents what can be
# set, and this file records what was.

__all__ = ["OVERRIDES_PATH", "apply"]

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
import dataclasses
import json
import logging
import pathlib

from unpythonic import fupdate, sym

logger = logging.getLogger(__name__)

# Where a machine's own answers live. Deliberately **not** derived from `raven.config.toplevel_userdata_dir`,
# and not itself overridable: this is the file that would have to say so, and it has to be found before it
# can say anything. Same reason a shell finds `~/.bashrc` at a fixed path.
OVERRIDES_PATH = pathlib.Path("~/.config/raven/overrides.json").expanduser()

# Returned by `_coerce` for a value it will not fit to the setting's shipped shape.
_refused = sym("refused")

# What marks a key as commented out, at either level. Chosen because JSON's own de facto comment idiom is
# `//`, and because no Python name can start with it, so a real setting can never be mistaken for one.
_COMMENT_MARKER = "//"

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
        live = sorted(key for key in data if not key.startswith(_COMMENT_MARKER))
        logger.info(f"_read: {path} has overrides for: {', '.join(live) if live else '(nothing)'}")
        for key in live:
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
    if default is None:  # nothing to fit to
        return value
    if isinstance(default, pathlib.PurePath):
        # Absolutized and `~`-expanded, but **not** resolved through symlinks — `canonical_path` rather
        # than `absolutize_filename`, the pair `raven.common.utils` draws that distinction for. Nearly
        # every path in these configs is a *root* that other paths are built under or checked against
        # (`llm_docs_dir`, and a document id is where a file sits beneath it), and resolving a symlink can
        # relocate such a root out of the tree that gave it its meaning. Someone whose `~/.config` is a
        # symlink into a dotfiles checkout is the ordinary way to meet that. Nothing is given up: the OS
        # follows symlinks on open and on stat, so the unresolved path still reaches the real file.
        #
        # Deferred because `raven.common.utils` reaches numpy, and `raven.config` — the first module of the
        # constellation to load — would otherwise pay for it before doing anything.
        from .common import utils as common_utils  # noqa: PLC0415 -- intentional deferred import
        return common_utils.canonical_path(value)
    if isinstance(default, bool) or isinstance(value, bool):
        # Asked before the numeric case below, because `bool` is a subclass of `int`: otherwise `true`
        # would quietly pass for a setting that wants a number, and `1` for one that wants a switch.
        if isinstance(default, bool) and isinstance(value, bool):
            return value
    elif isinstance(default, float) and isinstance(value, int):
        return float(value)  # JSON writes a whole number without its point; the setting still wants a float
    elif isinstance(default, tuple) and hasattr(default, "_fields"):
        # A `NamedTuple` — `client_config.network_timeout` is one. Copy-and-update from the shipped value,
        # so an override may name only the field it means: a connect timeout worth changing usually sits
        # beside a read timeout that is not. `_replace` is also what checks the field names, which is the
        # only validation a record of numbers can offer.
        #
        # Through its own class either way. `tuple(value)` would hand back a plain tuple, which reads
        # correctly at every index and raises on every *name*, somewhere far from here.
        try:
            if isinstance(value, dict):
                updates = value
            elif len(value) <= len(default._fields):
                updates = dict(zip(default._fields, value))  # a positional prefix, as a list has no names
            else:
                raise TypeError(f"{len(value)} values given for fields {default._fields}")
            return default._replace(**updates)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(f"_coerce: {where} is a {type(default).__name__}{default._fields} and the override does not fit it: {exc}; ignored, so the shipped default applies.")
            return _refused
    elif isinstance(default, Mapping) and isinstance(value, dict):
        return _coerce_mapping(default, value, where)
    elif isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    elif isinstance(value, type(default)):
        return value
    logger.warning(f"_coerce: {where} ships as {type(default)} and the override is {type(value)}; ignored, so the shipped default applies.")
    return _refused


def _coerce_mapping(default: Mapping, value: dict, where: str):
    """Fit a JSON object to the shape of the shipped mapping `default`, as a whole replacement. Return it, or `_refused`.

    JSON spells every key as a string, so the keys are converted to the type the shipped keys have. Each
    value whose key the shipped mapping also has is fitted to the shipped value, in turn; a new key's value
    is taken as it comes, there being nothing to fit it to.
    """
    # A mapping in a config file has one key type in practice, so the shipped keys say which — and when
    # they do not (none to look at, or a mix), a string could mean either, and guessing would quietly
    # produce a dict whose lookups all miss. `type(...)` rather than `isinstance`, so a `bool` key is not
    # counted as an `int` one; and only the types a string converts to without ambiguity.
    key_types = {type(key) for key in default}
    if value and (len(key_types) != 1 or not key_types <= {str, int, float}):
        logger.warning(f"_coerce_mapping: {where} ships with keys of type {sorted(t.__name__ for t in key_types) or 'unknown (it is empty)'}, so which type the override's keys should have cannot be told; ignored, so the shipped default applies.")
        return _refused
    key_type = key_types.pop() if key_types else str

    fitted = {}
    for written_key, written_value in value.items():
        try:
            key = key_type(written_key)
        except ValueError:
            logger.warning(f"_coerce_mapping: {where} has keys of type {key_type.__name__}, and the override's key '{written_key}' is not one; ignored, so the shipped default applies.")
            return _refused
        if key in default:
            written_value = _coerce(default[key], written_value, f"{where}[{key!r}]")
            if written_value is _refused:
                return _refused
        fitted[key] = written_value

    # Through its own class, as for a `NamedTuple` above. A class that cannot be built from a dict — an
    # `env`, whose fields are keyword arguments — is refused here: its set of fields is what the app reads,
    # and replacing it wholesale is not something an override should be able to do by accident.
    try:
        return type(default)(fitted)
    except TypeError as exc:
        logger.warning(f"_coerce_mapping: {where} is a {type(default).__name__}, which cannot be replaced as a whole ({type(exc)}: {exc}); name its entries with a dotted name instead. Ignored, so the shipped default applies.")
        return _refused


def _lookup(container, step: str):
    """Find the member of `container` that `step` names. Return `(key, member)`, or `None` if there is no unique one.

    A mapping's member is an entry, named by its key as a string — so `32` is named `"32"`, and a mapping
    holding both `32` and `"32"` has no unique member by that name. A sequence's member is an item, named
    by its index. Anything else is reached by attribute, which is also how a `NamedTuple`'s fields are named.
    """
    if isinstance(container, Mapping):  # an `env` too, whose fields are its keys
        matches = [key for key in container if str(key) == step]
        return (matches[0], container[matches[0]]) if len(matches) == 1 else None
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes)) and step.isdigit():
        index = int(step)
        return (index, container[index]) if index < len(container) else None
    # `hasattr` is asked rather than trusting the write to fail, because an `env` or an ordinary object
    # accepts a brand-new attribute without complaint — a misspelling would add a field nobody reads.
    if hasattr(container, step):
        return step, getattr(container, step)
    return None


def _store(container, key, member):
    """Store `member` in `container` under `key`. Return the container that now holds it.

    That is `container` itself when it could be updated in place, and a rebuilt copy when it is immutable —
    which then has to be stored in *its* parent in turn. Raise if neither can be done.
    """
    # Same division `unpythonic.mogrify` draws: a mutable container keeps its identity, and only an
    # immutable one is rebuilt, since there is no in-place update to be had for it. A `NamedTuple` is
    # genuinely immutable rather than merely lacking an API for it (3.13's `copy.replace` is its `_replace`,
    # a copy), so rebuilding is the only correct option there, not a shortcut.
    if isinstance(container, MutableMapping):
        container[key] = member
        return container
    if isinstance(container, Mapping):  # `frozendict`, say
        return type(container)({**container, key: member})
    if isinstance(container, MutableSequence):
        container[key] = member
        return container
    if isinstance(container, tuple) and isinstance(key, str):  # a `NamedTuple` field, by name
        return container._replace(**{key: member})
    if isinstance(container, Sequence):  # an immutable sequence, by index; `fupdate` rebuilds through `_make` where there is one
        return fupdate(container, key, member)
    try:
        setattr(container, key, member)
        return container
    except dataclasses.FrozenInstanceError:
        return dataclasses.replace(container, **{key: member})


def _apply_one(namespace: dict, dotted_name: str, value, module_name: str, path: pathlib.Path) -> bool:
    """Bind one setting, walking `dotted_name` into whatever holds it. Return whether it was applied."""
    where = f"'{dotted_name}' of '{module_name}'"

    # Walk down, keeping each container and the key that led out of it: a rebuilt immutable container has
    # to be stored back into its parent, and the trail is how the way back up is found.
    trail = []
    member = namespace
    for step in dotted_name.split("."):
        found = _lookup(member, step)
        if found is None:
            container_name = "that module" if member is namespace else repr(member)
            logger.warning(f"_apply_one: {where} names nothing: '{step}' is not in {container_name} (from {path}); ignored.")
            return False
        key, next_member = found
        trail.append((member, key))
        member = next_member

    fitted = _coerce(member, value, where)
    if fitted is _refused:
        return False

    # Up again, until a container takes the new member in place; everything above that still holds it.
    # Nothing is mutated before that last store, so a failure anywhere on the way leaves the module as
    # shipped rather than half-updated.
    try:
        for container, key in reversed(trail):
            updated = _store(container, key, fitted)
            if updated is container:
                break
            fitted = updated
    except Exception as exc:  # a container this cannot write to; an override file must not stop an app from starting
        logger.warning(f"_apply_one: {where} cannot be set ({type(exc)}: {exc}) (from {path}); ignored, so the shipped default applies.")
        return False
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
               if not name.startswith(_COMMENT_MARKER)
               and _apply_one(namespace, name, value, module_name, path)]
    if applied:
        logger.info(f"apply: {module_name}: overridden from {path}: {', '.join(applied)}")
    return applied
