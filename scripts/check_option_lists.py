#!/usr/bin/env python
"""Check that a list of options defined in code still appears wherever prose enumerates it.

Some lists exist twice by necessity. The definitive one is a Python literal — a dict of upscaler quality
settings, a table of hotkeys — and the copies are prose: a README, a comment above a config default, a
docstring. Prose cannot import anything, so the copies are hand-maintained, and nothing in the toolchain
has an opinion about them. A value added to the code and not to the prose is invisible to every reader who
learns the options from the documentation, which is most of them.

Two lists are checked today, and they arrived from opposite directions:

- **The avatar upscaler's `quality` settings.** They lived in nine places before `UPSCALE_QUALITIES` became
  the definitive one (2026-09-08). Four were machine-readable and now derive from it; the remaining five
  are prose and are what this checks.
- **Raven-librarian's hotkeys.** The F1 help card builds its table from `hotkey_info`, and the README
  repeats it for a reader who does not have the app open. The first hand-written copy of that table, read
  from the key handler rather than from the card, was missing an entire scope — the audio input panel's
  bare letters — which is what prompted this script.

**Direction matters, and it is one-way: everything in the code list must appear in the prose, never the
reverse.** Librarian's help card is knowingly incomplete — the card is a fixed-height window and the chat
graph's nineteen keys do not fit — so its README lists *more* than `hotkey_info` does. That is the correct
state, and a symmetric check would report it as a failure.

**A name counts as present only inside backticks or quotes**, not as bare prose. Both lists contain words
like `low` and `high` that occur incidentally in any English paragraph, so a substring search would pass
whatever the file said. Every real copy already writes them as `` `low` `` or `"low"`, so the delimiter
costs nothing and removes the whole class of false pass.

Entries the AST cannot resolve — a `key=` built by a function call, as Librarian's send-key label is —
are printed as notes rather than skipped in silence, and do not fail the run. Nothing can be concluded
about them either way; a checker that quietly ignores what it cannot read is how a list goes uncovered
without anybody noticing, and one that fails on a permanent blind spot is a checker nobody runs.

Exit status is 0 when clean, 1 when a documented copy is missing a value.
"""

import ast
import dataclasses
import pathlib
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# How to read the names out of the definitive literal.
DICT_KEYS = "dict-keys"          # `{"low": "...", ...}` -> the keys
ENV_KEY_KWARG = "env-key-kwarg"  # `(env(key="F1", ...), ...)` -> each `key=`


@dataclasses.dataclass(frozen=True)
class Target:
    """A prose file that repeats the list, optionally only within one section of it."""

    path: str
    section: Optional[str] = None  # a Markdown heading; the search is limited to it and what follows,
    #                                up to the next heading of the same level


@dataclasses.dataclass(frozen=True)
class Rule:
    what: str
    module: str
    name: str
    shape: str
    targets: Tuple[Target, ...]


RULES = (
    Rule(what="the avatar upscaler's quality settings",
         module="raven/common/video/upscaler.py",
         name="UPSCALE_QUALITIES",
         shape=DICT_KEYS,
         targets=(Target("raven/server/config.py"),
                  Target("raven/librarian/config.py"),
                  Target("raven/client/mayberemote.py"),
                  Target("raven/avatar/README.md"))),

    Rule(what="Raven-librarian's hotkeys",
         module="raven/librarian/app.py",
         name="hotkey_info",
         shape=ENV_KEY_KWARG,
         targets=(Target("raven/librarian/README.md", section="## Keyboard reference"),)),
)


def normalize(name: str) -> str:
    """Fold the spellings that differ only in spacing, case, or a layer of quoting.

    `Ctrl+Shift+Left` == `ctrl+shift+left`, and a name written as `` `"low"` `` -- a code span around a
    quoted string, which is how a docstring naming a Python literal reads -- folds to the same thing as
    `"low"` and as `` `low` ``.
    """
    return "".join(name.split()).lower().strip("\"'`")


def module_level_value(tree: ast.Module, name: str) -> Optional[ast.expr]:
    """Return the expression assigned to module-level `name`, or `None` if there is no such assignment."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    return None


def read_option_names(path: pathlib.Path, name: str, shape: str) -> Tuple[List[str], List[str]]:
    """Return `(names, unresolved)` from the literal assigned to `name` in the module at `path`.

    `unresolved` holds the source text of entries whose name is not a literal — a call, a variable — about
    which nothing can be checked.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    value = module_level_value(tree, name)
    if value is None:
        raise LookupError(f"{path}: no module-level assignment to `{name}`")

    names: List[str] = []
    unresolved: List[str] = []

    if shape == DICT_KEYS:
        if not isinstance(value, ast.Dict):
            raise TypeError(f"{path}: `{name}` is {type(value).__name__}, expected a dict literal")
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.append(key.value)
            else:
                unresolved.append(ast.unparse(key) if key is not None else "**expansion")
    elif shape == ENV_KEY_KWARG:
        if not isinstance(value, (ast.Tuple, ast.List)):
            raise TypeError(f"{path}: `{name}` is {type(value).__name__}, expected a tuple or list")
        for element in value.elts:
            if not isinstance(element, ast.Call):
                continue  # a sentinel such as `helpcard.hotkey_new_column`; carries no key by design
            for keyword in element.keywords:
                if keyword.arg != "key":
                    continue
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    if keyword.value.value:  # a blank entry is a spacer row
                        names.append(keyword.value.value)
                else:
                    unresolved.append(ast.unparse(keyword.value))
    else:
        raise ValueError(f"unknown shape {shape!r}")

    return names, unresolved


def section_of(text: str, heading: str) -> str:
    """Return `heading` and everything under it, up to the next heading of the same level."""
    start = text.find(heading)
    if start == -1:
        raise LookupError(f"no section {heading!r}")
    level = len(heading) - len(heading.lstrip("#"))
    rest = text[start + len(heading):]
    for match in re.finditer(r"^#{1,6} ", rest, flags=re.MULTILINE):
        if len(rest[match.start():match.end()].strip()) == level:
            return rest[:match.start()]
    return rest


# Anything a writer would put a name in: Markdown code spans, and the quotes a Python comment or docstring
# uses. A bare word does not count -- see the module docstring.
_DELIMITED = re.compile(r"`([^`\n]+)`|\"([^\"\n]+)\"|'([^'\n]+)'")


def delimited_tokens(text: str) -> Set[str]:
    """Return every backticked or quoted run in `text`, normalized."""
    tokens = set()
    for match in _DELIMITED.finditer(text):
        span = match.group(1) or match.group(2) or match.group(3)
        tokens.add(normalize(span))
    return tokens


def check(rule: Rule) -> Tuple[List[str], List[str]]:
    """Return `(problems, notes)` for `rule`. Problems fail the run; notes are printed and do not.

    An entry whose name is computed is a note rather than a problem: nothing can be concluded about it
    either way, and the two that exist -- Librarian's send and newline key labels, which depend on a
    setting -- are permanent. Failing on them every run would make the checker say "something is wrong"
    when the honest report is "there is a corner I cannot see into".
    """
    problems: List[str] = []
    module_path = REPO_ROOT / rule.module
    names, unresolved = read_option_names(module_path, rule.name, rule.shape)
    if not names:
        return ([f"{rule.module}: `{rule.name}` yielded no names, so this rule checks nothing"], [])

    notes = [f"{rule.module}: `{rule.name}` has an entry this cannot read, so it is not checked: {entry}"
             for entry in unresolved]

    wanted = [(name, normalize(name)) for name in names]
    for target in rule.targets:
        target_path = REPO_ROOT / target.path
        text = target_path.read_text(encoding="utf-8")
        where = target.path
        if target.section is not None:
            try:
                text = section_of(text, target.section)
            except LookupError:
                problems.append(f"{target.path}: no section {target.section!r} — "
                                f"the list of {rule.what} has nowhere to be checked against")
                continue
            where = f"{target.path} § {target.section.lstrip('# ')}"
        present = delimited_tokens(text)
        missing = [name for name, key in wanted if key not in present]
        if missing:
            listed = ", ".join(repr(name) for name in missing)
            problems.append(f"{where}: does not name {listed} — "
                            f"defined in {rule.module} as part of `{rule.name}`")
    return problems, notes


def main() -> int:
    problems: Dict[str, List[str]] = {}
    notes: Dict[str, List[str]] = {}
    checked = 0
    for rule in RULES:
        found, noted = check(rule)
        checked += len(rule.targets)
        if found:
            problems[rule.what] = found
        if noted:
            notes[rule.what] = noted

    for what, lines in notes.items():
        print(f"note, {what}:")
        for line in lines:
            print(f"  {line}")

    if not problems:
        print(f"OK: {len(RULES)} option list(s) named in code, and every value appears in each of the "
              f"{checked} place(s) that documents them.")
        return 0

    for what, found in problems.items():
        print(f"\n{what}:")
        for line in found:
            print(f"  {line}")
    total = sum(len(found) for found in problems.values())
    print(f"\n{total} problem(s). A value is documented where it appears in backticks or quotes; "
          "prose that merely contains the word does not count.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
