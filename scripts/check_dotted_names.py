#!/usr/bin/env python
"""Check that every backquoted, fully qualified `raven.…` name in this repository names something that exists.

Docstrings, comments and documentation point at code by its dotted name — `raven.librarian.scaffold.ai_turn`,
`raven.common.gui.utils.MinimumShowTime` — and nothing else in the toolchain reads those references. A
function renamed or moved leaves its old name in prose, still rendering, still read as current, until a
reader goes looking for it. A probe on 2026-10-02 found 33 of 354 distinct names unresolved, about ten of
them real rot in live source and docs.

What it checks: every backquoted span in a tracked text file that begins with a fully qualified `raven.`
name, resolved against the source tree without importing anything:

- packages and modules, as directories (with or without `__init__.py`) and `.py` files under `raven/`;
- then names bound at a module's top level: definitions, assignments, and what its imports bring in,
  star imports followed to the module they name;
- then the members of a class: what its body defines, and the `self.<name>` its methods assign.

A name resolves if every component does. Past a class member it stops asking, the type of an attribute
being something only a run could tell.

Names that are absent on purpose are marked where they are written, by a parenthetical right after the
closing backtick, which tells the reader as much as it tells this checker:

- `(planned)` for something a document proposes. Reported if it comes to exist, so the marker goes when the
  plan lands.
- `(stale)` for a reference kept as history: a name that moved, or never existed, quoted to say so.
  Reported if it resolves after all.

Not checked: `briefs/` and `investigations/`, which name what a brief plans and what a closed piece of work
used, both legitimately absent from the tree; the released sections of `CHANGELOG.md`, each describing its
own time; and short forms such as `guiutils.focus_item`, which need each file's import aliases to resolve.

Exit status is 0 when clean, 1 when anything is reported.
"""

import ast
import functools
import pathlib
import re
import subprocess
import sys

__all__ = ["find_references", "resolves", "unresolved", "main"]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Trees whose references are allowed to name what does not exist (yet, or any more).
EXEMPT_PREFIXES = ("briefs/", "investigations/")

BACKQUOTED_RE = re.compile(r"`([^`\n]+)`")
MARKER_RE = re.compile(r"\s*\((planned|stale)\)")
CHANGELOG = pathlib.Path("CHANGELOG.md")
DOTTED_RE = re.compile(r"raven(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def tracked_files() -> list[pathlib.Path]:
    """Every file git tracks, as paths relative to the repository root."""
    listing = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
                             capture_output=True, text=True, check=True).stdout
    return [pathlib.Path(name) for name in listing.split("\0") if name]


def find_references() -> dict[tuple[str, str | None], list[str]]:
    """Return `{(dotted name, marker or None): [where it appears, as "file:line"]}` over the tracked, non-exempt
    files. The marker is `"planned"` or `"stale"` where the reference carries one."""
    references: dict[tuple[str, str | None], list[str]] = {}
    for relative_path in tracked_files():
        if str(relative_path).startswith(EXEMPT_PREFIXES):
            continue
        try:
            text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):  # a binary asset, or a path git knows about and we cannot read
            continue
        released = False  # in `CHANGELOG.md`: past the in-progress release's section
        for lineno, line in enumerate(text.splitlines(), start=1):
            if relative_path == CHANGELOG and line.startswith("## "):
                released = "in progress" not in line
            if released:
                continue
            for span in BACKQUOTED_RE.finditer(line):
                match = DOTTED_RE.match(span.group(1))
                if match is not None:
                    maybe_marker = MARKER_RE.match(line, span.end())
                    key = (match.group(0), maybe_marker.group(1) if maybe_marker else None)
                    references.setdefault(key, []).append(f"{relative_path}:{lineno}")
    return references


@functools.cache
def _parse(path: pathlib.Path) -> ast.Module | None:
    if path.is_dir():
        return None
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _module_file(module: tuple[str, ...]) -> pathlib.Path | None:
    """The source for a dotted module path: `x.py`, a package's `__init__.py`, or the directory of a namespace
    package (most of Raven's have no `__init__.py`), which binds no names of its own."""
    base = REPO_ROOT.joinpath(*module)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.is_dir():
        return base
    return None


def _absolute(module: tuple[str, ...], node: ast.ImportFrom) -> tuple[str, ...]:
    """The module a `from … import` in `module` names, made absolute."""
    if not node.level:
        return tuple((node.module or "").split("."))
    is_package = _module_file(module) is not None and _module_file(module).name == "__init__.py"
    anchor = module if is_package else module[:-1]
    anchor = anchor[:len(anchor) - (node.level - 1)] if node.level > 1 else anchor
    return anchor + (tuple(node.module.split(".")) if node.module else ())


@functools.cache
def _top_level_names(module: tuple[str, ...]) -> frozenset[str]:
    """Every name bound at `module`'s top level, star imports followed."""
    path = _module_file(module)
    tree = _parse(path) if path is not None else None
    if tree is None:
        return frozenset()
    names: set[str] = set()

    def bind(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind(element)

    def visit(statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    bind(target)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                bind(node.target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        names.update(_top_level_names(_absolute(module, node)))
                    else:
                        names.add(alias.asname or alias.name)
            # Bindings inside a top-level `if`, `try` or `with` are still the module's.
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for block in ("body", "orelse", "finalbody"):
                    visit(getattr(node, block, []) or [])
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)

    visit(tree.body)
    return frozenset(names)


def _class_members(module: tuple[str, ...], class_name: str) -> frozenset[str] | None:
    """The members of class `class_name` in `module`, or `None` if it is not a class defined there."""
    path = _module_file(module)
    tree = _parse(path) if path is not None else None
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            members: set[str] = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    members.add(item.name)
                elif isinstance(item, ast.Assign):
                    members.update(t.id for t in item.targets if isinstance(t, ast.Name))
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
            for sub in ast.walk(node):  # `self.x = …` anywhere in its methods
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target] if isinstance(sub, ast.AnnAssign) else []
                for target in targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        members.add(target.attr)
            return frozenset(members)
    return None


def resolves(dotted: str) -> bool:
    """Whether `dotted` names a module, or something a module or one of its classes defines."""
    parts = tuple(dotted.split("."))
    # The longest prefix that is a module or package.
    for split in range(len(parts), 0, -1):
        if _module_file(parts[:split]) is not None:
            break
    else:
        return False
    module, rest = parts[:split], parts[split:]
    if not rest:
        return True
    if rest[0] not in _top_level_names(module):
        return False
    if len(rest) == 1:
        return True
    members = _class_members(module, rest[0])
    if members is None:
        return True  # not a class defined here (an import, an instance): past that, only a run could tell
    return rest[1] in members


def unresolved(references: dict[tuple[str, str | None], list[str]]) -> dict[tuple[str, str | None], list[str]]:
    """Return the subset of `references` that is wrong: unmarked and naming nothing in the tree, or marked as
    absent and naming something after all."""
    return {(dotted, maybe_marker): sites for (dotted, maybe_marker), sites in references.items()
            if resolves(dotted) == (maybe_marker is not None)}


def main() -> int:
    references = find_references()
    broken = unresolved(references)
    n = len(references)

    if broken:
        print(f"{len(broken)} of {n} backquoted `raven.…` reference{'s' if n != 1 else ''} wrong:", file=sys.stderr)
        for dotted, maybe_marker in sorted(broken, key=lambda key: (key[0], key[1] or "")):
            why = (f"marked ({maybe_marker}), but it exists: drop the marker" if maybe_marker
                   else "names nothing in the tree")
            print(f"  {dotted} -- {why}", file=sys.stderr)
            for site in broken[(dotted, maybe_marker)]:
                print(f"      {site}", file=sys.stderr)
        return 1

    print(f"OK: all {n} backquoted `raven.…` references resolve, or are marked as absent on purpose.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
