#!/usr/bin/env python3
"""Whether each module's `__all__` still lists what the module has, in the order the module has it.

`__all__` is read as a table of contents: the convention is that it mirrors the file, so a reader who
wants `recenter_window` knows to look after `park_offscreen`. Two things break that, and neither shows
up in a diff of the change that caused it:

  - **The order drifts.** A name is appended where it was convenient rather than where its definition
    sits, or a group is rearranged thematically, and afterwards the list no longer predicts anything.
  - **A name is missing.** `guiutils.DEFAULT_BUTTON_BG_COLOR` was absent from its `__all__` while
    `animation.py` imported it — so `from ... import *` did not bring it, and nothing said why.

**This reports; it never rewrites.** `__all__` carries comments, and their prose is positional — a
line reading `# ...ditto` means nothing once it has been moved, and a group heading describes the names
under it. Sorting that automatically would scramble the commentary while making the names right, which
is a worse artifact than the one it fixed. The wanted order is printed so the fix is a hand-move of
whole lines, comments attached.

Only a literal `__all__` can be checked. A computed one (`__all__ = _something()`, or built by a loop)
is skipped, as is a module with no `__all__` — adding one everywhere is a judgement about what is API,
not something to enforce from here.

Exits 1 if any list is misordered or names something the module does not define. Public definitions
*absent* from `__all__` are reported as notices and do not fail the run, since a public-looking name is
not necessarily API — a DPG callback bound by name, or a parser's internal classes, legitimately stay
out. `--strict` fails on those too.
"""

import argparse
import ast
import pathlib
import sys

# Trees that are not ours to hold to this, or that have no API surface to speak of.
SKIP_PARTS = {"vendor", "tests", "__pycache__"}


def find_all_assignment(tree: ast.Module) -> ast.Assign | None:
    """Return the module-level `__all__ = [...]` assignment, or `None` if there is none."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "__all__" for target in node.targets):
            return node
    return None


def definition_order(tree: ast.Module) -> list[str]:
    """Return the names the module binds at module level, in source order, first binding wins.

    Imports count: a module that re-exports something is offering it, and `__all__` lists it at the
    import's position. Definitions guarded by `if` or `try` count too — that is how an optional
    dependency's shim gets defined, and it is still where the reader will find it.
    """
    order: list[str] = []
    seen: set[str] = set()

    def remember(name: str) -> None:
        if name and name != "*" and name not in seen:
            seen.add(name)
            order.append(name)

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                remember(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        remember(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                remember(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    remember((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, (ast.If, ast.Try)):
                visit(node.body)
                visit(node.orelse)
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)
                visit(getattr(node, "finalbody", []))

    visit(tree.body)
    return order


def supplies_names_dynamically(tree: ast.Module) -> bool:
    """Whether the module can produce names this script cannot see, making "undefined" unanswerable.

    Two ways. `from x import *` hides where a name came from. A module-level `__getattr__` (PEP 562)
    *manufactures* one on access — `xdotwidget` uses it to hand out its widget class without importing
    DearPyGui when the package is imported, which is a good reason and not something to report.
    """
    star = any(isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
               for node in ast.walk(tree))
    module_getattr = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__"
                         for node in tree.body)
    return star or module_getattr


def public_definitions(tree: ast.Module) -> list[str]:
    """Names the module defines itself — not imports — that do not start with an underscore."""
    return [node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith("_")]


def check(path: pathlib.Path, strict: bool) -> tuple[int, int]:
    """Check one module. Return `(problems, notices)`."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        print(f"  UNPARSED  {path}: {type(exc).__name__}: {exc}")
        return (1, 0)

    assignment = find_all_assignment(tree)
    if assignment is None:
        return (0, 0)
    try:
        exported = list(ast.literal_eval(assignment.value))
    except ValueError:  # computed rather than written out; nothing to compare against
        return (0, 0)

    problems = notices = 0
    order = definition_order(tree)

    if not supplies_names_dynamically(tree):
        for name in exported:
            if name not in set(order):
                print(f"  UNDEFINED {path}: `__all__` names {name!r}, which the module does not define or import")
                problems += 1

    expected = [name for name in order if name in set(exported)]
    listed = [name for name in exported if name in set(order)]
    if listed != expected:
        print(f"  MISORDER  {path}: `__all__` does not follow the file")
        for a, b in zip(listed, expected):
            if a != b:
                print(f"              first divergence: lists {a!r} where the file defines {b!r}")
                break
        print(f"              wanted order: {', '.join(expected)}")
        problems += 1

    missing = [name for name in public_definitions(tree) if name not in set(exported)]
    if missing:
        label = "MISSING  " if strict else "notice   "
        print(f"  {label} {path}: defined and public, but not in `__all__`: {', '.join(missing)}")
        if strict:
            problems += 1
        else:
            notices += 1

    return (problems, notices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=["raven"],
                        help="files or directories to check (default: raven)")
    parser.add_argument("--strict", action="store_true",
                        help="also fail when a public definition is absent from `__all__`")
    opts = parser.parse_args()

    files: list[pathlib.Path] = []
    for entry in opts.paths:
        path = pathlib.Path(entry)
        if path.is_dir():
            files += [p for p in sorted(path.rglob("*.py")) if not (SKIP_PARTS & set(p.parts))]
        elif path.suffix == ".py":
            files.append(path)

    problems = notices = 0
    for path in files:
        found, noticed = check(path, opts.strict)
        problems += found
        notices += noticed

    print()
    if problems:
        print(f"{problems} problem(s) in {len(files)} module(s). Move the lines by hand, comments attached —")
        print("their prose is positional, which is why nothing here rewrites the list for you.")
        return 1
    if notices:
        print(f"No ordering problems in {len(files)} module(s); {notices} module(s) have a public definition "
              "outside `__all__`, which may well be deliberate. `--strict` treats those as problems.")
        return 0
    print(f"Every `__all__` in {len(files)} module(s) follows its file and names only what the module has.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
