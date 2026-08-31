#!/usr/bin/env python
"""Check that every `python -m raven...` in this repository names a module that exists.

Usage lines are documentation the reader is meant to *type*, and they rot the way any other reference to a
moved file does — except that nothing imports them, nothing lints them, and the test suite has no opinion
about them. So a module renamed or relocated leaves its old path sitting in a docstring, a README table or
a shell script, indistinguishable from a working one until somebody tries it.

Three were stale on 2026-08-31, the day this was written, and each had been wrong for a different reason:

- `raven/conference_timer/app.py` and `raven/xdot_viewer/app.py` both advertised the *package* rather than
  the module — `raven.xdot_viewer` where only `raven.xdot_viewer.app` runs, no package here carrying a
  `__main__.py`.
- `raven/common/image/tests/loader_bench.py` still gave `raven.cherrypick.tests.loader_bench`, from before
  the module moved packages.
- `convert-all-wos2bib.sh` called `raven.import_wos`, which has never existed in this repository — the
  module was `raven/wos2bib.py` on the day the script was committed, and is `raven/papers/wos2bib.py` now.

What it checks: for each distinct `python -m <dotted.path>` under the `raven` root, that the path resolves
to a `.py` file or to a package with a `__main__.py`. Tracked files only, so gitignored scratch is out of
scope, and text of any kind is in — the README's console-script table is exactly where a wrong one costs
the most.

**It deliberately stops at existence and does not ask whether the module *does* anything under `-m`.** The
tempting extra check is a `__main__` guard, and it is wrong here: six of the thirty-one paths have none,
and all six are apps whose module bodies run at import. Every one would be a false report.

The sibling rot in console-script names (`raven-foo` in prose, against `[project.scripts]`) is not checked
either, and that is a measurement rather than an oversight: matching the obvious pattern over this
repository yields thirty-odd hits and *no* true positives — Markdown heading anchors
(`raven-visualizer-visualize-research-literature`), hyphenated prose (`raven-style-guide`), and tools that
are proposed in briefs but not yet written. A checker at that signal-to-noise trains people to ignore it.

Exit status is 0 when clean, 1 when anything is reported.
"""

import pathlib
import re
import subprocess
import sys

__all__ = ["find_usages", "unresolvable", "main"]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# `python -m raven.<package>.<module>`, also matching `python3` and whatever prefix the line puts before it
# (`pdm run`, a `$` prompt, a Markdown table's arrow). Only first-party paths: a third-party module is
# not in this tree, so its absence here says nothing.
#
# The trailing lookahead requires the path to be *complete*, which is what keeps prose about this checker
# out of its own report: `python -m raven...` and `python -m raven.<module>` both stop at a dot that no
# identifier follows, so neither is read as a claim that `raven` is runnable. Write placeholders that way.
# The cost is a usage line ending in a sentence period mid-paragraph, which is not seen: usage lines here
# live in code fences.
USAGE_RE = re.compile(r"\bpython3?\s+-m\s+(raven(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?![\w.])")


def tracked_files() -> list[pathlib.Path]:
    """Every file git tracks, as paths relative to the repository root."""
    listing = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
                             capture_output=True, text=True, check=True).stdout
    return [pathlib.Path(name) for name in listing.split("\0") if name]


def find_usages() -> dict[str, list[str]]:
    """Return `{dotted module path: [where it appears, as "file:line"]}` over the tracked files."""
    usages: dict[str, list[str]] = {}
    for relative_path in tracked_files():
        try:
            text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):  # a binary asset, or a path git knows about and we cannot read
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in USAGE_RE.finditer(line):
                usages.setdefault(match.group(1), []).append(f"{relative_path}:{lineno}")
    return usages


def unresolvable(usages: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return the subset of `usages` whose module path names nothing in the tree."""
    def resolves(dotted: str) -> bool:
        base = REPO_ROOT / pathlib.Path(*dotted.split("."))
        return base.with_suffix(".py").is_file() or (base / "__main__.py").is_file()

    return {dotted: sites for dotted, sites in usages.items() if not resolves(dotted)}


def main() -> int:
    usages = find_usages()
    broken = unresolvable(usages)

    if broken:
        print(f"{len(broken)} of {len(usages)} `python -m` path(s) name nothing in the tree:",
              file=sys.stderr)
        for dotted in sorted(broken):
            print(f"  {dotted}", file=sys.stderr)
            for site in broken[dotted]:
                print(f"      {site}", file=sys.stderr)
        return 1

    print(f"OK: all {len(usages)} distinct `python -m raven...` paths resolve to a module.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
