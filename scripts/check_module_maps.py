#!/usr/bin/env python3
"""Whether the per-package `CLAUDE.md` module maps still describe the packages they map.

Each of those files opens with a table of modules and their sizes, which a reader uses to find the mass of a
package before reading any of it. Two things go wrong with such a table, and neither announces itself:

  - **A size drifts.** Nothing recomputes it, so a figure written once keeps its authority while the code
    moves out from under it. Measured 2026-08-24, the librarian map had been reading 30-45% low for three
    weeks, with `chat_controller.py` recorded at 2769 lines against an actual 3983.
  - **A module goes missing.** A new module is simply absent from the map, and absence is invisible: nothing
    in the table looks wrong, and a reader who trusts it concludes the module does not exist. `indexer.py`
    was unlisted from the day it was added.

Sizes are written rounded to two significant figures (`~1.9k`, `~450`), which is what this checks against —
a rounded figure is allowed to be 5% off, and anything worse means the code moved rather than that the
rounding was coarse. Rounding is the point: a figure that has to be exact has to be maintained, and this
table is read for the shape of a package rather than for a number to quote.

Exits 1 if any figure is off, any documented module is gone, or any module is undocumented.
"""

import pathlib
import re
import sys

# Package map -> the package it maps. Add a pair when a package grows a `CLAUDE.md` with a module table.
MODULE_MAPS = {"raven/librarian/CLAUDE.md": "raven/librarian",
               "raven/visualizer/CLAUDE.md": "raven/visualizer"}

# What a rounded figure may be off by. Two significant figures cannot be worse than 5% on its own, so
# anything beyond this is the code having moved.
TOLERANCE = 0.05

# `name.py (~1.9k)` and `name.py (~450)`, the two forms the tables use.
SIZE_CLAIM = re.compile(r"(\w+\.py)\s*\(~([\d.]+k?)")


def claimed_sizes(text: str) -> dict:
    """Return `{module name: claimed line count}` for every size in `text`."""
    sizes = {}
    for name, figure in SIZE_CLAIM.findall(text):
        sizes[name] = int(float(figure[:-1]) * 1000) if figure.endswith("k") else int(figure)
    return sizes


def actual_sizes(package: pathlib.Path) -> dict:
    """Return `{module name: line count}` for the package's own modules."""
    return {path.name: len(path.read_text(encoding="utf-8").splitlines())
            for path in sorted(package.glob("*.py"))}


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    problems = 0

    for document, package in MODULE_MAPS.items():
        claimed = claimed_sizes((root / document).read_text(encoding="utf-8"))
        actual = actual_sizes(root / package)
        print(f"{document}: {len(claimed)} modules mapped, {len(actual)} present")

        for name, size in sorted(claimed.items()):
            if name not in actual:
                print(f"  GONE      {name}: mapped, but not in {package}/")
                problems += 1
                continue
            error = abs(size - actual[name]) / actual[name]
            if error > TOLERANCE:
                print(f"  STALE     {name}: mapped as ~{size}, actually {actual[name]} ({100 * error:.0f}% off)")
                problems += 1

        for name in sorted(set(actual) - set(claimed)):
            # An undocumented module is the quiet failure: the map looks complete either way.
            print(f"  UNMAPPED  {name}: {actual[name]} lines, absent from the map")
            problems += 1

    if problems:
        print(f"\n{problems} problem(s). Re-measure with `wc -l`, round to two significant figures, and update the map.")
        return 1
    print("\nEvery mapped size is within 5% of the code, and every module is mapped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
