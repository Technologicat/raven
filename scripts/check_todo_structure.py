#!/usr/bin/env python
"""Check that `TODO_DEFERRED.md` still has the structure its readers and editors assume.

The file is ~4300 lines, is edited constantly, and its items are found by scanning `##` headings. Structural
damage is therefore both easy to cause and hard to see: an edit anchored on a heading that forgets to put
the heading back leaves the orphaned item's body silently appended to the item above it. Nothing renders
wrong, nothing errors, and the item is simply gone from every future scan of the headings — which is exactly
what happened on 2026-08-14, to the item this script was written after.

What it checks, all of it mechanical:

- Every metadata line (`*Cluster: … · Cost: … · Gate: … · Filed: …*`) is preceded by a `##` heading.
  This is the orphan detector, and the one that catches a lost heading.
- Every metadata line names the four required fields.
- Every `##` heading is preceded by a blank line, per the canonical format.
- No two items share a heading, since items are cited from briefs and from each other by title.

Exit status is 0 when clean, 1 when anything is reported.
"""

import pathlib
import re
import sys

__all__ = ["check", "main"]

TODO_PATH = pathlib.Path(__file__).resolve().parent.parent / "TODO_DEFERRED.md"

METADATA_RE = re.compile(r"^\*Cluster:.*\*$")
REQUIRED_FIELDS = ("Cluster:", "Cost:", "Gate:", "Filed:")

# Sections that are prose about the file rather than deferred items, and so carry no metadata line.
SECTIONS_WITHOUT_METADATA = {"Two things a triage pass should know",
                             "Declined",
                             "Waiting on upstream"}


def check(path: pathlib.Path) -> list[str]:
    """Return a list of complaints about `path`. Empty means the structure is intact."""
    lines = path.read_text(encoding="utf-8").splitlines()
    complaints = []
    seen_headings = {}

    for i, line in enumerate(lines):
        lineno = i + 1

        if line.startswith("## "):
            title = line[3:].strip()
            if i > 0 and lines[i - 1].strip():
                complaints.append(f"{lineno}: no blank line before heading {title!r}")
            if title in seen_headings:
                complaints.append(f"{lineno}: heading {title!r} duplicates line {seen_headings[title]}; "
                                  f"items are cited by title, so titles have to be unique")
            seen_headings[title] = lineno

        elif METADATA_RE.match(line):
            # The orphan check. A metadata line belongs to the heading two lines above it.
            if i < 2 or not lines[i - 2].startswith("## "):
                complaints.append(f"{lineno}: metadata line with no `##` heading above it — "
                                  f"an item's heading was probably lost by an edit")
            missing = [field for field in REQUIRED_FIELDS if field not in line]
            if missing:
                complaints.append(f"{lineno}: metadata line is missing {', '.join(missing)}")

    return complaints


def main() -> int:
    if not TODO_PATH.exists():
        print(f"not found: {TODO_PATH}", file=sys.stderr)
        return 1

    complaints = check(TODO_PATH)
    if complaints:
        print(f"{TODO_PATH.name}: {len(complaints)} structural problem(s):", file=sys.stderr)
        for complaint in complaints:
            print(f"  {complaint}", file=sys.stderr)
        return 1

    n_items = sum(1 for line in TODO_PATH.read_text(encoding="utf-8").splitlines()
                  if METADATA_RE.match(line))
    print(f"OK: {TODO_PATH.name} is structurally intact; {n_items} items carry metadata "
          f"(plus {len(SECTIONS_WITHOUT_METADATA)} prose sections that do not).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
