#!/usr/bin/env python
"""Check that the `dpg` skill's router still points at sections that exist.

`SKILL.md` is a router into `dpg-notes.md`: it holds no content of its own, only a table saying which section
answers which question. That makes it silently perishable in a way the notes are not — reword a heading and
the row still reads fine, still looks authoritative, and now sends the reader nowhere. Nothing at runtime
would ever notice, because a skill is prose.

So this asserts the one invariant that keeps a router honest: **every section name it cites resolves against
a real heading, and every path it mentions exists.** Run it after editing either file.

Lives beside the skill rather than in `scripts/` so that the checker travels with the thing it checks — the
same reason the skill lives beside the notes it indexes. If that trio ever moves to another repository, it
moves as a unit and keeps working.

Exit code 0 if the router is intact, 1 otherwise.
"""

import pathlib
import re
import sys

SKILL_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SKILL_DIR.parents[2]
NOTES = REPO_ROOT / "dpg-notes.md"
SKILL = SKILL_DIR / "SKILL.md"

# A router target: `*Section*`, optionally `→ Subsection`, optionally a trailing `(commentary)`.
# `·` separates targets within one table cell.
_TOP_LEVEL = re.compile(r"^\s*\*([^*]+)\*")
_TRAILING_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")
# Backticked spans, one line at a time — `[^`]*` across newlines pairs the closing backtick of one line with
# the opening backtick of another, and every such "path" is nonsense.
_BACKTICK_SPAN = re.compile(r"`([^`\n]+)`")


def headings(markdown: str) -> set[str]:
    """Every ATX heading in `markdown`, without its leading hashes."""
    return {re.sub(r"^#+\s+", "", line).strip()
            for line in markdown.splitlines() if line.startswith("#")}


def cited_sections(skill: str) -> list[tuple[str, tuple[str, ...]]]:
    """Every `(citation, acceptable_names)` the router's table names, in file order.

    `citation` is the raw chunk, kept so a failure can quote what the reader would have seen.
    `acceptable_names` holds one or more readings of the same citation; any one resolving is enough.
    """
    found = []
    for line in skill.splitlines():
        if not line.startswith("|") or line.startswith("|---") or "→" not in line and "*" not in line:
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        for chunk in cells[2].split("·"):
            match = _TOP_LEVEL.match(chunk)
            if not match:
                continue
            found.append((chunk.strip(), (match.group(1).strip(),)))
            if "→" in chunk:
                subsection = chunk.split("→", 1)[1].strip()
                # A trailing `(…)` is usually this table's own commentary — but some headings genuinely end
                # in a parenthetical ("… (the 517/518 trap)"), so accept either reading rather than forcing
                # the table to avoid a shape the notes actually use.
                candidates = (subsection, _TRAILING_PARENTHETICAL.sub("", subsection).strip())
                found.append((chunk.strip(), tuple(c for c in candidates if c)))
    return found


def cited_paths(skill: str) -> list[str]:
    """Every repository path the skill mentions in backticks.

    A span counts as a path if it holds no whitespace and either has a slash or names a Markdown file. The
    no-whitespace rule is what excludes a *command* that contains a path — "python .claude/…/check_router.py"
    is backticked the same way and is not a filename. (No path in this repository contains a space, so the
    rule costs nothing; it would need revisiting in a tree where one did.) Prose like `guiutils` and
    `split_frame` is excluded by having neither a slash nor a `.md`.
    """
    return sorted({span for span in _BACKTICK_SPAN.findall(skill)
                   if not any(c.isspace() for c in span) and ("/" in span or span.endswith(".md"))})


def main() -> int:
    if not NOTES.exists():
        print(f"FAIL: the reference itself is missing: {NOTES}")
        return 1
    known = headings(NOTES.read_text(encoding="utf-8"))
    skill = SKILL.read_text(encoding="utf-8")

    problems = []

    sections = cited_sections(skill)
    for citation, acceptable in sections:
        if not any(name in known for name in acceptable):
            problems.append(f"  section not found in dpg-notes.md: {acceptable[0]!r}\n      cited by: {citation}")

    paths = cited_paths(skill)
    for path in paths:
        if not (REPO_ROOT / path.rstrip("/")).exists():
            problems.append(f"  path does not exist: {path}")

    if problems:
        print(f"FAIL: the dpg skill's router has {len(problems)} broken pointer(s):")
        print("\n".join(problems))
        print("\nFix SKILL.md, or restore the heading it names. A router that points nowhere is worse than "
              "no router: it reads as an answer.")
        return 1

    print(f"OK: {len(sections)} section citations and {len(paths)} paths all resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
