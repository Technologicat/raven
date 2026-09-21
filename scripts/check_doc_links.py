#!/usr/bin/env python
"""Check that a Markdown document's own anchor links, and its table of contents, still resolve.

The README carries a hand-maintained table of contents, and nothing about adding a heading prompts
anyone to update it — so it drifts silently, and a reader only finds out by clicking a link that goes
nowhere. The same holds for cross-references in the prose: `[see below](#some-section)` keeps pointing
at a heading long after the heading has been renamed.

This is the docs-side sibling of `check_usage_paths.py`, which asks whether a `python -m raven...` line
still names a module. Same rot, same reason nothing else catches it: no import, no lint, no test.

**Scope is deliberately one document at a time.** Only in-document anchors (`](#target)`) are checked,
against that same file's headings. A link into *another* file's anchor is a different problem — it needs
the other file parsed and its path resolved — and is tracked fleet-wide in `~/.claude`'s
`TODO_DEFERRED.md` under the internal-reference-check item.

**The anchor rule is GitHub's, and its one surprising case is worth knowing**: punctuation is dropped
*before* spaces become hyphens, so `Install & run` yields `install--run` with two hyphens, the removed
`&` having left its surrounding spaces behind. A slug function that collapses whitespace instead gets
this wrong and reports working links as broken — which is how this checker's first draft produced two
false positives on a README that was entirely correct.

`slugify`, `heading_anchors`, `toc_anchors`, `dangling_links` and `toc_problems` are kept
character-for-character identical to the copies in `pyan/tests/test_docs.py`, **their code at least** —
a docstring may differ where it points at the other copy, since each says where the other one is. That
duplication is on purpose while there are two of them: the fleet-wide version will lift these out into
one shared module, and keeping the copies from drifting in the meantime is what makes that a move rather
than a merge. If you change one, change the other, and compare the *bodies* — `ast.dump` of each
function with its docstring dropped is the check, since a prose diff reports the cross-references as
drift every time.

Usage:

    python scripts/check_doc_links.py                 # every tracked .md file
    python scripts/check_doc_links.py CLAUDE.md ...   # just these

A file with no table of contents is checked for dangling links alone; only a document that *has* one can
have one that disagrees. Whether a TOC is expected to list *every* heading is per-document and recorded
in `COMPLETE_TOC` below — the component READMEs deliberately index their top-level sections only.

Exit status is 0 when clean, 1 when anything is reported.
"""

import pathlib
import re
import subprocess
import sys

__all__ = ["slugify", "heading_anchors", "toc_anchors", "dangling_links", "toc_problems",
           "tracked_markdown", "check_file", "main"]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# A list item that is nothing but a link to an anchor in this same document.
TOC_ENTRY = re.compile(r"^\s*- \[.*\]\(#([^)]+)\)\s*$")
ANCHOR_LINK = re.compile(r"\[[^\]]*\]\(#([^)]+)\)")
# An inline code span. A whole link inside one is being *quoted*, not made.
INLINE_CODE = re.compile(r"`[^`]*`")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")


def slugify(title):
    """GitHub's heading-anchor rule: lowercase, drop punctuation, spaces to hyphens.

    A link in the heading contributes its text and not its target, GitHub anchoring from what the
    heading *renders* as. Without this the URL survives as letters — `## 0.2.9 — ["Pleiades"](https://
    en.wikipedia.org/wiki/Pleiades) edition` would anchor as `029--pleiadeshttpsenwikipediaorg...`,
    and a TOC generated from the same function would agree with itself while disagreeing with GitHub.
    """
    title = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", title)
    return re.sub(r"[^\w\s-]", "", title.strip().lower()).replace(" ", "-")


def heading_anchors(lines):
    """Every heading's anchor, in document order, with GitHub's duplicate suffixes.

    Lines inside fenced code blocks are skipped: the shell examples are full of
    ``# comment`` lines that would otherwise read as top-level headings.
    """
    anchors = []
    counts = {}
    in_fence = False
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING.match(line)
        if not match:
            continue
        base = slugify(match.group(2))
        seen = counts.get(base, 0)
        counts[base] = seen + 1
        anchors.append(base if seen == 0 else f"{base}-{seen}")
    return anchors


def toc_anchors(lines):
    """The targets listed in the table of contents.

    The TOC is the first run of consecutive link-only list items, so an anchor
    link written anywhere else in the document is not mistaken for one.
    """
    block = []
    for line in lines:
        match = TOC_ENTRY.match(line)
        if match:
            block.append(match.group(1))
        elif block:
            break
    return block


def dangling_links(lines):
    """Anchor links anywhere in the prose that point at no heading.

    Inline code spans are removed before matching, so a link written *about* link syntax — a brief
    explaining that ``[text](#anchor)`` is what it walks — is prose rather than a reference to a
    heading called "anchor". Raven's briefs are where the gap showed up, pyan's README having no such
    sentence; `pyan/tests/test_docs.py` carries the same fix.
    """
    headings = set(heading_anchors(lines))
    found = []
    in_fence = False
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        found.extend(a for a in ANCHOR_LINK.findall(INLINE_CODE.sub("", line)) if a not in headings)
    return found


def toc_problems(lines):
    """Every way the table of contents can disagree with the headings."""
    headings = heading_anchors(lines)
    toc = toc_anchors(lines)
    problems = []
    if not toc:
        problems.append("no table of contents found")
    problems += [f"TOC entry points at no heading: {a}" for a in toc if a not in headings]
    problems += [f"heading missing from TOC: {a}" for a in headings if a not in toc]
    if toc != [a for a in headings if a in toc]:
        problems.append("TOC is not in document order")
    return problems


# Documents whose table of contents is meant to list *every* heading, so that one missing from it is
# a fault rather than an editorial choice. Add a file here when its TOC is meant to be exhaustive.
#
# Most of Raven's Markdown is deliberately not in this set. The component READMEs carry a TOC of their
# top-level sections and stop there — `raven/librarian/README.md` lists 28 of its 47 headings, and
# `raven/avatar/README.md` 13 of 14 — so demanding completeness of them reports twenty-odd faults that
# are all the author's intent. The main README is exhaustive, and being checked for it is what caught
# the one `#####` heading that had never been listed.
#
# `CHANGELOG.md` is deliberately partial too, and for a different reason worth stating so that nobody
# "completes" it later: its TOC lists releases and stops. Ten releases times three sections times up to
# eight component groups is a hundred lines of contents before a reader reaches any content — and the
# navigation that would buy is already free twice over, from GitHub's heading outline and from folding
# in an editor. What a reader scanning that file wants is which release, not which component of 0.2.4.
COMPLETE_TOC = {"README.md"}

TOC_COMPLETENESS_PREFIX = "heading missing from TOC: "


def tracked_markdown() -> list[pathlib.Path]:
    """Every tracked `.md` file, as absolute paths. Gitignored scratch is out of scope."""
    listing = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z", "*.md"],
                             capture_output=True, text=True, check=True).stdout
    return [REPO_ROOT / name for name in listing.split("\0") if name]


def check_file(path: pathlib.Path, *, require_complete_toc: bool) -> list[str]:
    """Everything wrong with one document's internal links, as lines ready to print.

    A document with no table of contents is not faulted for lacking one — most of this repository's
    Markdown has none, and only a document that has one can have one that disagrees.

    `require_complete_toc` selects whether a heading absent from the TOC counts. The filtering is done
    on `toc_problems`' output rather than inside it, so that function stays identical to pyan's copy;
    see this module's docstring on why that matters.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    problems = []
    if toc_anchors(lines):
        problems += toc_problems(lines)
        if not require_complete_toc:
            problems = [p for p in problems if not p.startswith(TOC_COMPLETENESS_PREFIX)]
    problems += [f"link points at no heading: #{a}" for a in dangling_links(lines)]
    return problems


def main() -> int:
    names = sys.argv[1:]
    paths = ([pathlib.Path(n) if pathlib.Path(n).is_absolute() else REPO_ROOT / n for n in names]
             if names else tracked_markdown())

    failed = 0
    for path in paths:
        if not path.is_file():
            print(f"{path}: no such file", file=sys.stderr)
            failed += 1
            continue
        relative = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        problems = check_file(path, require_complete_toc=str(relative) in COMPLETE_TOC)
        if problems:
            failed += 1
            print(f"{relative}: {len(problems)} problem{'s' if len(problems) != 1 else ''}:",
                  file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)

    if failed:
        print(f"\n{failed} of {len(paths)} documents have internal links that do not resolve.",
              file=sys.stderr)
        return 1
    print(f"OK: internal links and tables of contents agree across {len(paths)} "
          f"document{'s' if len(paths) != 1 else ''}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
