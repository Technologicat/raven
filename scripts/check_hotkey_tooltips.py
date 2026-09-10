#!/usr/bin/env python
"""Check that every hotkey an app binds is named on the control that triggers it.

Raven's policy (`raven-style-guide.md`, "Hotkey discoverability") is that a hotkey is surfaced twice: on the
F1 help card, and in the caption of the GUI control that does the same thing, bracketed — `"Open folder
[Ctrl+O]"`. `check_option_lists.py` covers the card half. This is the other one.

**It reports; mostly it does not fail.** The reason is a limit it cannot get past: a key with no control
naming it is either a gap or a key that has no control *to* name it, and nothing in the source distinguishes
the two. The Visualizer's `Esc` cancels an edit inside a text field and its arrow keys scroll a panel — no
widget does either, and none should be invented so a checker can be quiet. So the answer is a list for a
person to read, which is the whole job narrowed from forty keys to three.

Where an app has been read through by hand, it goes in `SIGNED_OFF` with the keys that legitimately have no
control, and from then on the run *does* fail for that app — on a newly bound key with no caption, which is
the thing worth catching. An app not on that list is inventory, not a gate.

**A modifier variant counts as named through its parent.** This is the rule the script exists to hold on to,
because deriving it again is what wasted an afternoon. Tooltips do not repeat a chord per variant; they name
the base key once and spell the modifiers out beneath it:

    "Select items currently on-screen in the plotter [F9]
         with Shift: add
         with Ctrl: subtract
         with Ctrl+Shift: intersect"

So `Shift+F9` is named here, and a check looking for a literal `[Shift+F9]` reports a gap that is not one.
A first pass at this over-reported fourteen gaps on the Visualizer where three were real.

**Only string literals count**, gathered from the AST rather than by reading the file as text. A key appears
in comments, in `mvKey_*` constants and in the hotkey table itself, and none of those is something a user
can see. The table's own lines are excluded by source range for the same reason: it must not answer for
itself.

**The keys come from `hotkey_info`, not from the key handler.** That is what the card shows, so it is the
list a reader is promised; a key bound in a handler and absent from the card is `check_option_lists.py`'s
business, from the other direction.

**How much of this is tractable at all**, since that was the question as much as the answer. Across the
eight apps: 195 key cells, of which 20 name several keys at once and no scan can match; of the 175 left, 148
come back named and 21 want a person. So the scan removes something like seven eighths of the reading and
cannot remove the last of it — which is worth having, and is not a gate.

It took three tries to get even that far, and each failure looked like a result:

1. Matching a bracketed literal reported **14 gaps on the Visualizer where 3 were real** — captions name a
   base key once and spell the modifiers beneath it.
2. Stripping *all* modifiers to find the base then missed `Ctrl+Shift+C`, whose parent is `Ctrl+C` and not
   `C`. The parent is whichever sibling the variant differs from.
3. Treating a hyphen as a range marker classified `Numpad -` as a range and declined to check it.

The one honest win came from a mismatch rather than an absence: the chat graph's zoom buttons read
`[numpad +]` where the card read `Numpad +`, which the style guide forbids and which no reader would ever
have reported. That is the shape of defect this catches — two spellings of one key, rather than none.

Exit status is 0 when every signed-off app is fully accounted for, 1 when one of them has a key that is
neither named nor listed as an exemption.
"""

import ast
import itertools
import pathlib
import sys
from typing import Dict, List, Set, Tuple

#: Apps read through by hand, with the keys that have no control to be named on. Only these can fail the run.
#: A key here is a claim that a person looked and found no widget doing the same thing — not that the tooltip
#: is missing. Adding a control for one of these is a *feature*, and the entry should then go.
SIGNED_OFF: Dict[str, Dict[str, str]] = {
    "raven/visualizer": {
        "Esc": "cancels the edit inside the search field; no widget does that",
        "Up arrow": "scrolls the info panel slightly; its buttons are top/bottom and page, not 'a little'",
        "Down arrow": "as Up arrow",
    },
    "raven/librarian": {
        "Page Up": "scrolls the chat log; `view.page_up` has no button, only the key handler",
        "Page Down": "as Page Up",
        "Any arrow": "the card's cell names a class of keys; the graph's captions name the members "
                     "individually ('press an arrow key', '[Up] and [Down]'), so there is nothing to match",
    },
}


def _is_composite(key: str) -> bool:
    """Whether this table cell names several keys rather than one, so no caption can match it.

    A card's *Key* column is prose, not an identifier: a row may cover a pair or a range — `"Up / Down"`,
    `"Arrows / WASD"`, `"Shift+1-9"` — because that is what reads well in a table one line per idea. Those
    are reported apart from the gaps, since "no caption spells this" is trivially true of them and says
    nothing about whether the keys they cover are discoverable.
    """
    # An en dash marks a range (`Shift+1-9`, written with U+2013). A hyphen-minus does not — it *is* a key,
    # as in `Numpad -`, which a hyphen test classifies as a range and then declines to check.
    return "/" in key or "–" in key


def _hotkey_tables(tree: ast.AST) -> List[Tuple[int, int]]:
    """Return the source line ranges of every `*hotkey_info` assignment in `tree`."""
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("hotkey_info"):
                    ranges.append((node.lineno, node.end_lineno))
    return ranges


def _keys_in(tree: ast.AST) -> List[str]:
    """Return every `key=` string in this module's hotkey tables, in order, without duplicates."""
    keys = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not any(name.endswith("hotkey_info") for name in names):
                continue
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                for kw in call.keywords:
                    if kw.arg == "key" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        if kw.value.value and kw.value.value not in keys:
                            keys.append(kw.value.value)
    return keys


def _visible_strings(package: pathlib.Path) -> Set[str]:
    """Return every string literal in `package` that is not part of a hotkey table.

    String literals only: a key also appears in comments and in `dpg.mvKey_*` names, and neither is
    something a user can read. Docstrings are included — they cost nothing here, since a key named only in
    one would still have to be named in a caption to satisfy a reader, and this check is a floor.
    """
    strings = set()
    for path in sorted(package.rglob("*.py")):
        if "tests" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"  could not parse {path}: {exc}")
            continue
        excluded = _hotkey_tables(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(low <= node.lineno <= high for low, high in excluded):
                    continue
                strings.add(node.value)
    return strings


def _literally_named(key: str, strings: Set[str]) -> bool:
    """Whether some control's caption spells `key` out.

    Bracketed is the convention, but a caption may list a key among others — `"[Enter, while the search
    field has focus]"` — so a comma or the closing bracket counts as a boundary too.
    """
    return any(f"[{key}" in s or f"{key}," in s or f"{key}]" in s for s in strings)


def _named_by(key: str, strings: Set[str]) -> str:
    """Return how `key` is named on some control, or the empty string if it is not named at all."""
    if _literally_named(key, strings):
        return "named"

    # A modifier variant, named through a *parent* key plus a "with <the difference>:" line beneath it. The
    # parent is whichever sibling the variant differs from, which is not necessarily the bare base key:
    # Librarian's `Ctrl+Shift+C` hangs off `Ctrl+C` and differs from it by Shift alone, so its caption reads
    # "[Ctrl+C] ... with Shift: include message node ID". Looking only for `with Ctrl+Shift:` misses it.
    parts = key.split("+")
    mods, base = parts[:-1], parts[-1]
    # Longest first, so the answer names the smallest difference — the one the caption actually spells.
    for dropped_count in range(1, len(mods) + 1):
        for dropped in itertools.combinations(range(len(mods)), dropped_count):
            kept = [m for i, m in enumerate(mods) if i not in dropped]
            parent = "+".join(kept + [base])
            difference = "+".join(mods[i] for i in dropped)
            if _literally_named(parent, strings) and any(f"with {difference}:" in s for s in strings):
                return f"via {parent} + 'with {difference}:'"
    return ""


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    packages = sorted({p.parent for p in repo.glob("raven/**/app.py")})
    failed = False
    totals = {"cells": 0, "composite": 0, "named": 0, "for a person": 0}
    for package in packages:
        rel = package.relative_to(repo).as_posix()
        tree = ast.parse((package / "app.py").read_text(encoding="utf-8"))
        keys = _keys_in(tree)
        if not keys:
            continue
        strings = _visible_strings(package)
        exemptions = SIGNED_OFF.get(rel, {})
        composite = [key for key in keys if _is_composite(key)]
        single = [key for key in keys if not _is_composite(key)]
        unnamed = [key for key in single if not _named_by(key, strings)]
        unexplained = [key for key in unnamed if key not in exemptions]
        named = len(single) - len(unnamed)

        gated = rel in SIGNED_OFF
        print(f"{rel}: {named} of {len(single)} single keys named on a control"
              f"{'' if gated else '  [not yet read through -- inventory only]'}")
        for key in unnamed:
            if key in exemptions:
                print(f"    ok, no control to name it: {key} -- {exemptions[key]}")
            else:
                print(f"    {'MISSING' if gated else 'unnamed'}: {key}")
        if composite:
            # Phrased so nothing has to agree with the count: "1 rows" and "1 row(s)" are both wrong, and a
            # participle sidesteps the verb as well as the noun.
            print(f"    not checked, each naming several keys at once so that no caption can match "
                  f"({len(composite)}): {', '.join(repr(k) for k in composite)}")
        if gated and unexplained:
            failed = True
        totals["cells"] += len(keys)
        totals["composite"] += len(composite)
        totals["named"] += named
        totals["for a person"] += len(unexplained)

    # What the scan is worth, printed rather than claimed: how much of the reading it actually removes.
    print(f"\nAcross {len(packages)} apps: {totals['cells']} key cells, of which {totals['composite']} name "
          f"several keys at once and cannot be matched by any scan. Of the {totals['cells'] - totals['composite']} "
          f"that can, {totals['named']} are named on a control and {totals['for a person']} are left for a "
          f"person to look at.")

    if failed:
        print("\nA signed-off app has a key named on no control. Either give the control a bracketed hint,\n"
              "or, if nothing triggers that key but the key itself, add it to SIGNED_OFF with the reason.")
        return 1
    print("\nOK: every signed-off app accounts for all of its keys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
