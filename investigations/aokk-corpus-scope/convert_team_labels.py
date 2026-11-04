"""Turn the AOKK team's marked held-for-review sheet into a table of decisions, keyed by record.

    python investigations/aokk-corpus-scope/convert_team_labels.py SHEET.xlsx OUT.tsv --flags FLAGS.tsv

The sheet is free-form: each reviewer's section repeats the header row, the decision is a free-text cell, and a
long reason may sit in the column meant for how AI was used. The team added five columns of their own on the left in
place of the tool's `mark`, and the headers from column F on sit one column off the data, as far as `evidence`. So
columns are read by position here.

`decision` comes out `keep`, `drop` or empty. Empty means the team has not settled it: their own answer was a maybe,
or `--flags` marks the row `check` (see `read_flags`). `--decisions` carries no guess from an empty cell.
"""

import argparse
import collections
import pathlib
import sys

import openpyxl

from raven.common import tabular

# By position, the sheet's actual contents.
REVIEWER, ANSWER, ABSTRACT_ENOUGH, AI_USE, REASON, TIER, KEY, LEVEL, POPULATION = range(9)
TITLE = 13
HEADER_MARK = "Tarkastajan nimi"


def read_flags(path: pathlib.Path) -> dict[str, tuple[str, str, str]]:
    """The rows needing a note, from the table at `path`: key -> (kind, note, proposed decision).

    A kind of `check` means the decision cannot be taken at face value until the team has looked, so it comes out empty
    with the proposal beside it. `ruled` means a rule the team set explicitly settles the row, so the proposal becomes the
    decision. Any other kind only adds the note: the decision stands. Kept in a table of its own, the keys being the
    private corpus's.
    """
    return {row["key"]: (row["kind"], row["note"], row.get("proposed", "")) for row in tabular.read_table(path)}


def decision_of(answer: str) -> str:
    """`keep`, `drop`, or empty for an answer that does not settle it."""
    word = answer.strip().lower().split(" ")[0] if answer.strip() else ""
    return {"kyllä": "keep", "ei": "drop"}.get(word, "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sheet")
    parser.add_argument("out")
    parser.add_argument("--flags", required=True, help="a table with 'key', 'kind' and 'note' columns; see read_flags")
    opts = parser.parse_args()

    flags = read_flags(pathlib.Path(opts.flags))
    worksheet = openpyxl.load_workbook(pathlib.Path(opts.sheet)).worksheets[0]
    rows = []
    for row in worksheet.iter_rows(min_row=2, values_only=True):
        if row[REVIEWER] in (None, HEADER_MARK):
            continue
        cell = lambda i: str(row[i]).strip() if row[i] is not None else ""  # noqa: B023 -- used within this iteration
        key = cell(KEY)
        # One reviewer's later rows carry the reason in the AI column, with the reason column empty.
        reason, ai_use = (cell(REASON), cell(AI_USE)) if cell(REASON) else (cell(AI_USE), "")
        kind, note, proposed = flags.get(key, ("", "", ""))
        decision = {"check": "", "ruled": proposed}.get(kind, decision_of(cell(ANSWER)))
        rows.append([key, decision, reason, cell(ANSWER), cell(REVIEWER), cell(ABSTRACT_ENOUGH), ai_use, kind, note, proposed,
                     cell(TIER), cell(LEVEL), cell(POPULATION), cell(TITLE)])
    tabular.write_table(opts.out, ["The AOKK team's decisions on the held-for-review tier, converted from their sheet."],
                        ["key", "decision", "reason", "answer as written", "reviewer", "abstract enough", "AI use", "flag", "note", "proposed",
                         "tier", "level (prototype)", "population (prototype)", "title"], rows)
    keys = [r[0] for r in rows]
    print(f"{len(rows)} rows, {len(set(keys))} distinct keys: {dict(collections.Counter(r[1] or 'unsettled' for r in rows))}")
    for r in rows:
        if not r[1]:
            print(f"    unsettled: {r[0]} | {r[3]} | proposed: {r[9] or '-'}")
    print("wrote", opts.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
