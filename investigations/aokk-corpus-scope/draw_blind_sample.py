"""Draw the stratified blind sample of `blind-sample-design.md` from a `raven-scopebib` run, and score it once rated.

    python investigations/aokk-corpus-scope/draw_blind_sample.py draw RUN_DIR OUT_DIR --exclude DECISIONS.tsv [--seed 17]
    python investigations/aokk-corpus-scope/draw_blind_sample.py score OUT_DIR RATED.xlsx

`draw` takes the run's three outcome tables (`03b-removed`, `03c-kept`, `03a-CHECK-BY-HAND`) and its rule table
(`02b-ruled`, for the kept records' tags), leaves out every record in `--exclude` (a table with a `key` column: what
people have rated already) and every record the sift removed (no abstract to decide from, and not the tool's judgement),
and draws by the tool's outcome, at random within each stratum: 30 removed, 20 kept, 10 sent to a person. Each record's
group is recorded in the key for reading the result: what decided a removal, a kept record's tags, a hold's reason.

It writes the sheet people fill in, with `--rules` on a second sheet of it,
(`blind-sample.xlsx`: shuffled, with title, abstract, venue and link and nothing of the stratum or the tool's answer, and
the columns the team added themselves to their own sheet) and the key (`blind-key.tsv`: each record's stratum, group and
the tool's outcome, plus each stratum's size in the pool, so that rates can be weighted back to it).

`score` reads a filled-in copy of the sheet against the key: per stratum, how often people agree with the tool, the
one-sided 95% upper bound on the share it got wrong (Clopper-Pearson), and the weighted share of in-scope records
removed across the pool. The sheet's `päätös` takes kyllä/ei or mukaan/pois, or keep/drop.
"""

import argparse
import collections
import pathlib
import random
import re
import sys

from raven.common import tabular
from raven.papers.scopebib import report

QUOTAS = {"removed": 30, "kept": 20, "to a person": 10}
# The rater's four columns carry the names the team gave them in their own sheet. Only the decision is needed; the
# check was promised to be fast, so the other three say they are optional.
OPTIONAL = " (valinnainen)"
SHEET_COLUMNS = ["key", "päätös", "riittikö abstrakti" + OPTIONAL, "tekoälyn käyttö" + OPTIONAL, "perustelu" + OPTIONAL,
                 "title", "authors", "year", "venue", "link", "abstract"]
BLANK_COLUMNS = set(SHEET_COLUMNS[1:5])  # the rater's to fill
KEY_COLUMNS = ["key", "stratum", "group", "tool outcome", "stratum size in pool"]
KEEP_WORDS = {"keep", "kyllä", "kylla", "mukaan", "yes", "k"}
DROP_WORDS = {"drop", "ei", "pois", "no", "e"}


def read_table_any(run_dir: pathlib.Path, stem: str) -> list[dict[str, str]]:
    """The table `stem` in `run_dir`, in whichever format the run wrote."""
    for ext in tabular.FORMATS:
        path = run_dir / f"{stem}.{ext}"
        if path.exists():
            return tabular.read_table(path)
    raise SystemExit(f"{run_dir}: no table '{stem}'")


def removal_group(row: dict[str, str]) -> str:
    """What decided a removal: the judge's tests that fired (a test asked the other way round fires on false), or the
    rule."""
    if row["removed by"] == "judge":
        tests = re.findall(r"(\w+) (?:true|false)", row["why"].split(" confidence")[0])
        return "judge: " + (" + ".join(tests) if tests else "?")
    return f"{row['removed by']}: {row['why'].split(' (')[0]}"


def add_rules_sheet(path: pathlib.Path, rules: pathlib.Path) -> None:
    """Put the scope rules raters decide by on a second sheet of `path`, one line of `rules` per row."""
    import openpyxl
    book = openpyxl.load_workbook(path)
    sheet = book.create_sheet(rules.stem)
    for line in rules.read_text(encoding="utf-8").splitlines():
        sheet.append([line])
    sheet.column_dimensions["A"].width = 140
    book.save(path)


def draw(run_dir: pathlib.Path, out_dir: pathlib.Path, exclude: pathlib.Path, seed: int,
         maybe_rules: pathlib.Path | None = None) -> int:
    excluded = {row["key"] for row in tabular.read_table(exclude)}
    tags = {row["key"]: row for row in read_table_any(run_dir, report.EXTRACT_AUDIT)}
    strata: dict[str, dict[str, list[str]]] = {name: collections.defaultdict(list) for name in QUOTAS}
    rows_by_key = {}
    for stem, stratum in ((report.REMOVED_AUDIT, "removed"), (report.KEPT_AUDIT, "kept"),
                          (report.CHECK_BY_HAND, "to a person")):
        for row in read_table_any(run_dir, stem):
            if row["key"] in excluded or row.get("removed by") == "sift":
                continue
            rows_by_key[row["key"]] = row
            if stratum == "removed":
                group = removal_group(row)
            elif stratum == "kept":
                tag = tags.get(row["key"], {})
                group = f"{tag.get('study_kind', '?')} / {tag.get('agent_kind', '?')}"
            else:
                group = row["why"].split(" (")[0].split(";")[0][:80]
            strata[stratum][group].append(row["key"])

    rng = random.Random(seed)
    key_rows, sheet_rows = [], []
    for stratum, quota in QUOTAS.items():
        size = sum(len(keys) for keys in strata[stratum].values())
        everyone = sorted((key, group) for group, keys in strata[stratum].items() for key in keys)
        chosen = rng.sample(everyone, min(quota, len(everyone)))
        print(f"{stratum}: {len(chosen)} drawn of {size}, from {len(strata[stratum])} groups")
        for key, group in chosen:
            key_rows.append({"key": key, "stratum": stratum, "group": group, "tool outcome": stratum,
                             "stratum size in pool": str(size)})
            row = rows_by_key[key]
            sheet_rows.append({column: ("" if column in BLANK_COLUMNS else row.get(column, "")) for column in SHEET_COLUMNS})
    rng.shuffle(sheet_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    tabular.write_table(out_dir / "blind-sample.xlsx", [], SHEET_COLUMNS,
                        [[row[column] for column in SHEET_COLUMNS] for row in sheet_rows])
    tabular.write_table(out_dir / "blind-key.tsv", [f"seed {seed}, from {run_dir}, excluding {exclude.name}"],
                        KEY_COLUMNS, [[row[column] for column in KEY_COLUMNS] for row in key_rows])
    if maybe_rules is not None:
        add_rules_sheet(out_dir / "blind-sample.xlsx", maybe_rules)
    print(f"wrote {out_dir / 'blind-sample.xlsx'} ({len(sheet_rows)} records) and {out_dir / 'blind-key.tsv'}")
    return 0


def upper_bound(k: int, n: int, level: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a rate, with `k` of `n` observed."""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta(k + 1, n - k).ppf(level))


def score(out_dir: pathlib.Path, rated: pathlib.Path) -> int:
    key = {row["key"]: row for row in tabular.read_table(out_dir / "blind-key.tsv")}
    decisions = {}
    for row in tabular.read_table(rated):
        word = row.get("päätös", row.get("decision", "")).strip().lower()
        if word in KEEP_WORDS:
            decisions[row["key"]] = "keep"
        elif word in DROP_WORDS:
            decisions[row["key"]] = "drop"
    print(f"{len(decisions)} of {len(key)} rated")
    weighted_missed = weighted_total = 0.0
    for stratum in QUOTAS:
        rows = [row for row in key.values() if row["stratum"] == stratum and row["key"] in decisions]
        if not rows:
            continue
        size = int(rows[0]["stratum size in pool"])
        keeps = sum(decisions[row["key"]] == "keep" for row in rows)
        n = len(rows)
        if stratum == "removed":
            print(f"removed:     {keeps} of {n} rated in scope; the share of removals that are wrong is at most "
                  f"{100 * upper_bound(keeps, n):.1f}% (95%, one-sided)")
        elif stratum == "kept":
            print(f"kept:        {n - keeps} of {n} rated out of scope; at most {100 * upper_bound(n - keeps, n):.1f}%")
        else:
            print(f"to a person: {keeps} of {n} rated in scope")
        # The in-scope share of each stratum, scaled to its size in the pool: an estimate of in-scope records per stratum.
        weighted_total += size * keeps / n
        if stratum == "removed":
            weighted_missed += size * keeps / n
    if weighted_total:
        print(f"\nestimated share of the pool's in-scope records that the tool removes: "
              f"{100 * weighted_missed / weighted_total:.1f}% (a point estimate; see the bound above)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p_draw = commands.add_parser("draw", help="draw the sample and write the sheet and the key")
    p_draw.add_argument("run_dir")
    p_draw.add_argument("out_dir")
    p_draw.add_argument("--exclude", required=True, help="a table whose 'key' column lists records already rated")
    p_draw.add_argument("--seed", type=int, default=17)
    p_draw.add_argument("--rules", default=None, help="a text file of the scope rules, put on a second sheet for raters")
    p_score = commands.add_parser("score", help="score a filled-in sheet against the key")
    p_score.add_argument("out_dir")
    p_score.add_argument("rated")
    opts = parser.parse_args()
    if opts.command == "draw":
        return draw(pathlib.Path(opts.run_dir), pathlib.Path(opts.out_dir), pathlib.Path(opts.exclude), opts.seed,
                    pathlib.Path(opts.rules) if opts.rules else None)
    return score(pathlib.Path(opts.out_dir), pathlib.Path(opts.rated))


if __name__ == "__main__":
    sys.exit(main())
