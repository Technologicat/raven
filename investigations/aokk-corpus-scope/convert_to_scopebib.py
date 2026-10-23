"""Convert this prototype's saved answers into `raven-scopebib`'s own ledgers, once, and check the conversion.

    python investigations/aokk-corpus-scope/convert_to_scopebib.py --bib $B

writes, beside this script (or into `--out-dir`), the ledgers the tool reads:

    judge-titles-<fp>.jsonl, judge-abstracts-<fp>.jsonl   from judged.jsonl
    extract-<fp>.jsonl                                    from extracted-446e1705.jsonl
    review-<fp>.jsonl                                     from drop-review-*.tsv

and prints the three fingerprints, which name the instruments to the tool. After this the tool needs no
importer: the prototype's answers are ledgers like any other. Each ledger opens with a header naming the
model that made the answers (`HEADER`), so the reports name it too.

**Each set of answers is stamped with the prototype's own instrument, never the tool's**: a hash of the
script that made them, as it stood at its last commit before the answers were last written (`SOURCES`).
So a report says which judge decided, and a run under the tool's instrument still asks everything afresh.
That stamp names the instrument's final state. The judge's prompt was fixed partway through its run, and
`judged-before-prompt-fix.jsonl` holds the answers from before, so not every answer in `judged.jsonl` was
made by exactly that text.

Three translations, each the prototype's spelling of something the tool spells differently:

  - a JSON `true`/`false` becomes `"yes"`/`"no"` where the field's vocabulary has those;
  - any value outside a vocabulary becomes `not_stated`, the original kept in `raw` (the prototype's
    `"unclear"`);
  - the extractor's single `"evidence"` becomes the quote of the definition's one quoted field, `level`.

**The conversion checks itself before it reports success**, against the prototype's own reading of its
answers, record by record: the judge's tests (which are true, which unanswered), the extractor's rules
against `filter_keeps.py`'s tiers, and the reviewer's `belongs`. A failed check names the first record that
disagrees and writes nothing it has not already written; rerun after fixing, into a fresh `--out-dir`.
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

from raven.common import tabular
from raven.papers.scopebib import definition as scope_definition
from raven.papers.scopebib import extract, judge, records, review, rules

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import filter_keeps  # noqa: E402 -- the path has to be set up first; this is the sibling apparatus

HERE = pathlib.Path(__file__).resolve().parent

#: Which script made each set of answers, and its last commit before they were last written.
SOURCES = {"judge": ("judge_scope.py", "f114ea0b"),
           "extract": ("extract_fields.py", "8bf1f408"),
           "review": ("review_drops.py", "d2b855e4")}

TESTS = ("no_ai", "not_education", "wrong_level")

#: What every set of answers was made with, as a sweep's ledger header records it, so that a report names
#: the model. The prototype's own files do not record it; the maintainer did (2026-10-23), for all three.
HEADER = {"backend_flavor": "lmstudio", "model_requested": "qwen3.6-35b-a3b", "model_id": "qwen3.6-35b-a3b",
          "label": "Qwen 3.6 35B-A3B", "quantization": "IQ4_NL_XL", "bits_per_weight": None, "load_config": None,
          "not_reported": ["bits_per_weight", "load_config"], "declared": "KV cache 4-bit",
          "recorded": "at conversion, from the maintainer's account; the prototype's files do not record it"}


def fingerprint(stage: str) -> str:
    """The instrument of a stage's answers: eight hex digits of the source of the script that made them."""
    script, commit = SOURCES[stage]
    path = (HERE / script).relative_to(HERE.parents[1])
    source = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=HERE.parents[1], check=True,
                            capture_output=True, text=True).stdout
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]


def lines(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"conversion check failed: {message}")


def convert_judge(definition, out_dir: pathlib.Path) -> str:
    fp = fingerprint("judge")
    titles, abstracts = judge.ledgers(out_dir, fp)
    titles.write_header(HEADER)
    abstracts.write_header(HEADER)
    latest = {}
    for entry in lines(HERE / "judged.jsonl"):
        latest[(entry["key"], entry["source"])] = entry
    for (key, source), entry in latest.items():
        (titles if source == "title" else abstracts).write_answer(key, judge.normalize(entry, definition, source))
    # The check: the same tests true and the same unanswered, for every answer of both passes.
    for (key, source), entry in latest.items():
        written = (titles if source == "title" else abstracts).answers()[key]["tests"]
        check(written == {name: entry[name] if entry[name] in (True, False) else None for name in TESTS},
              f"{key} ({source}): tests {written}, the prototype's {[entry[name] for name in TESTS]}")
    print(f"judge     {fp}  {len(titles.answers())} title answers, {len(abstracts.answers())} abstract answers")
    return fp


def convert_extract(definition, corpus: dict[str, records.Record], out_dir: pathlib.Path) -> str:
    fp = fingerprint("extract")
    ledger = extract.ledger(out_dir, fp)
    ledger.write_header(HEADER)
    entries = [entry for entry in lines(HERE / "extracted-446e1705.jsonl") if entry["key"] in corpus]
    for entry in entries:
        answer = {}
        for field in definition.fields:
            value = entry[field.name]
            if isinstance(value, bool) and field.values is not None and {"yes", "no"} <= set(field.values):
                value = "yes" if value else "no"
            answer[field.name] = value
        answer["level_quote"] = entry.get("evidence", "")
        ledger.write_answer(entry["key"], extract.normalize(answer, definition, corpus[entry["key"]]))
    # The check: every rule decision is the prototype's tier, by action and by label.
    actions = {filter_keeps.TIER_A: "remove", filter_keeps.TIER_B1: "remove", filter_keeps.TIER_B2: "hold",
               filter_keeps.TIER_C: "hold", None: rules.KEEP}
    written = ledger.answers()
    for entry in entries:
        tier = filter_keeps.tier_of(entry)
        decision = rules.decide(written[entry["key"]], definition)
        check((decision.action, decision.label) == (actions[tier], tier),
              f"{entry['key']}: rules say {decision.action} ({decision.label}), the prototype's tier {tier!r}")
    print(f"extract   {fp}  {len(written)} extractions")
    return fp


def convert_review(out_dir: pathlib.Path) -> str:
    fp = fingerprint("review")
    ledger = review.ledger(out_dir, fp)
    ledger.write_header(HEADER)
    rows = [row for path in sorted(HERE.glob("drop-review-*.tsv")) for row in tabular.read_table(path)]
    for row in rows:
        belongs = {"yes": True, "no": False}.get(row["belongs"].strip().lower())
        ledger.write_answer(row["key"], {"belongs": belongs, "case": " ".join(row["case"].split())[:300]})
    written = ledger.answers()
    for row in rows:
        check(written[row["key"]]["belongs"] == {"yes": True, "no": False}.get(row["belongs"].strip().lower()),
              f"{row['key']}: belongs {written[row['key']]['belongs']}, the table's {row['belongs']!r}")
    print(f"review    {fp}  {len(written)} reviews, controls included")
    return fp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bib", required=True, help="the corpus the prototype judged, for checking quotes")
    parser.add_argument("--out-dir", default=None, help="where the ledgers go (default: beside this)")
    opts = parser.parse_args()
    out_dir = pathlib.Path(opts.out_dir) if opts.out_dir else HERE
    definition = scope_definition.load_definition(HERE / "aokk.scope.toml")
    corpus = {record.key: record for record in records.load_records(opts.bib)}
    convert_judge(definition, out_dir)
    convert_extract(definition, corpus, out_dir)
    convert_review(out_dir)
    print(f"\nall three checks passed; ledgers in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
