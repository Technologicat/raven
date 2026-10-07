"""Did escalating the title-only drops rescue the records the reviewer had flagged?

Two instruments, asked opposite questions, neither seeing the other's answer:

- `review_drops.py` asked *make the strongest case that this record belongs*, and made one for 29 of the
  749 records dropped confidently from their titles.
- pass 2 re-asks the judge's own rubric with the abstract in hand, and either keeps or re-drops each.

So the flagged records give the escalation something to be measured against. If pass 2 keeps the flagged
ones at the same rate as the rest, the review's flags said nothing about what a second look would find —
which would mean one of the two is not working, and the pair cannot say which. A large gap is the two
agreeing from opposite directions, which neither could establish alone.

Three inputs, all beside this script:

- `judged.jsonl`, the judge's state file, read directly because the answers this compares are the
  *superseded* lines: a key's title-sourced answer and its later abstract-sourced one both live there, in
  order.
- `dropped-before-escalating-titles.tsv`, the drop list as it stood before this escalation, which is what
  says which records it escalated: every row there whose source was the title.
- `contested.tsv`, the reviewer's flags, as written by `score_review.py --contested`.
"""

import argparse
import collections
import json
import pathlib
import sys

from raven.common import tabular

HERE = pathlib.Path(__file__).resolve().parent


def verdict_of(answer: dict) -> str:
    halves = (answer["no_ai"], answer["not_education"], answer["wrong_level"])
    if True in halves:
        return "drop"
    return "unknown" if None in halves else "keep"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rescues", action="store_true",
                        help="also list the records pass 2 kept that the reviewer could make no case "
                             "for — two instruments with opposite biases disagreeing, which is where a "
                             "spurious rescue hides")
    opts = parser.parse_args()

    # Every line, in order, so a key's title answer and its later abstract answer are both visible.
    by_key = collections.defaultdict(list)
    for line in (HERE / "judged.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            answer = json.loads(line)
            by_key[answer["key"]].append(answer)

    # Which records this run escalated is defined by the drop list as it stood before it — every row there
    # whose source was "title". Reconstructing it from the state file instead does not work: a record can
    # be a high-confidence title drop and still have been escalated earlier, by the thin-title rule, and
    # such a record's rescue belongs to the original run rather than to this one. Counting those here
    # inflates the unflagged group with rescues that already existed, which is invisible except that the
    # record is missing from the old drop list.
    before_rows = {}
    for row in tabular.read_table(HERE / "dropped-before-escalating-titles.tsv"):
        before_rows[row["key"]] = {"source": row["source"], "why": row["why"], "title": row["title"]}
    todays = {key for key, row in before_rows.items() if row["source"] == "title"}

    escalated = {}
    for key in todays:
        last_abstract = next((a for a in reversed(by_key.get(key, [])) if a["source"] == "abstract"), None)
        if last_abstract:
            escalated[key] = (before_rows[key], last_abstract)

    # No flags would put every escalated record in "not flagged" and still print a comparison, which reads
    # as a result. So both ways of getting none — no file, or a file whose rows do not match — stop here.
    # In whichever format it was written, or saved in after marking.
    candidates = [HERE / f"contested.{fmt}" for fmt in tabular.FORMATS]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        print(f"no contested.tsv, .xlsx or .ods in {HERE}; write it first with `score_review.py --contested`",
              file=sys.stderr)
        return 1
    contested = set()
    for row in tabular.read_table(path):
        if row.get("cell") == "drop/high/title":
            contested.add(row["key"])
    if not contested:
        print(f"{path} has no rows in cell drop/high/title, so there are no flags to compare against",
              file=sys.stderr)
        return 1

    print(f"{len(escalated)} of {len(todays)} title-only drops re-judged from the abstract so far\n")

    for label, keys in (("flagged by the reviewer", contested & set(escalated)),
                        ("not flagged", set(escalated) - contested)):
        if not keys:
            continue
        outcomes = collections.Counter(verdict_of(escalated[key][1]) for key in keys)
        rescued = outcomes["keep"] + outcomes["unknown"]
        print(f"{label:<26}{len(keys):>5} records   "
              f"no longer dropped: {rescued:>4} ({100 * rescued / len(keys):5.1f}%)   "
              f"{dict(outcomes)}")

    # The place a spurious rescue hides: pass 2 kept it, and the reviewer — straining to find any reason
    # to keep it — could not. Two instruments with opposite biases disagreeing, so one of them is wrong,
    # and these are the records to read to find out which.
    if opts.rescues:
        unflagged_rescues = [key for key in set(escalated) - contested
                             if verdict_of(escalated[key][1]) != "drop"]
        print(f"\n{len(unflagged_rescues)} kept by pass 2 that the reviewer could make no case for:\n")
        for key in sorted(unflagged_rescues):
            before, after = escalated[key]
            print(f"{key}   ({verdict_of(after)})")
            print(f"  title      : {before['title'][:110]}")
            print(f"  from title : {before['why']}")
            print(f"  from abstr : {after['why']}\n")

    flagged = contested & set(escalated)
    rest = set(escalated) - contested
    if flagged and rest:
        def rescue_rate(keys):
            return sum(1 for k in keys if verdict_of(escalated[k][1]) != "drop") / len(keys)
        gap = rescue_rate(flagged) - rescue_rate(rest)
        print(f"\ngap: {100 * gap:.1f} points")
        if gap < 0.2:
            print("  The two rates are close: the review's flags did not predict what pass 2 would do.\n"
                  "  That is one instrument failing, and this comparison cannot say which.")
        else:
            print("  The reviewer's flags predicted the second look. Two instruments asked opposite\n"
                  "  questions, agreeing on which records the title-only verdict got wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
