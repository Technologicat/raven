"""Ask, of a record, whether a case can be made that it belongs in the corpus after all.

A second reader over the records `judge_scope.py` dropped — but asked a different question, not the same
one twice. Re-asking "was this verdict right?" invites agreement, and the answer is a rubber stamp bought
with an hour of compute. This asks the opposite: *make the strongest case that this record IS a study of
AI agents in higher education, and if there is no case, say so.*

Two things follow from putting it that way.

**The output is evidence rather than a verdict.** A one-line case is checkable against the abstract in
seconds, where "I confirm the drop" is not checkable at all. That matters at this scale — a corpus of
thousands cannot be re-read by a person, so the review has to hand back something a person can spot-check
rather than something they must trust.

**The reader is told nothing about where these records came from.** Not that they were judged, not that
they were dropped, not what the verdict was, and not that a model produced it. Anchoring is the obvious
reason; the specific one is that models rate their own family's output more highly, and the reviewing
model here is deliberately the same one that judged, so any hint of provenance would buy exactly the bias
this is meant to detect. It sees a record and a question.

    python review_drops.py --bib corpus.bib --dropped dropped.tsv --limit 300

**Read the negative control before the result.** Asked whether a case exists, a model can manufacture one
for anything, and a reviewer that says yes to everything is not discriminating — it just looks thorough.
So `--control N` mixes in records that were *kept*, unlabelled and shuffled among the rest. If the case
rate is the same for both groups, this instrument is measuring nothing, and that has to be visible before
any of its findings are worth reading.
"""

import argparse
import csv
import logging
import pathlib
import random
import sys

from unpythonic import timer
from unpythonic.env import env

from raven.librarian import agent, config as librarian_config, llmclient

# Kept in step with `judge_scope.SCOPE_QUESTION` by hand rather than imported, because the two scripts ask
# about the same corpus from opposite sides and a reader comparing them should see both questions written
# out. If one changes, this changes.
SCOPE_QUESTION = "studies on different aspects of the use of AI agents in higher education"

ABSTRACT_CHARS = 4000

INSTRUCTIONS = """\
You are helping assemble a literature review on: {question}.

For each numbered record below, answer one question: is there a credible case that this record belongs in \
that review?

A credible case rests on what the record actually says. It is not enough that a topic could conceivably \
touch education, or that AI could conceivably be involved - the record has to give you something to point \
at. If you find yourself reaching, there is no case, and saying so is the useful answer.

For each record, answer:
  "i"        the record's number, copied exactly
  "belongs"  true if a credible case exists, false if not
  "case"     if true, the case in at most twenty words, pointing at what the record says. If false, at \
most twenty words on what is missing.

Answer with a JSON array of objects and nothing else. One object per record, in order, no commentary, no \
markdown fences.

{items}
"""


def load_records(bib_path: pathlib.Path) -> dict[str, env]:
    """Every record of a `.bib`, keyed by citekey. Shares `judge_scope`'s reading of a record."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import judge_scope
    return {record.key: record for record in judge_scope.load_records(bib_path)}


def read_keys(tsv_path: pathlib.Path) -> list[str]:
    """The citekeys listed in a TSV, in file order, skipping the `#` header lines a run writes."""
    lines = [line for line in tsv_path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return [row["key"] for row in csv.DictReader(lines, delimiter="\t") if row.get("key")]


def format_record(index: int, record: env) -> str:
    """One record as the reviewer sees it: a number, and what the bibliography says. No provenance."""
    venue = f"Published in: {record.venue}\n" if record.venue else ""
    abstract = record.abstract[:ABSTRACT_CHARS] if record.abstract else "(none)"
    return (f"--- record {index} ---\n"
            f"Title: {record.title}\n"
            f"{venue}"
            f"Abstract: {abstract}\n")


def review_batch(llm_settings: env, batch: list[env]) -> dict[int, dict]:
    """Ask about one batch. Returns `{position: answer}`; an unresolvable index is dropped, not guessed."""
    items = "\n".join(format_record(i, record) for i, record in enumerate(batch))
    reply = agent.ask(llm_settings, INSTRUCTIONS.format(question=SCOPE_QUESTION, items=items))
    answers = agent.parse_json_reply(reply)
    if not isinstance(answers, list):
        raise ValueError(f"expected a JSON array, got {type(answers).__name__}")
    out = {}
    for answer in answers:
        if not isinstance(answer, dict) or "i" not in answer:
            continue
        try:
            i = int(answer["i"])
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(batch):
            out[i] = answer
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bib", required=True, help="the .bib the records come from")
    parser.add_argument("--dropped", required=True, help="the run's dropped.tsv, least-defended first")
    parser.add_argument("--kept", default=None,
                        help="the run's in-scope .bib, to draw the control group from")
    parser.add_argument("--limit", type=int, default=300, metavar="N",
                        help="review the first N dropped records. The list is sorted least-defended "
                             "first, so the head is where a case is most likely to exist")
    parser.add_argument("--control", type=int, default=40, metavar="N",
                        help="how many KEPT records to mix in, unlabelled. If the case rate is the same "
                             "for both groups this reviewer is not discriminating, and nothing it says "
                             "about the dropped ones means anything")
    parser.add_argument("--batch", type=int, default=10, help="records per model call")
    parser.add_argument("--seed", type=int, default=42, help="which control records, and the shuffle")
    parser.add_argument("--backend-url", default=None,
                        help=f"the LLM backend (default: {librarian_config.llm_backend_url})")
    parser.add_argument("--model", default=None, help="model id to review with")
    parser.add_argument("--out", default=None, help="where the review TSV goes")
    opts = parser.parse_args()

    logging.getLogger("bibtexparser").setLevel(logging.ERROR)
    here = pathlib.Path(__file__).resolve().parent
    records = load_records(pathlib.Path(opts.bib).expanduser().resolve())

    dropped_keys = read_keys(pathlib.Path(opts.dropped).expanduser().resolve())[:opts.limit]
    reviewed = [(key, "dropped") for key in dropped_keys if key in records]

    # The control, drawn from what the same run kept. Unlabelled and shuffled in among the rest, so the
    # reviewer cannot tell the groups apart and neither can its answers until they are scored.
    if opts.control and opts.kept:
        kept_keys = [key for key in read_keys_from_bib(pathlib.Path(opts.kept).expanduser().resolve())
                     if key in records and key not in set(dropped_keys)]
        control = random.Random(opts.seed).sample(kept_keys, min(opts.control, len(kept_keys)))
        reviewed += [(key, "kept") for key in control]
    random.Random(opts.seed).shuffle(reviewed)

    llm_settings = llmclient.setup(backend_url=opts.backend_url or librarian_config.llm_backend_url,
                                   quiet=True)
    if opts.model:
        llm_settings.request_data["model"] = opts.model
        llm_settings.model_id = opts.model
    print(f"backend: {llm_settings.model_id}")
    print(f"reviewing {sum(1 for _, g in reviewed if g == 'dropped')} dropped records "
          f"and {sum(1 for _, g in reviewed if g == 'kept')} kept ones, shuffled together")

    rows = []
    batches = [reviewed[i:i + opts.batch] for i in range(0, len(reviewed), opts.batch)]
    for n, batch in enumerate(batches, start=1):
        with timer() as tim:
            try:
                answers = review_batch(llm_settings, [records[key] for key, _ in batch])
            except Exception as exc:  # noqa: BLE001 -- one bad batch must not end the run
                print(f"  batch {n}/{len(batches)}: FAILED ({type(exc)}: {exc})", flush=True)
                continue
        for i, (key, group) in enumerate(batch):
            if i in answers:
                answer = answers[i]
                rows.append({"key": key, "group": group,
                             "belongs": "yes" if answer.get("belongs") is True else "no",
                             "case": " ".join(str(answer.get("case") or "").split()),
                             "title": records[key].title})
        print(f"  batch {n}/{len(batches)}: {len(answers)}/{len(batch)} answered in {tim.dt:.1f}s",
              flush=True)

    out_path = pathlib.Path(opts.out) if opts.out else here / "drop-review.tsv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("group\tbelongs\tkey\tcase\ttitle\n")
        for row in sorted(rows, key=lambda r: (r["belongs"] != "yes", r["group"], r["key"])):
            f.write(f"{row['group']}\t{row['belongs']}\t{row['key']}\t{row['case']}\t{row['title']}\n")

    print(f"\nwrote {out_path}")
    _report(rows)
    return 0


def read_keys_from_bib(bib_path: pathlib.Path) -> list[str]:
    """The citekeys of a `.bib`, for drawing the control group out of what a run kept."""
    from raven.papers import bibtex
    return [entry.key for entry in bibtex.parse_file(str(bib_path), split_names=False).entries]


def _report(rows: list[dict]) -> None:
    """Print the control comparison first, because it decides whether the rest means anything."""
    for group in ("kept", "dropped"):
        subset = [row for row in rows if row["group"] == group]
        if not subset:
            continue
        yes = sum(1 for row in subset if row["belongs"] == "yes")
        print(f"  {group:<8} {len(subset):>4} records, a case found for {yes:>4} "
              f"({100 * yes / len(subset):4.1f}%)")
    kept = [row for row in rows if row["group"] == "kept"]
    dropped = [row for row in rows if row["group"] == "dropped"]
    if kept and dropped:
        kept_rate = sum(1 for row in kept if row["belongs"] == "yes") / len(kept)
        dropped_rate = sum(1 for row in dropped if row["belongs"] == "yes") / len(dropped)
        if kept_rate - dropped_rate < 0.2:
            print("\n  The two rates are close. This reviewer is not telling the groups apart, so what it\n"
                  "  says about any single dropped record carries no weight — read that before the cases.")


if __name__ == "__main__":
    sys.exit(main())
