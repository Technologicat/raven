#!/usr/bin/env python3
"""Did the generator copy its source instead of asking about it?

Every generated question set carries the same hazard: a question that reuses its source's distinctive
phrasing turns retrieval into string matching, and the set then measures the matcher rather than the
retriever. Both generators forbid it in the prompt, which is an instruction rather than a guarantee.

For `questions.json` the check is indirect and already exists — `evaluate.py` reports the keyword-only
baseline, and BM25 scoring near-perfect means the questions are too close to their abstracts. **The fiction
set cannot use that check.** With 19 documents and k=20, finding the gold document is nearly free for any
retriever, so the BM25 baseline is uninformative there and the hazard would go unmeasured.

So measure the overlap directly: the longest run of consecutive words a question shares with the passage it
was written from. A question that asks about a passage reuses names and a technical term or two — runs of
two or three. A question that copies from it reuses a clause.

**Why this matters more than it looks.** Leakage does not merely flatter the retrieval scores. The threshold
being chosen in `sharpness.py` is read off *where the on-corpus similarity distribution sits*, so questions
that are partly copies of their sources shift that distribution upward and make the corpus look easier to
separate from off-corpus queries than it is. The number that comes out is then wrong in the unsafe
direction: a cut placed too high, rejecting real questions in the field.

Reads the corpus to recover each passage, so it needs the documents directory the set was generated from —
which is recorded in the set itself.

Usage:
    python check_leakage.py [worst_n]
"""

import collections
import json
import pathlib
import re
import sys

from raven.common import docextract

HERE = pathlib.Path(__file__).parent
QUESTIONS_PATH = HERE / "fiction_questions.json"

# Runs at or above this length read as copied phrasing rather than shared vocabulary. Not a tuned value —
# it is where "a noun phrase someone would naturally repeat" stops and "a clause" starts. The distribution
# is what to read; this only decides which rows get printed as suspects.
SUSPECT_RUN = 6

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def words(text: str) -> list[str]:
    """Case-folded word tokens, punctuation and digits dropped — what "shared phrasing" should mean here."""
    return [w.casefold() for w in _WORD.findall(text)]


def longest_shared_run(question: str, source: str) -> tuple[int, str]:
    """Length of the longest run of consecutive words shared with `source`, and the run itself.

    Compares against the source's word *n-grams* rather than its raw text, so that punctuation and casing
    differences do not hide a copied clause.
    """
    q, s = words(question), words(source)
    if not q or not s:
        return 0, ""
    # Longest first, returning on the first hit: the answer is the length, so there is nothing to gain by
    # continuing downward once one is found.
    for length in range(min(len(q), len(s)), 0, -1):
        source_ngrams = {tuple(s[i:i + length]) for i in range(len(s) - length + 1)}
        for i in range(len(q) - length + 1):
            if tuple(q[i:i + length]) in source_ngrams:
                return length, " ".join(q[i:i + length])
    return 0, ""


def main() -> None:
    worst_n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if not QUESTIONS_PATH.exists():
        print(f"no question set at {QUESTIONS_PATH}; run make_fiction_questions.py first")
        return
    payload = json.loads(QUESTIONS_PATH.read_text())
    questions = payload["questions"]
    passage_chars = payload["passage_chars"]

    # Each set records where it drew from, so the passages can be recovered without saving them into the
    # committed file — which would put third-party prose in the repository.
    corpus_dirs = [pathlib.Path(payload["corpus_dir"]).expanduser()]
    if payload.get("held_out_dir"):
        corpus_dirs.append(pathlib.Path(payload["held_out_dir"]).expanduser())

    texts: dict[str, str] = {}
    for directory in corpus_dirs:
        if not directory.is_dir():
            print(f"warning: corpus directory missing, its questions cannot be checked: {directory}")
            continue
        for path in directory.iterdir():
            if path.is_file() and docextract.is_supported(path):
                try:
                    texts[path.name] = docextract.extract_text(path) or ""
                except Exception as exc:  # noqa: BLE001 -- an unreadable file is data, not a crash
                    print(f"  skipping '{path.name}': {type(exc).__name__}: {exc}")

    rows = []
    for item in questions:
        source = texts.get(item["source"])
        if source is None:
            continue
        offset = item["source_offset"]
        passage = source[offset:offset + passage_chars]
        length, run = longest_shared_run(item["question"], passage)
        rows.append({"length": length, "run": run, "kind": item["kind"],
                     "question": item["question"], "source": item["source"]})

    if not rows:
        print("no questions could be checked (corpus directories missing?)")
        return

    histogram = collections.Counter(r["length"] for r in rows)
    print(f"longest shared word run, over {len(rows)} questions "
          f"(each against the {passage_chars}-character passage it was written from):\n")
    for length in sorted(histogram):
        marker = "  <-- copied phrasing" if length >= SUSPECT_RUN else ""
        print(f"  {length:>2} words: {'#' * histogram[length]:<40} {histogram[length]:>3}{marker}")

    suspects = sorted((r for r in rows if r["length"] >= SUSPECT_RUN),
                      key=lambda r: -r["length"])
    share = len(suspects) / len(rows)
    print(f"\n{len(suspects)} of {len(rows)} ({share:.0%}) share a run of {SUSPECT_RUN}+ words.")
    if share > 0.1:
        print("That is high enough to distrust the set: the questions are partly copies of their sources,\n"
              "which inflates on-corpus similarity and biases any threshold read off it upward.")
    else:
        print("Low enough that the questions are asking about their sources rather than quoting them.")

    if suspects:
        print(f"\nworst {min(worst_n, len(suspects))}:")
        for r in suspects[:worst_n]:
            print(f"  [{r['length']:>2}] {r['run'][:76]}")
            print(f"       {r['kind']:<9} {r['question'][:76]}")


if __name__ == "__main__":
    main()
