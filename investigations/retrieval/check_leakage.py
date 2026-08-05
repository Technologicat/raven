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

**The BibTeX sets need it too, for a reason the BM25 baseline cannot cover.** That baseline answers "are
these questions trivially findable", which is about retrieval scores. The threshold question is different
and sharper: a set whose questions are partly copies sits higher in similarity than an honest one, so the
cut read off it lands too high, and the harm shows up as real questions rejected in the field. Both need
checking, and the titles-only corpus needs it most — a question generated from a title alone has an order
of magnitude less room to paraphrase away from its source than one generated from an abstract.

Reads the corpus to recover each source, so it needs the documents directory the set was generated from.
Fiction records that directory in the set itself; the BibTeX sets are looked up by corpus name in
`make_questions.CORPORA`, and the source text is the title-plus-abstract the generator was shown.

Usage:
    python check_leakage.py [fiction|hydrogen|arxiv-ai|banichuk] [worst_n]
"""

import collections
import json
import pathlib
import re
import sys

from raven.common import docextract

import make_questions  # shared instrument: corpus profiles and record parsing are defined once

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


def fiction_rows(payload: dict) -> tuple[list[dict], str]:
    """Score the fiction set, whose source for a question is a passage at a recorded offset."""
    passage_chars = payload["passage_chars"]

    # The set records where it drew from, so the passages can be recovered without saving them into the
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
    for item in payload["questions"]:
        source = texts.get(item["source"])
        if source is None:
            continue
        offset = item["source_offset"]
        passage = source[offset:offset + passage_chars]
        length, run = longest_shared_run(item["question"], passage)
        rows.append({"length": length, "run": run, "kind": item["kind"],
                     "question": item["question"], "source": item["source"]})
    return rows, f"the {passage_chars}-character passage it was written from"


def bibtex_rows(corpus: str, payload: dict) -> tuple[list[dict], str]:
    """Score a BibTeX set, whose source for a question is the record the generator was shown.

    That is title-plus-abstract, or title alone for a titles-only corpus — reconstructed rather than
    stored, for the same reason fiction reconstructs its passages.
    """
    profile = make_questions.CORPORA[corpus]
    corpus_dir = profile["docs_dir"]
    use_abstracts = profile.get("use_abstracts", True)
    if not corpus_dir.is_dir():
        print(f"warning: corpus directory missing, cannot check: {corpus_dir}")
        return [], ""

    rows = []
    for item in payload["questions"]:
        for document_id in item["gold"]:
            entry = make_questions.load_entry(corpus_dir / document_id, require_abstract=use_abstracts)
            if entry is None:
                continue
            source = entry["title"] + " " + entry["abstract"]
            length, run = longest_shared_run(item["question"], source)
            rows.append({"length": length, "run": run, "kind": item["kind"],
                         "question": item["question"], "source": document_id})
    return rows, "its record's title" + (" and abstract" if use_abstracts else " (no abstract available)")


def main() -> None:
    corpus = sys.argv[1] if len(sys.argv) > 1 else "fiction"
    worst_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if corpus == "fiction":
        questions_path = QUESTIONS_PATH
    elif corpus in make_questions.CORPORA:
        questions_path = make_questions.CORPORA[corpus]["out_path"]
    else:
        raise SystemExit(f"unknown corpus '{corpus}'; expected 'fiction' or one of "
                         f"{', '.join(make_questions.CORPORA)}")

    if not questions_path.exists():
        print(f"no question set at {questions_path}; generate it first")
        return
    payload = json.loads(questions_path.read_text())

    rows, against = (fiction_rows(payload) if corpus == "fiction" else bibtex_rows(corpus, payload))
    if not rows:
        print("no questions could be checked (corpus directory missing?)")
        return

    histogram = collections.Counter(r["length"] for r in rows)
    print(f"corpus '{corpus}': longest shared word run, over {len(rows)} questions "
          f"(each against {against}):\n")
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
