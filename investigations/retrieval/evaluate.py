#!/usr/bin/env python3
"""Score `HybridIR` on the known-item set built by `make_questions.py`.

Answers the question brief 09 is built on: **does the hybrid rank track how good a result is, and
where does it stop tracking?** Every lever in that brief is a hypothesis about a different part of
the answer, and none of them is decidable by argument.

Reports, per question shape (`focused` / `rambling`) and overall:

    recall@k   fraction of questions whose gold document appears in the top k
    MRR        mean reciprocal rank of the gold document (0 if absent)
    rank hist  where the gold document actually lands

and runs the two engines *separately* alongside the fusion, which is the diagnostic that matters:
if BM25 alone matches or beats the fusion, RRF is losing information rather than adding it - and if
BM25 alone is near-perfect, the generated questions are too close to their source abstracts and the
set needs regenerating before any conclusion is drawn from it (see `make_questions.py`).

Requires a running raven-server (spaCy tokenization + embeddings) and the local document index. It
reads the index; it does not write to it.

Usage:
    python evaluate.py [k]
"""

import collections
import concurrent.futures
import json
import pathlib
import sys

from raven.client import api as client_api
from raven.client import config as client_config
from raven.librarian import config as librarian_config
from raven.librarian import hybridir

QUESTIONS_PATH = pathlib.Path(__file__).parent / "questions.json"

# Written on every run. A scoring run costs minutes of GPU time and its output is otherwise a terminal
# scrollback, which is not a place results live - so the per-question ranks are persisted here, and a later
# configuration can be compared against this one without re-running the baseline.
RESULTS_PATH = pathlib.Path(__file__).parent / "results.json"


def rank_of_gold(results: list[dict], gold: set[str]) -> int | None:
    """1-based rank of the first result whose document is in `gold`, or `None` if absent."""
    for rank, result in enumerate(results, start=1):
        if result["document_id"] in gold:
            return rank
    return None


def summarize(label: str, ranks: list[int | None], k: int) -> dict:
    found = [r for r in ranks if r is not None]
    n = len(ranks)
    return {"label": label,
            "n": n,
            "recall@k": len(found) / n if n else 0.0,
            "recall@1": sum(1 for r in found if r == 1) / n if n else 0.0,
            "recall@5": sum(1 for r in found if r <= 5) / n if n else 0.0,
            "mrr": sum(1.0 / r for r in found) / n if n else 0.0,
            "ranks": sorted(found)}


def report(rows: list[dict], k: int) -> None:
    print(f"\n{'condition':<44} {'n':>4} {'R@1':>7} {'R@5':>7} {f'R@{k}':>7} {'MRR':>7}")
    print("-" * 82)
    for row in rows:
        print(f"{row['label']:<44} {row['n']:>4} {row['recall@1']:>7.2f} {row['recall@5']:>7.2f} "
              f"{row['recall@k']:>7.2f} {row['mrr']:>7.3f}")


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    if not QUESTIONS_PATH.exists():
        print(f"no question set at {QUESTIONS_PATH}; run make_questions.py first")
        return
    payload = json.loads(QUESTIONS_PATH.read_text())
    questions = payload["questions"]
    print(f"{len(questions)} questions against a {payload['corpus_size']}-record corpus, k={k}")

    executor = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file,
                          executor=executor)
    hybridir.init(executor=executor)
    retriever = hybridir.HybridIR(datastore_base_dir=librarian_config.llm_database_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)

    # Per-question ranks under each condition. The two single-engine conditions are obtained by
    # starving the other one with an impossible threshold, so that no code path differs between
    # them and the fusion - the comparison is only meaningful if everything else is identical.
    # `multi_query` is stated explicitly on every condition rather than left to the default, so that the
    # baseline rows keep measuring the same thing when the default changes under them. The whole-message-only
    # rows are what the 2026-07-28 baseline in the README recorded.
    conditions = {"whole message only (the 2026-07-28 baseline)": {"multi_query": False},
                  "keyword only (BM25)": {"multi_query": False, "semantic_distance_threshold": -1.0},
                  "semantic only (vector)": {"multi_query": False, "keyword_score_threshold": 1e9},
                  "whole message + subqueries (lever 3)": {"multi_query": True}}

    # What the per-question progress line and the rank histogram below describe.
    MAIN_CONDITION = "whole message + subqueries (lever 3)"

    ranks: dict[str, dict[str, list]] = {name: collections.defaultdict(list) for name in conditions}
    for i, item in enumerate(questions, start=1):
        gold = set(item["gold"])
        for name, overrides in conditions.items():
            results = retriever.query(item["question"], k=k, return_extra_info=False, **overrides)
            rank = rank_of_gold(results, gold)
            ranks[name][item["kind"]].append(rank)
            ranks[name]["all"].append(rank)
        shown = ranks[MAIN_CONDITION]["all"][-1]
        print(f"  [{i}/{len(questions)}] {item['kind']:<9} gold rank {shown if shown else '-':<4} "
              f"{item['question'][:70]}")

    for kind in ("all", "focused", "rambling"):
        rows = [summarize(name, ranks[name][kind], k) for name in conditions if ranks[name][kind]]
        if not rows or not rows[0]["n"]:
            continue
        print(f"\n=== {kind} ===")
        report(rows, k)

    hist = collections.Counter(r for r in ranks[MAIN_CONDITION]["all"] if r is not None)
    missing = sum(1 for r in ranks[MAIN_CONDITION]["all"] if r is None)
    print(f"\ngold-document rank histogram (hybrid): "
          f"{dict(sorted(hist.items()))}, absent from top-{k}: {missing}")

    # Per-question ranks, not just the summary: a later configuration is compared against this run
    # question by question, and an aggregate cannot say *which* questions moved.
    RESULTS_PATH.write_text(json.dumps(
        {"k": k,
         "n_questions": len(questions),
         "corpus_size": payload["corpus_size"],
         "generator_model": payload.get("generator_model"),
         "summary": {kind: [summarize(name, ranks[name][kind], k) for name in conditions if ranks[name][kind]]
                     for kind in ("all", "focused", "rambling")},
         "per_question": [{"kind": item["kind"],
                           "question": item["question"],
                           "gold": item["gold"],
                           "rank": {name: ranks[name]["all"][i] for name in conditions}}
                          for i, item in enumerate(questions)]},
        indent=2, ensure_ascii=False) + "\n")
    print(f"wrote per-question results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
