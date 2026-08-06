"""Does per-query score sharpness predict *which retrieval arm* found the gold document?

The arm-selection oracle is wide — picking the better arm per query is worth +7 to +12 points of
recall@20 over fixed RRF fusion, more than any reranker offered. This asks whether that choice is
predictable from something we already compute.

The signal under test is the *difference in sharpness between the two arms on the same query*:

    signal = score_sharpness(bm25 candidates) - score_sharpness(vector candidates)

**A difference, deliberately.** The off-corpus work died because it needed an absolute threshold on a
similarity level, and levels do not transfer between corpora. Two sharpness values measured on the same
query, from the same candidate depth, are commensurable without any cross-corpus calibration — so the
question here is whether the *ordering* carries information, which is a strictly weaker thing to ask.

**This measures prediction quality, not a rule.** Output is AUROC of the signal against the label "the
BM25 arm ranked the gold document better than the vector arm did". 0.5 is no information and kills the
idea outright; anything meaningfully above leaves room for a rule, which is a separate question with its
own operating point. Asking this first is what the threshold work should have done.

**Ties are dropped, and there are many** — a third of queries on some corpora rank the gold identically
in both arms, so the label is undefined for them. Including them as either class would move AUROC
without meaning anything.

    python arm_signal.py <corpus> [--db-dir DIR]
"""

from __future__ import annotations

__all__ = ["RATIOS", "arm_signals", "auroc"]

import concurrent.futures
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sharpness  # noqa: E402

from raven.client import api as client_api  # noqa: E402
from raven.client import config as client_config  # noqa: E402
from raven.librarian import config as librarian_config  # noqa: E402
from raven.librarian import hybridir  # noqa: E402

# `min_ratio` values for `score_sharpness`. Swept rather than chosen, because the right one is not known
# and the cost of trying several is one list comprehension per query.
RATIOS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

DEPTH = 100

auroc = sharpness.auroc


def arm_signals(report, ratio: float) -> tuple[float, float]:
    """Sharpness of each arm's full candidate list for one query, as `(bm25, vector)`.

    Reads `report.per_query[0]`, not the top level: the top-level `keyword_scores` are the survivors of
    the score threshold, and `score_sharpness` needs the *full* candidate list — a candidate the
    threshold rejected is precisely one the best result left behind, and dropping it changes the
    denominator query by query. `per_query` is the pre-threshold record. Index 0 because this runs with
    `multi_query=False`, so there is exactly one subquery.

    Vector distances are converted to similarities first: `score_sharpness` requires bigger-is-better,
    and passing a cosine *distance* would measure the worst match instead of the best.
    """
    sub = report.per_query[0]
    kw = [float(s) for s in sub.candidate_keyword_scores]
    vec = [1.0 - float(d) for d in sub.candidate_vector_distances]
    return (hybridir.score_sharpness(kw, min_ratio=ratio) if kw else 0.0,
            hybridir.score_sharpness(vec, min_ratio=ratio) if vec else 0.0)


def standardized_top(scores: list[float]) -> float:
    """How far the best score stands above its own candidate distribution, in standard deviations.

    The comparable-across-arms statistic that `score_sharpness` is not. Sharpness counts candidates
    scoring at least `min_ratio` times the best, which presumes a score whose zero means "no match".
    That holds for BM25 and fails for cosine similarity, where an unrelated document still scores
    0.2–0.4 — so at a low `min_ratio` every vector candidate "keeps up" and the sharpness is ~0 whatever
    the arm found. Comparing the two arms' sharpness would then measure the scoring convention rather
    than the retrieval.

    A z-score of the top result within its own candidate list is dimensionless and location- and
    scale-invariant, so BM25 and cosine values are on the same footing by construction.
    """
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    if var <= 0.0:
        return 0.0
    return (max(scores) - mean) / (var ** 0.5)


def arm_z_signals(report) -> tuple[float, float]:
    """`standardized_top` of each arm's full candidate list, as `(bm25, vector)`."""
    sub = report.per_query[0]
    kw = [float(s) for s in sub.candidate_keyword_scores]
    vec = [1.0 - float(d) for d in sub.candidate_vector_distances]
    return standardized_top(kw), standardized_top(vec)


def main() -> None:  # pragma: no cover
    argv = sys.argv[1:]
    db_dir = None
    if "--db-dir" in argv:
        at = argv.index("--db-dir")
        db_dir = pathlib.Path(argv[at + 1]).expanduser()
        del argv[at:at + 2]
    corpus = argv[0] if argv else "hydrogen"
    if db_dir is None:
        db_dir = pathlib.Path.home() / f".config/raven/llmclient/rag_index_{corpus}"

    items, note = sharpness.build_workload(corpus)
    items = [i for i in items if i["on_corpus"] and i["gold"]]
    print(f"corpus '{corpus}': {len(items)} on-corpus questions, depth {DEPTH}")
    print(f"  index: {db_dir}")

    ex = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file, executor=ex)
    hybridir.init(executor=ex)
    retriever = hybridir.HybridIR(datastore_base_dir=db_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)

    def rank_in(seq, gold):
        seen = []
        for x in seq:
            did = x.get("document_id") if isinstance(x, dict) else None
            if did is not None and did not in seen:
                seen.append(did)
        for i, did in enumerate(seen, 1):
            if did in gold:
                return i
        return None

    rows = []
    for n, item in enumerate(items, 1):
        gold = set(item["gold"])
        _merged, rep = retriever.query(item["query"], k=DEPTH, multi_query=False,
                                       return_extra_info=True)
        kw_rank = rank_in(rep.keyword_results, gold)
        vec_rank = rank_in(rep.vector_results, gold)
        row = {"kind": item["kind"], "kw_rank": kw_rank, "vec_rank": vec_rank,
               "signals": {str(r): arm_signals(rep, r) for r in RATIOS},
               "z": arm_z_signals(rep)}
        rows.append(row)
        if n % 25 == 0:
            print(f"  [{n}/{len(items)}]", flush=True)

    MISS = 10 ** 9

    def better(row):
        """True if BM25 won, False if vector won, None on a tie (label undefined)."""
        a = row["kw_rank"] or MISS
        b = row["vec_rank"] or MISS
        return None if a == b else a < b

    labelled = [(row, better(row)) for row in rows]
    decided = [(row, lab) for row, lab in labelled if lab is not None]
    n_tie = len(labelled) - len(decided)
    n_kw = sum(1 for _row, lab in decided if lab)
    print()
    print(f"  decided queries: {len(decided)}   (bm25 won {n_kw}, vector won {len(decided) - n_kw}); "
          f"ties dropped: {n_tie}")
    if len(decided) < 20 or n_kw in (0, len(decided)):
        print("  too few decided queries, or one class empty — AUROC would not mean anything.")
        return

    print()
    print(f"  {'min_ratio':>10}  {'AUROC':>7}   (0.5 = the signal predicts nothing)")
    best = None
    for r in RATIOS:
        values = [row["signals"][str(r)][0] - row["signals"][str(r)][1] for row, _lab in decided]
        labels = [lab for _row, lab in decided]
        a = auroc(values, labels)
        print(f"  {r:>10}  {a:>7.3f}")
        if best is None or abs(a - 0.5) > abs(best[1] - 0.5):
            best = (r, a)
    print()
    print(f"  best separation at min_ratio={best[0]}: AUROC {best[1]:.3f}")

    labels = [lab for _row, lab in decided]
    z_values = [row["z"][0] - row["z"][1] for row, _lab in decided]
    print(f"  standardized-top difference:      AUROC {auroc(z_values, labels):.3f}   "
          f"(scale-free, so the two arms are comparable by construction)")

    out = pathlib.Path(__file__).parent / f"arm_signal_{corpus}.json"
    out.write_text(json.dumps({"corpus": corpus, "depth": DEPTH, "ratios": list(RATIOS),
                               "rows": rows}, indent=1), encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
