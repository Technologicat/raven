"""Does a cross-encoder reranker put the right document first?

The recall curve says there is room: on hydrogen the gold document is somewhere in the top 200 for 96%
of questions but ranked *first* for 38%. That gap is an ordering failure, and reordering is exactly what
a cross-encoder does — it reads the query and one candidate *together*, which the bi-encoder cannot,
having embedded the document long before the question existed.

The shape is standard retrieve-and-rerank: take a wide, cheap candidate set from the existing hybrid
retrieval, then spend a small expensive model only on reordering it. `k` for the candidate stage and `k`
for what reaches the LLM stopped being the same number the moment a reranker existed between them.

    python rerank.py <corpus> [--depth 100] [--device cpu|cuda] [--db-dir DIR]

Reports recall@k and MRR before and after reranking, over the same retrieval, so the comparison isolates
the reranker. **The baseline is the retrieval at the same depth**, which is a deliberate choice and not
the obvious one: fusion depth changes the fused order (a k=200 sweep scores 74.7% within its top 20
where a k=20 sweep scored 78%), so comparing against a *shallower* run would credit the reranker with
repairing damage the deeper retrieval caused. Both columns here come from one retrieval; only the
ordering differs.

Only on-corpus questions are scored. A negative has no gold document by construction, so a reranker
cannot be right or wrong about it — off-corpus detection is a separate question, parked elsewhere.

Requires a running raven-server (retrieval) and downloads the cross-encoder on first use (23M
parameters, ~90 MB).
"""

from __future__ import annotations

__all__ = [
    "MODEL_NAME",
    "DEPTHS",
    "rerank",
    "metrics",
]

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import sys
import time

from raven.client import api as client_api
from raven.client import config as client_config
from raven.librarian import config as librarian_config
from raven.librarian import hybridir

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sharpness  # noqa: E402 -- local investigation module, needs the path set above

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEPTHS = (1, 5, 10, 20)


def rerank(model, query: str, results: list[dict]) -> list[dict]:
    """Return `results` reordered by cross-encoder relevance to `query`, best first.

    Scoring is one batched call, since the pairs are independent and the model is small enough that
    per-call overhead dominates otherwise.
    """
    if not results:
        return results
    scores = model.predict([(query, result["text"]) for result in results])
    order = sorted(range(len(results)), key=lambda i: float(scores[i]), reverse=True)
    return [{**results[i], "rerank_score": float(scores[i])} for i in order]


def metrics(ranks: list[int | None], depths: tuple[int, ...] = DEPTHS) -> dict:
    """recall@k for each depth, plus MRR, over one ranking.

    MRR counts a miss as 0 rather than dropping it, so the two columns of a before/after comparison stay
    over the same denominator — a reranker that loses a document must be charged for it.
    """
    out = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / len(ranks)
           for k in depths}
    out["MRR"] = sum(1.0 / r for r in ranks if r is not None) / len(ranks)
    return out


def _format_row(label: str, values: dict, depths: tuple[int, ...]) -> str:
    cells = "  ".join(f"{values[f'recall@{k}']:>7.1%}" for k in depths)
    return f"  {label:<22} {cells}  {values['MRR']:>7.3f}"


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="hydrogen | fiction | arxiv-ai | banichuk")
    ap.add_argument("--depth", type=int, default=100,
                    help="Candidate-set size handed to the reranker (default: 100).")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (default: cuda).")
    ap.add_argument("--db-dir", type=pathlib.Path, default=librarian_config.llm_database_dir,
                    help="Index to query. Name it rather than renaming directories into the "
                         "configured slot; see the investigation README.")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("-o", "--output", type=pathlib.Path, default=None)
    args = ap.parse_args()

    items, note = sharpness.build_workload(args.corpus)
    items = [item for item in items if item["on_corpus"] and item["gold"]]
    print(f"corpus '{args.corpus}': {len(items)} on-corpus questions, candidate depth {args.depth}")
    print(f"  {note}")
    print(f"  index: {args.db_dir}")

    from sentence_transformers import CrossEncoder  # noqa: PLC0415 -- slow import, and only main needs it
    t0 = time.perf_counter()
    model = CrossEncoder(args.model, device=args.device)
    print(f"  reranker: {args.model} on {args.device}, loaded in {time.perf_counter() - t0:.1f} s")

    executor = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file,
                          executor=executor)
    hybridir.init(executor=executor)
    retriever = hybridir.HybridIR(datastore_base_dir=args.db_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)

    before: list[int | None] = []
    after: list[int | None] = []
    rerank_times: list[float] = []
    per_question = []
    for i, item in enumerate(items, start=1):
        gold = set(item["gold"])
        results, _report = retriever.query(item["query"], k=args.depth, multi_query=False,
                                           return_extra_info=True)
        rank_before = sharpness.rank_of_gold(results, gold)
        t0 = time.perf_counter()
        reranked = rerank(model, item["query"], results)
        rerank_times.append(time.perf_counter() - t0)
        rank_after = sharpness.rank_of_gold(reranked, gold)
        before.append(rank_before)
        after.append(rank_after)
        per_question.append({"kind": item["kind"], "query": item["query"], "gold": item["gold"],
                             "n_candidates": len(results),
                             "rank_before": rank_before, "rank_after": rank_after})
        arrow = "=" if rank_before == rank_after else ("^" if (rank_after or 10**9) < (rank_before or 10**9) else "v")
        print(f"  [{i}/{len(items)}] {str(rank_before):>5} -> {str(rank_after):<5} {arrow}  "
              f"{item['query'][:52]}")

    print()
    header = "  ".join(f"{'@' + str(k):>7}" for k in DEPTHS)
    print(f"  {'':<22} {header}  {'MRR':>7}")
    print(_format_row("retrieval only", metrics(before), DEPTHS))
    print(_format_row(f"+ reranked (top {args.depth})", metrics(after), DEPTHS))
    print()
    improved = sum(1 for b, a in zip(before, after)
                   if (a or 10**9) < (b or 10**9))
    worsened = sum(1 for b, a in zip(before, after)
                   if (a or 10**9) > (b or 10**9))
    print(f"  moved up: {improved}   moved down: {worsened}   unchanged: {len(items) - improved - worsened}")
    print(f"  rerank latency: median {statistics.median(rerank_times) * 1000:.0f} ms "
          f"for {args.depth} candidates on {args.device}")
    print()

    out_path = args.output or pathlib.Path(__file__).parent / f"rerank_results_{args.corpus}.json"
    out_path.write_text(json.dumps({"corpus": args.corpus, "depth": args.depth,
                                    "model": args.model, "device": args.device,
                                    "before": metrics(before), "after": metrics(after),
                                    "per_question": per_question}, indent=2), encoding="utf-8")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
