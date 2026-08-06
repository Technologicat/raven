"""Rerank one retrieval arm, then fuse — does keeping RRF's evidence diversity save the reranker?

Five conditions per question, all from one retrieval so nothing but the ordering differs:

  bm25            the keyword arm alone
  vector          the semantic arm alone
  fused           RRF over both — the shipped configuration
  rerank-bm25     RRF(reranked keyword arm, vector arm)     <- the classic BM25 -> cross-encoder shape
  rerank-vector   RRF(keyword arm, reranked vector arm)
  rerank-fused    reranking the fused list (measured before; included for comparison)
"""

import concurrent.futures
import json
import pathlib
import sys

sys.path.insert(0, "investigations/retrieval")
import sharpness  # noqa: E402

from raven.client import api as client_api  # noqa: E402
from raven.client import config as client_config  # noqa: E402
from raven.librarian import config as librarian_config  # noqa: E402
from raven.librarian import hybridir  # noqa: E402

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "hydrogen"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "cross-encoder/ms-marco-MiniLM-L6-v2"
DB_DIR = sys.argv[3] if len(sys.argv) > 3 else None
DEPTH = 100

# Where the gold document ends up for a *typical* question, which MRR does not tell you: MRR is a mean of
# reciprocal ranks, so a question already at rank 1 contributes 1.0 and one at rank 50 contributes 0.02.
# It is therefore dominated by the questions that already worked, and nearly blind to the ones we would
# most like to fix. `median rank` and `p75 rank` say what happens in the middle and in the tail; misses
# are held at `DEPTH + 1` so a condition cannot improve its median by losing documents entirely.
MISS_RANK = DEPTH + 1


def dedup_ids(seq):
    out = []
    for x in seq:
        did = x.get("document_id") if isinstance(x, dict) else None
        if did is not None and did not in out:
            out.append(did)
    return out


def rank_of(ids, gold):
    for i, did in enumerate(ids, 1):
        if did in gold:
            return i
    return None


def rrf(*lists, K=60):
    scores = {}
    for lst in lists:
        for rank, did in enumerate(lst, 1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (K + rank)
    return [did for did, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def main():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(MODEL, device="cuda")

    ex = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file, executor=ex)
    hybridir.init(executor=ex)
    db_dir = (pathlib.Path(DB_DIR).expanduser() if DB_DIR
              else pathlib.Path.home() / f".config/raven/llmclient/rag_index_{CORPUS}")
    r = hybridir.HybridIR(datastore_base_dir=db_dir,
                          embedding_model_name=librarian_config.qa_embedding_model)

    items = [i for i in sharpness.build_workload(CORPUS)[0] if i["on_corpus"] and i["gold"]]
    conditions = ["bm25", "vector", "fused", "rerank-bm25", "rerank-vector", "rerank-fused"]
    ranks = {c: [] for c in conditions}

    def rerank_ids(query, entries):
        """Score `entries` (chunk dicts) with the cross-encoder; return deduped document ids, best first."""
        if not entries:
            return []
        scores = model.predict([(query, e.get("text", "")) for e in entries])
        order = sorted(range(len(entries)), key=lambda i: float(scores[i]), reverse=True)
        return dedup_ids([entries[i] for i in order])

    for n, item in enumerate(items, 1):
        gold = set(item["gold"])
        merged, rep = r.query(item["query"], k=DEPTH, multi_query=False, return_extra_info=True)
        kw, vec = rep.keyword_results, rep.vector_results
        kw_ids, vec_ids = dedup_ids(kw), dedup_ids(vec)
        fused_ids = dedup_ids(merged)

        kw_rr = rerank_ids(item["query"], kw)
        vec_rr = rerank_ids(item["query"], vec)
        fused_rr = rerank_ids(item["query"], merged)

        ranks["bm25"].append(rank_of(kw_ids, gold))
        ranks["vector"].append(rank_of(vec_ids, gold))
        ranks["fused"].append(rank_of(fused_ids, gold))
        ranks["rerank-bm25"].append(rank_of(rrf(kw_rr, vec_ids), gold))
        ranks["rerank-vector"].append(rank_of(rrf(kw_ids, vec_rr), gold))
        ranks["rerank-fused"].append(rank_of(fused_rr, gold))
        if n % 20 == 0:
            print(f"  [{n}/{len(items)}]", flush=True)

    def mrr(rs):
        return sum(1.0 / x for x in rs if x) / len(rs)

    def rec(rs, k):
        return sum(1 for x in rs if x and x <= k) / len(rs)

    def pct_rank(rs, q):
        """Rank at percentile `q`, misses held at MISS_RANK so losing a document cannot flatter it."""
        vals = sorted(x if x else MISS_RANK for x in rs)
        return vals[min(len(vals) - 1, int(q * len(vals)))]

    print(f"\n  corpus: {CORPUS}   model: {MODEL}   n={len(items)}   depth={DEPTH}")
    print(f"  {'condition':<15} {'@1':>7} {'@5':>7} {'@10':>7} {'@20':>7} {'MRR':>7} {'med':>5} {'p75':>5}")
    for c in conditions:
        rs = ranks[c]
        print(f"  {c:<15} " + " ".join(f"{rec(rs, k):>6.1%}" for k in (1, 5, 10, 20))
              + f" {mrr(rs):>7.3f} {pct_rank(rs, 0.5):>5} {pct_rank(rs, 0.75):>5}")
    out = pathlib.Path(f"investigations/retrieval/arm_rerank_{CORPUS}.json")
    out.write_text(json.dumps({"model": MODEL, "depth": DEPTH, "ranks": ranks}, indent=1), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
