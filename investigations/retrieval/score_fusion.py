"""Does fusing *scores* beat fusing *ranks*, and should a document rank by its best chunk?

Two untried levers, measured together because they are the same operation — re-deciding the final order
from one retrieval — and because they interact: how much a score-based fusion gains may depend on how
chunk scores are rolled up to documents.

**Rank fusion versus score fusion.** RRF fuses *positions* and throws the scores away. That is the reason
this investigation's founding complaint is true almost by construction: *the hybrid rank does not track
how good a result is*, because nothing about how good a result is survives into the fused order. Every
tuning experiment so far has adjusted parameters **inside** RRF — the arms' relative weight, the constant
`K` — without asking whether rank-only fusion is the right primitive. `CombSUM` over per-query-normalized
scores is the standard alternative, needs no re-index and no new model, and yields a fused value that is a
*score* rather than a reciprocal-rank artifact, which is what a calibrated confidence signal would need.

**Best-chunk versus the whole document.** A document currently ranks where its single best chunk ranked,
and nothing else about it counts. Two alternatives cost nothing to compute:

  max    the shipped rule — the document is as good as its best passage
  sum    every matching chunk contributes, so a document matching in five places outranks one matching once
  count  how many chunks matched at all, ignoring how well

`sum` and `count` differ from `max` only when documents produce several matching chunks, so they are
near-no-ops on abstracts and should matter most on fulltext — which is the shape Librarian is pitched at.
Note `sum` has a known failure mode worth watching for rather than assuming away: it rewards *long*
documents, which have more chunks available to match.

Normalization is per query and per arm, because BM25 scores and cosine similarities share no scale — on one
sampled query the surviving BM25 scores spanned 3.07 to 3.65 while cosine similarities spanned 0.68 to
0.78. A chunk missing from one arm scores 0 there, which is `CombSUM`'s convention and treats absence as
"this engine found nothing", not as "average".

    python score_fusion.py <corpus> [--db-dir DIR] [--depth 100]
"""

from __future__ import annotations

__all__ = ["normalize", "fuse_scores", "aggregate"]

import concurrent.futures
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fusion_weight  # noqa: E402
import sharpness  # noqa: E402

DEFAULT_DEPTH = 100
WEIGHTS = (0.3, 0.5, 0.7)


def normalize(values: list[float], how: str) -> list[float]:
    """Map one arm's candidate scores onto a common scale, per query.

    `minmax` puts the best candidate at 1 and the worst at 0, which makes the two arms directly
    addable and is what `CombSUM` assumes. `zscore` instead measures each candidate against its own
    field's spread, so a query where everything scored alike contributes little either way — closer to
    what a confidence signal wants, and not bounded.

    A degenerate list (all equal, or one element) normalizes to all-zero rather than dividing by zero:
    an arm that cannot distinguish its own candidates should not get a vote.
    """
    if len(values) < 2:
        return [0.0] * len(values)
    lo, hi = min(values), max(values)
    if how == "minmax":
        return [0.0] * len(values) if hi == lo else [(v - lo) / (hi - lo) for v in values]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return [0.0] * len(values) if var <= 0.0 else [(v - mean) / var ** 0.5 for v in values]


def fuse_scores(keyword: dict[str, float], vector: dict[str, float], w: float) -> dict[str, float]:
    """`CombSUM`: weighted sum of normalized per-arm scores, keyed by chunk. Absence scores 0."""
    out: dict[str, float] = {}
    for key in set(keyword) | set(vector):
        out[key] = w * keyword.get(key, 0.0) + (1.0 - w) * vector.get(key, 0.0)
    return out


def aggregate(chunk_scores: dict[str, float], chunk_document: dict[str, str], how: str) -> list[str]:
    """Roll chunk scores up to documents and return document keys, best first."""
    per_document: dict[str, list[float]] = {}
    for key, score in chunk_scores.items():
        per_document.setdefault(chunk_document[key], []).append(score)
    if how == "max":
        scored = {d: max(v) for d, v in per_document.items()}
    elif how == "sum":
        scored = {d: sum(v) for d, v in per_document.items()}
    else:  # count: how many chunks matched at all, ignoring how well
        scored = {d: float(len(v)) for d, v in per_document.items()}
    return [d for d, _s in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)]


def main() -> None:  # pragma: no cover
    argv = sys.argv[1:]
    depth, db_dir = DEFAULT_DEPTH, None
    if "--depth" in argv:
        at = argv.index("--depth")
        depth = int(argv[at + 1])
        del argv[at:at + 2]
    if "--db-dir" in argv:
        at = argv.index("--db-dir")
        db_dir = pathlib.Path(argv[at + 1]).expanduser()
        del argv[at:at + 2]
    corpus = argv[0] if argv else "hydrogen"
    if db_dir is None:
        db_dir = pathlib.Path.home() / f".config/raven/llmclient/rag_index_{corpus}"

    from raven.client import api as client_api
    from raven.client import config as client_config
    from raven.librarian import config as librarian_config
    from raven.librarian import hybridir

    ex = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file, executor=ex)
    hybridir.init(executor=ex)
    retriever = hybridir.HybridIR(datastore_base_dir=db_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)

    items = [i for i in sharpness.build_workload(corpus)[0] if i["on_corpus"] and i["gold"]]
    print(f"corpus '{corpus}': {len(items)} on-corpus questions, depth {depth}")
    print(f"  index: {db_dir}\n")

    conditions = [("rrf", "max", None)]
    for how in ("minmax", "zscore"):
        for agg in ("max", "sum", "count"):
            for w in WEIGHTS:
                conditions.append((how, agg, w))
    ranks: dict[tuple, list] = {c: [] for c in conditions}

    for n, item in enumerate(items, 1):
        gold = {sharpness.document_key(g) for g in item["gold"]}
        _merged, rep = retriever.query(item["query"], k=depth, multi_query=False, return_extra_info=True)

        chunk_document: dict[str, str] = {}
        kw_raw, vec_raw = {}, {}
        for record, score in zip(rep.keyword_results, rep.keyword_scores):
            key = record["full_id"]
            chunk_document[key] = sharpness.document_key(record["document_id"])
            kw_raw[key] = float(score)
        for record, distance in zip(rep.vector_results, rep.vector_distances):
            key = record["full_id"]
            chunk_document[key] = sharpness.document_key(record["document_id"])
            vec_raw[key] = 1.0 - float(distance)  # distances are bigger-is-worse

        # The rank-fusion baseline, from the same candidate lists, deduplicated to documents the way the
        # shipped path does — a document ranks where its best chunk ranked.
        kw_docs = fusion_weight.dedup_ids(rep.keyword_results)
        vec_docs = fusion_weight.dedup_ids(rep.vector_results)
        ranks[("rrf", "max", None)].append(
            fusion_weight.gold_rank(fusion_weight.weighted_rrf(kw_docs, vec_docs, 0.5, 60), gold))

        for how in ("minmax", "zscore"):
            kw_keys, vec_keys = list(kw_raw), list(vec_raw)
            kw_norm = dict(zip(kw_keys, normalize([kw_raw[k] for k in kw_keys], how)))
            vec_norm = dict(zip(vec_keys, normalize([vec_raw[k] for k in vec_keys], how)))
            for w in WEIGHTS:
                fused = fuse_scores(kw_norm, vec_norm, w)
                for agg in ("max", "sum", "count"):
                    ordered = aggregate(fused, chunk_document, agg)
                    ranks[(how, agg, w)].append(fusion_weight.gold_rank(ordered, gold))
        if n % 25 == 0:
            print(f"  [{n}/{len(items)}]", flush=True)

    total = len(items)

    def recall(rs, k):
        return sum(1 for r in rs if r and r <= k) / total

    def mrr(rs):
        return sum(1.0 / r for r in rs if r) / total

    baseline = ranks[("rrf", "max", None)]
    print(f"\n  n={total}   baseline is RRF (w=0.5, K=60), best-chunk aggregation")
    print(f"  {'fusion':<9} {'aggregate':<10} {'w':>4} {'@20':>7} {'@50':>7} {'MRR':>7}"
          f" {'gain':>5} {'loss':>5} {'p':>7}")
    rows = []
    for cond in conditions:
        how, agg, w = cond
        rs = ranks[cond]
        gained, lost, p = fusion_weight.mcnemar(baseline, rs)
        label_w = "-" if w is None else f"{w:.1f}"
        marker = "  <- baseline" if how == "rrf" else ""
        print(f"  {how:<9} {agg:<10} {label_w:>4} {recall(rs, 20):>7.1%} {recall(rs, 50):>7.1%} "
              f"{mrr(rs):>7.3f} {gained:>5} {lost:>5} {p:>7.3f}{marker}")
        rows.append({"fusion": how, "aggregate": agg, "weight": w, "recall20": recall(rs, 20),
                     "recall50": recall(rs, 50), "mrr": mrr(rs), "gained": gained, "lost": lost, "p": p})

    print("\n  'gain'/'loss' are questions whose gold document enters/leaves the top 20 relative to the")
    print("  baseline, and p is the exact paired test over those. Recall differences without a paired")
    print("  test are not evidence at this sample size.")

    out = pathlib.Path(__file__).parent / f"score_fusion_{corpus}.json"
    out.write_text(json.dumps({"corpus": corpus, "depth": depth, "n": total, "rows": rows}, indent=1),
                   encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
