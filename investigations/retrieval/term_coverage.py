"""Is "how many distinct query terms matched" evidence that BM25 does not already carry?

The seventh axis on the evidence-accumulation list, and the cheapest: no new model, no new retrieval, no
re-tokenization. The index already stores `tokens` per chunk, lemmatized by the same spaCy pipeline the
query goes through, so a chunk's term coverage is a set intersection against the tokenized query.

**Why it might be independent of BM25 rather than a restatement of it.** BM25 sums per-term contributions
and saturates each one, so a chunk mentioning *one* query term ten times can outscore a chunk mentioning
*three* query terms once each — the saturation curve bounds how much any single term can contribute, but
nothing rewards breadth across terms directly. Coordination-level matching, the oldest idea in the book,
says a document matching more of the query is more likely to be about the query. Whether that adds
anything on top of BM25 in practice is the measurement.

**Fused as a third ranked list**, not as a weighted term, so there is no constant to tune: rank the
candidates by coverage and hand RRF three lists instead of two. That also keeps it honest about what is
being tested — the *independence* of the signal, not a tuned blend of it.

    python term_coverage.py <corpus> [--db-dir DIR] [--depth 100]
"""

from __future__ import annotations

__all__ = ["coverage_of"]

import concurrent.futures
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fusion_weight  # noqa: E402
import sharpness  # noqa: E402

DEFAULT_DEPTH = 100


def coverage_of(chunk_tokens: list[str], query_terms: set[str]) -> float:
    """Share of the query's distinct lemmas present in this chunk. 0 when the query tokenizes to nothing."""
    if not query_terms:
        return 0.0
    return len(query_terms & set(chunk_tokens)) / len(query_terms)


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
    corpus = argv[0] if argv else "arxiv-ai"
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

    conditions = ("bm25+vector", "bm25+vector+coverage", "coverage alone")
    ranks: dict[str, list] = {c: [] for c in conditions}
    spreads = []

    for n, item in enumerate(items, 1):
        gold = {sharpness.document_key(g) for g in item["gold"]}
        _merged, rep = retriever.query(item["query"], k=depth, multi_query=False, return_extra_info=True)
        query_terms = set(retriever._tokenize(item["query"]))

        scored = []
        with retriever.datastore_lock:
            for record in list(rep.keyword_results) + list(rep.vector_results):
                document = retriever.documents.get(record["document_id"])
                if document is None:
                    continue
                tokens = document.get("tokens") or []
                index = record.get("chunk_id")
                if not isinstance(index, int) or index >= len(tokens):
                    continue
                scored.append((record, coverage_of(tokens[index], query_terms)))

        # Deduplicate chunks that both arms returned, keeping one entry each.
        best: dict[str, tuple] = {}
        for record, value in scored:
            best[record["full_id"]] = (record, value)
        ranked = sorted(best.values(), key=lambda pair: pair[1], reverse=True)
        coverage_docs = fusion_weight.dedup_ids([record for record, _v in ranked])
        if ranked:
            spreads.append((ranked[0][1], ranked[-1][1]))

        kw_docs = fusion_weight.dedup_ids(rep.keyword_results)
        vec_docs = fusion_weight.dedup_ids(rep.vector_results)
        ranks["bm25+vector"].append(
            fusion_weight.gold_rank(fusion_weight.weighted_rrf(kw_docs, vec_docs, 0.5, 60), gold))
        ranks["bm25+vector+coverage"].append(
            fusion_weight.gold_rank(
                [d for d, _s in hybridir.reciprocal_rank_fusion(kw_docs, vec_docs, coverage_docs)], gold))
        ranks["coverage alone"].append(fusion_weight.gold_rank(coverage_docs, gold))
        if n % 25 == 0:
            print(f"  [{n}/{len(items)}]", flush=True)

    total = len(items)

    def recall(rs, k):
        return sum(1 for r in rs if r and r <= k) / total

    def mrr(rs):
        return sum(1.0 / r for r in rs if r) / total

    if spreads:
        top = sum(a for a, _b in spreads) / len(spreads)
        bottom = sum(b for _a, b in spreads) / len(spreads)
        print(f"\n  mean coverage of the best candidate {top:.1%}, of the worst {bottom:.1%} — if these are "
              f"equal\n  the signal is constant across candidates and cannot rank anything.")

    print(f"\n  n={total}")
    print(f"  {'condition':<24} {'@20':>7} {'@50':>7} {'MRR':>7} {'gain':>5} {'loss':>5} {'p':>7}")
    baseline = ranks["bm25+vector"]
    rows = []
    for name in conditions:
        rs = ranks[name]
        gained, lost, p = fusion_weight.mcnemar(baseline, rs)
        marker = "  <- baseline" if name == "bm25+vector" else ""
        print(f"  {name:<24} {recall(rs, 20):>7.1%} {recall(rs, 50):>7.1%} {mrr(rs):>7.3f} "
              f"{gained:>5} {lost:>5} {p:>7.3f}{marker}")
        rows.append({"condition": name, "recall20": recall(rs, 20), "recall50": recall(rs, 50),
                     "mrr": mrr(rs), "gained": gained, "lost": lost, "p": p})

    out = pathlib.Path(__file__).parent / f"term_coverage_{corpus}.json"
    out.write_text(json.dumps({"corpus": corpus, "depth": depth, "n": total, "rows": rows}, indent=1),
                   encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
