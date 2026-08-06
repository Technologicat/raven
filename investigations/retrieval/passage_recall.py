"""Did retrieval find the right *passage*, or merely the right document?

On a corpus of few long documents these are very different questions, and only the second has been
measured. Fiction's recall@20 reads 100%, which sounds like solved retrieval and is close to tautological:
19 documents, `k=20`, so a query that returns twenty results has returned essentially the collection. The
2977 chunks underneath are where the actual discrimination lives.

**The labels exist already.** `make_fiction_questions.py` samples a passage and writes the question from
it, recording `source` and `source_offset` — so the gold answer is not "this document" but "this document,
around this offset". Nothing new needs generating; the earlier measurements simply asked the easier
question because it was the one the arXiv sets could answer.

A result counts as a passage hit when it comes from the gold document and its span covers the gold offset.
Merged spans and bare chunks are both scored, since a span covers more ground and should therefore hit
more often — the interesting quantity is how much of the document-level score survives the stricter test.

**What this cannot do is compare corpora.** Only the fiction set carries offsets; the arXiv and hydrogen
questions are written from whole abstracts, where a document is one to three chunks and the two questions
nearly coincide. Extending this to the fulltext corpus needs questions generated from the PDFs' passages,
which is a generation run built on the *fiction* generator rather than on `make_questions.py` — the latter
samples whole records and has nowhere to record an offset.

    python passage_recall.py [--db-dir DIR] [--k 20] [--corpus fiction]
"""

from __future__ import annotations

__all__ = ["covers_offset", "passage_hit"]

import concurrent.futures
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sharpness  # noqa: E402


def covers_offset(result: dict, offset: int) -> bool:
    """Whether `result`'s span in its document contains `offset`.

    A retrieved span runs from its own `offset` for as many characters as its text; `merge_contiguous_spans`
    preserves both fields, so this reads the same for a merged span and for a bare chunk.
    """
    start = result.get("offset")
    if start is None:
        return False
    return start <= offset < start + len(result.get("text", ""))


def passage_hit(results: list[dict], gold_documents: set[str], gold_offset: int) -> bool:
    """Whether any result is from a gold document *and* covers the passage the question came from."""
    keys = {sharpness.document_key(g) for g in gold_documents}
    return any(sharpness.document_key(r["document_id"]) in keys and covers_offset(r, gold_offset)
               for r in results if r.get("document_id"))


def main() -> None:  # pragma: no cover
    argv = sys.argv[1:]

    def opt(name, default):
        if name in argv:
            at = argv.index(name)
            value = argv[at + 1]
            del argv[at:at + 2]
            return value
        return default

    corpus = opt("--corpus", "fiction")
    k = int(opt("--k", "20"))
    db_dir = opt("--db-dir", None)
    db_dir = (pathlib.Path(db_dir).expanduser() if db_dir
              else pathlib.Path.home() / f".config/raven/llmclient/rag_index_{corpus}")

    from raven.client import api as client_api
    from raven.client import config as client_config
    from raven.librarian import config as librarian_config
    from raven.librarian import hybridir

    items = [i for i in sharpness.build_workload(corpus)[0]
             if i["on_corpus"] and i["gold"] and i.get("source_offset") is not None]
    if not items:
        print(f"corpus '{corpus}' carries no `source_offset` labels; only passage-sampled sets can be "
              f"scored this way (see the module docstring).")
        return

    ex = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file, executor=ex)
    hybridir.init(executor=ex)
    retriever = hybridir.HybridIR(datastore_base_dir=db_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)
    print(f"corpus '{corpus}': {len(items)} questions carrying a passage offset, k={k}")
    print(f"  index: {db_dir}\n")

    tally = {arm: {"document": 0, "passage": 0} for arm in ("merged", "chunks")}
    rows = []
    for n, item in enumerate(items, 1):
        gold = set(item["gold"])
        offset = int(item["source_offset"])
        merged = retriever.query(item["query"], k=k, multi_query=False)
        chunks = retriever.query(item["query"], k=k, multi_query=False, merge=False)
        row = {"gold": sorted(gold), "source_offset": offset}
        for arm, results in (("merged", merged), ("chunks", chunks)):
            found_document = sharpness.rank_of_gold(results, gold) is not None
            found_passage = passage_hit(results, gold, offset)
            tally[arm]["document"] += int(found_document)
            tally[arm]["passage"] += int(found_passage)
            row[arm] = {"document": found_document, "passage": found_passage}
        rows.append(row)
        if n % 20 == 0:
            print(f"  [{n}/{len(items)}]", flush=True)

    total = len(items)
    print(f"\n  {'arm':<10} {'document@k':>12} {'passage@k':>12} {'gap':>8}")
    for arm in ("merged", "chunks"):
        d = tally[arm]["document"] / total
        p = tally[arm]["passage"] / total
        print(f"  {arm:<10} {d:>11.1%} {p:>12.1%} {d - p:>+8.1%}")
    print("\n  'gap' is how much of the document-level score does not survive asking for the right passage.")

    out = pathlib.Path(__file__).parent / f"passage_recall_{corpus}.json"
    out.write_text(json.dumps({"corpus": corpus, "k": k, "n": total, "tally": tally, "rows": rows},
                              indent=1), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
