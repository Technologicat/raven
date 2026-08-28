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


def covers_offset(result: dict, offset: int, length: int) -> bool:
    """Whether `result`'s span in its document overlaps the passage `[offset, offset + length)`.

    **Overlap, not containment of the start point**, and the difference is not pedantic. The source passage
    is `passage_chars` long — 4000 in the fiction set — while a chunk is 1000, so a passage spans four or
    five chunks and the question may have been written from any part of it. Asking whether a result
    contains the passage's *start* therefore scores a chunk covering the relevant text as a miss whenever
    the relevant text is not in the first quarter.

    It also biases the arm comparison, which is worse than being merely strict: a merged span is longer, so
    it reaches back to the passage start more often than a bare chunk does, and the start-containment
    version credited merging for that. Overlap treats both arms alike.

    A retrieved span runs from its own `offset` for as many characters as its text; `merge_contiguous_spans`
    preserves both fields, so this reads the same for a merged span and for a bare chunk.
    """
    start = result.get("offset")
    if start is None:
        return False
    end = start + len(result.get("text", ""))
    return start < offset + length and offset < end


def passage_hit(results: list[dict], gold_documents: set[str], gold_offset: int, passage_length: int) -> bool:
    """Whether any result is from a gold document *and* overlaps the passage the question came from."""
    keys = {sharpness.document_key(g) for g in gold_documents}
    return any(sharpness.document_key(r["document_id"]) in keys
               and covers_offset(r, gold_offset, passage_length)
               for r in results if r.get("document_id"))


def passage_coverage(results: list[dict], gold_documents: set[str], gold_offset: int,
                     passage_length: int) -> float:
    """What fraction of the source passage reaches the model, over the union of retrieved spans.

    **This is the measurement the boolean versions were reaching for, and it is the one that matches how
    the results are used** (Juha, 2026-08-06). The model does not receive "the chunk that overlapped"; it
    receives every retrieved result *in full*. So whether it can answer depends on how much of the passage
    the question was written from is in front of it — not on whether the particular chunk holding the
    answer happened to be the one retrieved.

    That also explains why the two boolean metrics disagreed so violently. Requiring the passage's start
    point scores 39.8% and favours long spans; requiring any overlap scores 89.8% and treats a single
    1000-character chunk of a 4000-character passage as a hit. Neither is wrong about what it measures;
    both are the wrong question. A fraction needs no threshold and no tie-breaking convention, and it
    separates the arms on the axis that actually differs between them — how much text each delivers.

    Union rather than sum, because merged spans and adjacent chunks overlap each other and double-counting
    would let an arm exceed 1.0 by returning the same text twice.
    """
    keys = {sharpness.document_key(g) for g in gold_documents}
    covered = [False] * passage_length
    for r in results:
        if not r.get("document_id") or sharpness.document_key(r["document_id"]) not in keys:
            continue
        start = r.get("offset")
        if start is None:
            continue
        end = start + len(r.get("text", ""))
        lo = max(0, start - gold_offset)
        hi = min(passage_length, end - gold_offset)
        for i in range(lo, hi):
            covered[i] = True
    return sum(covered) / passage_length if passage_length else 0.0


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
              else pathlib.Path.home() / f".config/raven/librarian/rag_index_{corpus}")

    from raven.client import api as client_api
    from raven.client import config as client_config
    from raven.librarian import config as librarian_config
    from raven.librarian import hybridir

    workload, note = sharpness.build_workload(corpus)
    items = [i for i in workload
             if i["on_corpus"] and i["gold"] and i.get("source_offset") is not None]
    # Read from the set rather than assumed: the generator records the passage length it sampled with, and
    # the scoring depends on it — a wrong value silently mis-scores every question in the same direction.
    passage_length = note.get("passage_chars")
    if passage_length is None:
        raise SystemExit(f"question set for '{corpus}' records no `passage_chars`; cannot score passages "
                         "without knowing how long the sampled passage was")
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
    coverage = {arm: [] for arm in ("merged", "chunks")}
    rows = []
    for n, item in enumerate(items, 1):
        gold = set(item["gold"])
        offset = int(item["source_offset"])
        merged = retriever.query(item["query"], k=k, multi_query=False)
        chunks = retriever.query(item["query"], k=k, multi_query=False, merge=False)
        row = {"gold": sorted(gold), "source_offset": offset}
        for arm, results in (("merged", merged), ("chunks", chunks)):
            found_document = sharpness.rank_of_gold(results, gold) is not None
            found_passage = passage_hit(results, gold, offset, passage_length)
            fraction = passage_coverage(results, gold, offset, passage_length)
            tally[arm]["document"] += int(found_document)
            tally[arm]["passage"] += int(found_passage)
            coverage[arm].append(fraction)
            row[arm] = {"document": found_document, "passage": found_passage, "coverage": fraction}
        rows.append(row)
        if n % 20 == 0:
            print(f"  [{n}/{len(items)}]", flush=True)

    total = len(items)
    print(f"\n  {'arm':<10} {'document@k':>12} {'passage@k':>12} {'mean cover':>12} {'>=50%':>8} {'>=90%':>8}")
    for arm in ("merged", "chunks"):
        d = tally[arm]["document"] / total
        p = tally[arm]["passage"] / total
        cov = coverage[arm]
        print(f"  {arm:<10} {d:>11.1%} {p:>12.1%} {sum(cov) / total:>12.1%} "
              f"{sum(1 for c in cov if c >= 0.5) / total:>8.1%} {sum(1 for c in cov if c >= 0.9) / total:>8.1%}")
    print("\n  'mean cover' is the share of the source passage the model actually receives, over the union of")
    print("  retrieved spans. That is what decides whether it has the material to answer, since it sees each")
    print("  result in full rather than only the part that matched — so the booleans left of it are proxies.")

    out = pathlib.Path(__file__).parent / f"passage_recall_{corpus}.json"
    out.write_text(json.dumps({"corpus": corpus, "k": k, "n": total, "tally": tally, "rows": rows},
                              indent=1), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
