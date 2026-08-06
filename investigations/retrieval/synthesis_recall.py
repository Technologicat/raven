"""Does a question that needs several documents behave differently from one that needs a single document?

This is the measurement adaptive `k` rests on, and until the `synthesis` question class existed it could
not be made: every question in every set was a known-item question with exactly one right document, so
*specificity had no variance* and nothing could be learned about telling a narrow question from a broad
one.

Two things are measured, and only the second is the point:

- **Set recall at `k`** — what fraction of a synthesis question's gold *set* is retrieved. Every other
  scorer here asks whether *any* gold document appeared, which for a four-document gold set is nearly
  free and would report a solved problem.
- **How the recall curve differs in shape** between the `focused` questions of the same corpus and the
  `synthesis` ones. This is the hypothesis: a broad question should keep gaining from a larger `k` after a
  narrow one has stopped, because its answer is spread across documents that rank at different depths. If
  the two curves have the same shape, adaptive `k` has nothing to exploit no matter how well a broad
  question can be *detected*.

That framing matters, because the detection question and the payoff question are separate and only the
second is settled here. A signal that perfectly identifies broad questions buys nothing if broad questions
do not actually want more results.

**The metric understates, in the way this whole directory understates.** Other documents in the corpus may
contribute to a synthesis answer without being in the gold set, so set recall is a floor — the same
limitation as known-item recall being a floor on precision, arriving in the same shape.

    python synthesis_recall.py <corpus> [--db-dir DIR] [--depth 200]
"""

from __future__ import annotations

__all__ = ["K_POINTS", "set_recall"]

import concurrent.futures
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sharpness  # noqa: E402

K_POINTS = (5, 10, 20, 50, 100, 200)
DEFAULT_DEPTH = 200


def set_recall(results: list[dict], gold: set[str], k: int) -> float:
    """Fraction of `gold` whose documents appear among the first `k` results.

    Deduplicated to documents first, so `k` counts documents rather than chunks — otherwise a corpus of
    long documents would report a low set recall merely because one document filled the window.
    """
    if not gold:
        return 0.0
    keys = {sharpness.document_key(g) for g in gold}
    seen: list[str] = []
    for r in results:
        did = r.get("document_id")
        if did is None:
            continue
        key = sharpness.document_key(did)
        if key not in seen:
            seen.append(key)
        if len(seen) >= k:
            break
    return len(keys & set(seen[:k])) / len(keys)


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

    workload = sharpness.build_workload(corpus)[0]
    groups = {"focused": [i for i in workload if i["on_corpus"] and i["gold"] and i["kind"] == "focused"],
              "synthesis": [i for i in workload if i["on_corpus"] and i["gold"] and i["kind"] == "synthesis"]}
    if not groups["synthesis"]:
        print(f"corpus '{corpus}' has no synthesis questions yet — generate them with\n"
              f"    make_questions.py --append {corpus} <base_url> <model> 0 0 --synthesis N")
        return

    ex = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file, executor=ex)
    hybridir.init(executor=ex)
    retriever = hybridir.HybridIR(datastore_base_dir=db_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)

    gold_sizes = {name: sorted({len(i["gold"]) for i in items}) for name, items in groups.items()}
    print(f"corpus '{corpus}': {len(groups['focused'])} focused, {len(groups['synthesis'])} synthesis; "
          f"gold set sizes {gold_sizes}")
    print(f"  index: {db_dir}   depth: {depth}\n")

    curves = {name: {k: [] for k in K_POINTS} for name in groups}
    for name, items in groups.items():
        for n, item in enumerate(items, 1):
            results = retriever.query(item["query"], k=depth, multi_query=False)
            gold = set(item["gold"])
            for k in K_POINTS:
                curves[name][k].append(set_recall(results, gold, k))
            if n % 25 == 0:
                print(f"  {name} [{n}/{len(items)}]", flush=True)

    print(f"\n  mean set recall at k  ({' '.join(str(k) for k in K_POINTS)})")
    print(f"  {'group':<12} " + " ".join(f"{('k=' + str(k)):>8}" for k in K_POINTS) + f" {'5->200':>9}")
    summary = {}
    for name in ("focused", "synthesis"):
        means = [sum(curves[name][k]) / max(len(curves[name][k]), 1) for k in K_POINTS]
        summary[name] = dict(zip((str(k) for k in K_POINTS), means))
        print(f"  {name:<12} " + " ".join(f"{m:>8.1%}" for m in means) + f" {means[-1] - means[0]:>+9.1%}")

    print("\n  The last column is what adaptive `k` would be buying: how much a group still gains between")
    print("  k=5 and k=200. If the two are equal, detecting a broad question buys nothing, however well")
    print("  it can be detected.")

    out = pathlib.Path(__file__).parent / f"synthesis_recall_{corpus}.json"
    out.write_text(json.dumps({"corpus": corpus, "depth": depth, "k_points": list(K_POINTS),
                               "n": {name: len(items) for name, items in groups.items()},
                               "mean_set_recall": summary}, indent=1), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
