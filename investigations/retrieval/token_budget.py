"""At an equal prefill budget, is the model better served by merged spans or by bare chunks?

`k` is not the budget. Prefill time is, and it is paid per *token* — so the question "how many results
should we send" is really "how should a fixed number of tokens be spent". `merge_contiguous_spans` spends
them on making each result longer: a hit and its neighbouring chunks are stitched into one span, so a
merged result carries context the chunk alone did not. That is a purchase, and this measures what it buys.

The comparison holds the budget fixed and varies the unit:

  merged    contiguous chunks stitched with no length limit — the behaviour before the cap
  capped    the same, with `max_span_length` — what ships
  chunks    the fused ranking unmerged, best chunk first

All three come from `HybridIR.query` itself rather than from a re-implementation here, which matters: the
first version of this script rebuilt the chunk-level fusion in the harness, so a divergence between the two
would have shown up as a finding. `merge=False` exists for exactly this.

Each is filled from its own retrieval until it reaches the budget, so the only difference is what a
"result" is. The metric is whether the gold document is anywhere in the material the model would see.

**Budgeted in characters, not tokens.** A token count would need the LLM backend, one call per
measurement, which is both slow and unavailable while the backend is generating question sets. Both arms
here are the same kind of text from the same corpus, so their characters-per-token ratios are equal and
the *comparison* is unaffected; only the axis label changes. For orientation, the measured hydrogen
retrieval runs about 4 characters per token, so a 30000-character budget is roughly the `k=20` prefill.

**What this deliberately does not measure: whether the reply is any good.** Merging exists so that the
model receives readable passages rather than fragments cut mid-sentence, and a chunk-level win on recall
per character says nothing about that. Read a win here as "there is a trade to consider", not as "ship
chunks".

    python token_budget.py <corpus> [--db-dir DIR] [--depth 100]
"""

from __future__ import annotations

__all__ = ["BUDGETS", "fill_to_budget", "gold_in"]

import concurrent.futures
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sharpness  # noqa: E402

# Characters. Spans the range the prefill measurement covers: ~30k is about the k=20 cost, ~300k about
# k=200, which the same measurement showed to be unusable in conversation.
BUDGETS = (7500, 15000, 30000, 60000, 120000, 300000)

DEFAULT_DEPTH = 100


def fill_to_budget(results: list[dict], budget: int) -> list[dict]:
    """Take results in rank order until the next one would not fit.

    Stops rather than skips ahead: taking a later, shorter result because the next one overflows would
    measure a packing strategy nobody has proposed, and would flatter whichever arm has more small results
    lying around.
    """
    taken, used = [], 0
    for r in results:
        size = len(r.get("text", ""))
        if used + size > budget:
            break
        taken.append(r)
        used += size
    return taken


def gold_in(results: list[dict], gold: set[str]) -> bool:
    """Whether any result is a gold document, compared by `sharpness.document_key`.

    Keyed on the id without its extension, so a question written from `paper.bib` matches the same paper
    indexed as `paper.pdf` — which is the whole point of holding the arXiv corpus in both forms. Comparing
    raw ids reports a clean 0.0% at every budget instead of failing.
    """
    keys = {sharpness.document_key(g) for g in gold}
    return any(sharpness.document_key(r["document_id"]) in keys for r in results if r.get("document_id"))


def main() -> None:  # pragma: no cover
    argv = sys.argv[1:]
    depth, db_dir, label = DEFAULT_DEPTH, None, None
    # The output filename is keyed on the corpus, and two *indexes* can share one question set — the arXiv
    # abstracts and the arXiv fulltexts hold the same documents under the same ids, which is the whole point
    # of that pair. Without a label the second run silently overwrites the first.
    if "--label" in argv:
        at = argv.index("--label")
        label = argv[at + 1]
        del argv[at:at + 2]
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

    arms = ("merged", "capped", "chunks")
    hits = {arm: {b: 0 for b in BUDGETS} for arm in arms}
    counts = {arm: {b: [] for b in BUDGETS} for arm in arms}
    rows = []
    for n, item in enumerate(items, 1):
        gold = set(item["gold"])
        # Three retrievals rather than one derived three ways, so every arm is the shipped code path.
        merged = retriever.query(item["query"], k=depth, multi_query=False, max_span_length=None)
        capped = retriever.query(item["query"], k=depth, multi_query=False,
                                 max_span_length=librarian_config.docs_max_result_length)
        chunks = retriever.query(item["query"], k=depth, multi_query=False, merge=False)

        row = {"budgets": {}}
        for budget in BUDGETS:
            for arm, results in (("merged", merged), ("capped", capped), ("chunks", chunks)):
                taken = fill_to_budget(results, budget)
                found = gold_in(taken, gold)
                hits[arm][budget] += int(found)
                counts[arm][budget].append(len(taken))
                row["budgets"].setdefault(str(budget), {})[arm] = {"n": len(taken), "hit": found}
        rows.append(row)
        if n % 20 == 0:
            print(f"  [{n}/{len(items)}]", flush=True)

    n_items = len(items)

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2] if s else 0

    print(f"\n  recall (gold document present), n={n_items}")
    print(f"  {'chars':>8} {'~tokens':>8} | {'merged':>8} {'results':>8} | {'capped':>8} {'results':>8} "
          f"| {'chunks':>8} {'results':>8} | {'delta':>7}")
    censored = []
    for b in BUDGETS:
        m = hits["merged"][b] / n_items
        c = hits["chunks"][b] / n_items
        n_merged = median(counts["merged"][b])
        # `query` returns at most `k` results, so once a budget is large enough to hold all of them the
        # merged arm stops growing while the chunk arm keeps going. The delta on such a row is measuring
        # that cap, not the merging — flag it rather than let it read as the largest effect in the table.
        flag = "  <- merged arm capped at k, not comparable" if n_merged >= depth else ""
        if flag:
            censored.append(b)
        print(f"  {b:>8} {b // 4:>8} | {m:>8.1%} {n_merged:>8} | "
              f"{hits['capped'][b] / n_items:>8.1%} {median(counts['capped'][b]):>8} | "
              f"{c:>8.1%} {median(counts['chunks'][b]):>8} | {c - m:>+7.1%}{flag}")
    if censored:
        print(f"\n  {len(censored)} row(s) censored by the depth cap ({depth}); read the rows above them.")

    out = pathlib.Path(__file__).parent / f"token_budget_{label or corpus}.json"
    out.write_text(json.dumps({"corpus": corpus, "depth": depth, "budgets": list(BUDGETS),
                               "hits": {a: {str(k): v for k, v in d.items()} for a, d in hits.items()},
                               "n": n_items, "rows": rows}, indent=1), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
