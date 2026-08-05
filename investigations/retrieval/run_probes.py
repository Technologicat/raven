#!/usr/bin/env python3
"""Score the hand-written probe set in `probes.json` against the live fiction index.

The generated question sets measure one thing well — given a question a corpus can answer, is the right
document found — and are blind to two others. Every generated question is answerable **by construction**,
so nothing in them exercises a question the corpus cannot answer; and every one is written *from* a passage,
so nothing exercises a question about the document as a whole. The probes fill both gaps, at the cost of
being few and hand-labelled.

What this reports, and why each part exists:

**By class.** The probes are stratified by *where the answer lives* rather than by topic — in a chunk, in
the document but stated, in the document but only exhibited, or outside the corpus entirely. Retrieval
should succeed on the first two, and the last is expected to miss: `asimov-pastiche` asks for something not
present in any form, so a miss there is the correct answer, and this script scores it that way rather than
counting it against the engine.

**Phrasing spread.** Each probe carries several wordings of one information need, so ground truth is held
constant and the confidence signal's variation is attributable to phrasing alone. This is the measurement
that most constrains lever 1: on the Switzerland probe all six wordings retrieve at rank 1 while the signal
spans 0.229, which is wider than the gap between the two corpora that a fixed threshold has to straddle.
A signal that moves more with wording than with whether the corpus can answer is not one to threshold
globally.

Requires a running raven-server and the fiction corpus indexed. Reads the index, does not write to it.
**It cannot check that the right corpus is loaded**; pointing it at the hydrogen index silently scores
every probe as a miss.

Usage:
    python run_probes.py [k]
"""

import collections
import concurrent.futures
import json
import pathlib
import statistics
import sys

from raven.client import api as client_api
from raven.client import config as client_config
from raven.librarian import config as librarian_config
from raven.librarian import hybridir

HERE = pathlib.Path(__file__).parent
PROBES_PATH = HERE / "probes.json"
RESULTS_PATH = HERE / "probe_results.json"


def document_rank(results: list[dict], gold: set[str]) -> int | None:
    """1-based rank of the gold document among the distinct documents returned, or `None` if absent.

    Documents rather than spans: a probe asks "which story", so several chunks of one story are one answer.
    """
    seen = []
    for result in results:
        if result["document_id"] not in seen:
            seen.append(result["document_id"])
    return next((i for i, doc in enumerate(seen, start=1) if doc in gold), None)


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    payload = json.loads(PROBES_PATH.read_text())
    probes = [p for p in payload["probes"] if p.get("verified")]
    print(f"{len(probes)} verified probes, "
          f"{sum(len(p['phrasings']) for p in probes)} phrasings, k={k}")
    if payload.get("excluded"):
        print(f"  ({len(payload['excluded'])} excluded for unsettled ground truth: "
              f"{', '.join(e['id'] for e in payload['excluded'])})")

    executor = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file,
                          executor=executor)
    hybridir.init(executor=executor)
    retriever = hybridir.HybridIR(datastore_base_dir=librarian_config.llm_database_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)
    with retriever.datastore_lock:
        print(f"  live index holds {len(retriever.documents)} documents\n")

    rows = []
    for probe in probes:
        gold = set(probe["gold"])
        expects_miss = probe.get("expectation") == "not-retrievable-by-design"
        per_phrasing = []
        for phrasing in probe["phrasings"]:
            results, report = retriever.query(phrasing, k=k, multi_query=False, return_extra_info=True)
            similarity = max((1.0 - d for d in report.per_query[0].candidate_vector_distances), default=0.0)
            rank = document_rank(results, gold)
            per_phrasing.append({"phrasing": phrasing, "rank": rank, "similarity": similarity})
        sims = [x["similarity"] for x in per_phrasing]
        ranks = [x["rank"] for x in per_phrasing]
        found = sum(1 for r in ranks if r == 1)
        rows.append({"id": probe["id"], "class": probe["class"], "expects_miss": expects_miss,
                     "gold": probe["gold"], "phrasings": per_phrasing,
                     "rank1": found, "n": len(ranks),
                     "sim_min": min(sims), "sim_max": max(sims),
                     "sim_median": statistics.median(sims), "spread": max(sims) - min(sims)})

    print(f"{'probe':<28} {'class':<26} {'rank1':>7} {'sim min–max':>16} {'spread':>7}")
    print("-" * 92)
    for row in rows:
        verdict = ""
        if row["expects_miss"]:
            verdict = "  (a miss here is correct)"
        elif row["rank1"] < row["n"]:
            verdict = "  <-- misses"
        print(f"{row['id']:<28} {row['class']:<26} {row['rank1']:>3}/{row['n']:<3} "
              f"{row['sim_min']:>7.3f}–{row['sim_max']:<7.3f} {row['spread']:>7.3f}{verdict}")

    print("\nby class:")
    by_class: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)
    for name, group in sorted(by_class.items()):
        hit = sum(r["rank1"] for r in group)
        total = sum(r["n"] for r in group)
        note = " — expected to miss" if all(r["expects_miss"] for r in group) else ""
        print(f"  {name:<26} {hit:>3}/{total:<4} phrasings retrieved the gold document at rank 1{note}")

    widest = max(rows, key=lambda r: r["spread"])
    print(f"\nwidest phrasing spread: {widest['id']} at {widest['spread']:.3f} "
          f"({widest['sim_min']:.3f}–{widest['sim_max']:.3f}), "
          f"gold at rank 1 for {widest['rank1']}/{widest['n']} wordings.")
    print("A signal that moves this much with wording, while retrieval quality does not, cannot carry a "
          "global threshold.")

    RESULTS_PATH.write_text(json.dumps({"k": k, "probes": rows}, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote per-phrasing results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
