#!/usr/bin/env python3
"""Choose the off-corpus cut for whichever collection is currently indexed, without labelled questions.

`sharpness.py` established that the absolute best vector similarity separates on-corpus from off-corpus
queries, and that **no single constant travels**: the cut that rejected none of 99 hydrogen questions
rejects 24 of 88 fiction ones. The conclusion recorded in brief 09 was "per collection", which leaves open
what a per-collection number is *made of*. A setting the user types is not an answer — the value cannot be
guessed from anything a user knows, and getting it wrong fails in the direction that hurts (an answerable
question marked ungrounded reads as a confident refusal).

So calibrate it from the collection itself, at index time. The trick is which side to calibrate on:

- **The positive side is not available.** "What does an on-corpus question score" needs on-corpus
  questions, and at index time there are none. Chunks can stand in as pseudo-queries, but a chunk is long
  and expository where a question is short and oblique — the same mismatch that makes the dramatized-text
  probes fail — so the estimate is biased in an unknown direction.
- **The negative side is available, and is corpus-independent by definition.** "What does a query with no
  answer here score" can be measured against *any* index with a fixed probe set, because a probe about
  sourdough is off-corpus for every collection anyone would build. That is what this script does: run the
  probes, read the top of the resulting distribution, and put the cut there.

Measured on both corpora (numbers in `README.md`), against the fixed 0.45 the hydrogen run proposed:

    estimator            hydrogen: on-corpus lost    fiction: on-corpus lost
    fixed 0.45                    0 / 99  (0.0%)            24 / 88  (27.3%)
    p75 of the probes             0 / 99  (0.0%)             3 / 88  ( 3.4%)

Two caveats that belong with any number this prints. The probe set is **12 queries**, so a quantile of it
is itself a noisy statistic — a larger and more varied set would tighten it, and is the obvious next step
if this ships. And the estimator was selected on the same two corpora it is reported on, which is the
usual way a tuned constant sneaks back in under a different name; a third collection is what would test it.

Requires a running raven-server and an indexed collection. Reads the index, does not write to it.

Usage:
    python calibrate.py [hydrogen|fiction]

The argument is optional and only names the labelled result file to *validate* against, if one is present.
The calibration itself needs no labels — that is the point.
"""

import concurrent.futures
import json
import pathlib
import statistics
import sys

from raven.client import api as client_api
from raven.client import config as client_config
from raven.librarian import config as librarian_config
from raven.librarian import hybridir

import sharpness  # shared instrument: the probe set is defined once, next to the eval that validated it

HERE = pathlib.Path(__file__).parent

# Quantile of the probe distribution to cut at. Measured against `max`, `2nd highest` and
# `median + 2*stdev`; p75 dominated all three on both corpora, because the top probe is an outlier
# ("What is the capital of Mongolia?" scores 0.479 against hydrogen abstracts, 0.074 clear of the next)
# and any estimator anchored on the maximum inherits that one query's bad luck.
CUT_QUANTILE = 0.75


def recommended_cut(probe_scores: list[float]) -> float:
    """The cut, as a quantile of the probe distribution. Separated out so the choice has one home."""
    return statistics.quantiles(probe_scores, n=100)[int(100 * CUT_QUANTILE) - 1]


def main() -> None:
    maybe_corpus = sys.argv[1] if len(sys.argv) > 1 else None

    executor = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file,
                          executor=executor)
    hybridir.init(executor=executor)
    retriever = hybridir.HybridIR(datastore_base_dir=librarian_config.llm_database_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)
    with retriever.datastore_lock:
        n_documents = len(retriever.documents)
    print(f"calibrating against the live index: {n_documents} documents\n")

    # Pleasantries earn their place in the probe set rather than padding it: against a *narrative* corpus
    # they are the hardest universal negative, because conversational filler looks like dialogue. On
    # fiction the three top-scoring probes are all pleasantries; on hydrogen abstracts they sit mid-pack.
    scored = []
    for kind, probe in sharpness.PROBES:
        if kind == "adjacent_science":  # corpus-specific by construction — not universal, so not calibration material
            continue
        _, report = retriever.query(probe, k=20, multi_query=False, return_extra_info=True)
        similarity = max((1.0 - d for d in report.per_query[0].candidate_vector_distances), default=0.0)
        scored.append((similarity, kind, probe))

    print(f"{'score':>7}  {'kind':<11} probe")
    print("-" * 72)
    for similarity, kind, probe in sorted(scored, reverse=True):
        print(f"{similarity:>7.3f}  {kind:<11} {probe}")

    scores = [s for s, _, _ in scored]
    cut = recommended_cut(scores)
    print(f"\n{len(scores)} probes: min {min(scores):.3f}, median {statistics.median(scores):.3f}, "
          f"max {max(scores):.3f}")
    print(f"recommended cut (p{int(100 * CUT_QUANTILE)}): {cut:.3f}")

    if maybe_corpus is None:
        return
    results_path = HERE / f"sharpness_results_{maybe_corpus}.json"
    if not results_path.exists():
        print(f"\n(no labelled results at {results_path}, so nothing to validate against)")
        return

    # Validation half: what the recommended cut would have cost on the labelled set. Only meaningful when
    # the named corpus is the one actually indexed, which nothing here can check.
    per_query = json.loads(results_path.read_text())["per_query"]
    on_corpus = [o["signals"]["vector best score"] for o in per_query if o["on_corpus"]]
    far = [o["signals"]["vector best score"] for o in per_query if o["kind"] == "other_corpus"]
    print(f"\nvalidating against {results_path.name} — assumes '{maybe_corpus}' is the indexed collection:")
    print(f"  on-corpus questions rejected: {sum(1 for x in on_corpus if x < cut)} / {len(on_corpus)}")
    if far:
        print(f"  other-corpus negatives caught: {sum(1 for x in far if x < cut)} / {len(far)}")


if __name__ == "__main__":
    main()
