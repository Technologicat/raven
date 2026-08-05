#!/usr/bin/env python3
"""Score a fixed off-corpus probe set against whichever collection is currently indexed.

**Its original purpose is refuted; it is kept as the instrument, not the answer.** This was written to
*choose* a per-collection cut by taking a quantile of the probe distribution, on the evidence of two
corpora. A third — 1268 arXiv AI/ML abstracts — showed the selected estimator (p75) letting 72.8% of
off-corpus negatives through, and showed a single global constant near 0.40 matching the best
probe-calibrated estimator on both axes. So there is no recommendation to make here any more. What the
script still does honestly is measure where a corpus puts queries that have no answer in it, which is what
produced the comparison table; `README.md` has the table and the reasoning.

The design argument it was built on, preserved because the argument is sound and the conclusion still did
not follow:

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

**Where that goes wrong** is the last step, not the reasoning before it. The probes do roughly span the
negative range on all three corpora (only 2, 16 and 3 negatives respectively exceed the probe maximum), so
the instrument is sound. What does not hold is that a *quantile of the probe distribution* tracks where the
on-corpus distribution starts. The probe spread varies with corpus content in a way unrelated to that:
"What is the capital of Mongolia?" scores 0.479 against hydrogen abstracts and 0.210 against arXiv ones, so
p75 lands near the on-corpus floor on two corpora and far below it on the third — which is a coincidence
being read as a mechanism.

The deeper reason there was little to recover: the arXiv run shows the signal reads *topical match* rather
than register (the near well separates at 0.999, identical to the far one), so its scale is not
per-collection in the way the "no single constant travels" premise assumed.

Requires a running raven-server and an indexed collection. Reads the index, does not write to it.

Usage:
    python calibrate.py [hydrogen|fiction|arxiv-ai]

The argument is optional and only names the labelled result file to report against, if one is present.
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

# The constant the three-corpus comparison settles on (see `README.md`), reported so a run can be read
# against what would actually ship. It is not derived from the probes — that is the whole correction.
SHIPPING_CUT = 0.40

# Quantiles reported for the probe distribution. Several rather than one, because no single quantile of it
# turned out to predict where the on-corpus distribution starts, and printing one would re-imply that it does.
REPORTED_QUANTILES = (50, 75, 90)


def probe_quantiles(probe_scores: list[float]) -> dict[int, float]:
    """The reported quantiles of the probe distribution, as `{percentile: score}`."""
    percentiles = statistics.quantiles(probe_scores, n=100)
    return {p: percentiles[p - 1] for p in REPORTED_QUANTILES}


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
    quantiles = probe_quantiles(scores)
    print(f"\n{len(scores)} probes: min {min(scores):.3f}, max {max(scores):.3f}, "
          + ", ".join(f"p{p} {v:.3f}" for p, v in quantiles.items()))
    print(f"the shipping cut is {SHIPPING_CUT:.2f}, chosen across three corpora rather than from these "
          f"probes — see README.md")

    if maybe_corpus is None:
        return
    results_path = HERE / f"sharpness_results_{maybe_corpus}.json"
    if not results_path.exists():
        print(f"\n(no labelled results at {results_path}, so nothing to report against)")
        return

    # What each candidate would cost on the labelled set. Only meaningful when the named corpus is the one
    # actually indexed, which nothing here can check.
    per_query = json.loads(results_path.read_text())["per_query"]
    on_corpus = [o["signals"]["vector best score"] for o in per_query if o["on_corpus"]]
    negatives = [o["signals"]["vector best score"] for o in per_query if not o["on_corpus"]]
    print(f"\nagainst {results_path.name} — assumes '{maybe_corpus}' is the indexed collection:")
    print(f"{'cut':>22} {'on-corpus lost':>16} {'negatives missed':>18}")
    candidates = [(f"p{p} of probes", v) for p, v in quantiles.items()] + [("shipping constant", SHIPPING_CUT)]
    for label, cut in candidates:
        lost = sum(1 for x in on_corpus if x < cut)
        missed = sum(1 for x in negatives if x >= cut)
        print(f"{f'{label} ({cut:.3f})':>22} {f'{lost}/{len(on_corpus)}':>16} {f'{missed}/{len(negatives)}':>18}")


if __name__ == "__main__":
    main()
