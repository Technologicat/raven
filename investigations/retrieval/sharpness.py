#!/usr/bin/env python3
"""Does `hybridir.score_sharpness` mean anything? Two ways of asking, because two consumers want
different things from it.

Brief 09's lever 1 proposes reading a query's confidence off the shape of its own score distribution,
before fusion flattens every score into a rank. The design argument is in the brief and in
`score_sharpness`'s docstring; this script is what decides whether the signal discriminates, and at
what ratio-to-best. Nothing in Librarian's retrieval path consumes it until this says it should.

**Measurement A — does sharpness predict retrieval success?** Over the known-item set: is the signal
higher on questions whose gold document was found than on those where it was not? This is what
adaptive `k` and per-subquery gating (lever 3's retry) need — both are decisions about a query the
corpus *can* answer.

**Measurement B — does sharpness separate on-corpus from off-corpus?** The known-item questions were
written *from* the corpus, so every one of them is answerable and measurement A cannot see the case
that matters most: a question the corpus has nothing to say about. Brief 10's grounding marker ships
today with a silent failure of exactly that kind — asked "what is 2 + 2?", retrieval returned
electrolysis documents and the marker read that as grounding. So B scores the known-item questions
against the probes below, which have no answer in a hydrogen corpus.

Both are reported as **AUROC**: the probability that a randomly chosen positive scores above a randomly
chosen negative, ties counted half. 0.5 is a coin flip, i.e. a signal that discriminates nothing.
Several candidate signals are scored side by side, including two constant-free alternatives to the
`min_p` reading, so that the ratio sweep is not being compared only against itself.

Requires a running raven-server (spaCy tokenization + embeddings) and the local document index. It
reads the index; it does not write to it.

Usage:
    python sharpness.py [k]
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

QUESTIONS_PATH = pathlib.Path(__file__).parent / "questions.json"
RESULTS_PATH = pathlib.Path(__file__).parent / "sharpness_results.json"

# Ratios to sweep. Wide and coarse: the brief is explicit that the values in circulation for `min_p`
# sampling carry no information about what to use here, so this is a search rather than a refinement.
RATIOS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)

# Queries with no answer in a hydrogen-production corpus, which is what the grounding marker has to
# recognize. Three kinds, because they fail differently and the hard one is last:
#
#   off_corpus  — a real question about something else entirely
#   pleasantry  — conversational filler, which is also what lever 3's splitter hands the fusion
#   adjacent    — science, plausibly phrased, still not in this corpus. The honest test: these are
#                 the ones that will retrieve *something* topical-looking.
PROBES = [("off_corpus", "What is 2 + 2?"),
          ("off_corpus", "How do I make a good sourdough starter?"),
          ("off_corpus", "What is the plot of Hamlet?"),
          ("off_corpus", "Which team won the 1998 FIFA World Cup?"),
          ("off_corpus", "How do I change a bicycle tyre?"),
          ("off_corpus", "What is the capital of Mongolia?"),
          ("off_corpus", "What year did the Beatles break up?"),
          ("off_corpus", "How do I sort a list of dictionaries in Python?"),
          ("pleasantry", "Good evening! How are you doing today?"),
          ("pleasantry", "Thanks, that was really helpful."),
          ("pleasantry", "Hi there."),
          ("pleasantry", "Could you say a bit more about that?"),
          ("adjacent", "What is the tensile strength of carbon fibre composites?"),
          ("adjacent", "How does CRISPR gene editing work in plants?"),
          ("adjacent", "What are the failure modes of lithium iron phosphate cells?"),
          ("adjacent", "How is atmospheric methane measured from satellites?")]


def rank_of_gold(results: list[dict], gold: set[str]) -> int | None:
    """1-based rank of the first result whose document is in `gold`, or `None` if absent."""
    for rank, result in enumerate(results, start=1):
        if result["document_id"] in gold:
            return rank
    return None


def auroc(values: list[float], labels: list[bool]) -> float:
    """Probability that a positive outranks a negative, ties counted half. 0.5 discriminates nothing.

    Equivalently the Mann-Whitney U statistic, normalized. Computed pairwise because the sample is a
    hundred items and clarity is worth more here than an O(n log n) formulation.
    """
    positives = [value for value, label in zip(values, labels) if label]
    negatives = [value for value, label in zip(values, labels) if not label]
    if not positives or not negatives:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def signals_from_report(report) -> dict[str, float]:
    """Every candidate confidence signal, read off one query's report. Bigger must mean more confident.

    The vector arm reports cosine *distances*, so they are converted to similarities first — a
    ratio-to-best test on distances measures the worst match instead of the best one.
    """
    entry = report.per_query[0]  # the whole message; subqueries are not in play for either measurement
    keyword = list(entry.candidate_keyword_scores)
    vector = [1.0 - distance for distance in entry.candidate_vector_distances]

    signals = {}
    for name, scores in (("keyword", keyword), ("vector", vector)):
        for ratio in RATIOS:
            signals[f"{name} sharpness @ {ratio}"] = hybridir.score_sharpness(scores, min_ratio=ratio)

        # Constant-free alternatives, so the swept family is not the only thing on the table. Both are
        # scale-free for the same reason sharpness is: they only ever compare a query's results to each
        # other. `best / mean` reads the whole distribution, `(best - second) / best` only its head.
        positive = [s for s in scores if s > 0.0]
        best = max(scores) if scores else 0.0
        mean = statistics.fmean(positive) if positive else 0.0
        signals[f"{name} best/mean"] = (best / mean) if mean > 0.0 else 0.0
        ordered = sorted(scores, reverse=True)
        signals[f"{name} top-two gap"] = ((ordered[0] - ordered[1]) / ordered[0]
                                          if len(ordered) >= 2 and ordered[0] > 0.0 else 0.0)

        # The absolute reading. For the vector arm the brief argues this is meaningful (a normalized
        # embedder calibrates cosine distance), and for the keyword arm it should not be (BM25 moves with
        # IDF and document length) — so the keyword row is a control, and it failing is the expected result.
        signals[f"{name} best score"] = best

    return signals


def report_auroc(title: str, rows: list[tuple[str, float, int, int]]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'signal':<28} {'AUROC':>7} {'pos':>5} {'neg':>5}")
    print("-" * 48)
    for name, score, n_pos, n_neg in rows:
        print(f"{name:<28} {score:>7.3f} {n_pos:>5} {n_neg:>5}")


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    if not QUESTIONS_PATH.exists():
        print(f"no question set at {QUESTIONS_PATH}; run make_questions.py first")
        return
    payload = json.loads(QUESTIONS_PATH.read_text())
    questions = payload["questions"]
    print(f"{len(questions)} known-item questions + {len(PROBES)} off-corpus probes, "
          f"against a {payload['corpus_size']}-record corpus, k={k}")

    executor = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file,
                          executor=executor)
    hybridir.init(executor=executor)
    retriever = hybridir.HybridIR(datastore_base_dir=librarian_config.llm_database_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)

    # `multi_query=False` throughout: both measurements are about the whole-message query. Per-subquery
    # gating is a separate question, and it is downstream of this one being answered at all.
    observations = []
    for i, item in enumerate(questions, start=1):
        results, report = retriever.query(item["question"], k=k,
                                          multi_query=False, return_extra_info=True)
        rank = rank_of_gold(results, set(item["gold"]))
        observations.append({"kind": item["kind"],
                             "query": item["question"],
                             "on_corpus": True,
                             "rank": rank,
                             "signals": signals_from_report(report)})
        print(f"  [{i}/{len(questions) + len(PROBES)}] {item['kind']:<9} "
              f"gold rank {rank if rank else '-':<4} {item['question'][:60]}")

    for j, (kind, probe) in enumerate(PROBES, start=1):
        _results, report = retriever.query(probe, k=k, multi_query=False, return_extra_info=True)
        observations.append({"kind": kind,
                             "query": probe,
                             "on_corpus": False,
                             "rank": None,
                             "signals": signals_from_report(report)})
        print(f"  [{len(questions) + j}/{len(questions) + len(PROBES)}] {kind:<9} "
              f"{'probe':<9} {probe[:60]}")

    signal_names = list(observations[0]["signals"])

    # --- Measurement A: does sharpness predict retrieval success? ---------------------------------
    summary = {}
    for population in ("all", "focused", "rambling"):
        subset = [o for o in observations
                  if o["on_corpus"] and (population == "all" or o["kind"] == population)]
        labels = [o["rank"] is not None for o in subset]
        rows = sorted(((name, auroc([o["signals"][name] for o in subset], labels),
                        sum(labels), len(labels) - sum(labels))
                       for name in signal_names),
                      key=lambda row: -row[1])
        summary[f"A: found vs. missed ({population})"] = rows
        report_auroc(f"A: gold document found vs. missed — {population} questions", rows)

    # --- Measurement B: does sharpness separate on-corpus from off-corpus? ------------------------
    for probe_kind in ("all probes", "off_corpus", "pleasantry", "adjacent"):
        subset = [o for o in observations
                  if o["on_corpus"] or probe_kind == "all probes" or o["kind"] == probe_kind]
        labels = [o["on_corpus"] for o in subset]
        rows = sorted(((name, auroc([o["signals"][name] for o in subset], labels),
                        sum(labels), len(labels) - sum(labels))
                       for name in signal_names),
                      key=lambda row: -row[1])
        summary[f"B: on-corpus vs. {probe_kind}"] = rows
        report_auroc(f"B: known-item question vs. {probe_kind}", rows)

    RESULTS_PATH.write_text(json.dumps(
        {"k": k,
         "n_questions": len(questions),
         "n_probes": len(PROBES),
         "corpus_size": payload["corpus_size"],
         "ratios": list(RATIOS),
         "auroc": {title: [{"signal": name, "auroc": score, "n_positive": n_pos, "n_negative": n_neg}
                           for name, score, n_pos, n_neg in rows]
                   for title, rows in summary.items()},
         "per_query": observations},
        indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote per-query signals to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
