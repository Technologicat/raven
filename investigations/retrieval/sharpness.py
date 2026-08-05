#!/usr/bin/env python3
"""Does a retrieval confidence signal mean anything, and does its threshold survive a change of corpus?

Brief 09's lever 1 proposes reading a query's confidence off its own scores, before fusion flattens every
score into a rank. The design argument is in the brief and in `hybridir.score_sharpness`; this script is
what decides whether a candidate signal discriminates, and at what value. Nothing in Librarian's retrieval
path consumes any of it until this says it should.

Two questions, because the consumers want different things:

**A — does the signal predict retrieval success?** Over the known-item questions, is it higher where the
gold document was found than where it was missed? This is what adaptive `k` and per-subquery gating need,
both of which are decisions about a query the corpus *can* answer.

**B — does the signal separate on-corpus from off-corpus?** Known-item questions are written *from* the
corpus, so every one is answerable and A structurally cannot see the case brief 10's grounding marker
exists for. So the questions are scored against negatives with no answer in the indexed corpus.

Both are reported as **AUROC**: the probability that a randomly chosen positive scores above a randomly
chosen negative, ties counted half. 0.5 is a coin flip. Several candidate signals are scored side by side,
including constant-free alternatives to the `min_p`-style shape reading, so the sweep is not compared only
against itself.

**The corpus argument is the point of the second run.** Measured on hydrogen abstracts alone, the winning
signal was the absolute best vector similarity with a cut near 0.45. The standing objection to any such
constant is that the scale of "close" belongs to the collection — which one corpus cannot test. So:

    hydrogen   the Web of Science corpus is indexed. Positives are `questions.json`; negatives are the
               fiction questions (none of those stories is in this index) plus the built-in probes.
    fiction    the Optimalverse corpus is indexed. Positives are the `on_corpus` entries of
               `fiction_questions.json`; negatives are its `adjacent` entries — questions written from
               stories deliberately held out of the index, which is as hard as a negative gets — plus all
               the hydrogen questions and the built-in probes.

**Whichever corpus is actually indexed has to match the argument**, and nothing here can check that for
you: pointing this at the wrong index silently relabels every question. Both directions need a full
re-index, so run one, score it, then swap.

Requires a running raven-server (spaCy tokenization + embeddings) and the local document index. It reads
the index; it does not write to it.

Usage:
    python sharpness.py <hydrogen|fiction> [k]
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
HYDROGEN_QUESTIONS = HERE / "questions.json"
FICTION_QUESTIONS = HERE / "fiction_questions.json"

# Ratios to sweep. Wide and coarse: the brief is explicit that the values in circulation for `min_p`
# sampling carry no information about what to use here, so this is a search rather than a refinement.
RATIOS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)

# Hand-written negatives, kept from the first run so the two corpora are scored against a common set as
# well as against each other. Three kinds, and note that the labels are written from the hydrogen corpus's
# point of view: `adjacent_science` is genuinely near-miss for a corpus of electrolysis abstracts and
# plainly off-topic for one of fan fiction. The fiction run has its own, much harder adjacent group — the
# held-out stories — so nothing here needs to stand in for it.
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
          ("adjacent_science", "What is the tensile strength of carbon fibre composites?"),
          ("adjacent_science", "How does CRISPR gene editing work in plants?"),
          ("adjacent_science", "What are the failure modes of lithium iron phosphate cells?"),
          ("adjacent_science", "How is atmospheric methane measured from satellites?")]


def load_json(path: pathlib.Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def build_workload(corpus: str) -> tuple[list[dict], dict]:
    """Return the queries to run and a note about the corpus, or raise if the question sets are missing.

    Each item is `{"kind", "query", "on_corpus", "gold"}`. `kind` names the group it is reported under;
    `on_corpus` is the label measurement B discriminates on.
    """
    hydrogen = load_json(HYDROGEN_QUESTIONS)
    fiction = load_json(FICTION_QUESTIONS)

    if corpus == "hydrogen":
        if hydrogen is None:
            raise SystemExit(f"no question set at {HYDROGEN_QUESTIONS}; run make_questions.py first")
        items = [{"kind": q["kind"], "query": q["question"], "on_corpus": True, "gold": q["gold"]}
                 for q in hydrogen["questions"]]
        if fiction is not None:
            items += [{"kind": "other_corpus", "query": q["question"], "on_corpus": False, "gold": []}
                      for q in fiction["questions"]]
        note = {"corpus_size": hydrogen["corpus_size"], "questions_from": str(HYDROGEN_QUESTIONS)}
    elif corpus == "fiction":
        if fiction is None:
            raise SystemExit(f"no question set at {FICTION_QUESTIONS}; run make_fiction_questions.py first")
        items = [{"kind": q["kind"], "query": q["question"], "on_corpus": q["on_corpus"], "gold": q["gold"]}
                 for q in fiction["questions"]]
        if hydrogen is not None:
            items += [{"kind": "other_corpus", "query": q["question"], "on_corpus": False, "gold": []}
                      for q in hydrogen["questions"]]
        note = {"corpus_dir": fiction.get("corpus_dir"), "questions_from": str(FICTION_QUESTIONS)}
    else:
        raise SystemExit(f"unknown corpus '{corpus}'; expected 'hydrogen' or 'fiction'")

    items += [{"kind": kind, "query": probe, "on_corpus": False, "gold": []} for kind, probe in PROBES]
    return items, note


def rank_of_gold(results: list[dict], gold: set[str]) -> int | None:
    """1-based rank of the first result whose document is in `gold`, or `None` if absent."""
    for rank, result in enumerate(results, start=1):
        if result["document_id"] in gold:
            return rank
    return None


def auroc(values: list[float], labels: list[bool]) -> float:
    """Probability that a positive outranks a negative, ties counted half. 0.5 discriminates nothing.

    Equivalently the Mann-Whitney U statistic, normalized. Computed pairwise because the sample is a few
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


def report_auroc(title: str, rows: list[tuple[str, float, int, int]], limit: int | None = None) -> None:
    print(f"\n=== {title} ===")
    print(f"{'signal':<28} {'AUROC':>7} {'pos':>5} {'neg':>5}")
    print("-" * 48)
    for name, score, n_pos, n_neg in (rows[:limit] if limit else rows):
        print(f"{name:<28} {score:>7.3f} {n_pos:>5} {n_neg:>5}")


def report_distribution(observations: list[dict], signal: str) -> None:
    """Where each group's values actually sit — which is what a threshold has to be chosen from."""
    print(f"\n=== distribution of '{signal}', by group ===")
    print(f"{'group':<18} {'n':>4} {'min':>7} {'p25':>7} {'median':>7} {'max':>7}")
    print("-" * 56)
    groups: dict[str, list[float]] = collections.defaultdict(list)
    for o in observations:
        groups["ON-CORPUS" if o["on_corpus"] else o["kind"]].append(o["signals"][signal])
    for key in sorted(groups, key=lambda k: (k != "ON-CORPUS", k)):
        values = sorted(groups[key])
        print(f"{key:<18} {len(values):>4} {values[0]:>7.3f} {values[len(values) // 4]:>7.3f} "
              f"{statistics.median(values):>7.3f} {values[-1]:>7.3f}")


def report_cuts(observations: list[dict], signal: str) -> list[dict]:
    """What each candidate threshold would cost and buy. The operating point is chosen from this table."""
    on = sorted(o["signals"][signal] for o in observations if o["on_corpus"])
    off = sorted(o["signals"][signal] for o in observations if not o["on_corpus"])
    print(f"\n=== candidate cuts on '{signal}' ===")
    print(f"{'cut':>6} {'on-corpus rejected':>20} {'negatives rejected':>20}")
    print("-" * 48)
    rows = []
    for cut in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        false_negatives = sum(1 for v in on if v < cut)
        true_negatives = sum(1 for v in off if v < cut)
        rows.append({"cut": cut, "on_corpus_rejected": false_negatives, "on_corpus_total": len(on),
                     "negatives_rejected": true_negatives, "negatives_total": len(off)})
        print(f"{cut:>6.2f} {f'{false_negatives} / {len(on)}':>20} {f'{true_negatives} / {len(off)}':>20}")
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    corpus = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    items, note = build_workload(corpus)
    n_positive = sum(1 for i in items if i["on_corpus"])
    print(f"corpus '{corpus}': {n_positive} on-corpus questions, {len(items) - n_positive} negatives, k={k}")
    print(f"  {note}")

    executor = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file,
                          executor=executor)
    hybridir.init(executor=executor)
    retriever = hybridir.HybridIR(datastore_base_dir=librarian_config.llm_database_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)
    with retriever.datastore_lock:
        print(f"  live index holds {len(retriever.documents)} documents")

    # `multi_query=False` throughout: both measurements are about the whole-message query. Per-subquery
    # gating is a separate question, and it is downstream of this one being answered at all.
    observations = []
    for i, item in enumerate(items, start=1):
        results, report = retriever.query(item["query"], k=k, multi_query=False, return_extra_info=True)
        rank = rank_of_gold(results, set(item["gold"])) if item["gold"] else None
        observations.append({**item, "rank": rank, "signals": signals_from_report(report)})
        print(f"  [{i}/{len(items)}] {item['kind']:<16} {'gold rank ' + str(rank) if item['gold'] else 'negative':<14} "
              f"{item['query'][:56]}")

    signal_names = list(observations[0]["signals"])
    summary = {}

    def ranked(subset: list[dict], labels: list[bool]) -> list[tuple[str, float, int, int]]:
        return sorted(((name, auroc([o["signals"][name] for o in subset], labels),
                        sum(labels), len(labels) - sum(labels))
                       for name in signal_names),
                      key=lambda row: -row[1])

    # --- Measurement A: does the signal predict retrieval success? --------------------------------
    for population in ("all", "focused", "rambling"):
        subset = [o for o in observations
                  if o["on_corpus"] and o["gold"] and (population == "all" or o["kind"] == population)]
        if not subset:
            continue
        labels = [o["rank"] is not None for o in subset]
        if not any(labels) or all(labels):
            print(f"\n(A skipped for {population}: every question {'found' if all(labels) else 'missed'} "
                  f"its gold document — no contrast to measure. Expected on a small corpus, where k >= the "
                  f"document count makes known-item retrieval trivial.)")
            continue
        rows = ranked(subset, labels)
        summary[f"A: found vs. missed ({population})"] = rows
        report_auroc(f"A: gold document found vs. missed — {population} questions", rows, limit=8)

    # --- Measurement B: does the signal separate on-corpus from off-corpus? ------------------------
    negative_kinds = sorted({o["kind"] for o in observations if not o["on_corpus"]})
    for group in ["all negatives", *negative_kinds]:
        subset = [o for o in observations
                  if o["on_corpus"] or group == "all negatives" or o["kind"] == group]
        labels = [o["on_corpus"] for o in subset]
        rows = ranked(subset, labels)
        summary[f"B: on-corpus vs. {group}"] = rows
        report_auroc(f"B: on-corpus question vs. {group}", rows, limit=6)

    best_signal = summary[f"B: on-corpus vs. {'all negatives'}"][0][0]
    print(f"\nbest separator overall: {best_signal}")
    report_distribution(observations, best_signal)
    cuts = report_cuts(observations, best_signal)
    # The absolute vector reading is what the hydrogen run selected, so always show it too — otherwise a run
    # where something else wins by a hair silently stops reporting the number the two corpora are compared on.
    if best_signal != "vector best score":
        report_distribution(observations, "vector best score")
        report_cuts(observations, "vector best score")

    out_path = HERE / f"sharpness_results_{corpus}.json"
    out_path.write_text(json.dumps(
        {"corpus": corpus, "k": k, "note": note, "ratios": list(RATIOS),
         "n_positive": n_positive, "n_negative": len(items) - n_positive,
         "best_separator": best_signal,
         "cuts": cuts,
         "auroc": {title: [{"signal": name, "auroc": score, "n_positive": n_pos, "n_negative": n_neg}
                           for name, score, n_pos, n_neg in rows]
                   for title, rows in summary.items()},
         "per_query": observations},
        indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote per-query signals to {out_path}")


if __name__ == "__main__":
    main()
