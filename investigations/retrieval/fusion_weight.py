"""Is the 50:50 blend of the two retrieval arms the right one, and is the answer the same on every corpus?

Everything measured so far treats fusion as a switch: BM25, vector, or RRF over both with equal votes.
That is three points on a continuum nobody has swept. It matters because the one corpus where fusion
*loses* fails for a reason a weight would fix — on banichuk the keyword arm scores MRR 0.090 against the
vector arm's 0.201, and blending them equally drags the good arm down to 0.169. "Use less of the bad arm"
is the obvious response and has never been measured.

The per-query version of this question is closed: no signal we have predicts which arm will win a given
query (AUROC ~0.53 where the headroom is). A *per-corpus* weight needs no such predictor. It asks the
strictly easier question of whether one arm deserves more of the vote in this collection than in that one.

**Two phases, because the expensive half is not the interesting half.** `sweep` runs retrieval once per
question and records both arms' full ranked lists; `report` then computes weighted RRF for any weight, any
RRF constant and any `k` as pure arithmetic over that record. So the grid is free after the first pass, and
a new question about the same data costs no GPU at all.

    python fusion_weight.py sweep <corpus> [--db-dir DIR] [--depth 100]
    python fusion_weight.py report [corpus ...]

Weighted RRF here is `score(d) = w/(K + rank_bm25(d)) + (1-w)/(K + rank_vector(d))`, missing ranks
contributing nothing. One free parameter rather than two: RRF scores are compared only against each other,
so scaling both weights leaves the order untouched and only the ratio can matter. `w=1` is the keyword arm
alone, `w=0` the semantic arm alone, `w=0.5` what ships.
"""

from __future__ import annotations

__all__ = ["WEIGHTS", "K_CONSTANTS", "weighted_rrf", "gold_rank", "load_sweep", "mcnemar"]

import concurrent.futures
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sharpness  # noqa: E402

DEFAULT_DEPTH = 100

# Swept rather than reasoned about. Fine near the ends because that is where a corpus with one weak arm
# would land, and a coarse middle because 0.4-0.6 is one decision ("roughly equal") wearing three hats.
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# The RRF constant damps how much rank 1 outweighs rank 10; 60 is the literature default and what ships.
# Swept alongside the weight because the two interact — a small K makes the head of each list dominate, so
# it changes what a given weight actually buys. Extended below 10 once smaller values kept winning: the
# limit K -> 0 is plain reciprocal rank, where a rank-1 hit outvotes a rank-2 hit two to one, so a trend
# that runs all the way down says something different from an optimum in the middle.
K_CONSTANTS = (1, 3, 10, 30, 60, 120)


def weighted_rrf(kw: list[int], vec: list[int], w: float, K: int) -> list[int]:
    """Fuse two ranked lists of document indices into one, best first.

    `w` is the keyword arm's share of the vote and `1 - w` the semantic arm's. A document absent from a
    list contributes nothing from it, which is what makes `w=0` and `w=1` degenerate to the single arms
    exactly rather than approximately.
    """
    scores: dict[int, float] = {}
    for lst, weight in ((kw, w), (vec, 1.0 - w)):
        if weight == 0.0:
            continue
        for rank, did in enumerate(lst, 1):
            scores[did] = scores.get(did, 0.0) + weight / (K + rank)
    return [did for did, _score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def gold_rank(ids: list[int], gold: set[int]) -> int | None:
    """1-based position of the first gold document, or `None` if it is not in the list at all."""
    for i, did in enumerate(ids, 1):
        if did in gold:
            return i
    return None


def mcnemar(a: list[int | None], b: list[int | None], k: int = 20) -> tuple[int, int, float]:
    """Paired exact test of "gold within top `k`" between two orderings of the same retrievals.

    Returns `(gained, lost, p)` — questions `b` rescues that `a` missed, questions `b` loses that `a` had,
    and the two-sided exact binomial p over those discordant pairs alone. Every question where both agree
    is discarded, which is the point: the conditions share the retrieval, so the agreements carry no
    information about the difference and counting them would only dilute it.

    Exact rather than the chi-square approximation because the discordant counts here run to single digits,
    where the approximation is not to be trusted.
    """
    gained = sum(1 for x, y in zip(a, b) if not (x and x <= k) and (y and y <= k))
    lost = sum(1 for x, y in zip(a, b) if (x and x <= k) and not (y and y <= k))
    n = gained + lost
    if n == 0:
        return (0, 0, 1.0)
    # Two-sided exact binomial at p=0.5: total probability of a split at least as lopsided as this one.
    def comb(n, r):
        out = 1
        for i in range(r):
            out = out * (n - i) // (i + 1)
        return out
    observed = min(gained, lost)
    tail = sum(comb(n, i) for i in range(observed + 1)) / (2 ** n)
    return (gained, lost, min(1.0, 2 * tail))


def paired_coverage(a: list[float], b: list[float]) -> tuple[int, int, float, float]:
    """Paired test of passage coverage between two orderings of the same retrievals.

    Returns `(improved, worsened, mean_delta, p)` — the number of questions where `b` covers more of the
    gold passage than `a` and fewer, the mean difference in coverage over *all* questions, and the
    two-sided p over the questions where the two differ.

    The companion to `mcnemar`, for the corpora where that test has nothing to work with. Document recall
    is a coin flip per question, and on a 19-document corpus at `k=20` it comes up heads every time: every
    condition scores 100%, every pair is concordant, and the test correctly reports that it has seen no
    evidence. Coverage is continuous and does not saturate, so it still separates conditions there — but a
    difference in mean coverage is not evidence on its own, which is exactly the trap this exists to close.

    Questions where the two conditions cover identically are discarded, on the same reasoning `mcnemar`
    discards concordant pairs: the conditions share the retrieval, so an agreement says nothing about the
    difference between them and counting it only dilutes what does.

    The test itself is Wilcoxon signed-rank — the paired, distribution-free one, which reads the *ranks* of
    the differences rather than their sizes. That matters here because coverage differences are bounded,
    lumpy (a chunk enters the top 20 or it does not) and nowhere near normal, so a paired t-test would be
    assuming a shape the data does not have.
    """
    deltas = [y - x for x, y in zip(a, b)]
    if not deltas:
        return (0, 0, 0.0, 1.0)
    mean_delta = sum(deltas) / len(deltas)
    nonzero = [d for d in deltas if d != 0.0]
    improved = sum(1 for d in nonzero if d > 0)
    worsened = sum(1 for d in nonzero if d < 0)
    if not nonzero:
        return (0, 0, mean_delta, 1.0)
    import scipy.stats  # local: only this function needs it, and the scripts around it are import-light
    p = float(scipy.stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)
    return (improved, worsened, mean_delta, p)


def dedup_ids(seq) -> list[str]:
    """Document keys in first-appearance order — a document ranks where its best chunk did.

    Keyed by `sharpness.document_key` (the id without its extension), so a gold label naming one
    representation of a document matches the same document indexed as another — the abstract and fulltext
    arXiv corpora hold the same papers as `.bib` and `.pdf`. Comparing raw ids across that pair reports a
    clean zero rather than failing.
    """
    out: list[str] = []
    for x in seq:
        did = x.get("document_id") if isinstance(x, dict) else None
        if did is not None:
            key = sharpness.document_key(did)
            if key not in out:
                out.append(key)
    return out


def sweep(corpus: str, db_dir: pathlib.Path, depth: int) -> pathlib.Path:  # pragma: no cover
    """Retrieve once per question and record both arms' rankings, plus what ships, plus the gold labels.

    Document IDs are stored as indices into a per-question vocabulary rather than as filenames. The file is
    read only by `report`, which never needs the names, and at depth 100 across five corpora the repeated
    strings would otherwise be most of the bytes.
    """
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
    print(f"  index: {db_dir}")

    rows = []
    for n, item in enumerate(items, 1):
        merged, rep = retriever.query(item["query"], k=depth, multi_query=False, return_extra_info=True)
        kw_ids = dedup_ids(rep.keyword_results)
        vec_ids = dedup_ids(rep.vector_results)
        shipped_ids = dedup_ids(merged)

        vocab: dict[str, int] = {}

        def index_of(did: str) -> int:
            if did not in vocab:
                vocab[did] = len(vocab)
            return vocab[did]

        row = {"kw": [index_of(d) for d in kw_ids],
               "vec": [index_of(d) for d in vec_ids],
               "shipped": [index_of(d) for d in shipped_ids],
               "gold": [index_of(sharpness.document_key(d)) for d in item["gold"]]}
        rows.append(row)
        if n % 20 == 0:
            print(f"  [{n}/{len(items)}]", flush=True)

    out = pathlib.Path(__file__).parent / f"fusion_weight_{corpus}.json"
    out.write_text(json.dumps({"corpus": corpus, "depth": depth, "rows": rows}, indent=1), encoding="utf-8")
    print(f"  wrote {out}")
    return out


def load_sweep(corpus: str) -> dict:
    path = pathlib.Path(__file__).parent / f"fusion_weight_{corpus}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def report(corpus: str) -> None:  # pragma: no cover
    data = load_sweep(corpus)
    rows, depth = data["rows"], data["depth"]
    n = len(rows)
    miss_rank = depth + 1

    def score(ranks: list[int | None]) -> tuple[float, float, float]:
        """`(recall@20, recall@50, MRR)` — the two operating points that ship, plus the head-weighted mean."""
        rec20 = sum(1 for r in ranks if r and r <= 20) / n
        rec50 = sum(1 for r in ranks if r and r <= 50) / n
        mrr = sum(1.0 / r for r in ranks if r) / n
        return rec20, rec50, mrr

    def ranks_for(w: float, K: int) -> list[int | None]:
        return [gold_rank(weighted_rrf(row["kw"], row["vec"], w, K), set(row["gold"])) for row in rows]

    shipped = [gold_rank(row["shipped"], set(row["gold"])) for row in rows]
    oracle = []
    for row in rows:
        gold = set(row["gold"])
        a, b = gold_rank(row["kw"], gold), gold_rank(row["vec"], gold)
        oracle.append(min(a or miss_rank, b or miss_rank) if (a or b) else None)

    print(f"\n=== {corpus} (n={n}, depth={depth}) ===")
    print(f"  {'w(bm25)':>8}" + "".join(f"{'K=' + str(K):>26}" for K in K_CONSTANTS))
    print(f"  {'':>8}" + "".join(f"{'@20':>8}{'@50':>9}{'MRR':>9}" for _K in K_CONSTANTS))
    best = None
    for w in WEIGHTS:
        cells = ""
        for K in K_CONSTANTS:
            rec20, rec50, mrr = score(ranks_for(w, K))
            cells += f"{rec20:>8.1%}{rec50:>9.1%}{mrr:>9.3f}"
            if best is None or mrr > best[2]:
                best = (w, K, mrr, rec20, rec50)
        print(f"  {w:>8.1f}{cells}")

    s20, s50, smrr = score(shipped)
    o20, o50, omrr = score(oracle)
    print(f"\n  shipped (merged spans, w=0.5, K=60): @20 {s20:.1%}  @50 {s50:.1%}  MRR {smrr:.3f}")
    print(f"  best swept point: w={best[0]:.1f} K={best[1]}  @20 {best[3]:.1%}  @50 {best[4]:.1%}  MRR {best[2]:.3f}")
    print(f"  per-query arm oracle (cheats):       @20 {o20:.1%}  @50 {o50:.1%}  MRR {omrr:.3f}")

    # The grid above was optimized on these same questions, so its winning cell is not a result. These two
    # comparisons are, because each was named as a hypothesis before the numbers were read: does the RRF
    # constant matter at the *shipped* weight, and does the shipped path's chunk-level fusion plus span
    # merging cost anything against plain document-level fusion?
    print("\n  paired tests at the shipped weight (w=0.5), gold within top 20:")
    for label, a, b in (("K=60 -> K=10", ranks_for(0.5, 60), ranks_for(0.5, 10)),
                        ("shipped -> document-level RRF, K=60", shipped, ranks_for(0.5, 60))):
        gained, lost, p = mcnemar(a, b)
        print(f"    {label:<38} gained {gained:>2}  lost {lost:>2}  p = {p:.3f}")


def main() -> None:  # pragma: no cover
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return
    mode, argv = argv[0], argv[1:]

    if mode == "report":
        for corpus in (argv or ["hydrogen"]):
            report(corpus)
        return

    if mode != "sweep":
        print(f"unknown mode '{mode}'; expected 'sweep' or 'report'")
        return

    depth = DEFAULT_DEPTH
    db_dir = None
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
        db_dir = pathlib.Path.home() / f".config/raven/librarian/rag_index_{corpus}"
    sweep(corpus, db_dir, depth)


if __name__ == "__main__":
    main()
