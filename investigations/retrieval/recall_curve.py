"""Read recall@k off the gold ranks recorded by `sharpness.py`.

This sizes the candidate stage of a retrieve-deep-then-rerank pipeline, which is the one number the
reranker design actually waits on. A cross-encoder can only reorder what the first stage handed it, so
recall@k at the depth we are willing to rerank is the ceiling on everything downstream — a reranker fed
a candidate set that does not contain the gold document cannot do anything but reorder wrong answers.

Reads the `per_query` records `sharpness.py` writes, so it needs no index, no server and no GPU. Run
the sweep once at a depth you will not want to exceed, then read every shallower k off the same file:

    python sharpness.py hydrogen 200 --db-dir ~/.config/raven/llmclient/rag_index_hydrogen
    python recall_curve.py sharpness_results_hydrogen.json

**Only on-corpus questions count.** A negative has no gold document by construction, so including one
would be scoring a question that cannot be answered right — it would depress every number by a constant
that says nothing about retrieval.

**`rank` is censored at the sweep's k.** `sharpness.py` records `None` both for "gold is below rank k"
and for "gold is not in the corpus at all", and nothing in the file distinguishes them. So the curve is
valid up to the sweep depth and silent past it: a run at k=20 cannot tell you recall@50, and reading its
flat tail as saturation is the mistake this file exists to prevent. The reported depth is printed with
the curve for that reason.

**Recall@k is a fraction of a corpus, not an absolute.** Raven retrieves chunks, so the denominator is
the corpus chunk count — k=200 is 0.3% of hydrogen's ~60000 chunks and 37% of the hand-built BibTeX set.
At the second figure "retrieve deep" stops meaning selective retrieval and starts meaning "read most of
the corpus", where a high recall says little. Chunk counts are recorded in the investigation README, not
here, since this file only sees ranks.
"""

from __future__ import annotations

__all__ = [
    "DEPTHS",
    "recall_at",
    "curve",
]

import json
import sys
from pathlib import Path

DEPTHS = (1, 5, 10, 20, 50, 100, 200)


def recall_at(ranks: list[int | None], k: int) -> float:
    """Fraction of questions whose gold document is at rank `k` or better."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def curve(records: list[dict], depths: tuple[int, ...] = DEPTHS) -> list[tuple[int, float, float]]:
    """Return [(k, recall@k, gain over the previous depth), ...] for on-corpus questions."""
    ranks = [record["rank"] for record in records]
    out = []
    previous = 0.0
    for k in depths:
        value = recall_at(ranks, k)
        out.append((k, value, value - previous))
        previous = value
    return out


def _report(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    swept_to = data["k"]
    on_corpus = [record for record in data["per_query"] if record["on_corpus"] and record["gold"]]
    depths = tuple(k for k in DEPTHS if k <= swept_to)

    print()
    print(f"{data['corpus']}  ({len(on_corpus)} on-corpus questions, swept to k={swept_to})")
    print()
    print(f"  {'k':>5}  {'recall@k':>9}  {'gain':>7}")
    for k, value, gain in curve(on_corpus, depths):
        print(f"  {k:>5}  {value:>8.1%}  {gain:>+7.1%}")

    kinds = sorted({record["kind"] for record in on_corpus})
    if len(kinds) > 1:
        print()
        print(f"  by question kind ({', '.join(kinds)}):")
        for kind in kinds:
            subset = [r for r in on_corpus if r["kind"] == kind]
            cells = "  ".join(f"@{k}:{value:>5.1%}" for k, value, _gain in curve(subset, depths))
            print(f"    {kind:<16} n={len(subset):<4} {cells}")

    missed = sum(1 for record in on_corpus if record["rank"] is None)
    if missed:
        print()
        print(f"  {missed} question(s) never found their gold document within k={swept_to} "
              f"({missed / len(on_corpus):.1%}) — the ceiling no reranker can lift.")
    print()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    paths = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        here = Path(__file__).parent
        paths = sorted(here.glob("sharpness_results_*.json"))
    if not paths:
        print(__doc__)
        return
    for path in paths:
        _report(path)


if __name__ == "__main__":
    main()
