"""Check that a migrated index gave each chunk its own tokens back.

Usage: python check_alignment.py <index directory>

The migration tokenizes in batches that cross document boundaries and then deals the tokens back out, so an
off-by-one there would hand a chunk its neighbour's tokens without anything else noticing. For a sample of
chunks this asks whether most of each chunk's tokens occur in that chunk's own text, and then asks the same
of the tokens shifted by one chunk — the negative control, without which a clean result could mean only
that the check cannot tell the difference.
"""

import json
import pathlib
import random
import re
import sys

SAMPLE = 850


def flagged(pairs) -> int:
    """How many (chunk, tokens) pairs have fewer than half their tokens in the chunk's own text."""
    count = 0
    for chunk, tokens in pairs:
        words = set(re.findall(r"[a-z0-9]+", chunk["text"].lower()))
        if tokens and len([t for t in tokens if t in words]) < 0.5 * len(tokens):
            count += 1
    return count


def main() -> None:
    data_file = pathlib.Path(sys.argv[1]) / "fulldocs" / "data.json"
    docs = list(json.loads(data_file.read_text(encoding="utf-8"))["documents"].values())
    print(f"documents whose token-list count differs from their chunk count: "
          f"{sum(len(d['tokens']) != len(d['chunks']) for d in docs)} of {len(docs)}")

    pairs = [(chunk, tokens) for doc in docs for chunk, tokens in zip(doc["chunks"], doc["tokens"])]
    shifted = [(pairs[i][0], pairs[i - 1][1]) for i in range(1, len(pairs))]
    random.seed(1)
    n = min(SAMPLE, len(shifted))
    print(f"as stored:           {flagged(random.sample(pairs, n))} of {n} sampled chunks flagged")
    print(f"shifted by one chunk: {flagged(random.sample(shifted, n))} of {n} flagged (should be most)")


if __name__ == "__main__":
    main()
