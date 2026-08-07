"""Carve a topically contiguous slice out of an indexed corpus, as its own document directory.

**What this is for.** Corpus size and topical crowding move together across the corpora this investigation
has on hand — the 12k-record hydrogen collection is also the narrowest — so a result that looks like "large
corpora behave differently" cannot be told apart from "crowded corpora behave differently". Separating them
needs a corpus that is *small and still crowded*, which is what this builds.

**Why a ball and not a uniform sample.** Taking 2500 records at random from 12000 lowers the document count
*and* thins the crowding, in proportion, so both explanations predict the same improvement and the
experiment answers nothing. Taking one dense neighbourhood instead holds crowding roughly fixed — it is if
anything higher than the corpus average, which makes the test stringent in the right direction — while
cutting the count.

**Why a ball and not a cluster.** A ball needs no cluster count, no algorithm and no tuning: it is the
nearest N neighbours of one seed. Clustering would answer a question nobody asked here (*how many groups
are there*) at the cost of several that would then need answering.

**The seed matters more than it looks.** Seeding at a random document risks landing on an outlier, whose
"nearest neighbours" are merely the least distant of a crowd that is nowhere near it — a ball in name that
is really a uniform sample wearing one. Seeding on a topic known to be well represented avoids that, and
the tightness report below is what confirms it worked rather than assuming it did.

Usage:
    python make_subcorpus.py <source-corpus> <seed query> [--size 2500] [--name SUFFIX]

    python make_subcorpus.py hydrogen "photocatalytic water splitting" --size 2500 --name photocat

Writes `~/.config/raven/llmclient/documents_<source><SUFFIX>/`, ready for `raven-indexer`. The source
documents are copied, not moved; the original corpus is untouched.
"""

from __future__ import annotations

__all__ = ["nearest_documents", "build_subcorpus"]

import pathlib
import shutil
import statistics
import sys

DEFAULT_SIZE = 2500


def nearest_documents(retriever, seed_text: str, size: int) -> tuple[list[str], list[float]]:
    """The `size` document ids nearest `seed_text`, with the vector distance each was reached at.

    A document ranks at its *best* chunk, which is the rule the shipped retrieval path uses, so the slice
    is the one a reader following the search would have seen. Asks for several chunks per document wanted,
    since an abstract chunks into one or two and a fulltext into many.
    """
    best: dict[str, float] = {}
    depth = size * 3
    _merged, rep = retriever.query(seed_text, k=depth, multi_query=False, return_extra_info=True)
    for record, distance in zip(rep.vector_results, rep.vector_distances):
        key = record["document_id"]
        distance = float(distance)
        if key not in best or distance < best[key]:
            best[key] = distance
    ordered = sorted(best.items(), key=lambda kv: kv[1])[:size]
    return [key for key, _d in ordered], [d for _key, d in ordered]


def build_subcorpus(source_docs: pathlib.Path, target_docs: pathlib.Path,
                    document_ids: list[str]) -> tuple[int, int]:
    """Copy `document_ids` from `source_docs` into a fresh `target_docs`. Returns `(copied, missing)`."""
    target_docs.mkdir(parents=True, exist_ok=True)
    copied = missing = 0
    for document_id in document_ids:
        source = source_docs / document_id
        if not source.exists():
            missing += 1
            continue
        shutil.copy2(source, target_docs / source.name)
        copied += 1
    return copied, missing


def main() -> None:  # pragma: no cover
    argv = sys.argv[1:]
    size, name = DEFAULT_SIZE, "subset"
    if "--size" in argv:
        at = argv.index("--size")
        size = int(argv[at + 1])
        del argv[at:at + 2]
    if "--name" in argv:
        at = argv.index("--name")
        name = argv[at + 1]
        del argv[at:at + 2]
    if len(argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    source, seed_text = argv[0], " ".join(argv[1:])

    from raven.client import api as client_api
    from raven.client import config as client_config
    from raven.librarian import config as librarian_config
    from raven.librarian import hybridir
    import concurrent.futures

    base = pathlib.Path("~/.config/raven/llmclient").expanduser()
    source_docs = base / f"documents_{source}"
    target_docs = base / f"documents_{source}_{name}"
    if target_docs.exists():
        raise SystemExit(f"{target_docs} already exists; move it aside first rather than mixing two slices")

    executor = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file, executor=executor)
    retriever = hybridir.HybridIR(datastore_base_dir=base / f"rag_index_{source}",
                                  embedding_model_name=librarian_config.qa_embedding_model)
    print(f"source '{source}': documents at {source_docs}")
    print(f"seed: {seed_text!r}")

    document_ids, distances = nearest_documents(retriever, seed_text, size)
    if not document_ids:
        raise SystemExit("no documents matched the seed; is the index built?")

    # How tight the ball came out. A slice whose farthest member sits about as far away as an unrelated
    # document would is not a neighbourhood, and the experiment it was built for would measure nothing.
    print(f"\n  selected {len(document_ids)} documents")
    print(f"  vector distance: nearest {distances[0]:.3f}, median {statistics.median(distances):.3f}, "
          f"farthest {distances[-1]:.3f}")
    print("  a tight ball has the farthest member well inside the corpus's typical distance; a flat "
          "profile means the seed was an outlier and this is a uniform sample in disguise.")

    copied, missing = build_subcorpus(source_docs, target_docs, document_ids)
    print(f"\n  copied {copied} documents to {target_docs}" + (f" ({missing} missing)" if missing else ""))
    print(f"\nNext: raven-indexer {target_docs} -d {base / f'rag_index_{source}_{name}'}")
    executor.shutdown(wait=False)


if __name__ == "__main__":  # pragma: no cover
    main()
