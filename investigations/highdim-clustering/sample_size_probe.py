#!/usr/bin/env python
"""Does describing a large cluster by 60 of its entries lose anything that 200 would find?

The importer's LLM keyword pass sends at most `max_prompt_entries` (60) of a cluster's entries, spread evenly
from its center outwards. On a cluster of a thousand or more that is a sparse sample, and the question is
whether the keywords would change with a larger one.

The model is not deterministic, so one run per setting would compare noise with noise. Each setting runs
`--replicates` times, and the reading is whether two runs at *different* sample sizes disagree more than two
runs at the *same* size do. Agreement is the Jaccard index of two keyword sets, case-folded.

Every turn is saved whole to `--out` as JSONL, thinking trace included, and streamed to `--stream` as it
runs, so a turn that runs away still leaves its text behind.

    python sample_size_probe.py --vectors V1.npz [V2.npz ...] --dataset D.pickle --out turns.jsonl \\
        --stream turns.log [--sizes 60 200] [--replicates 3] [--clusters N]
"""

import argparse
import itertools
import json
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import clusterlab


def pick(original, members, n):
    """The importer's sample: `n` members spread evenly along their order of closeness to the cluster's center."""
    center = clusterlab.normalize(original[members].mean(axis=0, keepdims=True))[0]
    ordered = members[np.argsort(-(original[members] @ center))]
    if len(ordered) <= n:
        return ordered
    return ordered[np.linspace(0, len(ordered) - 1, n).round().astype(int)]


def keywords_of(reply):
    reply = (reply or "").strip()
    if reply.lower() == "keyword extraction failed":
        return []
    return [k.strip() for k in reply.split(",") if k.strip()]


def jaccard(a, b):
    a, b = {k.casefold() for k in a}, {k.casefold() for k in b}
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", nargs="+", required=True, help="one corpus; several caches are concatenated")
    parser.add_argument("--dataset", required=True, help="dataset aligned with --vectors, for titles and abstracts")
    parser.add_argument("--cut", type=int, default=100)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60, 200])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--clusters", type=int, default=2, help="how many of the largest clusters to probe")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--backend-url", default=None)
    args = parser.parse_args()

    from raven.librarian import agent, config as librarian_config, llmclient
    from raven.visualizer import config as visualizer_config

    original = clusterlab.normalize(np.concatenate([clusterlab.load_vectors(v)[0] for v in args.vectors]))
    entries = clusterlab.load_entries(args.dataset)
    if len(entries) != len(original):
        sys.exit(f"dataset has {len(entries)} entries, vectors {len(original)}: not aligned")
    labels = AgglomerativeClustering(n_clusters=args.cut, metric="cosine",
                                     linkage="average").fit_predict(clusterlab.center(original))
    cluster_ids = sorted(np.unique(labels), key=lambda c: -np.sum(labels == c))[:args.clusters]

    llm_settings = llmclient.setup(backend_url=args.backend_url or librarian_config.llm_backend_url, quiet=True)
    print(f"probing with {llm_settings.model_id}", file=sys.stderr)
    prompt_head = visualizer_config.clusters_llm_keyword_extraction_prompt

    results = {}  # (cluster, size) -> [keyword list per replicate]
    with open(args.out, "w", encoding="utf-8") as out, open(args.stream, "w", encoding="utf-8") as stream:
        for cid in cluster_ids:
            members = np.flatnonzero(labels == cid)
            for size in args.sizes:
                picks = pick(original, members, size)
                body = "\n\n\n".join(clusterlab.format_for_keyword_extraction(*entries[i]).strip() for i in picks)
                for replicate in range(args.replicates):
                    label = f"cluster {cid} ({len(members)} entries), {len(picks)} sent, replicate {replicate + 1}"
                    print(label, file=sys.stderr, flush=True)
                    record = agent.turn(llm_settings, f"{prompt_head}\n-----\n\n{body}",
                                        use_character_card=False, tools_enabled=False, internet_enabled=False,
                                        docs_enabled=False, markup=None,
                                        on_progress=agent.stream_log(stream, label))
                    keywords = keywords_of(record.reply)
                    results.setdefault((int(cid), size), []).append(keywords)
                    out.write(json.dumps({"cluster": int(cid), "cluster_size": len(members), "sent": len(picks),
                                          "replicate": replicate + 1, "keywords": keywords,
                                          "turn": record.to_dict()}, default=str) + "\n")
                    out.flush()
                    print(f"    {', '.join(keywords)}", file=sys.stderr, flush=True)

    print("\ncluster  pair                 mean Jaccard")
    for cid in cluster_ids:
        cid = int(cid)
        for size in args.sizes:
            runs = results[(cid, size)]
            within = [jaccard(a, b) for a, b in itertools.combinations(runs, 2)]
            print(f"{cid:7d}  {size} vs {size}  (within)  {np.mean(within):.2f}")
        for s1, s2 in itertools.combinations(args.sizes, 2):
            across = [jaccard(a, b) for a in results[(cid, s1)] for b in results[(cid, s2)]]
            print(f"{cid:7d}  {s1} vs {s2}  (across)  {np.mean(across):.2f}")


if __name__ == "__main__":
    main()
