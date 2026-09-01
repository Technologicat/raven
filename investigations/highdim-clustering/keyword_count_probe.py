#!/usr/bin/env python
"""Does asking the model for twelve keywords give twelve, or six and six of padding?

The keyword-pools design wants more than six per cluster, so that filtering the corpus-common ones out
still leaves a cluster with something to show. Whether that is possible is a question about the model:
human keyword sets run to five or six, so a request for twelve may be out of distribution and answered
by padding rather than by six more real keywords.

This runs the same clusters at both settings and prints them side by side, because the judgement is a
reader's. Two numbers are computed to aim that reading:

    distinctiveness  the fraction of a position's keywords that appear in only one cluster. Padding
                     should be *more* generic than the real ones, so if positions 7-12 are filler their
                     distinctiveness will fall rather than rise.
    grounding        the fraction of keywords whose words appear in the cluster's own titles. Low
                     grounding is not itself damning - a good keyword is often an abstraction over the
                     titles rather than a quotation from them - but a *drop* between the two halves is
                     a sign the model has started inventing.

Usage:

    python keyword_count_probe.py --vectors PATH.npz --dataset PATH.pickle [--clusters 12]
"""

import argparse
import re
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import clusterlab
from show_clusters import drop_undersized_clusters


# The prompt spells its keyword count as an English word, so the comparison is driven by words and the
# reporting needs the numbers back. Only the counts worth asking for are here.
WORD_NUMBERS = {"six": 6, "eight": 8, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20}


def ask_for_keywords(llm_settings, agent, base_prompt, texts, count_word):
    """Run one keyword extraction, with the prompt's keyword count swapped to `count_word`."""
    prompt_text = base_prompt.replace("up to six keywords", f"up to {count_word} keywords")
    if f"up to {count_word} keywords" not in prompt_text:
        raise SystemExit("could not find the keyword count in the prompt; has its wording changed?")
    body = "\n\n\n".join(t.strip() for t in texts)
    record = agent.turn(llm_settings,
                        f"{prompt_text}\n-----\n\n{body}",
                        use_character_card=False,
                        tools_enabled=False,
                        internet_enabled=False,
                        docs_enabled=False,
                        markup=None)
    reply = (record.reply or "").strip()
    if reply.lower() == "keyword extraction failed":
        return []
    return [k.strip() for k in reply.split(",") if k.strip()]


def distinctiveness(keyword_lists, positions):
    """Fraction of the keywords at `positions` that occur in exactly one cluster."""
    counts = {}
    for keywords in keyword_lists:
        for k in {k.casefold() for k in keywords}:
            counts[k] = counts.get(k, 0) + 1
    picked = [k.casefold() for keywords in keyword_lists for k in keywords[positions[0]:positions[1]]]
    if not picked:
        return float("nan")
    return sum(1 for k in picked if counts[k] == 1) / len(picked)


def grounding(keywords, titles, positions):
    """Fraction of the keywords at `positions` whose words all appear somewhere in the cluster's titles."""
    haystack = " ".join(titles).casefold()
    picked = keywords[positions[0]:positions[1]]
    if not picked:
        return float("nan")
    hits = 0
    for k in picked:
        words = [w for w in re.findall(r"[a-z]+", k.casefold()) if len(w) > 3]
        if words and all(w in haystack for w in words):
            hits += 1
    return hits / len(picked)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-clusters", type=int, default=100, help="agglomerative cut level")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=12, help="how many clusters to probe")
    parser.add_argument("--counts", nargs=2, default=["six", "twelve"],
                        help="the two keyword counts to compare, spelled as the prompt spells them "
                             "(e.g. --counts twelve fifteen)")
    parser.add_argument("--max-prompt-entries", type=int, default=60)
    parser.add_argument("--backend-url", default=None)
    args = parser.parse_args()

    from raven.librarian import agent, config as librarian_config, llmclient
    from raven.visualizer import config as visualizer_config

    vectors, _model = clusterlab.load_vectors(args.vectors)
    entries = clusterlab.load_entries(args.dataset)
    original = clusterlab.normalize(vectors)
    labels = AgglomerativeClustering(n_clusters=args.n_clusters, metric="cosine",
                                     linkage="average").fit_predict(clusterlab.center(original))
    labels = drop_undersized_clusters(labels, args.min_cluster_size)

    url = args.backend_url or librarian_config.llm_backend_url
    llm_settings = llmclient.setup(backend_url=url, quiet=True)
    print(f"probing with {llm_settings.model_id}", file=sys.stderr)

    # Largest clusters first: those are the ones the design most needs more keywords for, since they are
    # the ones whose six are most likely to be corpus-common.
    cluster_ids = sorted(np.unique(labels[labels >= 0]), key=lambda c: -np.sum(labels == c))[:args.clusters]

    six, twelve, titles_by_cluster = [], [], []
    for cid in cluster_ids:
        members = np.flatnonzero(labels == cid)
        center = clusterlab.normalize(original[members].mean(axis=0, keepdims=True))
        ordered = members[np.argsort(-(original[members] @ center[0]))]
        picks = ordered[np.linspace(0, len(ordered) - 1, min(len(ordered), args.max_prompt_entries)).round().astype(int)]
        texts = [clusterlab.format_for_keyword_extraction(*entries[i]) for i in picks]
        titles_by_cluster.append([entries[i][0] for i in members])

        base = visualizer_config.clusters_llm_keyword_extraction_prompt
        got6 = ask_for_keywords(llm_settings, agent, base, texts, args.counts[0])
        got12 = ask_for_keywords(llm_settings, agent, base, texts, args.counts[1])
        six.append(got6)
        twelve.append(got12)
        print(f"cluster {cid} ({len(members)} entries)")
        print(f"    {args.counts[0]:<7}[{len(got6):>2}]: {', '.join(got6)}")
        print(f"    {args.counts[1]:<7}[{len(got12):>2}]: {', '.join(got12)}")
        print(f"            first six: {', '.join(got12[:6])}")
        print(f"            the tail : {', '.join(got12[6:])}")
        print(flush=True)

    # The boundary between "head" and "tail" is wherever the smaller request stopped, so that the tail
    # is exactly the keywords the larger request added. Hardcoding 6 and 12 would misreport any other
    # pair, and the whole point of the comparison is the added ones.
    head = WORD_NUMBERS[args.counts[0]]
    full = WORD_NUMBERS[args.counts[1]]

    print("=" * 78)
    print(f"asked for {args.counts[0]}: mean returned {np.mean([len(k) for k in six]):.1f}")
    print(f"asked for {args.counts[1]}: mean returned {np.mean([len(k) for k in twelve]):.1f}")
    print()
    print(f"distinctiveness, {args.counts[0]}-run positions 1-{head}: {distinctiveness(six, (0, head)):.2f}")
    print(f"distinctiveness, {args.counts[1]}-run positions 1-{head}: {distinctiveness(twelve, (0, head)):.2f}")
    print(f"distinctiveness, {args.counts[1]}-run positions {head + 1}-{full}: "
          f"{distinctiveness(twelve, (head, full)):.2f}")
    print(f"   (a fall from the head to positions {head + 1}-{full} is what padding would look like)")
    print()
    g_head = np.nanmean([grounding(k, t, (0, head)) for k, t in zip(twelve, titles_by_cluster)])
    g_tail = np.nanmean([grounding(k, t, (head, full)) for k, t in zip(twelve, titles_by_cluster)])
    print(f"grounding in the cluster's own titles, positions 1-{head}: {g_head:.2f}")
    print(f"grounding in the cluster's own titles, positions {head + 1}-{full}: {g_tail:.2f}")


if __name__ == "__main__":
    main()
