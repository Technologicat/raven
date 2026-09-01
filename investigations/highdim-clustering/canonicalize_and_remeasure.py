#!/usr/bin/env python
"""How much of the high-IDF tail is spelling variants rather than specific topics?

Ranking cluster keywords by IDF inherits the classical frequency-analysis failure: a term occurring in
one cluster scores highest, and "occurs once" describes a genuinely specific topic and a one-off variant
equally well. A string test cannot separate them - it flags `logical reasoning` against `reasoning`,
which is a specialization and exactly what should rank first - so the question needs a judge that reads
meaning.

The canonicalization pass is that judge, and running it answers the question as a side effect: a
singleton that *merges* was a variant, and one that survives is more likely a real topic. This applies
it to keyword files already produced and re-measures, which is why it needs one LLM call per corpus
rather than a re-run of the extraction.

It reuses the shipped prompt and the shipped parser (`importer._parse_canonicalization_mapping`), so
what is measured is what the importer would actually do, rather than a second implementation that might
be kinder to itself.

Usage:

    python canonicalize_and_remeasure.py FILE.txt [FILE.txt ...]
"""

import argparse
import re
from collections import Counter


def read_clusters(path):
    """Read the per-cluster keyword lists out of a `show_clusters.py --llm-keywords` dump."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*KEYWORDS: (.*)", line)
            if m:
                out.append([k.strip() for k in m.group(1).split(",") if k.strip()])
    return out


def measure(clusters):
    """The numbers the keyword design is decided on, for one labelling."""
    cdf = Counter(k.casefold() for ks in clusters for k in set(k.casefold() for k in ks))
    singles = [k for k, c in cdf.items() if c == 1]
    n = len(clusters)
    common = {k for k, c in cdf.items() if c >= max(2, 0.10 * n)}
    burn = [sum(1 for k in ks if k.casefold() in common) for ks in clusters]
    burn.sort()
    return {"distinct": len(cdf),
            "singletons": len(singles),
            "singleton_fraction": len(singles) / max(1, len(cdf)),
            "common_pool": len(common),
            "burn_median": burn[len(burn) // 2] if burn else 0,
            "burn_p90": burn[int(0.9 * (len(burn) - 1))] if burn else 0}


def canonicalize(clusters, backend_url=None):
    """Run the shipped canonicalization pass over these keyword lists. Returns `(new_clusters, mapping)`."""
    from raven.librarian import agent, config as librarian_config, llmclient
    from raven.visualizer import config as visualizer_config
    from raven.visualizer import importer

    vocabulary = {k for ks in clusters for k in ks}
    vocabulary.discard("<unknown topic>")
    listing = "\n".join(sorted(vocabulary))
    prompt = f"{visualizer_config.clusters_llm_keyword_canonicalization_prompt}\n-----\n\n{listing}"

    url = backend_url or librarian_config.llm_backend_url
    llm_settings = llmclient.setup(backend_url=url, quiet=True)
    record = agent.turn(llm_settings, prompt, use_character_card=False, tools_enabled=False,
                        internet_enabled=False, docs_enabled=False, markup=None)
    mapping = importer._parse_canonicalization_mapping(record.reply or "", vocabulary)

    canonicalized = []
    for keywords in clusters:
        seen = {}
        for k in keywords:
            seen.setdefault(mapping.get(k, k), None)
        canonicalized.append(list(seen))
    return canonicalized, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--backend-url", default=None)
    args = parser.parse_args()

    for path in args.files:
        clusters = read_clusters(path)
        name = path.split("/")[-1]
        before = measure(clusters)
        canonicalized, mapping = canonicalize(clusters, args.backend_url)
        after = measure(canonicalized)

        # A singleton that merged was a variant; the count is the answer the string test could not give.
        singles_before = {k.casefold() for k, c in Counter(k.casefold() for ks in clusters
                                                           for k in set(k.casefold() for k in ks)).items() if c == 1}
        merged_singletons = sum(1 for k in mapping if k.casefold() in singles_before)

        print(f"=== {name} ({len(clusters)} clusters) ===")
        print(f"  {len(mapping)} replacements applied, of which {merged_singletons} merged a keyword that "
              f"had occurred in exactly one cluster")
        print(f"  {'':<22}{'before':>9}{'after':>9}")
        for label, key in (("distinct keywords", "distinct"), ("singletons", "singletons"),
                           ("common pool", "common_pool"), ("burn median", "burn_median"),
                           ("burn p90", "burn_p90")):
            print(f"  {label:<22}{before[key]:>9}{after[key]:>9}")
        print(f"  {'singleton fraction':<22}{before['singleton_fraction']:>8.0%}{after['singleton_fraction']:>9.0%}")
        if mapping:
            shown = sorted(mapping.items())[:6]
            print("  sample merges: " + "; ".join(f"{a!r} -> {b!r}" for a, b in shown))
        print(flush=True)


if __name__ == "__main__":
    main()
