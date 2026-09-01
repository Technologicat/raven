#!/usr/bin/env python
"""Of the keywords occurring in exactly one cluster, how many are real topics and how many are noise?

Ranking by IDF puts the rarest keywords first, and 82-90% of a corpus's keyword vocabulary occurs in
exactly one cluster. That tail is where the ranking's value and its classical failure mode both live: a
term occurring once describes a genuinely specific topic and a one-off artifact equally well, and no
measurement in this bundle can tell those apart. Canonicalization cannot either - it merges *spelling*
variants and leaves conceptual near-duplicates alone, which is why it barely touched the tail.

So this asks a model to judge, which is the only instrument available for the question. Read the result
knowing what it is: one model grading another model's output, and therefore weaker evidence than
anything else here.

Two things happen before the judging, both at Juha's suggestion (2026-09-01):

  - **Case is unified deterministically first**, rather than being left to the LLM. Most of what the
    canonicalization pass actually did was case fixes, and a rule does that more reliably than a model
    while costing nothing - which leaves the model only the judgements that need one.
  - **The judge's own output is put through the same normalization**, so a verdict on `Water Splitting`
    is not lost because the vocabulary spells it `Water splitting`.

Usage:

    python judge_singletons.py FILE.txt [FILE.txt ...] [--batch 60]
"""

import argparse
from collections import Counter

from canonicalize_and_remeasure import read_clusters


VERDICTS = {
    "topic": "a specific research topic, worth showing as a label",
    "generic": "too vague to tell one cluster from another",
    "duplicate": "means the same as another keyword in the list",
    "artifact": "not a topic at all - a fragment, an instruction, or noise",
}

JUDGE_PROMPT = """**Instructions**

Below is the full keyword vocabulary of a clustered document dataset, followed by a subset of those
keywords to judge. Each keyword in the subset occurs in exactly one cluster of the dataset.

For each keyword in the subset, decide which of these it is:

    topic     - a specific research topic, worth showing as a cluster label
    generic   - a real term, but too vague to tell one cluster from another
    duplicate - means substantially the same as another keyword in the vocabulary
    artifact  - not a topic at all: a sentence fragment, an instruction, or noise

Use the full vocabulary to decide "duplicate": a keyword is a duplicate only if some *other* keyword in
the vocabulary means substantially the same thing.

As your response, after you are done thinking, write one line per keyword in the subset, in the format:

    keyword -> verdict

IMPORTANT: The result will be read by a computer program. Use " -> " as the separator, copy the keyword
exactly as given, and write one of the four verdict words and nothing else after the arrow. Do not add
commentary, headings, or bullet points."""


def unify_case(clusters):
    """Collapse keywords that differ only in case, keeping the most frequent surface form.

    Returns `(new_clusters, mapping)`. Deterministic, and cheaper and more reliable than asking a model
    to notice that `AI safety` and `AI Safety` are the same words - which is most of what the LLM
    canonicalization pass was spending itself on.

    Ties go to the form that sorts first, so the result does not depend on iteration order.
    """
    forms = {}
    for keywords in clusters:
        for k in keywords:
            forms.setdefault(k.casefold(), Counter())[k] += 1
    canonical = {folded: min(counter.most_common(), key=lambda kv: (-kv[1], kv[0]))[0]
                 for folded, counter in forms.items()}
    mapping = {k: canonical[k.casefold()]
               for keywords in clusters for k in keywords if k != canonical[k.casefold()]}
    out = []
    for keywords in clusters:
        seen = {}
        for k in keywords:
            seen.setdefault(canonical[k.casefold()], None)
        out.append(list(seen))
    return out, mapping


def judge(vocabulary, singletons, batch_size, backend_url=None):
    """Ask the model to classify each singleton. Returns `{keyword: verdict}`."""
    from raven.librarian import agent, config as librarian_config, llmclient

    url = backend_url or librarian_config.llm_backend_url
    llm_settings = llmclient.setup(backend_url=url, quiet=True)

    # The judge's replies are matched back case-insensitively, for the same reason the vocabulary was
    # unified: a verdict on `Water Splitting` should not be dropped because the list says `Water splitting`.
    by_fold = {k.casefold(): k for k in vocabulary}
    verdicts = {}
    for start in range(0, len(singletons), batch_size):
        chunk = singletons[start:start + batch_size]
        prompt = (f"{JUDGE_PROMPT}\n-----\n\nFULL VOCABULARY:\n" + "\n".join(sorted(vocabulary))
                  + "\n\nKEYWORDS TO JUDGE:\n" + "\n".join(chunk))
        record = agent.turn(llm_settings, prompt, use_character_card=False, tools_enabled=False,
                            internet_enabled=False, docs_enabled=False, markup=None)
        for line in (record.reply or "").splitlines():
            keyword, separator, verdict = line.partition("->")
            if not separator:
                continue
            keyword = keyword.strip().lstrip("-*").strip()
            verdict = verdict.strip().casefold()
            if keyword.casefold() in by_fold and verdict in VERDICTS:
                verdicts[by_fold[keyword.casefold()]] = verdict
    return verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--batch", type=int, default=60, help="how many singletons to judge per request")
    parser.add_argument("--backend-url", default=None)
    args = parser.parse_args()

    for path in args.files:
        clusters, _ = unify_case(read_clusters(path))
        cdf = Counter(k for ks in clusters for k in set(ks))
        vocabulary = sorted(cdf)
        singletons = sorted(k for k, c in cdf.items() if c == 1)

        verdicts = judge(vocabulary, singletons, args.batch, args.backend_url)
        counts = Counter(verdicts.values())
        judged = len(verdicts)

        print(f"=== {path.split('/')[-1]} ({len(clusters)} clusters) ===")
        print(f"  vocabulary {len(vocabulary)} after case unification; {len(singletons)} occur in one "
              f"cluster; {judged} judged")
        for verdict, description in VERDICTS.items():
            n = counts.get(verdict, 0)
            print(f"    {verdict:<10} {n:>4}  {n / max(1, judged):>5.0%}   {description}")
        unjudged = [k for k in singletons if k not in verdicts]
        if unjudged:
            print(f"  {len(unjudged)} not judged (no parsable verdict): {unjudged[:4]}")
        for verdict in ("artifact", "duplicate", "generic"):
            examples = [k for k, v in verdicts.items() if v == verdict][:5]
            if examples:
                print(f"  sample {verdict}: {examples}")
        print(flush=True)


if __name__ == "__main__":
    main()
