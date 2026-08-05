#!/usr/bin/env python3
"""Build a known-item retrieval evaluation set from a local BibTeX corpus.

Retrieval quality arguments are undecidable without labels, and full relevance judgments are
expensive: twenty questions against twenty results each is four hundred human judgments before
anyone learns anything. This builds the cheap half instead.

**Known-item retrieval.** Sample an abstract, write a question that abstract answers, and the label
comes free: that document is relevant to that question. The metric is then "did the retriever find
the paper the question was written from", scored as recall@k and MRR, with no judging at all.

What that buys and what it costs, stated plainly because the limitation is structural:

- It *understates* precision. Other papers in a 12k-document corpus on one topic will often also
  answer the question, and they count as misses here. The absolute numbers are therefore a floor.
- It is nonetheless *unbiased across configurations*, which is the property that matters for the
  question being asked. Comparing score-aware fusion against plain RRF only needs the same set of
  questions and the same gold documents on both sides.
- Full judgments can be layered on later by pooling the top-N of each configuration and judging the
  union. This file's output is the seed for that, not a competitor to it.

**The questions are LLM-written, and that is a hazard to watch.** A question that reuses the
abstract's distinctive phrasing turns the task into string matching, and BM25 will ace it while
telling you nothing. The prompt below forbids verbatim phrases, and `evaluate.py` reports the
keyword-only baseline for exactly this reason: if BM25 alone scores near-perfect, the questions are
too easy and the set needs regenerating, not celebrating.

**Two question shapes**, because the levers under evaluation fail differently:

- `focused` - one question from one abstract, the ordinary case.
- `rambling` - several sentences of context from several abstracts, ending in a question aimed at
  one of them. This is what the multiline composer produces in real use, and it is the shape that
  the "embed the whole message and get its centroid" failure needs in order to show up at all.

**Copyright.** The corpus is Web of Science records and does not enter the repository. Only the
generated questions and the WoS accession numbers they point at are written out - identifiers and
new text, not the abstracts. Keep it that way.

Usage:
    python make_questions.py <base_url> <model> [n_focused] [n_rambling]

    python make_questions.py http://localhost:1234 qwen3.6-35b-a3b 24 8
"""

import json
import pathlib
import random
import re
import sys
import urllib.request

import bibtexparser

CORPUS_DIR = pathlib.Path("~/.config/raven/llmclient/documents").expanduser()
OUT_PATH = pathlib.Path(__file__).parent / "questions.json"

# Fixed, so that a rerun samples the same papers and the set stays comparable across regenerations.
SEED = 20260728

# Abstracts shorter than this tend to be truncated records with nothing specific to ask about.
MIN_ABSTRACT_CHARS = 600

TIMEOUT = 600

FOCUSED_PROMPT = """Below is the title and abstract of a scientific paper.

Write ONE question that this paper answers, as a researcher might type it into a search box.

Requirements:
- The question must be answerable from this paper, and specific enough that a random other paper on
  hydrogen production would not answer it.
- Do NOT reuse distinctive phrases from the abstract. Rephrase in your own words. A reader who has
  the abstract in front of them should recognize the question as being about it, but a keyword
  search for the question's exact words should not trivially land on it.
- Do not mention the authors, the journal, or the year.
- Output the question and nothing else. No preamble, no quotation marks.

Title: {title}

Abstract: {abstract}"""

RAMBLING_PROMPT = """Below are the titles and abstracts of several scientific papers.

Write a short message (4 to 6 sentences) of the kind a researcher types into a chat assistant when
they are thinking out loud. It should wander across the topics of ALL the papers shown, and end with
a specific question that ONLY the paper marked TARGET can answer.

Requirements:
- Do NOT reuse distinctive phrases from any abstract. Rephrase in your own words.
- The wandering part should be genuine context, not filler: mention what the other papers are about
  as things the researcher is already thinking about.
- Do not mention authors, journals, or years.
- Output the message and nothing else.

{papers}"""


def load_entry(path: pathlib.Path) -> dict | None:
    """Return `{"id", "title", "abstract"}` for a BibTeX file, or `None` if it has no usable abstract."""
    try:
        library = bibtexparser.parse_file(str(path))
    except Exception:  # noqa: BLE001 -- a malformed record is data, not a crash
        return None
    if not library.entries:
        return None
    fields = {field.key: field.value for field in library.entries[0].fields}
    abstract = (fields.get("Abstract") or "").strip()
    title = (fields.get("Title") or "").strip()
    if len(abstract) < MIN_ABSTRACT_CHARS or not title:
        return None
    # The document id HybridIR reports is the filename, which is what the labels must key on.
    return {"id": path.name, "title": clean(title), "abstract": clean(abstract)}


def clean(text: str) -> str:
    """Undo the BibTeX/LaTeX-isms that would otherwise show up in a generated question."""
    text = re.sub(r"</?sub>|</?sup>", "", text)
    text = text.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    text = re.sub(r"[{}]", "", text)
    return " ".join(text.split())


def ask(base: str, model: str, prompt: str) -> str:
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps({"model": model,
                                                  "messages": [{"role": "user", "content": prompt}],
                                                  "max_tokens": 4000,
                                                  "temperature": 0.7}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        body = json.loads(r.read().decode("utf-8"))
    choice = body.get("choices", [{}])[0]
    text = (choice.get("message", {}).get("content") or "").strip()
    # A thinking model that runs out of budget mid-deliberation returns empty content; that is a
    # dropped sample, not a question.
    if choice.get("finish_reason") == "length" and not text:
        return ""
    return " ".join(text.split()).strip('"')


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    base, model = sys.argv[1], sys.argv[2]
    n_focused = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    n_rambling = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    files = sorted(CORPUS_DIR.glob("*.bib"))
    if not files:
        print(f"no .bib files in {CORPUS_DIR}")
        return
    print(f"corpus: {len(files)} records in {CORPUS_DIR}")

    rng = random.Random(SEED)
    rng.shuffle(files)

    # Draw more candidates than needed; short or malformed records are skipped.
    needed = n_focused + 3 * n_rambling
    entries = []
    for path in files:
        entry = load_entry(path)
        if entry is not None:
            entries.append(entry)
        if len(entries) >= needed + 20:
            break
    print(f"usable records drawn: {len(entries)}\n")

    questions = []

    def save() -> None:
        """Write everything generated so far. Called after each question, not once at the end.

        The run costs about an hour of GPU time, and a crash near the end that had written nothing is the
        expensive kind of failure — the whole set has to be regenerated, and the seed guarantees only that
        the same *papers* are drawn, not that the model says the same thing about them.
        """
        OUT_PATH.write_text(json.dumps(
            {"corpus": "Web of Science hydrogen-production records, local to the developer machine "
                       "(not in this repository)",
             "corpus_size": len(files),
             "seed": SEED,
             "generator_model": model,
             "questions": questions},
            indent=2, ensure_ascii=False) + "\n")

    for i in range(n_focused):
        entry = entries[i]
        text = ask(base, model, FOCUSED_PROMPT.format(title=entry["title"], abstract=entry["abstract"]))
        if not text:
            print(f"  focused {i + 1}: SKIPPED (empty reply)", flush=True)
            continue
        questions.append({"kind": "focused", "question": text, "gold": [entry["id"]],
                          "gold_titles": [entry["title"]]})
        save()
        print(f"  focused {i + 1}/{n_focused}: {text[:110]}", flush=True)

    offset = n_focused
    for i in range(n_rambling):
        group = entries[offset + 3 * i: offset + 3 * i + 3]
        if len(group) < 3:
            break
        target = group[0]
        blocks = [f"{'TARGET - ' if j == 0 else ''}Title: {e['title']}\nAbstract: {e['abstract']}"
                  for j, e in enumerate(group)]
        text = ask(base, model, RAMBLING_PROMPT.format(papers="\n\n".join(blocks)))
        if not text:
            print(f"  rambling {i + 1}: SKIPPED (empty reply)", flush=True)
            continue
        questions.append({"kind": "rambling", "question": text, "gold": [target["id"]],
                          "gold_titles": [target["title"]],
                          "distractors": [e["id"] for e in group[1:]]})
        save()
        print(f"  rambling {i + 1}/{n_rambling}: {text[:110]}", flush=True)

    save()
    print(f"\nwrote {len(questions)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
