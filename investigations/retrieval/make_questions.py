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

**Two BibTeX corpora share this generator**, because they are the same *shape* of thing — one record per
file, title plus abstract, academic register — and differ only in subject. That is exactly what makes the
second one worth having: fiction tested the retrieval signals against a corpus as far from hydrogen
abstracts as anything gets, which is the flattering direction. arXiv AI/ML is the near well, where genre,
length and phrasing all match and only the topic moves. See `CORPORA` below.

(Fiction has its own generator, `make_fiction_questions.py`, and that split is not duplication for its own
sake: prose has no abstracts to sample, so both the sampling and the prompts differ down to the bone.)

Usage:
    python make_questions.py <hydrogen|arxiv-ai> <base_url> <model> [n_focused] [n_rambling]

    python make_questions.py hydrogen http://localhost:1234 qwen3.6-35b-a3b 24 8
"""

import json
import pathlib
import random
import re
import sys
import urllib.request

import bibtexparser

HERE = pathlib.Path(__file__).parent

# Per-corpus profiles. `sibling_topic` goes into the focused prompt as the thing a *different* paper in the
# same collection would be about, which is what stops the generator writing questions so broad that half
# the corpus answers them. It has to name the collection's subject, not the individual paper's.
CORPORA = {
    # Note the asymmetry in `docs_dir`: hydrogen reads the *live* Librarian docs slot, which now rotates
    # between three corpora, so generating for it while another one is swapped in silently draws from the
    # wrong collection. arXiv reads a stable path and cannot. Check what is in the slot before regenerating.
    "hydrogen": {"docs_dir": pathlib.Path("~/.config/raven/llmclient/documents").expanduser(),
                 "out_path": HERE / "questions.json",
                 "sibling_topic": "hydrogen production",
                 "description": "Web of Science hydrogen-production records, local to the developer "
                                "machine (not in this repository)"},
    # `use_abstracts: False` is not a degraded mode — it is what a hand-built BibTeX database looks like.
    # This one was typed by hand between 2007 and 2016, partly predating routine online abstracts, so 537
    # of its 541 records are title, authors and year. Asking "which paper was the one about X" over such a
    # collection is a plausible thing to want, and it is the case where a QA-type embedder has roughly a
    # tenth of the surface to match a question against.
    "banichuk": {"docs_dir": pathlib.Path("~/Documents/koodit/raven/00_stuff/datasets/banichuk").expanduser(),
                 "out_path": HERE / "banichuk_questions.json",
                 "sibling_topic": "axially moving materials and structural mechanics",
                 "use_abstracts": False,
                 "description": "an axially-moving-materials bibliography accumulated over a working "
                                "career and typed by hand, 541 records spanning 1766-2013, almost all of "
                                "them titles only; local to the developer machine (not in this repository)"},
    "arxiv-ai": {"docs_dir": pathlib.Path("~/Documents/koodit/raven/00_stuff/datasets/ai_papers/burst").expanduser(),
                 "out_path": HERE / "arxiv_ai_questions.json",
                 "sibling_topic": "AI and machine learning",
                 "description": "arXiv AI/ML abstracts as actually accumulated by one reader, so a "
                                "minority of cosmology, astronomy and speculative-physics records are "
                                "mixed in — strays from saving everything to one folder, not planted "
                                "confounders; local to the developer machine (not in this repository)"},
}

# Fixed, so that a rerun samples the same papers and the set stays comparable across regenerations.
SEED = 20260728

# Abstracts shorter than this tend to be truncated records with nothing specific to ask about.
MIN_ABSTRACT_CHARS = 600

TIMEOUT = 600

FOCUSED_PROMPT = """Below is the title and abstract of a scientific paper.

Write ONE question that this paper answers, as a researcher might type it into a search box.

Requirements:
- The question must be answerable from this paper, and specific enough that a random other paper on
  {sibling_topic} would not answer it.
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


TITLES_FOCUSED_PROMPT = """Below is the title of a scientific paper. There is no abstract available.

Write ONE question that this paper is likely to answer, as a researcher might type it into a search box
when trying to find a paper they half-remember.

Requirements:
- The question must be about what the title says the paper is about, and specific enough that a random
  other paper on {sibling_topic} would not be an equally good answer.
- Do NOT reuse the title's distinctive noun phrases. Rephrase in your own words — describe the same
  subject the way someone would who recalled the topic but not the wording.
- Do not mention the authors, the journal, or the year.
- Output the question and nothing else. No preamble, no quotation marks.

Title: {title}"""

TITLES_RAMBLING_PROMPT = """Below are the titles of several scientific papers. No abstracts are available.

Write a short message (4 to 6 sentences) of the kind a researcher types into a chat assistant when they
are thinking out loud. It should wander across the subjects of ALL the titles shown, and end with a
specific question that ONLY the paper marked TARGET can answer.

Requirements:
- Do NOT reuse distinctive noun phrases from any title. Rephrase in your own words.
- The wandering part should be genuine context, not filler: mention what the other papers are about as
  things the researcher is already thinking about.
- Do not mention authors, journals, or years.
- Output the message and nothing else.

{papers}"""


def load_entry(path: pathlib.Path, require_abstract: bool = True) -> dict | None:
    """Return `{"id", "title", "abstract"}` for a BibTeX file, or `None` if it is not usable.

    `require_abstract`: when True, a record whose abstract is missing or too short is rejected — those tend
                        to be truncated records with nothing specific to ask about. When False, only a
                        title is required, and `abstract` comes back as the empty string.
    """
    try:
        library = bibtexparser.parse_file(str(path))
    except Exception:  # noqa: BLE001 -- a malformed record is data, not a crash
        return None
    if not library.entries:
        return None
    fields = {field.key: field.value for field in library.entries[0].fields}
    abstract = (fields.get("Abstract") or "").strip()
    title = (fields.get("Title") or "").strip()
    if not title or (require_abstract and len(abstract) < MIN_ABSTRACT_CHARS):
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
    if len(sys.argv) < 4:
        print(__doc__)
        return
    argv = sys.argv[1:]
    append = "--append" in argv
    if append:
        argv.remove("--append")
    corpus, base, model = argv[0], argv[1], argv[2]
    if corpus not in CORPORA:
        raise SystemExit(f"unknown corpus '{corpus}'; expected one of {', '.join(CORPORA)}")
    profile = CORPORA[corpus]
    corpus_dir, out_path = profile["docs_dir"], profile["out_path"]
    n_focused = int(argv[3]) if len(argv) > 3 else 24
    n_rambling = int(argv[4]) if len(argv) > 4 else 8

    files = sorted(corpus_dir.glob("*.bib"))
    if not files:
        print(f"no .bib files in {corpus_dir}")
        return
    print(f"corpus '{corpus}': {len(files)} records in {corpus_dir}")

    # In append mode the counts are how many to *add*, and the questions already in the file are kept
    # verbatim. Keeping them is the point: every result committed under `investigations/retrieval/` was
    # scored against those exact questions, so regenerating them would silently invalidate the lot — the
    # seed fixes which *papers* are drawn, not what the model says about them (it is sampled at
    # temperature). Growing the set instead leaves every past number comparable and only narrows the
    # confidence intervals.
    questions = []
    used_ids: set[str] = set()
    if append:
        if not out_path.exists():
            raise SystemExit(f"--append: nothing to append to; '{out_path}' does not exist")
        questions = json.loads(out_path.read_text(encoding="utf-8"))["questions"]
        # Reconstructed from the questions themselves rather than from a recorded count, because a
        # generation that came back empty consumed its paper without leaving a question behind. Counting
        # kinds would therefore drift by exactly those, and the drift would show up as a second question
        # about a paper already in the set — not fatal, but a correlated sample nobody asked for.
        for q in questions:
            used_ids.update(q.get("gold", []))
            used_ids.update(q.get("distractors", []))
        print(f"appending to {len(questions)} existing questions, over {len(used_ids)} already-used records")

    rng = random.Random(SEED)
    rng.shuffle(files)

    # Draw more candidates than needed; short or malformed records are skipped, and in append mode the
    # records already spoken for are skipped too — so the pool has to be deep enough to clear them first.
    needed = n_focused + 3 * n_rambling
    use_abstracts = profile.get("use_abstracts", True)
    entries = []
    for path in files:
        entry = load_entry(path, require_abstract=use_abstracts)
        if entry is not None and entry["id"] not in used_ids:
            entries.append(entry)
        if len(entries) >= needed + 20:
            break
    print(f"usable records drawn: {len(entries)}\n")
    if len(entries) < needed:
        print(f"  note: only {len(entries)} unused records available, wanted {needed} — "
              f"the corpus is running out of papers to ask about.")

    def save() -> None:
        """Write everything generated so far. Called after each question, not once at the end.

        The run costs about an hour of GPU time, and a crash near the end that had written nothing is the
        expensive kind of failure — the whole set has to be regenerated, and the seed guarantees only that
        the same *papers* are drawn, not that the model says the same thing about them.
        """
        out_path.write_text(json.dumps(
            {"corpus": profile["description"],
             "corpus_size": len(files),
             "seed": SEED,
             "generator_model": model,
             "questions": questions},
            indent=2, ensure_ascii=False) + "\n")

    for i in range(n_focused):
        entry = entries[i]
        prompt = (FOCUSED_PROMPT.format(title=entry["title"], abstract=entry["abstract"],
                                        sibling_topic=profile["sibling_topic"]) if use_abstracts
                  else TITLES_FOCUSED_PROMPT.format(title=entry["title"],
                                                    sibling_topic=profile["sibling_topic"]))
        text = ask(base, model, prompt)
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
        blocks = [f"{'TARGET - ' if j == 0 else ''}Title: {e['title']}"
                  + (f"\nAbstract: {e['abstract']}" if use_abstracts else "")
                  for j, e in enumerate(group)]
        template = RAMBLING_PROMPT if use_abstracts else TITLES_RAMBLING_PROMPT
        text = ask(base, model, template.format(papers="\n\n".join(blocks)))
        if not text:
            print(f"  rambling {i + 1}: SKIPPED (empty reply)", flush=True)
            continue
        questions.append({"kind": "rambling", "question": text, "gold": [target["id"]],
                          "gold_titles": [target["title"]],
                          "distractors": [e["id"] for e in group[1:]]})
        save()
        print(f"  rambling {i + 1}/{n_rambling}: {text[:110]}", flush=True)

    save()
    print(f"\nwrote {len(questions)} questions to {out_path}")
    if append:
        print("  note: the cross-corpus negatives in `sharpness.py` are built from whatever question "
              "files exist when it runs, so re-score every corpus before comparing any two of them.")


if __name__ == "__main__":
    main()
