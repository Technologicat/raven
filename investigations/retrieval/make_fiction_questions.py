#!/usr/bin/env python3
"""Build a known-item evaluation set from a prose fiction corpus — the second corpus the set has wanted.

Sibling of `make_questions.py`, which does the same job over a BibTeX corpus of scientific abstracts. Kept
as a separate script rather than a mode of that one: the corpus loading, the sampling and the prompts all
differ, what is left to share is a dozen lines of HTTP, and an investigation script that reads standalone
years later is worth more than one that saves those lines.

**Why a second corpus at all.** Brief 09's lever 1 measured a retrieval confidence signal and found the
usable one to be the *absolute* best vector similarity, with a threshold near 0.45 separating questions the
corpus can answer from questions it cannot. The objection to any such constant is that the scale of "close"
is a property of the collection, and one corpus of hydrogen-production abstracts cannot test that. Prose
fiction is about as far from scientific abstracts as a document set gets while still being something a
researcher might plausibly index, so it is the natural adversary: if 0.45 survives here, it travels.

**Two sets come out of one run**, and the second is the more valuable:

- `on_corpus` — questions written from passages of stories that are *in* the index. These are the positives.
- `adjacent` — questions written from stories deliberately *held out* of the index. Same universe, same
  site, same author community, same generator, same prompts. They differ from the positives in exactly one
  respect: the answer is not in the corpus. Hand-writing negatives that hard is not really possible, and the
  first run of `sharpness.py` had to make do with four of them, written by the implementer.

  **They are a lower bound, deliberately.** Fan fiction in one shared universe overlaps: a question about a
  held-out story may well be answerable from an indexed one, which mislabels it as a negative and costs the
  signal AUROC it should have had. That biases *against* the signal, so a separation measured here is real
  rather than flattered. The prompt asks for story-specific detail to keep the overlap down.

The far negatives are free and are not generated here: the 99 hydrogen questions already in
`questions.json` are off-corpus for a fiction index by construction, and vice versa.

**Copyright.** The stories are third-party fan fiction and do not enter the repository — only the generated
questions (new text) and the filenames they were drawn from (identifiers). Same rule as the BibTeX corpus.

Usage:
    python make_fiction_questions.py <base_url> <model> <corpus_dir> [held_out_dir] [n_focused] [n_rambling]

    python make_fiction_questions.py http://localhost:1234 qwen3.6-35b-a3b \
        ~/.config/raven/llmclient/documents ~/Downloads/held-out 78 24
"""

import json
import pathlib
import random
import sys
import urllib.request

from raven.common import docextract

OUT_PATH = pathlib.Path(__file__).parent / "fiction_questions.json"

# Fixed, so a rerun draws the same passages and the set stays comparable across regenerations.
SEED = 20260805

# How much of a story to show the generator. Long enough to carry a scene worth asking about, short enough
# that the model does not have to choose which of several to write from.
PASSAGE_CHARS = 4000

# Passages are drawn from the corpus by *length*, not one per story: a random passage of the corpus is what a
# reader's question is drawn from too, and weighting by document would over-sample the very short stories.
TIMEOUT = 600

FOCUSED_PROMPT = """Below is a passage from a work of fiction.

Write ONE question that this passage answers, as a reader might type it into a search box when trying to
find this part of the story again.

Requirements:
- The question must be answerable from this passage, and specific to what happens *here*. These stories all
  share a setting, characters and premise, so a question about the premise itself ("what is Equestria
  Online?") is useless — ask about the particular events, people, arguments or details of this passage.
- Do NOT reuse distinctive phrases from the passage. Rephrase in your own words. A reader who has the
  passage in front of them should recognize what the question is about, but a keyword search for the
  question's exact words should not trivially land on it.
- Do not mention the title or the author, and do not refer to "the passage" or "this excerpt" — write it as
  a question about the story's content, standing on its own.
- Output the question and nothing else. No preamble, no quotation marks.

Passage:

{passage}"""

RAMBLING_PROMPT = """Below are several passages from works of fiction.

Write a short message (4 to 6 sentences) of the kind a reader types into a chat assistant when they are
thinking out loud about what they have been reading. It should wander across the content of ALL the
passages shown, and end with a specific question that ONLY the passage marked TARGET can answer.

Requirements:
- Do NOT reuse distinctive phrases from any passage. Rephrase in your own words.
- The wandering part should be genuine context, not filler: mention what the other passages are about as
  things the reader is already turning over.
- These stories share a setting and characters, so the closing question must turn on the particular events
  of the TARGET passage rather than on the shared premise.
- Do not mention titles or authors, and do not refer to "the passage" or "this excerpt".
- Output the message and nothing else.

{passages}"""


def load_passages(corpus_dir: pathlib.Path, rng: random.Random, count: int,
                  by_length: bool = True) -> list[dict]:
    """Draw `count` passages from the documents in `corpus_dir`.

    `by_length`: Whether to weight the draw by document length, so that a passage is a uniform sample of the
                 *corpus* rather than of its file list. True for the questions being asked *of* a corpus — a
                 reader's question comes from wherever they were reading, and they read more of the long
                 stories. False when the draw is building a *negative* class, where the job is to represent
                 off-corpus content broadly and one 2.6-million-character story would otherwise take three
                 quarters of the sample.

    Returns `[{"id", "offset", "text"}, ...]`, where `id` is the filename — which is the document id
    `HybridIR` reports, and therefore what the gold labels have to key on.
    """
    documents = []
    for path in sorted(corpus_dir.iterdir()):
        if not path.is_file() or not docextract.is_supported(path):
            continue
        try:
            text = docextract.extract_text(path) or ""
        except Exception as exc:  # noqa: BLE001 -- an unreadable file is data about the corpus, not a crash
            print(f"  skipping '{path.name}': {type(exc).__name__}: {exc}")
            continue
        if len(text) >= PASSAGE_CHARS:
            documents.append({"id": path.name, "text": text})
    if not documents:
        return []
    print(f"corpus: {len(documents)} documents, {sum(len(d['text']) for d in documents)} characters")

    weights = [len(d["text"]) for d in documents] if by_length else None
    passages = []
    for i in range(count):
        if by_length:
            document = rng.choices(documents, weights=weights, k=1)[0]
        else:  # round-robin, so every document contributes before any contributes twice
            document = documents[i % len(documents)]
        text = document["text"]
        start = rng.randrange(0, len(text) - PASSAGE_CHARS + 1)
        # Snap to a paragraph boundary so the passage does not open mid-sentence, which produces questions
        # about a fragment the retriever will never see as one.
        boundary = text.find("\n\n", start)
        if boundary != -1 and boundary - start < PASSAGE_CHARS // 4:
            start = boundary + 2
        passages.append({"id": document["id"], "offset": start,
                         "text": text[start:start + PASSAGE_CHARS]})
    return passages


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
    # A thinking model that runs out of budget mid-deliberation returns empty content; that is a dropped
    # sample, not a question.
    if choice.get("finish_reason") == "length" and not text:
        return ""
    return " ".join(text.split()).strip('"')


def generate(base: str, model: str, passages: list[dict], n_focused: int, n_rambling: int,
             label: str, on_corpus: bool) -> list[dict]:
    """Turn drawn passages into questions. `label` prefixes the progress lines and names the group."""
    questions = []

    for i in range(min(n_focused, len(passages))):
        passage = passages[i]
        text = ask(base, model, FOCUSED_PROMPT.format(passage=passage["text"]))
        if not text:
            print(f"  {label} focused {i + 1}: SKIPPED (empty reply)")
            continue
        questions.append({"kind": "focused" if on_corpus else "adjacent",
                          "group": label,
                          "on_corpus": on_corpus,
                          "question": text,
                          "gold": [passage["id"]] if on_corpus else [],
                          "source": passage["id"],
                          "source_offset": passage["offset"]})
        print(f"  {label} focused {i + 1}/{n_focused}: {text[:100]}")

    offset = n_focused
    for i in range(n_rambling):
        group = passages[offset + 3 * i: offset + 3 * i + 3]
        if len(group) < 3:
            break
        target = group[0]
        blocks = [f"{'TARGET - ' if j == 0 else ''}Passage {j + 1}:\n\n{p['text']}"
                  for j, p in enumerate(group)]
        text = ask(base, model, RAMBLING_PROMPT.format(passages="\n\n".join(blocks)))
        if not text:
            print(f"  {label} rambling {i + 1}: SKIPPED (empty reply)")
            continue
        questions.append({"kind": "rambling" if on_corpus else "adjacent",
                          "group": label,
                          "on_corpus": on_corpus,
                          "question": text,
                          "gold": [target["id"]] if on_corpus else [],
                          "source": target["id"],
                          "source_offset": target["offset"],
                          "distractors": [p["id"] for p in group[1:]]})
        print(f"  {label} rambling {i + 1}/{n_rambling}: {text[:100]}")

    return questions


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        return
    base, model = sys.argv[1], sys.argv[2]
    corpus_dir = pathlib.Path(sys.argv[3]).expanduser()
    held_out_dir = pathlib.Path(sys.argv[4]).expanduser() if len(sys.argv) > 4 else None
    n_focused = int(sys.argv[5]) if len(sys.argv) > 5 else 78
    n_rambling = int(sys.argv[6]) if len(sys.argv) > 6 else 24

    rng = random.Random(SEED)

    print(f"--- indexed corpus: {corpus_dir} ---")
    passages = load_passages(corpus_dir, rng, n_focused + 3 * n_rambling)
    if not passages:
        print(f"no usable documents in {corpus_dir}")
        return
    questions = generate(base, model, passages, n_focused, n_rambling,
                         label="on-corpus", on_corpus=True)

    held_out_questions = []
    if held_out_dir is not None and held_out_dir.is_dir():
        # Two thirds as many as the positives is plenty: these are only ever the negative class, and the
        # cost of a negative is a full generation each.
        n_adjacent_focused = max(1, n_focused // 3)
        n_adjacent_rambling = max(1, n_rambling // 3)
        print(f"\n--- held-out corpus (adjacent negatives): {held_out_dir} ---")
        held_out_passages = load_passages(held_out_dir, rng,
                                          n_adjacent_focused + 3 * n_adjacent_rambling,
                                          by_length=False)
        held_out_questions = generate(base, model, held_out_passages,
                                      n_adjacent_focused, n_adjacent_rambling,
                                      label="adjacent", on_corpus=False)

    payload = {"corpus": "Optimalverse fan fiction saved from fimfiction.net, local to the developer "
                         "machine (not in this repository)",
               "corpus_dir": str(corpus_dir),
               "held_out_dir": str(held_out_dir) if held_out_dir else None,
               "seed": SEED,
               "passage_chars": PASSAGE_CHARS,
               "generator_model": model,
               "questions": questions + held_out_questions}
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(questions)} on-corpus and {len(held_out_questions)} adjacent questions "
          f"to {OUT_PATH}")


if __name__ == "__main__":
    main()
