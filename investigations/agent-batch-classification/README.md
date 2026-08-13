# Classifying a real paper pile with `raven.librarian.agent`

Measured 2026-08-13, against a working researcher's own collection of ~1600 papers accumulated over years.

The task, in the user's words: go through the pile and say which field of science each paper belongs to, in a
form that supports moving the strays out — the collection is supposed to be about AI research, and it has
drifted. That is a plausible thing to actually want, which is why it was worth doing against a real pile
rather than a tidy fixture.

It doubles as an exercise for `raven.librarian.agent`, the scripting surface over the agent loop: no GUI, no
character card, no tools, ~1600 classifications, and a shape (batch in, structured out) that is not what the
chat frontends exercise.

## What the outputs are, and why they are not here

The script writes a TSV, a resumable JSONL and a generated mover script. **None of them are committed**, and
should not be: they are a list of one person's filenames, which is a description of what they read, and this
repository is public. Re-run the script to regenerate them locally.

The same reasoning is why `--dir` is a required argument rather than a default — a personal path as a default
is a personal path waiting to be committed.

## Method

**Titles carry almost all of the signal.** 1545 of 1578 filenames in the pile follow "Authors YEAR - Title
[- id][- note]", so the model is asked about *filenames*, in batches of 40, and only entries whose filename
says too little are escalated to reading the document's first page. Reading every document would have cost
~1600 pypdf extractions to learn something the filename already said.

**Batches are index-keyed, and short answers are tolerated rather than trusted.** Each item is numbered, the
model returns a JSON array carrying those numbers back, and only answers whose index resolves are recorded.
A batch that comes back one item short leaves that item unrecorded, and therefore in the to-do list for the
next run — which is also what makes a re-run the recovery path for a failed batch. The first trial did drop
one of forty, so this is not theoretical.

**The run is resumable.** Every answer is appended to a JSONL as it arrives, and a re-run skips what is
already there. At ~95 s per batch of 40 the full pile is about an hour; losing that to a backend hiccup at
minute 50 would be the difference between a usable tool and a demo.

**Escalation triggers on either a low-confidence answer or a filename with too little prose in it** — the
second condition independently of what the model said about the first, which turned out to matter (below).
2.3% of the pile trips the filename condition.

## The finding worth keeping: a model will read a filename it cannot read

The first trial classified `2006.05563.pdf` as AI with **high** confidence, explaining *"ArXiv ID matches
GPT-3 foundational AI paper"*. GPT-3 is 2005.14165. Two other bare-identifier filenames got the same
treatment — a confident topic, a confident field, and a stated reason that amounted to claiming recall of an
arXiv ID-to-paper mapping.

This is worth naming precisely, because it is not the usual "the model was wrong". The model was **asked to
judge from a filename, given a filename with no topic in it, and answered as though it had one** — supplying
not just a guess but a justification for the guess. Confidence was not merely miscalibrated; it was inverted,
sitting highest exactly where the input carried least.

Two defences, and the interesting part is that only one of them is a prompt:

- **Tell it that an identifier is not a description**, explicitly, including "do not claim to recognize which
  paper an arXiv id refers to; you cannot". After this, every bare-ID filename came back `low` with reasons
  like *"Filename is just an arXiv ID, carries no topic"*. That is the right answer and the model reaches it
  readily — it simply does not volunteer it.
- **Do not let the model's own confidence be the only escalation trigger.** The structural condition (does
  this filename contain prose?) is computed in Python and cannot be talked out of firing. Had escalation
  depended on the confidence field alone, the fabricated answers would have been the *least* likely to be
  checked.

The second is the more durable lesson: when a model's self-report is the thing that decides whether to look
harder, the failure mode where the self-report is wrong is precisely the one that goes unexamined.

## Files

| File | What it is |
|---|---|
| `classify_papers.py` | The classifier. `--dir` is required; `--limit` runs a trial batch; `--no-escalate` skips document reading; re-running resumes. |

Generated at runtime and deliberately not committed: `papers-classified.jsonl` (resumable state),
`papers-classified.tsv` (the classification), `move-stragglers.sh` (dry run unless given `--commit`).
