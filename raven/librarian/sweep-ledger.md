# The sweep ledger

`raven.librarian.sweep` asks one question of every item in a set, and keeps the answers in a **ledger**: a
JSONL file that a run appends to as answers arrive, so that an interrupted run resumes and a finished one
needs no backend to be read again. Raven's batch tools keep their LLM answers this way (`raven-scopebib`,
`raven-deduplicate --judge`, `raven-pdf2bib`), and so does the lorebook, for its terms' inflected forms
(`raven.librarian.lorebook_expansion`).

This describes the files, for whoever reads them without Raven: a port of the lorebook to another frontend,
an analysis script, a person checking what a run did. What the *answers* hold is each tool's own business;
the lorebook's is in [`lorebook-porting.md`](lorebook-porting.md#the-forms-raven-generated).

## The files

A ledger is three files in one directory, named for the run and its **instrument**:

| file | what it holds |
|---|---|
| `{name}-{fingerprint}.jsonl` | the answers, one JSON object per line, appended to |
| `{name}-{fingerprint}-traces.jsonl` | every model call, whole, one line per batch |
| `{name}-{fingerprint}.state.json` | how far the latest run has got, rewritten as it goes |

`name` says what is being asked (`expansions`, for the lorebook). `fingerprint` is eight hex digits naming the
instrument, below. Ledgers for several instruments sit side by side, one set of files each.

## The fingerprint

**A hash of everything that decides what an answer means**, so that a change to any of it starts a fresh
ledger instead of mixing two measurements in one file:

```python
sha256("\x00".join([material, f"model={model_id}", f"quantization={quantization}", f"declared={declared}"]))[:8]
```

- `material` is the caller's half: the prompt, with anything else that changes what is asked.
- `model_id` and `quantization` are what the backend reported; either may be `None`, written as such.
- `declared` is what the operator stated that the backend cannot report (a KV cache quantization, say), or
  `None`.
- A run that offers the model tools appends `f"tools={json.dumps(tools, sort_keys=True)}"` as a fifth part.

So a ledger belongs to **one prompt on one model**. Switching models finds no answers and asks again; switching
back finds the earlier ledger where it was left.

## The lines of the answers file

Three kinds, each carrying `"v"`, the fingerprint, so a line copied in from another ledger is recognized and
ignored:

- **A run's header**, at the start of each run that asks anything:
  `{"header": {...}, "v": "...", "started": "2026-11-23T09:46:48"}`. The header records the backend's URL, the
  model's whole record as the backend reported it, and what was declared.
- **An answer**: `{"id": "...", "v": "...", "status": "ok", "answer": ..., "d": "..."}`. `answer` is anything JSON
  can hold. `d`, when present, is the **digest** of the item as it was asked, below.
- **A failure**: `{"id": "...", "v": "...", "status": "failed", "error": "..."}`. The item was tried and is to be
  tried again; a failure is also how an answer is retired on request ("asked again on request").

**A later line for an item supersedes an earlier one**, so the file is a history and its current state is the
last line per `id`. Nothing is ever rewritten: an answer asked again stays in the file above its successor.

**A line that does not parse is skipped.** A run killed mid-write leaves one, and its item is simply asked again;
later runs start a fresh line after it.

## The digest

**The fingerprint says the question has not changed; the digest says the item has not either.** It is the first
sixteen hex digits of the SHA-256 of the item *as the prompt shows it*:

```python
sha256(rendered_item.encode("utf-8")).hexdigest()[:16]
```

An answer whose `d` differs from the item's digest now, or that has none, is about other text, and reads as not
yet answered: the next run asks it again. A reader of the ledger should apply the same test, rendering each item
as the tool that wrote the ledger did.

## Reading a ledger

1. Read the answers file line by line, skipping what does not parse, and what has another `v` or no `id`.
2. Keep the last line for each `id`.
3. Of those, the answers are the ones with `"status": "ok"` — and, if you can render the items, whose `d`
   matches.

`raven.librarian.sweep.Ledger` does exactly this (`Ledger.answers`), and `python -m raven.librarian.sweep DIR`
reports how far each ledger in a directory has got, from its state file.
