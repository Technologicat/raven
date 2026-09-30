# Brief: the per-document LLM pass

**Unnumbered**, following the convention adopted after brief 15 — `markdown-block-rendering`,
`wake-word-voice-input` and `ligature-repair` are all unnumbered, and numbering was abandoned at 16.

> **Previously "brief 17".** Scoped out of brief 15 on 2026-08-04, given a reserved number, and never
> written. The reservation is now retired: refer to this by name. The dangling reference in
> `briefs/done/researchers-night/done/15_headless-agent-driver-brief.md` (`:362`, `:946`) stays as it stands —
> `done/` is a historical record and is not retconned — but the live content is here.

> **Line numbers are as of 2026-08-12** and want verifying.

**Gate: `next`.** Not exhibit work. But see the user count below: this is the most-pointed-at unwritten
thing in the project, and two of its users now carry measured costs.

## What it is

A **per-document LLM pass**: run the same question over every item in a set, with **retry, cache, resume and
progress**.

It sits one level above `raven.librarian.agent` (brief 15). That surface answers "run one turn and tell me
what happened"; this answers "run one turn per document over two thousand documents, and survive the
afternoon."

## Why now: eight users, none of which knew about the others

Each surfaced from a different direction. That is the pattern that says a shared primitive is missing,
rather than six features being wanted.

1. **`raven-pdf2bib`** — eight `perform_throwaway_task` call sites, each wrapped in its own hand-written
   retry loop: the same six lines, eight times, in one 1058-line file. No caching and no resume, so a crash
   at document 2400 restarts from zero.
2. **`rag_live_corpus`'s persistence layer** — a `PersistentForest` per sample plus a JSONL ledger, worth
   lifting wholesale. Those runs take an hour and the machines reboot.
3. **`briefs/design/corpus-interrogation-sketch.md`'s map stage** — `summaries = map(summarize, docs)`. Note
   `summarize` is *shipped code that is currently switched off*, sitting in the importer with progress, ETA
   and caching already written; what is missing is `synthesize` and a place for both to live.
4. **Mid-run LLM backend recovery for batch tools** — the model-loaded work made `raven-pdf2bib` and
   `raven-importer` stop at *start time* on a backend that is unreachable or has no model loaded, and
   explicitly deferred the mid-run case. CC's three deferred questions are this brief's scope exactly: how
   long to wait before giving up, whether to resume or restart, and what to do with the documents already
   written.
5. **A crash during ingest loses the whole run** — the measured version. The delayed-commit coalescer defers
   a commit one second after each finished read, so on a large corpus it never fires until the reads *stop*:
   **~40 minutes on the 1268-PDF fulltext corpus (2026-08-06)**, with every extracted document pending in
   memory and nothing on disk.
6. **Two shapes found while implementing brief 15** — a VLM pass over page images, and "here is a fulltext
   PDF, what does it say about X?" over a set. Brief 15 names the batch mechanics of both as this work.
7. **A title for every document, upon import** (Juha, 2026-09-29). `chatutil.document_label` names a document
   from its own content — exact for a BibTeX record, and for anything else the first substantial line, which
   is weak on fulltext and on fiction. Since 2026-09-29 that label is on every document search match's handle
   in the chat log, so where it is weak the reader sees it. An LLM asked for the title once per document at
   import, stored with the document, would make the label good in the general case. Waits for the unified DB
   (brief 13), where the import is.
8. **The corpus filter** — `briefs/corpus-filter-brief.md`, generalizing `investigations/aokk-corpus-scope/`.
   Two of its three model-calling scripts carry their own copy of this loop — a JSONL appended per
   answer, keyed on citekey, with a re-run skipping what is already there — and the third is driven slice
   by slice with `--skip`. **This brief goes first** (Juha, 2026-09-29), so that the filter is built on
   the primitive rather than lifting a third copy. It brings two requirements the other
   seven do not state, both measured there rather than predicted — see *Cache key* and *Batching* below.

Corpus sizes make several of these concrete rather than prospective: ~12k hydrogen abstracts already
ingested, ~2500 one-page ECCOMAS 2024 conference abstracts, an arXiv AI fulltext set of 1200+ full papers.

## What it is not

**Not an orchestration framework**, and it must not become one — no agent-role DSL, no supervisor
abstraction, no declarative pipeline. The scope is a loop with durability. Brief 15's scope note applies
here, with the correction recorded in the triage decisions: what is ruled out is a *framework*, not
scripting, and the scripting language is Python.

**Not the map-reduce engine either.** The corpus-interrogation sketch is explicit that `summarize` already
exists and what changes the size of that job is *"lift `summarize` out of the importer into the library, add
the reduce, and let both run against a scope"*. This brief is the lifting-and-durability half. `synthesize`
belongs to that sketch.

## Design starting points

### Resume is the load-bearing feature

Everything else here is convenience; resume is what makes an hour-long run survivable. Two of the eight users
exist *only* because it is missing.

The shape follows from what already works: `rag_live_corpus` keeps a JSONL ledger beside a
`PersistentForest`, and that pairing is worth lifting rather than redesigning. A ledger of completed items,
appended per item, gives resume, progress and the caching story at once — resume is "skip what the ledger
already has".

### Reset between documents, not one shared context

Settled 2026-08-11 while discussing the multi-agent question. A map stage processes documents independently,
so **a fresh `Forest` per item** gives isolation as well as bounded memory — no chance of one document's
context leaking into the next. The memory bound falls out of correct semantics rather than being arranged
for. This is what `pdf2bib` already did.

Persist per item where the run is worth keeping (`PersistentForest`), reset where it is not.

### The failure taxonomy is the interesting part

Not all failures are the same and the item's three deferred questions are really about telling them apart:

- **A bad document** — one item fails, the rest are fine. Record and continue.
- **A backend that has gone away** — every remaining item will fail. Stopping is right; the questions are
  how long to wait first, and whether to resume or restart afterwards.
- **A crash of the run itself** — nothing gets to decide anything, which is why the ledger has to be on disk
  before it is needed rather than written at the end.

Conflating the first two is the current failure mode: a batch run against a dead backend produces a
thousand "failed" documents that were never tried properly.

## Step 0: one model record in `llmclient`, shared with Librarian

Decided 2026-09-30 (Juha): the instrument stamp needs the model's identity, and Librarian already builds
one for its character card, so both come from one function rather than two copies. It is the first step
of this build, not a separate item, because the stamp is what needs it.

- **One reader of the model record, returning structured data.** `llmclient` reads LM Studio's
  `/api/v1/models` for the loaded model: id, quantization name and bits per weight, and the loaded
  instance's whole `config` block. Librarian formats its label from that; the stamp hashes it and writes
  the block into report headers (see *Decided 2026-09-29*, the cache key).
- **What moves.** `llmclient` reads LM Studio's `v0` in two places: backend-flavor detection, which keys
  on the `state` field, and `_resolve_model_info`. Loaded or not, vision, and context length all move to
  `v1`'s shapes — `loaded_instances` for `state`, `capabilities.vision` for `type == "vlm"`, the instance
  config's `context_length` for `loaded_context_length` (field names checked against a live instance,
  2026-09-30). About ten fixtures in `test_llmclient.py` mock `v0` and change with it.
- **Open: whether older LM Studio versions lack `v1`.** If they do, `v0` stays as a fallback, which means
  two parsers to keep and test. Find out before choosing.
- **Cost: moderate.** A foundation-layer change with a real test surface, so its own commit ahead of the
  pass itself.

## Decided 2026-09-29

Six of the seven questions below, settled in discussion with Juha. Question 5, progress reporting, was
settled the next day; see its own section after this one.

- **Where it lives (1): a sibling module of `raven.librarian.agent`**, built on `agent.ask_record`, as the AOKK scripts are. The name is
  still open, to come from what the module turns out to do. The AOKK scripts already import `agent` from
  outside the librarian package, so that dependency direction is in use.
- **The ledger (2): one JSONL for results and progress together**, appended one line per item as each
  answer arrives, a later line for an item superseding an earlier one. Reasoning traces go to a sidecar
  file beside it. This is the AOKK scripts' format, run there over about 4300 records.
- **The cache key (3): a caller-supplied item id plus an instrument fingerprint.** The id is whatever the
  caller has — citekey, path, content hash. The fingerprint hashes the prompt and anything else that
  decides what an answer means, and goes into the filename as well as into each line
  (`extract_fields.py`, `instrument_fingerprint`).
  - **A ledger can be seeded from answers made elsewhere** (Juha, 2026-09-30), under the fingerprint of
    the instrument that actually made them. The corpus filter needs this to replay the prototype's
    judgements through the finished tools; see `corpus-filter-brief.md`, question 6. With the
    fingerprint in the key, seeded answers never pass for the new instrument's. So the caller needs two
    things: to build its outputs from a *named* instrument's answers without asking anything (the replay),
    and to have a run under a new instrument re-ask rather than reuse them, since they are different
    measurements.
  - **The model is part of the instrument** (Juha, 2026-09-30), so the fingerprint hashes the model's id
    and quantization along with the prompt. Without them, switching models would silently reuse the old
    model's answers as current. LM Studio reports both per model; `llmclient._format_lmstudio_model_label`
    already reads them for Librarian's model identity, and the loaded context length with them.
    - **Record the whole load configuration LM Studio reports**, from `/api/v1/models` rather than the
      `/api/v0/models` that `llmclient` reads. Checked against a live instance on 2026-09-30: `v1` gives
      each loaded instance's `config` — context length, batch sizes, `parallel`, flash attention,
      speculative decoding and its draft settings, KV cache offload — beside the quantization's name and
      bits per weight. Which of these can change an answer is not established here, and taking the
      whole block into the report header costs nothing and settles nothing prematurely. Whether the
      fingerprint hashes all of it, or only id and quantization, wants deciding when this is built.
    - **What the backend does not report has to be declared.** The KV cache quantization is the known
      case: the same check found no field for it anywhere in the `v1` listing, and it is set by hand (the
      maintainer runs `q4_0`, since the context that fits otherwise is too short to be useful). So a run
      takes a free-text declaration of such settings, which goes into the fingerprint and into every
      report's header. Unstated, the report says so, rather than implying there was nothing to state.
    - A generic OpenAI-compatible backend reports neither id nor quantization reliably. There the stamp
      records what it could learn and says what it could not, on the same principle.
- **The backend policy (4): on any failure, probe the backend once, and stop if it is the backend.**
  `llmclient.reconnect(settings)` re-probes and returns a `backend_status`. Anything other than
  `backend_ready` stops the run, with `describe_backend_status`'s message, which is the wording batch
  tools already use. `backend_ready` means the fault was the document's: record it and go on.
  - **Stopping, not waiting, is the policy** (Juha). Against a local backend a failure usually means it
    stays down until the operator looks at it, and a batch run is typically one they have walked away
    from, so a retry loop would be waiting on nobody.
  - **Resume or restart stops being a question.** Resuming is re-running the same command: the ledger
    skips what is done, and a document recorded as failed is retried.
  - No thresholds are needed — no count of consecutive failures, no timeout to tune.
- **The first user (6): a port of `investigations/aokk-corpus-scope/extract_fields.py`**, not
  `raven-pdf2bib`. It is 363 lines against 1058, and its outputs are on disk, so re-running it on the
  primitive has a reference to compare against. `raven-pdf2bib` follows, as the heavier test.
- **Batching (7): the unit of work is a function from a list of items to `{id: answer}`**, with the batch
  size a parameter and the ordinary per-document case a batch of one. The ledger stays per item, and a
  failed batch is recorded as that many failed items, all retried on the next run.

## Decided 2026-09-30: progress reporting (5)

Settled with Juha. The aim is UX: give the user up-to-date information whenever there is some, without
spamming the log.

- **Two levels, kept apart.** *Within an item*, the caller's `on_progress` passes straight through to
  `agent.turn` (streamed chunks; `llmclient.make_console_progress_handler`, `agent.stream_log`). *Across
  items*, the pass reports at the granularity its calls actually operate at — per batch, a batch of one
  being the per-document case.
- **Push: an `on_item(event)` callback after each batch**, carrying done, total and failed counts, the
  ids just finished, the elapsed time, and the `unpythonic.ETAEstimator` instance itself rather than a
  string from it — `.formatted_eta` is the usual want, but a programmatic caller may want the numbers.
- **Resume counts only what is left.** At startup, read how many items the ledger already has, and give
  the estimator the remaining count as its total, counting this session's items from zero. Counting the
  resumed items as done would make the ETA wildly optimistic.
- **Pull: the latest progress is queryable** from the run while it runs — a status line for a GUI label,
  as the Visualizer importer's window shows, and the numbers for a progress bar. A GUI polls; the
  callback is for pushing to a console or a log.
  - The importer maps onto it directly: `progress.set_micro_count(total)` once, then `tick()` per item,
    which makes `_summarize` the natural second user after `extract_fields.py`.
- **The console default is a log line per batch**, not a progress bar: runs are long and often
  unattended, and a line leaves a history where a bar keeps only its latest state. **Rate-limited**, so a
  run of small batches does not spam the log.
- **Every failed item is logged individually, and never rate-limited** — a failed document is what a
  watcher of the run wants to see when it happens.
- **Cancellation reaches into the running batch.** A batch typically takes ~30 s, too long to wait for.
  So the run has a thread-safe `cancel()`, and the pass wraps the caller's `on_progress`: while the flag
  is set, every streamed chunk answers `llmclient.action_stop`, which interrupts generation and lets the
  turn finalize with what it had. The pass then drops that batch unwritten and stops; earlier batches are
  already in the ledger, so a resume starts from the one that did not finish.
  - Latency is one chunk, except while the backend is still reading the prompt and emitting nothing —
    then the cancel waits for the first chunk. Not measured; probably seconds for batched abstracts.
  - The same flag is checked between batches, which covers a cancel that lands between turns.

## What this brief must settle before implementation

1. **Where it lives, and its name.** Beside `agent` as a sibling module, or as a layer in the same file.
   Brief 15 left its own naming open for the same reason and the answer came from what the module did; do
   the same here.
2. **The ledger format and location.** JSONL beside the datastore is the existing precedent. Whether it is
   the same file for progress and for results, or two.
3. **Cache key.** What makes two runs "the same item" — content hash, path, or a caller-supplied id. The
   sidecar store is already content-addressed, which argues for the first, but a caller re-asking a
   *different question* about the same document must not hit the cache.
   - **The corpus filter has a working answer for the question half.** `extract_fields.py`'s
     `instrument_fingerprint` hashes the prompt together with the vocabularies it offers, stamps each answer
     with it, and names the output file after it. The prompt is in the hash because an edit to the
     instructions alone moved records from one value to another there, so answers from before and after it
     are different measurements and must not be pooled. The filename half is what saves every consumer
     from having to filter on the stamp — one of them did not. The item half is a caller-supplied id
     there (the citekey).
4. **The mid-run backend policy**: how long to wait, resume or restart, and what happens to documents
   already written. CC deferred these deliberately; they are the reason this brief exists rather than a
   detail of it.
5. **Progress reporting shape.** `summarize` in the importer already has progress, ETA and caching — read it
   before designing, since lifting may be most of the work.
6. **Whether `raven-pdf2bib` is converted as part of this or after.** It is the loudest user (eight
   hand-rolled retry loops) and the best test that the API is right; it is also a 1058-line file that
   nothing else depends on this brief to fix.
7. **Batching: whether the unit of work can be several items per model call.** Every LLM pass the corpus
   filter makes is batched — forty titles per call in the judge's first pass, ten records in the others.
   For the judge's second pass the reason is measured: one call per record put it at several hours,
   against the first pass's one (`judge_scope.py`, `judge_abstracts`). That sits awkwardly with *a fresh
   `Forest` per item* above. The ledger
   stays per item. What changes is a failure: one malformed field in the reply loses the whole batch, so a
   failed batch has to be recorded as that many failed items, all retryable. The reasoning trace covers
   the whole call, so `extract_fields.py` writes one trace entry per call, naming the keys that shared
   it. Whether batching belongs in the primitive or in the caller is the question; the requirement is
   that the primitive does not rule it out.
