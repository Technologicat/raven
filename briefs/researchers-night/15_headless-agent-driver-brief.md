# Brief 15: a scripting surface over the scaffold

> **Line numbers are as of 2026-08-04 and want verifying.** They were read from a shallow clone that was
> already a couple of hours behind a moving tree, and the files most often cited here (`app.py`,
> `chat_controller.py`) were among those being committed to that day. Treat every `file.py:NNN` below as a
> pointer to a thing that exists, not as a coordinate.

**v0.2.9 work, and first in that queue — ahead of 05.** Written 2026-08-04, during the deferred-TODO triage.
Not admitted to v0.2.8: it is not a defect, so the freeze covers it by the letter, and the argument that it
would make 09's own validation easier is exactly the "one more thing" the freeze exists to stop. It goes first
after the tag, and the reason is timing rather than value — see below.

**This brief consolidates rather than invents.** The design was largely already written, in `TODO.md` under
Librarian → Core features ("Make the scaffold scriptable", 2026-07-29). What follows lifts it into brief form,
adds the import-side prerequisite, and cuts one piece out into a brief of its own.

**This brief is now the authority, and the `TODO.md` entry is redundant** (2026-08-11). It said the opposite
— that `TODO.md` was right where the two disagreed — which had it backwards: `TODO.md` is a queue and is
ephemeral by design, so nothing should persist only there. Everything in that entry has been checked against
this brief and is either here or deliberately scoped out; the last piece to be moved across is the wire-level
exclusion, below. The entry can be retired whenever `TODO.md` is next edited.

**Filed in two places under different names**, which is how the poorer copy came to be the one that got
deferred: `TODO_DEFERRED.md`, *"Headless scaffold mode for `ai_turn` (scriptable agent layer)"* (filed
2026-06-03) is the same work as the `TODO.md` entry, minus the two-entry-point split and the result record.
Both close against this brief, along with *"Lazy `api.initialize` in `llmclient` and `hybridir`"*.

**Not absorbed: *"scaffold: collect `ai_turn`'s callbacks into a single bundle object"***. It looked like a
prerequisite and is not — see the callback section below. It remains a GUI-side ergonomics cleanup, and is
now independent of this work.

Deferred items are cited by **heading text, never by line number**. The line numbers this brief originally
carried had all three drifted by 2026-08-10, and one of them (`:1825`) had landed on a different item's real
heading — which reads as valid and is the failure worth avoiding. The file is about to get shorter, so any
line number written into it is already wrong.

## The problem

`llmclient` is already an LLM *scripting* layer, used by `raven-pdf2bib` and friends. There is no equivalent
one level up. Driving the full agent loop — LLM plus tool-calling, branching chat tree, RAG — requires either
a GUI or a TUI. `perform_throwaway_task` is the only scripting entry point and is deliberately thin: no
datastore, so no attachments, no branch, no retrieval.

The cost recurs. Every batch experiment re-implements a slice of `scaffold.ai_turn` by hand, and a replica
diverges from the real thing precisely where the interesting behaviour lives.

**`manual_tests/rag_live_corpus.py` is the specification by example.** Written 2026-07-29 to measure one
behaviour, it ended up hand-rolling most of what this surface would provide. This is not speculative API
design; it is code that already exists in the wrong place.

**The acceptance test is a set of existing scripts, not a checklist.** The probes the `TODO.md` survey
enumerates live in two trees — the prompt-shape and wire-level ones under `investigations/context-injects/`,
the full-turn ones under `briefs/librarian-extension/manual_tests/`. This work is done when
`inject_shapes`, `assembled_shape`, `absent_fact` and `rag_placement` can be rewritten against Part A without
reaching through a private door, and `rag_live_corpus` and `rag_tool_rescue` against Part B without
hand-rolling a branch walk.

**A third group is deliberately excluded, and must stay excluded.** The survey that produced those two entry
points sorted twelve probes three ways, and the third are the *wire-level* ones — `backend_capabilities`,
`gemma4_reasoning_roundtrip`, `vision_check`, the `webfetch_*` set, `datetime_inject`. Those post raw to
`/v1/chat/completions` on purpose: what they measure is the **backend**, and routing them through `llmclient`
would put the thing under test behind the thing doing the testing. A scripting surface is not for them, and
"port the remaining probes onto it" is the shape of a later mistake — one that would quietly turn a set of
backend measurements into measurements of Raven's handling of the backend.

Rewriting them is not required by this brief — they work, and `investigations/` exists so that measurements
stay reproducible as they were made. But *being able to* is the criterion, and checking it against real
scripts costs nothing beyond reading them.

## Why it goes first

Seven weeks of investigation-heavy work sit ahead of it — the markdown renderer set, the turn-sequencing race,
the auto-RAG-as-mistake bug, table layout, equation scoping — and every one wants a scriptable driver. Landing
it first amortizes it across all of them. Force multipliers reliably lose to deadline items and then get built
after the deadline, when the work they would have multiplied is already done.

It also lands immediately before three consecutive agent-loop features (05 lorebook, 04 MCP, 06 Hindsight)
whose behaviour is what one would most want to script.

## What this is not

Not a generic agent harness, and it must not become one: no plugin system, no workflow DSL, no orchestration
layer. What is not commodity here is what a generic harness cannot bring — a local corpus with its index, the
branching chattree, the provenance machinery. The surface is programmatic access to *those*. The agentic part
is only that a per-document pass may call tools rather than being one shot.

**MCP is not a user-plugin system**, and the distinction belongs in writing here too. MCP is tool *supply*:
an external process offering capabilities inward. User plugins would be app *extension*: third-party code
inside the process. Not wanted. Conflating the two is how a library grows an extension system by accident.

## It is a programming library, not a product

Visualizer and Librarian are what ships to researchers; the engines under them are useful for things those
products do not do, and that is what the surface is for. Being a library means `__all__`, docstrings that
stand as the documentation, and some statement about what may move. Mostly not code.

This is also the diagnosis for the `_perform_injects` problem below rather than a separate observation: it is
*already* a library API and has simply never been told so.

## The callback wall stays, and the surface has no callbacks

`ai_turn`'s mandatory callbacks are deliberate, on the fail-fast principle, and they earn that for a GUI
client — one that forgets `on_llm_progress` is genuinely broken and should say so loudly. `minichat` passes
eight explicit `None`s (`minichat.py:608–619`, in the `scaffold.ai_turn` call at `:598`) for the same reason.

They earn nothing from a script that will never draw a progress bar, and the wall bites there instead: a probe
silently rotted when brief 10 removed `on_nomatch_done`.

**So the fix is not defaults.** Defaults would weaken the wall where it protects someone in order to help
someone it cannot protect. The scripting surface takes **no callbacks at all** — the events become the
returned record of what happened, which is the thing every probe currently reconstructs by walking the branch
afterwards. Keep the wall where it works; do not propagate it to callers it cannot serve.

## Part 0 — lazy `api.initialize` in `llmclient`

> **Landed 2026-08-10.** `api.initialize` moved out of module scope into `_client_api()`, called on first use
> by the only two things in `llmclient` that reach the server — `websearch_wrapper` and `webfetch_wrapper`,
> the tool entrypoints for the model's `websearch` and `webfetch`. The four apps that need the client stack
> initialize it explicitly
> (Librarian, minichat, `pdf2bib`, and Visualizer in its LLM keyword mode), which is the honest form: an app
> declares what it will use. The description below is of the state *before* that, and its `llmclient.py:81`
> pointer no longer resolves.
>
> The dividend was larger than the item: importing `llmclient` no longer drags in `spacy`, `transformers` or
> `av`, so `test_scaffold` runs in CI instead of being skipped there, and `llmclient.configure` can be called
> from a test. `raven/librarian/tests/conftest.py` still carries a comment claiming the opposite — it was
> written when that was true, and its fixture could now use the real `configure`.

The one prerequisite, and it is import-side rather than design-side.

`llmclient.py:81` calls `api.initialize(...)` at module top, so importing `llmclient` both requires the full
`raven.client.api` chain to succeed — `spacy`, and `av` via the vendored Kokoro streaming writer — and runs
the side effect. (**Not** qoi, and **not** torch: CI installs both, torch as the CPU wheel, which imports
fine. The original wording named those two and sent a reader chasing dependencies that were already
present — corrected 2026-08-10, at the source in `test_scaffold.py` as well.) `scaffold` imports
`llmclient` at module level and inherits it, which is why `test_scaffold.py` carries a `pytest.importorskip`
and why scaffold coverage is invisible in the minimal-deps CI job.

A probe that must boot a TTS stack to test tool-calling is a probe nobody writes.

The natural seam is `llmclient.setup` — every caller already goes through it, so initialization can move there
(idempotent, called once). While in there, audit `llmclient`'s and `hybridir`'s other module-top imports for
side effects; `scaffold.py`'s `TYPE_CHECKING` import of `hybridir` is the model.

**Falls out**: remove the `importorskip` from `test_scaffold.py`, and scaffold's ~119 statements start
contributing to CI coverage.

**Measured 2026-08-10, because this part of the plan asserts an outcome rather than a fact — and it needed
one correction.** Moving the *call* into `setup` does not by itself remove the `importorskip`: the skip is
caused by the module-level `from ..client import api`, not by `api.initialize(...)`. What the measurement
settles is that deferring that one import is sufficient. Per-module, the heavy top-level packages pulled in
by an import are:

| module | pulls |
|---|---|
| `chatutil`, `chattree` | numpy |
| `librarian.config` | numpy, **torch** |
| `common.netutil` | — |
| `client.api` | av, kokoro, pygame, qoi, sentence_transformers, **spacy**, torch, transformers |
| `librarian.llmclient` | *exactly `client.api`'s set* |

So `client.api` is the only heavy contributor beyond `config`'s torch, and torch is installed in CI. `api` is
used in `llmclient` at three sites only — the bootup call, and inside `websearch_wrapper` and
`webfetch_wrapper`, both of which run only when a network tool is actually invoked. Deferring the import to
those three is therefore enough, and small.

Note `librarian.config` pulls torch independently (via `common.video.colorspace`). That is not a problem to
solve here — CI installs the CPU torch wheel, which imports fine — but it does mean "scaffold imports
nothing heavy" will not become true, and should not be written down as the goal. The goal is that it imports
nothing CI lacks.

## Part A — build the turn's prompt and hand it back

> **Landed 2026-08-11.** `_perform_injects` is `scaffold.build_turn_prompt`: public, in `__all__`, returning
> a new list instead of mutating the caller's. The two private doors it still reached through are public too:
> `scaffold.make_tool_context` (`build_turn_prompt` requires a tool context, so while only a private function
> could build one the public entry point could not be called without a private call, and "public" was
> nominal) and `llmclient.serialize_history_for_wire`. The latter went public as-is rather than behind a
> narrower wrapper: its `datastore=` is already optional and defaults to `None`, which is the no-attachments
> case a prompt-shape probe has, and a caller with a real chat needs the full function anyway — so a wrapper
> would have been a second name serving only the case the default already serves.
>
> **The criterion is checked by a test, not by inspection.** `TestPromptAssemblyFromOutside` in
> `test_scaffold.py` asserts the four names are in their modules' `__all__` and drives the whole path —
> `configure` → `make_tool_context` → `build_turn_prompt` → `serialize_history_for_wire` — with no backend.
> Re-privatizing any of them is not a signature change, so nothing else would have failed.
>
> `inject_shapes` and `rag_placement` are already clean, and always were: they build their own candidate
> shapes on purpose, which is why the criterion should never have named them (recorded above).
>
> **The prose sweep is complete, including the frozen TODO files.** Renaming a symbol inside body text moves
> no heading, and headings staying put is what the freeze is protecting while triage runs elsewhere — checked
> by line and heading counts before and after. The closed briefs under `librarian-extension/done/` keep the
> old name: they are historical records and are not repointed.
>
> **The migration made the probes measure Raven.** `absent_fact` and `assembled_shape` were also moved onto
> `llmclient.configure`, so they build Raven's real settings instead of forging seven of twenty-one fields
> with `system_prompt="You are a helpful assistant."`. That is the durable gain, independent of any single
> run's output.
>
> The re-run that followed found something worth knowing, and it is written up in
> `investigations/context-injects/README.md`. Three samples per arm at T=0 gave 2484, 30757 and 29684
> characters of reasoning **from identical requests** — so T=0 is not reproducible on this stack, and the
> one-sample comparison first written here could not have supported any conclusion. What survives is that
> the *as-shipped* wording is the one that runs away (3 of 4), while the two alternatives do not (0 of 3
> each) — which inverts the rationale that rejected `closing-note`, though that was measured on a different
> model. The probe now also sends Raven's own samplers rather than a bare temperature, which is the same
> "measure what Raven sends" correction one level down.

No backend involved. This is what the prompt-shape probes want: the prompt Raven *would* send, so they can
send it themselves and measure the backend without Raven confounding the result.

All of them reach into `scaffold._perform_injects` — a private function which, counted 2026-08-04, has
**seven callers outside its own module**, across three trees: `llmclient.py`, `test_scaffold.py`, four
probes under `investigations/context-injects/`, and `manual_tests/rag_live_corpus.py`. (The `TODO.md` note
says four; it undercounts.) That is a public API which has not been declared one. It gained a parameter this
session (`tools_are_spent`) and its callers survived only because the parameter defaults — next time the
change may not be so kind. Declaring it, naming it, and documenting the returned shape *is* most of Part A.

## Part B — run the turn and tell me what happened

> **Landed 2026-08-11.** `raven.librarian.agent` is `TurnRecord` (frozen dataclass), `turn` (run one, get
> one back) and `describe_turn` (build one by walking a branch). `turn` calls the real `scaffold.ai_turn`
> rather than reimplementing the loop, passing `on_prompt_ready` to itself so the record carries the wire
> histories — one per model call, `prompts[-1]` being the one that produced the reply.
>
> Both acceptance scripts are rewritten against it and both got shorter by more than the record's own size:
> `rag_tool_rescue`'s `run_once` went from a datastore, a factory reset, a `user_turn`, a fifteen-argument
> `ai_turn` and a branch walk to three lines, and `rag_live_corpus`'s phase F lost its `_ai_turn` closure
> and both of its walks. `rag_tool_rescue` was also *broken* — it still passed `tools_enabled=` and
> `speculate=`, retired by `37bf3c7` and `c24c89e` — so the same silent rot Part A found in the
> prompt-shape probes had reached the full-turn ones too. Rewriting them onto the surface is what fixes it
> for good: a probe that no longer names fifteen parameters cannot be broken by the fifteenth changing.
>
> **Three things the implementation settled that the brief did not ask about:**
>
> - **`reply` strips the persona prefix**, as both frontends do at render time (`Aria: ` is part of how a
>   message is stored, not part of what was said). Every hand-rolled walk read the stored form and so
>   reported the prefix; the record reports what a person reads, and `messages[-1]` still has the raw form.
> - **The turn's span stops at the head it started from.** Every existing walk ran to the root, which on a
>   multi-turn chat totals every turn on the branch — against a cap that is per-turn. `describe_turn` takes
>   `since_node_id` and `turn` passes it, so the counts answer the question the cap is asked. The 24
>   recorded `tool_budget` samples were rechecked per turn and are unaffected (their first turn called
>   nothing); noted in that investigation's README so it is not re-derived.
> - **Attachments need a file-backed datastore**, since a sidecar is a file beside it and the in-memory
>   default has nowhere to put one. `staged_images`/`staged_files` pass through to `user_turn` and are
>   refused up front rather than failing inside `imagestore` partway through the turn.
>
> What is *not* here, deliberately: the scripted backend (already out of scope, below), and any callback.

### Open: should an in-memory chat be able to carry attachments? (raised 2026-08-11)

`agent.turn` currently refuses them on a `chattree.Forest` and asks for a `PersistentForest`, because the
sidecar store is defined on the latter. The question is whether that is essential or incidental — a sidecar
is content-addressed bytes plus a description, and nothing about the *concept* needs a filesystem.

**Checked, and the two attachment kinds do not answer the same way**, which is what makes this a design
session rather than a small change:

- **Images would work in memory today.** `imagestore` touches the store only through `store_sidecar` and
  `read_sidecar`, both of which are bytes in, bytes out. `sidecar_url_to_data_url` builds a `data:` URL
  from those bytes and needs no path.
- **Documents would not, and the reason turned out to be wrong** (corrected 2026-08-11). The first reading
  said `docextract` is "path-based for every format", which described the signatures rather than any
  constraint: `pypdf`, `python-docx`, `python-pptx`, `odfpy` and `trafilatura` all read a stream, checked
  by feeding each one a `BytesIO`. The path-basedness was `docextract`'s own choice.

  So it was changed instead of worked around. `docextract.extract_text_from_bytes(raw, name)` is the
  in-memory entry point — the name selects the reader, since bytes do not announce their format — and
  `extract_text(path)` is now the thin one, opening the file and delegating. `sidecar_to_text` reads bytes
  and no longer asks for a path. **Both attachment kinds are therefore bytes-only at the store boundary**,
  which is the asymmetry gone.

Everything else that wants a real path is either GUI ("open the saved copy", "show it in the file manager")
or maintenance (`cleanup`'s `stat().st_size`, `appstate`'s existence check), and none of it is in a
script's way.

So the options are: leave the refusal (the error names the fix, and the fix is one argument); give `Forest`
a bytes-backed store and let documents materialize a temp file on read; or push a bytes interface down into
`docextract`, which is the clean version and the largest.

**Two use cases decide it, and they pull in different directions** (Juha, 2026-08-11). Both are one
throwaway turn per item, in a batch, whose output is text — which is what makes them scripting rather than
chat, and why the refusal is in their way at all:

- **A VLM pass over page images.** Rasterize a paper's pages, hand each to the model, get back what is on
  it — a modern VLM reads an equation photograph into LaTeX, so this is a conversion pipeline and not a
  demo. The bytes are generated in memory and nobody wants them afterwards. Under the refusal, a 40-page
  paper means a `PersistentForest` plus 40 content-addressed files to sweep.
- **"Here is a fulltext PDF, what does it say about XXX?"** The document is *already a file*, usually one
  of a stash the script is iterating over. Under the refusal, asking one question about each paper copies
  the entire corpus into a sidecar store — megabytes per item, duplicating what is already on disk and
  already durable.

  Distinct from the RAG path, which is the other way to ask this: `raven-indexer` chunks a corpus and the
  retrieval brings back passages. Attaching folds the *whole* document in, which is what a large context
  window is for and what "what does it say about XXX" wants when the answer is spread across the paper.

**So one mechanism does serve both**, which the earlier reading of this section denied: with `docextract`
taking bytes, every consumer of the sidecar store on the *read* side wants bytes and nothing else. A
bytes-backed sidecar store on `Forest` would make both attachment kinds work in an in-memory chat, with no
temp file anywhere and no second mechanism.

**The `chattree` half landed 2026-08-11**, to the decisions below. The sidecar store is split by what
actually differs: `Forest` owns the *policy* — content addressing, first-write-wins descriptions,
mark-and-sweep GC — over a dict-backed storage, and `PersistentForest` overrides only the members that
touch the filesystem. All 101 existing `chattree` tests pass unchanged, which is the invariant that
mattered; seven new ones pin the two backends answering alike — including that identical bytes get the same
content-addressed name in either, so a chat is not tied to where its attachments happen to live.

`sidecar_size` and `has_sidecar` are new, and between them they retire both non-GUI callers of
`sidecar_path` — `cleanup`'s size lookup and `appstate`'s existence check. `rescue_to_staging` copies bytes
rather than a file, which costs nothing since it already read both sides to compare them. What still asks
for a path is `chat_controller` and `cleanup_dialog`, reached only from a running Librarian, which always
has a file-backed datastore.

### Decisions (Juha, 2026-08-11)

- **The sidecar store moves up to `Forest`, backed by a dict**, with `PersistentForest` overriding the
  members that touch the filesystem. One store, both attachment kinds, no temp files.
- **`sidecar_path` and `sidecar_dir` are present-and-raising on the in-memory store**, with an error saying
  what to use instead. Absent would be the other option and is worse: `hasattr` then becomes the way to ask
  a question the type already answers, and every caller that wants a real file grows a branch.
- **`cleanup` stops asking for a path in order to get a size.** A sidecar's size is `len` of its bytes on
  either backend, so this becomes a store member rather than a `stat()` at the call site — which also
  removes one of `sidecar_path`'s two non-GUI callers. The remaining GUI ones are reached only from a
  running Librarian, which always has a file-backed datastore.
- **An in-memory store still wants a `sidecar_extractor`**, and arguably more than a file-backed one: its
  sidecars occupy RAM for the life of the process, and nothing else ever reclaims them. A long batch
  attaching a page image per item is exactly where that matters.
- **No option to persist the sidecars of an in-memory chat** — raised as a repeatability question, and the
  answer is that it would produce exactly the state the GC exists to delete. A sidecar is content-addressed
  bytes whose meaning lives in the payload referencing it, so writing the sidecars while the tree stays in
  RAM leaves a directory of hash-named orphans that nothing points at. It would also make "is this chat
  persistable" a question with three answers where the choice of class answers it once, and the upgrade
  path is already one argument: pass a `PersistentForest` and everything is kept, sidecars included.

  And repeatability does not need it in the first place, which is the part that settles it: a script's
  inputs are the script's own, and they outlive the run without Raven's help. The files are on the user's
  disk already — the fulltext PDF, the page images — and the next run offers the same ones again. What
  makes such a run reproducible is the input plus the script, which is where this repo keeps
  reproducibility anyway; Raven's copy of the bytes is a cache, and a cache is not an archive.

  **What re-attaching does cost is a fresh extraction**, and for a PDF that is `pypdf` parsing the whole
  document again. `textfilestore` memoizes on the content-addressed filename, so it is once per document
  per *process* however many turns use it — but a second run of the script pays it again. That is an
  argument for a persistent datastore on its own terms, though not the one it looks like: what is saved is
  not the extraction, which is not cached on disk either, but the asking. The previous run's answers are
  still in the chat, so the questions already answered do not have to be put again.

Worth knowing while doing it: **the GC already fails safe** without an extractor
(`prune_unreferenced_sidecars` logs and deletes nothing), so a partially wired store cannot lose
attachments.

**Two things these need that are separate from the sidecar question**, so that neither gets folded into it
by mistake:

- **Nothing in Raven rasterizes a PDF page.** `docextract._extract_pdf` reads the text layer through
  `pypdf`; there is no `pymupdf`, `pdf2image` or poppler anywhere in the tree or in `pyproject.toml`
  (checked). Page images have to come from somewhere, and that is a dependency decision. The fulltext case
  needs none of this — `docextract` already reads a PDF.
- **The batch mechanics are brief 17**, the per-document LLM pass already scoped out of this brief — retry,
  cache, resume, progress. Both of these are precisely its shape, and it now has two more prospective
  users.

**Landed meanwhile, since it was one check and this is what made it visible:** `agent.turn` refuses
`staged_images` when `llm_settings.model_is_vlm is False`. Librarian's attach button has always done this;
a script had nothing, so a page-image batch against a text-only model would have paid for every call and
got an answer about nothing. `None` still passes — it means the backend did not say.

**A turn returns what happened, not a node id.** Every probe re-implements the same branch walk afterwards:
count the tool nodes by name, count the *rounds* (an assistant message asking for tools, however many calls it
asks for), collect the reasoning that never reached `content`, find the reply. That walk is the actual result
of a turn, it is written out longhand each time, differently, and the round-versus-call distinction has been
got wrong at least once. Returning it removes the walk and fixes the distinction in one place.

**Start from `rag_live_corpus`'s copy, which is the one that gets it right.** Its `tool_calls` dict and
`rounds` counter implement exactly the distinction settled above, and it says so in an inline comment — "a
*round* is one assistant message asking for tools, however many it asks for". So Part B is largely lifting a
working implementation into a returned record, not designing one; the other copies are what it should replace.

Two things the record must cover that are awkward today:

- **The prompt.** The wire history is reachable only through `on_prompt_ready`, so a script that wants to
  assert on what was actually sent has to build a closure to catch it. In a callback-free surface, the
  assembled prompt belongs in the record.
- **Per-run overrides.** A/B-ing a knob currently means monkeypatching — `rag_live_corpus` swaps out a
  `chatutil` formatter to run its control arm. Fine in a probe, and a sign that per-run overrides belong in
  the surface rather than in module globals.

**The two parts are not layers of each other**: A must not talk to a backend at all, and B is useless without
one. A surface offering only B would leave the prompt-shape probes still reaching through the private door.

### What queues behind Part B (2026-08-11)

Three items, and **only the first is ordered by a dependency** — the rest is a judgement about value, stated
as such so that a later reader can overrule it without having to work out whether something would break.

1. **`perform_throwaway_task` moves to `agent` and is renamed.** **Unblocked 2026-08-11 — Part B landed, and
   the answer is that it dissolves**, but not until the surface grows two things. Read against the code
   rather than guessed: its first four lines are `agent.turn`'s one-liner path exactly (`chattree.Forest()`
   → `factory_reset_datastore` → user node → `linearize_chat`), and what remains differs in three ways, of
   which only two matter.

   - **It passes `tools_enabled=False`, and the surface cannot say that.** `internet_enabled=False,
     docs_enabled=False` still offers `get_current_time`, since it answers to neither switch —
     `maybe_tool_names_for_turn` returns the ungated group rather than an empty tuple. A one-shot with no
     tools at all therefore needs a way to withdraw them, and that is a real addition rather than a spelling.
   - **All ten call sites pass a progress callback**, each with its own symbol (`*`, `A`, `a`, `T`, `K`,
     `p`, and four `.`), so the dots on a `raven-pdf2bib` run say *which stage* is working. This is the one
     genuine counter-example to "the surface takes no callbacks": streaming progress is the single thing a
     returned record cannot express, because it is over by the time the record exists. The callback wall
     argument was about *event* callbacks a script will never draw; a progress spinner is not that, and the
     brief should not be read as having ruled it out.
   - Its `(raw_output_text, scrubbed_output_text)` return does *not* matter: `raw` exists only to
     reassemble an inline `<think>…</think>` from a time when reasoning was not separated. The record
     carries `reply` and `reasoning` as separate fields, which is the same information in better shape.

   **Both landed 2026-08-11**, and the conversion of the ten call sites is what remains — held back for the
   reason below rather than for effort.

   - `tools_enabled` on `scaffold.ai_turn` and `agent.turn`, sitting *above* the two group switches. A
     blanket switch was removed from `ai_turn` once, when the user-facing "Tools" toggle became the two
     group switches; the grounds were that a GUI user is never served by one, and that still holds, since
     nothing here is reachable from the GUI. It is a boolean rather than the tool-name list the old marker
     wished for, because a list would decide the same question as the group switches by a second route.
   - `on_progress` on `agent.turn`, the one callback the surface takes. Streaming progress is the single
     thing a record cannot carry, and an hour of silence on a local model is indistinguishable from a hang.
   - **And a bug the switch would otherwise have introduced** (Juha): the clock is delivered as a synthetic
     call to `get_current_time`, unconditionally. With every tool withdrawn, that stages a call to a tool
     the request does not declare — the shape models handle badly, and the reason the clock tool is normally
     offered whatever the switches say. So `tools_enabled=False` withholds the clock too. A scripted job
     rarely wants the time in any case.

#### Before the ten call sites move: what the same-system-prompt intuition is worth today

`perform_throwaway_task` runs its task under Librarian's own system prompt, character card and greeting,
and that is deliberate: **it was measured to improve results on the models current in 2025** (Juha). The
question the migration raises is which parts of that still earn their place, because the bundle has aged
unevenly:

- **A real system prompt rather than a stub** — still right, and Part A found the same thing from the other
  side: a probe forging `system_prompt="You are a helpful assistant."` measures a replica of Raven.
- **The character card and persona** — being the assistant character adds nothing to keyword extraction,
  and its conversational styling is the opposite of what a parsed output wants.
- **The greeting node**, which is what makes the exchange a chat at all.
- **The per-turn injects**, which *did not exist in this form* when the choice was made. `date_now` is
  harmless; **`reminder_to_write_conversationally` is the sharp one** — it instructs chat prose in a task
  whose output `raven-pdf2bib` parses. This is the real delta between the two paths now that the clock is
  handled, and it is why the conversion is not a drop-in.

**Settle it by measurement rather than by intuition — anyone's.** What made the 2025 choice good was that
it was measured, and the apparatus for redoing it is exactly what this brief just built: `agent.turn` with
`tools_enabled=False`, a `TurnRecord` per document, and a stash of real PDFs. Four arms — full injects, no
conversational reminder, no injects, and a bare prompt — scored on whether the extracted BibTeX parses and
is correct. Until that runs, converting the ten call sites would change what a working pipeline sends on
the strength of a guess about which half of a year-old finding survived.
2. **The "is a model loaded?" check.** Blocked by nothing, and lopsided: the *check* is cheap — the state is
   already in the `/api/v0/models` response `detect_backend_flavor` fetches and discards — while **the pill
   that shows it is a UX problem**, and that is the half that will take the time. The sketch below settles
   its behaviour (poll only while known-bad, green then hide) but not its appearance or its place in the
   layout. Second on the list because a wrong answer here meets the user at the worst moment: the first
   message of a session, against a backend with nothing loaded.
3. **The variant sweep across the model fleet.** Blocked by nothing either, and needs no code: four arms,
   three or more samples each, per
   `investigations/context-injects/README.md`. Last because it costs backend time rather than attention, and
   because it can run while something else is being written. It is what turns the two anecdotes recorded
   there into a result.

**Fixed 2026-08-11, and no longer outstanding:** the chat view's follow-tail bug, written up end to end in
`investigations/follow-tail-drift/` — symptom, the two dead ends, the causal fix, and why it was
intermittent.

**Not this brief's, and no action needed:** rasterizing a PDF's pages so a VLM can read them. It belongs
with the off-diagonal cells of the import matrix — OCR text out of an image, and a page range of a PDF as
images — which `TODO_DEFERRED.md` already carries as one item. It will surface when that is tackled;
recorded here only so the page-image use case above does not read as though it were waiting on this brief.

## Adopted 2026-08-10: "is a model actually loaded?", and the reconnect it implies

Came out of the `TODO_DEFERRED` triage — *"Librarian doesn't check that the LLM backend has a model
loaded"*, which named `llmclient.setup` as its natural home. It lands here rather than in its own brief
because Part 0 just built the thing it needs: `setup` is now a probe plus a pure `configure`, so "ask the
backend what it has" and "build the prompt from that" are separate, named, and separately callable.

### What the backend can tell us

LM Studio's `/api/v0/models` returns `state` per model — verified live 2026-08-10: one `loaded`, nine
`not-loaded`. **`detect_backend_flavor` already fetches that endpoint and reads `state`**, using it only to
identify the flavor and then discarding it. ooba and generic backends have no equivalent. So this is a
tri-state exactly like `model_is_vlm`: `True` / `False` / `None` for "cannot tell", and it belongs as a
`loaded` field on the `model_info` env that `_resolve_model_info` returns — which keeps `configure` pure,
since it receives backend facts as data.

### Decisions (Juha, 2026-08-10)

- **Librarian starts anyway, and says so.** Reachable-but-empty is recoverable in two clicks, and Librarian
  is useful without a model — past chats, the cleanup dialog, settings. Refusing to open a window for it
  would be the existing `sys.exit(255)` path applied to a condition that does not warrant it.
- **Batch tools exit; `minichat` warns.** `raven-pdf2bib` and `raven-importer` can run for hours, so failing
  at document 1 with a precise diagnosis beats discovering it mid-run. `minichat` is interactive, so it
  behaves like Librarian and keeps the REPL.
- **No retry mechanism is needed for generation** — one exists and is better than a poll. A send against a
  dead or empty backend already produces an error as an AI message, and reroll retries it. That is
  user-initiated, free when idle, and already built. What the new check buys is a *specific* message
  instead of a generic failure.

### The UX, and the one place a poll is unavoidable

**Librarian has no model readout at all today**, so this needs building: a pill reading something like
"LLM backend not connected", which turns green and announces the connection before hiding itself
(`dpg.hide_item`, per the usual pattern).

That is where the "no polling" answer runs out. Reroll covers *generation*, because the user acts. Nothing
makes a pill go green on its own. So the resolution is asymmetric, and worth stating as the rule rather than
discovering it later: **poll only while known-bad.** No request at all in the healthy state; a cheap
`_resolve_model_info` (one HTTP call) on a timer only while the pill is up, stopping the moment it clears.
The cost is bounded by the duration of a condition the user is actively fixing.

### The catch: the system prompt depends on the backend — and probably should not

**The prompt is not independent of what the probe learns.** `configure` builds `system_prompt` and
`character_card` from `template_vars` = (`user`, `char`, `model`, `context_length`), and the last two come
from the backend — the card tells the model its own identity and context size. A Librarian that started
without a backend holds a card built from placeholders, so connecting is not only a UI state change.

The mechanical repair is available: re-probe → `configure` → `appstate._refresh_system_prompt` → clear the
pill. Note what that last step does, because it is easy to assume otherwise: it adds a revision **and
deletes the previous one**. Deliberately — the node is refreshed on every app start, so keeping the old
revisions would grow a pile of them, one per launch, none of which anyone wants. So the placeholder version
is not preserved, and nothing is lost by that.

**But the better answer is probably to stop putting these two in the card at all** (Juha, 2026-08-10). The
model name and the context length are exactly the shape of thing the *date* is, and the date was moved out
for the same reason — the comment in `configure` says so in its own words:

> No date here, deliberately: this text is built once, at app start, so a date written into it goes wrong at
> the first midnight the session survives.

A model name written into a card built at app start goes wrong the moment the loaded model changes, which is
precisely the reconnect case, and also the case of a user switching models in LM Studio mid-session without
restarting anything. Moving both into the per-turn injects — where they are recomputed every turn anyway —
**dissolves the reconnect problem rather than solving it**: there is no stale card to rebuild, and no node to
rewrite. It also retires the open question of what the card should claim while nothing is loaded, since a
turn taken with no model loaded does not happen.

That is a change to what the model is told and where, so it wants measuring before it is adopted — the
inject-shape probes in `investigations/context-injects/` are the apparatus, and they now run on real
settings, which is what makes such a measurement trustworthy.

**One thing that is *not* a blocker either way**, since it looks like one: what the card should claim while
nothing is loaded. Nothing reads it — with no model connected there is no model to mislead, and the card is
rebuilt before any turn can be taken. So the "never state a guess to the model" rule is not in tension here
under either design; the question is academic, and should not be allowed to hold up the choice.

### Raised, and not this brief's to fix: the chat log does not show the injects

The chat view shows the stored conversation. It does not show the date inject, the clock tool call, the
retrieval tool call, or the reminders — all of which are really sent, every turn. **That breaks the WYSIWYG
promise the log otherwise makes** (Juha, 2026-08-10), and moving the model name and context length into the
injects would put more of what the model is told behind that same curtain.

**In scope for this brief after all** (Juha, 2026-08-10). The first reading filed it as a chat-view question
to be handed off, and that was the wrong cut: the injects are precisely what Part A makes addressable. Once
`build_turn_prompt` returns the assembled history, "show the user what was actually sent" is a *reader* of
the same function the probes read, not a separate excavation into the render path. A brief whose whole
subject is making the turn's prompt inspectable programmatically should not stop one step short of making it
inspectable to the person the log is shown to.

It is also the counterweight to the change above. Moving the model name and context length into the injects
improves correctness and, on today's chat view, hides two more things the model is told — so the two want
deciding together rather than in sequence.

**That objection is now spent (2026-08-11).** Both are already injected, just at a different moment: they are
`{model}` and `{context_length}` in the character-card template, substituted by `configure` at app start. What
made them visible was never that they are authored rather than injected — it is that the substitution happens
*before* the node is stored, and the log renders stored nodes. The per-turn injects were invisible for the
mirror-image reason.

So moving them changes which mechanism carries them, and the display follows automatically: the chat view
renders whatever `build_system_injects` returns. The remaining argument for the move is the one it always
was — a backend reconnect can change both, and a value baked in at start is then stale.

Juha is taking the item itself to the TODO triage so it is recorded there as well; this note is the scoping
decision, not the item.

#### Done 2026-08-11: the system message shows its injects

`scaffold.build_system_injects` was split out of `build_turn_prompt`, and the chat view renders its result
under the stored system prompt. One call, two readers — the prompt is built from it and the log displays it,
so the two cannot drift into the log claiming one thing while the wire carries another. Pinned by
`test_the_injects_the_view_shows_are_the_ones_the_prompt_carries`.

Two properties of the display worth knowing, both consequences of what the system-prompt node already is:

- **It is live, not recorded.** `appstate` overwrites the stored system prompt at every app start rather
  than keeping a revision per session, so that node has never been a record of a past turn. What is shown
  is what the *next* turn will send.
- **Which makes a midnight rollover visible**, and it is handled rather than documented away: a session
  left open past midnight would send the new date while the log showed the old one — the very divergence
  this is meant to remove. `DPGChatController.refresh_system_injects_if_stale` compares the drawn injects
  against a fresh call and redraws through `rebuild_in_place` when they differ. It runs at the start of a
  turn, so the display and the wire change together. Between turns the display can lag a rollover; nothing
  is being sent then.

Only the unconditional injects are shown. The two conditional ones are not knowable before the turn runs,
and a line appearing and vanishing between rebuilds would read as instability rather than as information.

#### What to show, and the two exceptions (Juha, 2026-08-11)

Show the system prompt as it is actually sent — the date, and the model name and context length once those
move into the injects. Two things stay hidden, for reasons particular to each:

- **The clock's synthetic tool call.** The turn presents the time *as if* the model had called
  `get_current_time`, and that framing exists for the model's benefit; shown to a user it only raises the
  question of who made a call they did not see. The *time* is worth showing, the staged call is not.
- **The retrieval results.** `k=50` was adopted as the strongest single retrieval-quality lever, so the
  matches are long by design and would bury the conversation they are supporting.

Both are exceptions to WYSIWYG rather than refinements of it, which is the honest way to hold them: the log
is faithful except here, and here is where a reader should be told there is more underneath.

#### Record the outputs, not the settings that produced them

Noted while making the formatters overridable, since it constrains Part B's record. `settings` cannot be
snapshotted as data — it holds callables (`tool_entrypoints`, now `formatters`) and a live tokenizer — and
an A/B arm's override is typically a lambda with no name worth recording. Nothing serializes settings today
and nothing should start.

What answers "what was in force?" is the assembled prompt, which Part B's record already carries and which
this section wants to display. The two uses want the same artifact, so build it once.

## Explicitly out of scope

**The scripted backend.** *"Headless scaffold mode for `ai_turn`"* bundles a scripted backend — canned model turns, for driving
the real `ai_turn` deterministically — with the driver. Split (Juha, 2026-08-04). v1 targets a real backend,
because that is what unblocks probes today. The scripted backend buys CI determinism and is a natural v2 once
the driver exists to hang it off. Deferred rather than dropped.

**The per-document LLM pass — proposed as brief 17.** Retry, cache, resume, progress. It has three users and
they are not this brief's users: `raven-pdf2bib` (eight `perform_throwaway_task` call sites, each wrapped in
its own hand-written retry loop — the same six lines, eight times, in one 1058-line file; no caching and no
resume, so a crash at document 2400 restarts from zero), `rag_live_corpus`'s persistence layer (a
`PersistentForest` per sample plus a JSONL ledger, worth lifting wholesale — these runs take an hour and the
machines reboot), and `briefs/design/corpus-interrogation-sketch.md`'s map stage. Three users is the bar for
building it, but it is a batch-execution primitive rather than a scripting surface, and folding it in here
would make this brief two briefs wearing one number.

## Verification pass, 2026-08-10

The brief asked for its own pointers to be checked. They were, and three claims did not survive.

**The pointers.** `api.initialize` is at `llmclient.py:81`, not `:79`. `llmclient.setup` is at `:733` and spans
223 lines. `minichat`'s explicit `None`s are around `:612`, not `:628–649`. `_perform_injects` is at
`scaffold.py:355`, called internally at `:995`.

**"Seven callers outside its own module" is wrong, and it was the evidence for the central claim.** Counting
actual call sites rather than mentions: `absent_fact.py` (1), `assembled_shape.py` (1), `test_scaffold.py`
(16). That is three files, and sixteen of the eighteen sites are tests. `llmclient.py` and `chatutil.py`
contain *comments* naming the function, not calls. `inject_shapes.py` and `rag_placement.py` do not call it
either — they deliberately **reimplement** the placement, because their job is to measure candidate shapes
*against* the one Raven uses.

So the "it is already a public API" argument rests on two probes, not seven callers. **The argument survives
on different evidence, which is stronger than what the brief cited:** both of those probes were *broken*, by
two independent signature changes four days apart, and stayed broken silently.

- `7c350a4` (2026-08-03 01:56) made `tool_context` required. The probes were relocated into
  `investigations/` 22 minutes later without being fixed.
- `c24c89e` (2026-08-07) removed `speculate` when the grounding marker was made honest.

Each call therefore carried an unknown keyword *and* omitted a required argument. Repaired 2026-08-10
(`198fecb`); `grounded=True` is the verified equivalent of the old `speculate=False`, since the gate went
from `if not speculate and grounding_material_exists:` to `if grounding_material_exists:`.

**Both then re-run live against qwen3.6-35b-a3b (LM Studio, IQ4_NL_XL, 128 Ki) and both still measure what
they were written to measure.** `assembled_shape` passes all four checks — the planted figure is used, the
absent fact is declined with documents present, an unrelated question is answered plainly, and the injected
clock is believed. `absent_fact` reports **0 of 6** attempts to search again, across its three variants at
both T=0 and T=1. So the repair restored the probes rather than merely satisfying the signature.

One caveat found in the running: `assembled_shape`'s date check tests for `today.isoformat()` in the reply
and the model answered "Monday, August 10, 2026", so a correct answer prints as `CHECK` rather than `OK`.
The probe under-reports its own success, and the prose form is what the inject deliberately supplies —
weekday included, so the model never does calendar arithmetic.

**The acceptance criterion names two scripts it cannot serve.** It asks that `inject_shapes`,
`assembled_shape`, `absent_fact` and `rag_placement` become expressible against Part A. The first and last
never reach through the private door and should not: they are the control arm, and a Part A that served them
would be a Part A that had absorbed their job. **The criterion is the other two**, which say in their own
docstrings that they build the history through Raven's own code "rather than reimplementing the shape".

### The requirement the brief missed: a real `settings` without a backend

Both Part A probes forge their own `settings = env(user=..., char=..., model="test-model", ...)` — **7 of the
21 fields `setup` returns** — because `setup` needs a live backend. Among the forged fields is
`system_prompt="You are a helpful assistant."`, standing in for Raven's real card-derived prompt.

So `assembled_shape`'s claim to measure "what Raven actually sends" is half true: the injects are real and
the prompt they are injected into is not. That is the brief's own thesis — a replica diverging where the
interesting behaviour lives — one layer above where the brief was looking, and it makes a real `settings`
object a Part A *deliverable* rather than a convenience.

**The split is not local-versus-remote, which is the tempting reading and does not work.** `model` and
`context_length` are backend-derived and then feed `template_vars`, which builds `system_prompt` and
`character_card` — the card tells the model its own identity and context size. There is no local half that
yields a usable settings object.

The split that works is **probe, then configure**:

- `setup(backend_url)` keeps its signature and behaviour: perform the two network calls
  (`detect_backend_flavor`, `_resolve_model_info`), then delegate. Two lines of the 223 touch the network.
- `configure(model_info, backend_flavor)` is everything else, pure, taking the backend's facts as data.

A probe calls `configure` with a synthetic `model_info` and gets a settings object that is *real* — real
system prompt, character card, tool tables, tokenizer, `request_data` template — differing from production
only in what it was told about the model.

## Decisions taken, 2026-08-10 (Juha)

Answering the questions below; the list is kept for its reasoning, and these are the rulings.

1. **`api.initialize` lands in `setup`.** Verified as safe: every entry point already goes through it
   (`app`, `minichat`, `pdf2bib`, `importer`, two investigations), and the library modules that do not
   (`scaffold`, `chat_controller`) run under one that does. The probes bypass `setup` entirely — which the
   `configure` split above resolves, so no separate `ensure_initialized()` is needed.
2. **The result record is a frozen dataclass.** Type-checkable, and a mistyped field fails loudly, which is
   what a probe asserting on `rounds` needs. It is the odd one out against the codebase's `env` habit, and
   that is accepted deliberately: this is a declared library surface, not an internal namespace.
   **Vocabulary, fixed here because hand-rolled walks get it wrong:** a **round** is one assistant message
   asking for tools, however many calls it asks for; a **call** is one tool invocation.
3. **Offline by default** — the brief's own suggestion, taken. A probe that silently hits the network is the
   more expensive mistake.
4. **Per-run overrides become fields on `settings`.** The thing being overridden today is a module-global
   `chatutil` formatter, monkeypatched by `rag_live_corpus` to run its control arm. Of the three candidates —
   thread an override dict through `run_turn`, wrap the call in a context manager, or promote the knob to
   `settings` — the third is the one that matches the standard the rest of the project sets: `settings` is
   already precisely "what this run is configured as", and it is what `configure` builds. The other two leave
   the value in a module global and add a mechanism for reaching around it.

   **The cost is honest and belongs in the estimate: this is an audit, not a field.** Which module-globals
   deserve promotion is a larger question than one probe's control arm, and the answer is not "all of them" —
   a constant that no run would ever vary stays a constant. The test is whether two runs of the same code
   could reasonably want different values. Formatters that shape what the model is told qualify; buffer sizes
   and cache paths do not.

   Corollary worth stating, since it is the reason this is worth the audit: **a probe monkeypatching a module
   global is a design signal, not a probe smell.** It is how the codebase currently says "this should have
   been configurable", and each instance is a candidate found by someone who needed it.

   **Done 2026-08-11. The audit's answer is eight formatters, and they went into one namespace.**
   `settings.formatters`, built by `chatutil.default_formatters()` and set by `configure`. The eight are the
   ones whose output reaches the model: `date_now`, `time_now`, the two reminders, the two tools-are-spent
   notices, `docs_matches`, `consulted_documents`. Six further `format_*` functions in `chatutil` write for
   the chat log or an export, where the reader is a person and no run would vary them, and they stayed put.

   A namespace rather than eight fields on `settings`: the group has one meaning worth naming, `settings`
   already carries twenty-two fields, and an override then reads
   `settings.formatters.notice_that_tools_are_spent = lambda: ""` — the same shape as the monkeypatch it
   replaces, without the global mutation or the `finally` that put it back. The names drop the `format_`
   prefix, which the namespace has already said.

   Two details the implementation forced, both worth knowing before extending this:

   - **`default_formatters()` returns a fresh namespace per call**, so an override belongs to one settings
     object. A shared one would leak an experiment's arm into every other run in the process, which is the
     failure being removed. This surfaced through `test_configure_reproduces_setup_given_the_same_facts`,
     which was comparing settings field by field *as reprs* — and `repr(env)` leads with the object's own
     address, so an env-valued field can never match itself across two calls. `env.__eq__` compares
     contents and gets it right; the test now compares values.
   - **Tool entrypoints reach formatters through `dyn.tool_context`, and fall back to the defaults.** A tool
     context legitimately carries `llm_settings=None` — the documented shape for a caller not running tools
     that need settings — and two entrypoints format their result without otherwise wanting settings. The
     fallback keeps those working rather than making formatters the reason a probe needs a full settings
     object.
5. **The surface constructs a `chattree.Forest` by default and accepts one.** In-memory, so constructing
   costs nothing and keeps the one-liner probe a one-liner.
6. **It lives in `raven.librarian.agent`.** The brief's warning against becoming a generic agent harness
   stands as a constraint on the *contents*, not on the name.
7. **The mutation goes.** The public function returns a new list; `scaffold`'s internal call site rebinds.
   `perform_` then names a side effect that no longer exists, so the rename is forced rather than optional —
   which is what the brief predicted would follow from settling the shape first.

### Raised 2026-08-11: `perform_throwaway_task` belongs in `agent` too

It is the scripting entry point that already exists, and `TODO.md` says so in as many words — "the only
scripting entry point, and deliberately thin: no datastore, so no attachments, no branch, no retrieval".
Those four absences are the list of things this brief adds, so the two are the same surface at different
stages of growth, currently one layer apart.

The layering argues the same way. `llmclient` is the backend-protocol layer, and `perform_throwaway_task`
does orchestration inside it: it assembles a history from the system prompt, the greeting and the
instruction, and then calls `invoke`. That is scaffold-shaped work. Its ten call sites are all first-party
and sit in two files — `papers/pdf2bib.py` (eight) and `visualizer/importer.py` (two) — both of which are
scripts driving an LLM with no chat in the picture, which is precisely what `agent` is for.

**Settled 2026-08-11: it dissolves.** The question was whether it *widens* — gaining the optional datastore,
so a throwaway task can carry an attachment, which is the narrow fix `TODO.md` asks for — or *dissolves*
into a one-liner over the new surface with the name going away. Widening is now moot: `agent.turn` already
takes attachments, in memory as well as on disk, so the thing that would have been added is present. What
is left of `perform_throwaway_task` is the two gaps listed in the queue above (no way to withdraw all
tools; ten callers wanting a progress symbol), and neither is a reason to keep a second one-shot path — they
are the price of retiring it. Moving it unchanged remains the one option to reject.

Either way the ten call sites change shape, since they currently unpack `Tuple[str, str]` and the surface
returns a record. Two files, first-party, greppable.

**The name is worth revisiting in the move, but not for the reason it might look.** "Throwaway" is doing a
real job: it says the invocation is one-off, which is the property a caller most needs to know. The weak
half is "perform a task", which says nothing that "it is an LLM call" did not already say. So the target is
a shorter name that keeps the one-off sense, rather than a rewrite that discards it.

Direction (not settled): a single descriptive word, paired with whatever Part B's full-loop entry point is
called, so the two read as the contrast they are — one shot with no tools and no history, versus the agent
loop with a record of what it did. `agent.ask` against `agent.turn` is one such pair, and its weakness is
exactly the one above in reverse: `ask` is short and plain but carries no hint that it is one-off.

## What this brief must settle before implementation

1. **Where `api.initialize` lands** — `setup`, or an explicit `ensure_initialized()` for callers that do not
   go through `setup`.
2. **The result record's shape and type.** `unpythonic.env`, a frozen dataclass, or a plain dict. Consistency
   argues for `env`; a probe asserting on fields argues for something type-checkable. Settle the
   round-versus-call vocabulary here, in writing, since it is the thing hand-rolled walks get wrong.
3. **Real tool execution.** A run with `tools_enabled=True` performs *real* tool calls: `webfetch` actually
   fetches, `websearch` actually searches. Correct when the probe meant to be online; wrong as a default while
   iterating on loop structure. Decide the knob — offline mode, tool allowlist, or fake registry — and which
   way the default points. Suggest offline by default: a probe that silently hits the network is the more
   expensive mistake.
4. **How per-run overrides are expressed**, given that the thing being overridden today is a module-global
   formatter.
5. **Whether the caller supplies the datastore or the surface constructs one.** `chattree.Forest` is
   in-memory (`PersistentForest` is the file-backed sister), so constructing one by default costs nothing and
   is what makes the one-liner probe a one-liner. Accepting one is what `rag_live_corpus`-shaped work needs.
6. **Where it lives, and its name.** Beside `scaffold` as a driver module, or as a programmatic sibling of
   `minichat`. Deliberately left open in the original item, on the grounds that the name should be picked
   against what the module ends up doing.
7. **`_perform_injects`: shape first, then name.** It is to be renamed and made public (Juha, 2026-08-04);
   updating all callers, investigations included, is not a problem — "supposed to stay as they were run"
   covers scientific reproducibility, and this is an internal organizational change. A probe stranded on a
   renamed API is *less* reproducible, not more.

   But the shape decides the name. Today it takes `history: List[Dict],  # mutated!` and returns `None`.
   That is serviceable internally and poor as a library surface, and it contradicts what Part A is for: a
   caller wanting "the prompt Raven would send" has to assemble the history, pass it in, and then read its
   own variable back. Two ways out — keep the mutating function private under a wrapper that *returns* the
   assembled history, or drop the mutation and return a new list. Decide that first; if the mutation stays
   internal the rename largely dissolves, and if it goes then `perform_` is the wrong verb regardless, since
   it names the side effect being removed.

   **The sweep includes prose.** The name appears in `raven/librarian/CLAUDE.md` (in the `config.py` entry of
   the module map), `TODO.md` (four places),
   `TODO_DEFERRED.md` (two), and `investigations/context-injects/context-inject-shape-measurements.md` three
   times, including its opening sentence. That write-up is the standing explanation of why the injects are
   shaped as they are; a dead name in it sends a reader to code that no longer exists.
