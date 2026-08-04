# Brief 15: a scripting surface over the scaffold

> **Line numbers are as of 2026-08-04 and want verifying.** They were read from a shallow clone that was
> already a couple of hours behind a moving tree, and the files most often cited here (`app.py`,
> `chat_controller.py`) were among those being committed to that day. Treat every `file.py:NNN` below as a
> pointer to a thing that exists, not as a coordinate.

**v0.2.9 work, and first in that queue — ahead of 05.** Written 2026-08-04, during the deferred-TODO triage.
Not admitted to v0.2.8: it is not a defect, so the freeze covers it by the letter, and the argument that it
would make 09's own validation easier is exactly the "one more thing" the freeze exists to stop. It goes first
after the tag, and the reason is timing rather than value — see below.

**This brief consolidates rather than invents.** The design is largely already written, in `TODO.md` under
Librarian → Core features ("Make the scaffold scriptable", 2026-07-29). What follows lifts it into brief form,
adds the import-side prerequisite, and cuts one piece out into a brief of its own. Where the two disagree,
`TODO.md` is right and this brief has been corrected to match.

**Filed in two places under different names**, which is how the poorer copy came to be the one that got
deferred: `TODO_DEFERRED.md:1825` ("Headless scaffold mode", filed 2026-06-03) is the same work as the
`TODO.md` entry, minus the two-entry-point split and the result record. Both close against this brief, along
with `TODO_DEFERRED.md:1474` (lazy `api.initialize`).

**Not absorbed: `TODO_DEFERRED.md:1813`** (collect `ai_turn`'s callbacks into a bundle). It looked like a
prerequisite and is not — see the callback section below. It remains a GUI-side ergonomics cleanup, and is
now independent of this work.

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
the full-turn ones under `briefs/summer_2026_librarian_extension/manual_tests/`. This work is done when
`inject_shapes`, `assembled_shape`, `absent_fact` and `rag_placement` can be rewritten against Part A without
reaching through a private door, and `rag_live_corpus` and `rag_tool_rescue` against Part B without
hand-rolling a branch walk.

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
eight explicit `None`s (`minichat.py:628–649`) for the same reason.

They earn nothing from a script that will never draw a progress bar, and the wall bites there instead: a probe
silently rotted when brief 10 removed `on_nomatch_done`.

**So the fix is not defaults.** Defaults would weaken the wall where it protects someone in order to help
someone it cannot protect. The scripting surface takes **no callbacks at all** — the events become the
returned record of what happened, which is the thing every probe currently reconstructs by walking the branch
afterwards. Keep the wall where it works; do not propagate it to callers it cannot serve.

## Part 0 — lazy `api.initialize` in `llmclient`

The one prerequisite, and it is import-side rather than design-side.

`llmclient.py:79` calls `api.initialize(...)` at module top, so importing `llmclient` both requires the full
`raven.client.api` chain to succeed — qoi, spaCy, Kokoro TTS — and runs the side effect. `scaffold` imports
`llmclient` at module level and inherits it, which is why `test_scaffold.py` carries a `pytest.importorskip`
and why scaffold coverage is invisible in the minimal-deps CI job.

A probe that must boot a TTS stack to test tool-calling is a probe nobody writes.

The natural seam is `llmclient.setup` — every caller already goes through it, so initialization can move there
(idempotent, called once). While in there, audit `llmclient`'s and `hybridir`'s other module-top imports for
side effects; `scaffold.py`'s `TYPE_CHECKING` import of `hybridir` is the model.

**Falls out**: remove the `importorskip` from `test_scaffold.py`, and scaffold's ~119 statements start
contributing to CI coverage.

## Part A — build the turn's prompt and hand it back

No backend involved. This is what the prompt-shape probes want: the prompt Raven *would* send, so they can
send it themselves and measure the backend without Raven confounding the result.

All of them reach into `scaffold._perform_injects` — a private function which, counted 2026-08-04, has
**seven callers outside its own module**, across three trees: `llmclient.py`, `test_scaffold.py`, four
probes under `investigations/context-injects/`, and `manual_tests/rag_live_corpus.py`. (The `TODO.md` note
says four; it undercounts.) That is a public API which has not been declared one. It gained a parameter this
session (`tools_are_spent`) and its callers survived only because the parameter defaults — next time the
change may not be so kind. Declaring it, naming it, and documenting the returned shape *is* most of Part A.

## Part B — run the turn and tell me what happened

**A turn returns what happened, not a node id.** Every probe re-implements the same branch walk afterwards:
count the tool nodes by name, count the *rounds* (an assistant message asking for tools, however many calls it
asks for), collect the reasoning that never reached `content`, find the reply. That walk is the actual result
of a turn, it is written out longhand each time, differently, and the round-versus-call distinction has been
got wrong at least once. Returning it removes the walk and fixes the distinction in one place.

Two things the record must cover that are awkward today:

- **The prompt.** The wire history is reachable only through `on_prompt_ready`, so a script that wants to
  assert on what was actually sent has to build a closure to catch it. In a callback-free surface, the
  assembled prompt belongs in the record.
- **Per-run overrides.** A/B-ing a knob currently means monkeypatching — `rag_live_corpus` swaps out a
  `chatutil` formatter to run its control arm. Fine in a probe, and a sign that per-run overrides belong in
  the surface rather than in module globals.

**The two parts are not layers of each other**: A must not talk to a backend at all, and B is useless without
one. A surface offering only B would leave the prompt-shape probes still reaching through the private door.

## Explicitly out of scope

**The scripted backend.** `TODO_DEFERRED.md:1825` bundles a scripted backend — canned model turns, for driving
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

   **The sweep includes prose.** The name appears in `raven/librarian/CLAUDE.md:24`, `TODO.md` (four places),
   `TODO_DEFERRED.md` (two), and `investigations/context-injects/context-inject-shape-measurements.md` three
   times, including its opening sentence. That write-up is the standing explanation of why the injects are
   shaped as they are; a dead name in it sends a reader to code that no longer exists.
