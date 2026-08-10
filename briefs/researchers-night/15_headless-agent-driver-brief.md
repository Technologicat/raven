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

> **Landed 2026-08-10.** `_perform_injects` is now `scaffold.build_turn_prompt`: public, in `__all__`,
> returning a new list instead of mutating the caller's. All 19 call sites updated, and the prose sweep done
> except the two TODO files, which are frozen pending their own brief, and the closed briefs, which are
> historical records and are not repointed.
>
> **The migration paid for itself immediately, in the way the brief predicted and worse.** `absent_fact` and
> `assembled_shape` were also moved onto `llmclient.configure`, so they build Raven's real settings instead
> of forging seven of twenty-one fields with `system_prompt="You are a helpful assistant."`. Re-run against
> qwen3.6-35b-a3b, two results changed at once: `absent_fact` as-shipped at T=0 went from a clean answer to
> `finish=length` with **31726 characters of reasoning and no reply**, and `assembled_shape`'s absent-fact
> check went from declining cleanly to emitting literal `<tool_call>` text. Both are failures those probes
> exist to catch, and both were invisible while the prompt around the injects was a placeholder. Recorded in
> `investigations/context-injects/README.md`; the runaway wants a look before Researchers' Night.

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

### Raised, and not this brief's to fix: the chat log does not show the injects

The chat view shows the stored conversation. It does not show the date inject, the clock tool call, the
retrieval tool call, or the reminders — all of which are really sent, every turn. **That breaks the WYSIWYG
promise the log otherwise makes** (Juha, 2026-08-10), and moving the model name and context length into the
injects would put more of what the model is told behind that same curtain.

Recorded here because it surfaced here. It is not part of this brief — it is a chat-view question, and
belongs with the other chat-log work — but it should not be lost, and the TODO files are frozen pending
their own pass.

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
5. **The surface constructs a `chattree.Forest` by default and accepts one.** In-memory, so constructing
   costs nothing and keeps the one-liner probe a one-liner.
6. **It lives in `raven.librarian.agent`.** The brief's warning against becoming a generic agent harness
   stands as a constraint on the *contents*, not on the name.
7. **The mutation goes.** The public function returns a new list; `scaffold`'s internal call site rebinds.
   `perform_` then names a side effect that no longer exists, so the rename is forced rather than optional —
   which is what the brief predicted would follow from settling the shape first.

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
