# Brief: expose the document database as a tool the model may call

**Status: steps 1 and 2 built, 2026-07-28.** The tool surface, the per-turn tool gate, the agent-loop cap,
grounding-by-declaration and the grounding UX are in (`d04bd97`, `2050b88`, `797b4ca`, `1b7f234`,
`7dc854c`, and the commit carrying this line). Step 3 — the truncation budget with `fetch_document` and
`list_consulted_documents` — is designed here and not yet built. Phase 1 of the Researchers' Night list (`TODO.md`), and the only item
on it that is new construction rather than repair.

Three things were decided while building rather than before, and are folded in below: the `docs_enabled`
split (§2), the accepted full-prompt reprocess when the cap fires (§4), and the rejection of a tool-call
countdown (§4). Live behaviour is exercised by `manual_tests/rag_tool_rescue.py`, the successor to the
`absent_fact.py` probe that measured the failure this work fixes.

**What:** give the LLM `search_documents` and `fetch_document`, alongside the existing auto-search that
Raven runs on the user's behalf. Keep both — they buy different things.

**Why now:** measured. Asked something the auto-injected matches do not answer, Qwen3.6-27B reaches for a
second, better-aimed search — a query that only exists *after* reading pass 1. Having no such tool, it
writes the call out as literal `<tool_call>` text and the user gets that instead of an answer, roughly one
turn in three (Q11 in `briefs/context-inject-shape-measurements.md`). Talking it out of asking was tried
and made things much worse. The model's instinct is right; the fix is to let it have the tool.

## Why both, and not just the tool

- **Auto-search** buys a zero-latency first pass from a cheap heuristic query (today: the user's message
  text, verbatim). No round trip.
- **The tool** buys a model-authored refinement, written by something that has read the first pass.
- **And the tool's results last longer.** Auto-inject results are ephemeral — built in `_perform_injects`,
  sent, discarded, never persisted. Tool results are real `role="tool"` nodes, so they are still in the
  linearized history on later turns. A follow-up question stays grounded if the *model* searched, and does
  not if *Raven* searched for it. This asymmetry was not one of the original arguments for the tool; it
  emerged from the grounding-lifetime analysis below, and it is the strongest of the three.

## The five decisions that shape the implementation

### 1. The retriever reaches the entrypoint through `dyn.tool_context`

`llmclient.setup` runs *before* `hybridir.setup` in both clients (`app.py:229`/`246`,
`minichat.py:98`/`112`), so a tool entrypoint cannot close over the retriever at registration time. The
per-turn request context already exists for exactly this class of harness-supplied state, and its comment
already says to grow the one env rather than scatter dyn vars.

**Consequence:** `tool_context` moves from per-*round* (built inside `_perform_and_store_tool_calls`) up to
per-*turn* (built in `ai_turn`), because two of its new fields have to accumulate across tool rounds.
Fields it gains: `retriever`, the grounding flag (§3), and the free-token budget (§5).

Duck-typed `.query()` with a `TYPE_CHECKING`-only import, exactly as `scaffold` already does — so
`llmclient` does not acquire a runtime dependency on the `chromadb`/`bm25s`/`watchdog` stack.

### 2. `tools_enabled` becomes a per-name gate

Docs-off must remove `search_documents` while leaving websearch, so the advertised tool list has to vary
per turn. `invoke` currently sends the whole list or none. This retires the standing
`# TODO: tools_enabled is a blunt hammer` at `scaffold.py:455` — that refinement is not optional here, it
is what the gate requires.

Both document tools are gated on docs-enabled **and** tools-enabled.

**Found while building it: there was no "docs enabled" signal to gate on.** Both clients collapsed the
user's docs toggle into `docs_query=None` before calling `ai_turn`, which was lossless only while the
automatic search was the sole route to the documents. It is not lossless now: `retry_tool_calls` passes
`docs_query=None` legitimately (a continuation runs no *new* automatic search), and gating tool
availability on that would make the tools vanish between rounds of one agent loop — a shape models read
as noise. And the "do we still need the autosearch?" question is live, so a turn that searches
automatically and a turn that permits the model to search must be separately expressible.

So `ai_turn` and `retry_tool_calls` take an explicit `docs_enabled`, and the clients pass their toggle
instead of collapsing it. `docs_enabled` answers *may the documents be used*; `docs_query` answers *search
for this before the model runs*. The collapse now happens in one place (`ai_turn`) rather than being
duplicated in each client.

The retriever then goes into the tool context **only when the documents are in play**, which makes its
presence the single gate the entrypoints read — and makes the gate fail closed: a model that calls a tool
that was never advertised finds no retriever and gets a refusal, rather than reaching around the switch.

**Cache cost, and why the loadout must not wobble.** Tool definitions are expanded into the *system block*
by the model's own chat template, so changing the advertised list invalidates the prompt prefix from the
very beginning — a full reprocess, not the tail reprocess that a changed inject costs. (Front-of-prompt
placement is how the mainstream templates do it; not verified against the specific model in use. To check:
`prefill` returns the backend's own `prompt_tokens`, so one call with `tool_names=None` and one with a
restricted set gives the delta directly.)

That is affordable only because the list is a function of `docs_enabled and retriever is not None`, so it
is constant within a conversation and changes only when the user deliberately flips the toggle — one full
reprocess per toggle, not per turn. Two rules follow, and both are already load-bearing: keep the list
stable *within* a turn (see `invoke`'s `tool_names` docstring), and pass the same list to `prefill`, or the
warm-up primes a prefix the real turn never sends and the reprocess gets paid twice.

### 3. Grounding is declared at the source, and scoped by lifetime

`_context_is_present` currently infers grounding from message shape: any `role="tool"` message after the
latest user message counts. Once a document search can return *nothing*, that heuristic starts lying — an
empty result is non-empty as a message, so the "base your claims on the documents" reminder would fire
with no documents present. That is precisely the configuration measured at 5–37× deliberation, with one
model never terminating (brief 08 / Q4).

**Replace inference with declaration.** Each tool entrypoint reports whether its result is grounding
material, via the `(output, metadata)` tuple return that already exists; the flag accumulates in the
per-turn `tool_context`. Tools that do not declare fall back to "non-empty result = grounding".

`webfetch` is the case that must declare explicitly: a denied fetch returns the canonical refusal string,
which is non-empty, so the default heuristic would score a refusal as grounding.

**Scoping is mechanical, not semantic:**

| Source | Counts as grounding for | Why |
|---|---|---|
| User attachments (image, `text_file`) | the whole branch | persisted; sitting in the window |
| Any tool result (document, websearch, webfetch) | the whole branch | persisted nodes; in the history verbatim |
| Document *auto-search* results | this turn only | never persisted; genuinely gone next turn |

The rule is simply *is this material still in the context* — a question with a checkable answer, unlike
"has it gone stale", which is a judgment none of this code can make.

**An earlier draft scoped web results to one turn and document results to the branch, and that was
wrong.** The argument was that a SERP is the answer to one question and goes stale, while a document match
is durable material. But the example that made it convincing — last week's weather grounds nothing about
today's instrument-calibration question — is a *topic change*, and topic changes strand a document match
exactly as thoroughly. Nothing here can see topical relevance, so both scopings were proxies; the
mechanical one at least measures what it claims to.

Two things the earlier draft leaned on have also moved. The harm it was avoiding — the reminder stuck on
for the rest of a conversation — was severe when that reminder read as a ban on general knowledge, and is
mild now that it says *"Answer general questions normally"*. And it cited `compute_auto_allowed_hosts`'s
one-hop rule as precedent, which does not transfer: that rule governs which hosts may be fetched without
asking, a security decision where conservative is right by default. Grounding is not a security question.

The scope stays declaration-based either way. `_record_grounding` writes each tool's declaration into the
tool node's `generation_metadata`, so a branch walk reads what the tool said rather than re-inferring from
message shape — which is what keeps an empty search from counting.

Implemented by seeding: `ai_turn` calls `_branch_grounding_is_present` when it creates the turn's context,
so every downstream reader is unchanged and still just asks the flag. The walk goes over stored *nodes*
rather than the linearized history, since `linearize_chat` hands out bare message dicts while the
declaration lives in the node's `generation_metadata`.

### 4. The agent loop gets a cap, and the final generation is tool-free

There is no bound on `ai_turn`'s `while True` today. It has been fine because websearch and webfetch are
self-limiting, but a search tool the model can rephrase against is exactly the shape that loops.

Configurable in `librarian/config.py` (pending settings dialogs across Raven).

**What happens at the cap is the load-bearing part.** Stopping the loop the moment the cap is hit leaves
the history ending on a tool call with no result — the paused-agent-loop shape measured in brief 08, which
is what makes Qwen answer with *another* call instead of an answer. So: at the cap, still run the
requested calls, then make the final invocation with `tools` stripped. The cap counts rounds of tool
calling; the last generation is always tool-free and therefore always an answer.

**Considered and not built: telling the model how many calls it has left.** Three arguments against, and
one variant that survives them.

- *Placement makes it expensive.* A countdown changes every round. Put where instructions go — the leading
  system block — it invalidates the prompt prefix from position 0 on **every round**, which is the same
  cache damage a wobbling tool loadout causes, except paid continuously instead of once per toggle.
- *It is prohibition-shaped.* "You have 2 tool calls remaining" is a constraint for a literal-minded model
  to reason about, and that is the exact class of wording brief 08 measured at 5–37× the deliberation.
- *It is a backstop, not a budget.* A model that reaches the cap is already looping; one that is not will
  never see the number. Announcing it every turn optimizes a path that should not be taken.

The variant worth revisiting: stay silent until the **last permitted round**, then attach a single notice
to *that round's tool results* — tail position, data-shaped, rare. That buys the one real benefit (a model
mid-plan wraps up instead of being truncated) without the per-turn cost. Unmeasured; if built, measure it
the way brief 08 measured the other inject wordings.

**Accepted cost: the final tool-free invocation is a full prompt reprocess.** Withdrawing the tools changes
the system block, so the one invocation that ends a capped turn cannot reuse the prefix — and it pays that
on the longest prompt of the turn. Accepted anyway, because it happens only on the pathological path, and
because it is what makes the cap a guarantee instead of a request: the prefix-preserving alternative (leave
the tools advertised, add a "no further searches this turn" notice) can simply be ignored, and then the
hard stop is needed regardless, one round later. If the reprocess ever proves expensive in practice, that
notice is the escalation to try first, with the strip kept as the fallback. (How expensive it actually is
has not been measured; `prefill` reports the backend's own prompt-token count, so it can be.)

### 5. Budgeted text, shared by the tool and by attachments

One truncation engine, used by `fetch_document` and by the `text_file` attachment wire-fold. The
attachment path is currently *unguarded* — a large PDF is folded into the message text wholesale and
simply blows the window — so this is a repair there, not only symmetry.

```
allowed = min(per_document_fraction × context_window,
              free_tokens − reserve_fraction × context_window)
allowed <= 0  →  error out, with a canonical string (cf. CANONICAL_NOT_ON_ALLOWLIST)
otherwise     →  truncate the middle to `allowed`
```

The per-document cap is what you truncate *to*; the error is for when the conversation has already eaten
the headroom. Middle-truncation with an explicit `[... N characters omitted ...]` marker, so the model
knows the text is incomplete rather than inferring the paper stops mid-sentence — and on a fulltext it
keeps the abstract/intro and the conclusions, which is the right half.

Both fractions configurable. **The reserve is doing real work, not slack:** the estimate cannot see what
the model generates *after* the fetch — its own reasoning, which on a thinking model is the largest single
consumer. Say so in the config comment, or someone will tune it to 5% reasoning it is pure margin.

`free_tokens` reaches the entrypoint via `tool_context`; the entrypoint has no view of the history. This
falls out well: the context is rebuilt once per tool round against the current head, so the number is
fresh each round rather than stale from turn start.

## Tool surface

**`search_documents(query)`** — no `k`. Minimal surface, fewer ways to emit a malformed call; result count
is host configuration, not a model decision.

**`fetch_document(document_id, offset=None, length=None)`** — offsets and lengths in **characters**, said
explicitly in the description, because a model will otherwise assume tokens. Requested length clamps to
the §5 budget; an offset past EOF clamps and says it did. A `document_id` is the document's path relative
to `docs_dir` (`HybridIRFileSystemEventHandler._make_document_id_from_path`: `str(relp)`), which the model
only learns from search results — so this is search-then-fetch by construction, the same workflow as
websearch → webfetch, against the internal engine.

**Naming.** Surveyed the librarian: `documents` (107) and `document_id` (87) for the *data*, `docs_*`
exclusively for feature/toggle/parameter names (`docs_query`, `docs_enabled`, `docs_dir`,
`docs_num_results`, `on_docs_start`). The split is already consistent — *docs* is the feature, *documents*
are the data — and a tool name names the data. The shape asymmetry against `websearch`/`webfetch` is real
but cosmetic; verb-first is what the model reads, and `search_documents` is already in shipped, measured
history via the synthetic inject.

## The shared match formatter

The inject and the tool must format a match identically, or the model will be told two different things
about what a match is. One formatter, in `chatutil` (layer 1, below both `scaffold` and `llmclient`).

It must carry `offset` and length, which the current inject formatter drops. Nothing changes on the
retrieval side — `HybridIR.query` already returns `offset` in every result; only the formatter discards
it. Without it the model cannot aim a `fetch_document` at the region a match came from.

**Trap:** the current formatter `.strip()`s the match text, so the emitted block starts a few characters
after the reported offset. Report the true (unstripped) document offset and leave a comment saying so, or
the mismatch gets "fixed" in the wrong direction later.

## Grounding UX: mark, don't hide

The anti-confabulation bypass fires *before* the LLM runs when docs are on, the search found nothing, and
speculation is off. Moving it to the end of the agent loop is required — once the model can re-query, an
empty pass 1 is the *strongest* signal that the heuristic query was bad, and it is currently the one case
where the model is forbidden to improve it.

But the end-of-loop position cannot preserve the original property, because generation streams: by the
time we know the answer was ungrounded, the user has watched it arrive. And the guard's false-positive
rate is near 100% on general-knowledge questions — `_search_docs_with_bypass` has no question-type check,
so in the default state (`docs_enabled: True`, `speculate_enabled: False`) "what is 2+2?" already gets
*"No matches in document database. Please try another query."* today. That is why the toggle has been
getting flipped constantly in practice.

The two halves of the policy already disagree, as of brief 08: the reminder says *"Base claims about the
provided documents on those documents. Answer general questions normally"*, while the bypass blocks
exactly those general questions. Whatever is done, they have to agree.

**Resolution: delete the bypass; mark the answer instead.** HEAD stays on the model's reply, which carries
an unintrusive `[general knowledge]` marker. No branch, no HEAD switch, no re-render.

Deliberately styled as a *marker*, not a warning. On a general-knowledge question the ungrounded state is
correct and expected, and most such questions land there — no document database answers "what is 2+2?" —
so a red badge would be crying wolf on the common case. Muted colour, with the explanation in a tooltip.

#### The marker measures presence, not relevance — and that is a hole, found on the first test drive

Asked "what is 2 + 2?" with documents on and speculation off, the marker did **not** appear. Retrieval had
returned hydrogen-electrolysis documents, so material *was* present and `grounded` went true. The model's
own reasoning named the problem: *"The documents provided are about hydrogen production and do not contain
this answer."*

So `grounded` answers **was there material**, while the marker's text claims **did the answer come from the
corpus**. Those two questions come apart exactly here, and two things follow:

- **The label overclaims.** What is measured is closer to "no documents matched" than to "answered from
  general knowledge".
- **Worse: against a real corpus the marker will almost never fire.** Retrieval nearly always returns
  *something* — `semantic_distance_threshold` defaults to 0.8 cosine, which is permissive, and BM25 matches
  on common words. The failure is therefore silent and in the dangerous direction: a feature that looks
  like it works while never warning. `docs_num_results` changes only how much noise arrives, not whether.

**This blocks on brief 09.** The signal the marker actually needs — *are these matches any good?* — is
lever 1 (let the scores survive fusion) plus the `min_p`-style survivor count. RRF discards absolute scores
today, so there is nothing to threshold on. That moves the query-side work from a parallel quality track to
a prerequisite for a phase-1 feature already shipped.

The other route is to ask the model instead of inferring: inline citations, validated against the result
set. The transcript above is evidence it would work, since the model plainly knew the matches were
irrelevant — it simply had no channel to say so in a form Raven could read.

**Do not tighten `semantic_distance_threshold` as a stopgap.** It would look like a fix and is not one: the
right cutoff depends on the corpus (scientific papers vs. fan fiction vs. general English), which is an
open question in brief 09, not a constant waiting to be picked.

This gives the toggle a better meaning than it has:

- **speculate off** → *tell me when you go off-corpus* (reminder injected, badge shown)
- **speculate on** → *I don't care* (no reminder, no badge)

instead of today's "off = refuse anything the corpus does not cover". A badge that correctly distinguishes
"this came from your papers" from "this came from the model" is also a better thing to demo than a refusal
— it shows the system knows which is which.

**Mechanically this deletes more than it adds:**

- `_search_docs_with_bypass` → plain `_search_docs`.
- `_create_synthetic_assistant_node` loses one of its two callers (keeps the backend-error one; its
  docstring says "Two cases use this" and needs updating).
- `on_nomatch_done` goes dead — consumed by both clients (`minichat.py:608`, `chat_controller.py:2359`),
  so removing it touches all three files.

The badge needs no end-of-loop pass: grounding is known *before* the final generation, since the
auto-search and every tool round have already reported in. It goes into `generation_metadata` at node
creation. Set the field **only when speculation is off** — absent means "nothing to say", which beats a
tri-state. The renderer shows the badge when the field is present, false, and the message has no
`tool_calls`.

**Cost:** the toggle's user-visible meaning changes, so its tooltip and the F1 help card change with it.

## Closing the follow-up hole

The auto-search's ephemerality has a user-visible consequence:

1. User asks about X → 2. auto-search returns documents about X → 3. model answers, grounded →
4. user asks a follow-up about a detail → 5. the material is gone, and the model has no idea where step 3
came from.

The information is not lost, only never shown again: every assistant node already carries
`retrieval: {"query": ..., "results": [...]}` (`create_ai_payload`, gated on `docs_query is not None`).

**Fix: inject the pointers, not the text.** Walk the branch's assistant nodes, collect and dedupe the
`document_id`s, and give the model a compact "documents consulted earlier in this conversation" list.
Re-injecting the text would grow without bound; a list of relative paths does not.

This works only because `fetch_document` exists — a pointer plus a way to follow it. It is the second
justification for that tool, and a better one than symmetry with `webfetch`.

Note what it does *not* do: an ID list is not grounding material, so the badge still fires at step 5
unless the model actually fetches. That is correct — at step 5 it has not re-read anything. The badge says
so, and the remedy is sitting right there.

### `list_consulted_documents`, and it is both a real tool and an inject

"Consulted" is deliberately agnostic about *who* consulted: the list merges documents Raven auto-searched
with documents the model fetched itself, and `list_searched_documents` / `list_retrieved_documents` would
each smuggle in an actor. It also leaves the short names free for the tools that must not collide with it
— `list_documents` and a topics/scopes lister both mean "what exists in the database", which is a
different question from "what has this conversation looked at".

**Real tool from the start, *and* auto-injected.** These are not alternatives:

- As a *tool*, because the observed behaviour says it will get called. Qwen reaches for retrieval it does
  not have when it feels a gap — that is the entire Q11 finding — so a tool that answers "what have we
  already looked at" is one it will use unprompted.
- As an *inject*, because the model cannot always detect the gap. At step 5 its own transcript shows it
  answering from documents, so nothing signals that the material is gone. The push covers the case where
  the model does not know to pull. Occasional redundancy between the two is an accepted cost.

**Content per entry: the `document_id`, and the query that surfaced it.** Both are already in the stored
`retrieval` payload. List only *previously* consulted documents, excluding this turn's matches — those are
present with their full text right beside it, so including them is redundancy in the one place that has to
stay compact.

**Unresolved: how to label an entry usefully.** The query is a weak label *today* because
`docs_query` is the user's raw message text, which is often an essay — and a `document_id` is a relative
path, which is opaque for a corpus filed under accession numbers rather than titles. So an entry may carry
nothing the model can judge "is this worth fetching?" by. Note the field is not the problem: it is the
slot where a *good* query goes once brief 09's lever 3 lands, and the tool path already writes short
model-authored queries into it. Candidate stopgaps, in preference order: truncate the query for display
(becomes unnecessary as queries improve); or derive a pseudo-title from the document's first line, which
is cheap since `retriever.documents[id]["text"]` is right there. Decide against a real corpus rather than
in the abstract.

## Build order

1. **Tool surface** — per-name gate; `tool_context` per-turn with `retriever` + grounding flag + budget;
   shared match formatter with offsets; `search_documents`; round cap with tools-stripped final call;
   grounding flag replaces the `_context_is_present` message-shape heuristic. *The measured Q11 fix, and
   it stands alone.*
2. **Grounding UX** *(built)* — bypass deleted, `grounded` metadata on the reply, marker in both
   frontends, toggle semantics, tooltip, F1 card.
3. **Truncation engine, `fetch_document`, provenance list** — budget helper, `(offset, length)`, applied
   to the attachment wire-fold, plus the consulted-documents injection. *Largest, least demo-critical.*

PRF was briefly listed here as a fourth step and belongs to
`09_retrieval-query-side-brief.md` instead, which is where the query-side levers live and where the
measurement caveats already are. Nothing in it depends on this brief beyond the observation recorded there:
the tool changes what a weak first pass costs, so the measurement that decides PRF should be taken after
step 1, in conversation, not standalone.

## Deliberately out of scope

- **Scoped search, and a topics/scopes tool.** Both depend on document scopes, which do not exist yet.
- **Attachment search/fetch tools.** Not the same problem in a different hat: documents live in the RAG
  index, attachments are per-chat sidecars that were never indexed. The right answer is a *separate
  HybridIR store* — a per-chat store is a scope, so this wants the same machinery scopes want, and belongs
  with that work. Truncating attachments (§5) makes these tools *wanted*, which is the reason to keep them
  near the front of that queue rather than filed and forgotten.
