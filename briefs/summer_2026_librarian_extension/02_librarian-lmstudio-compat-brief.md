# Brief: LM Studio / generic OpenAI-compat support in Librarian

**Scope:** `raven/librarian/llmclient.py` (and a few config keys in `config.py`). `chattree`,
`scaffold`, `hybridir` need no changes — they never touch the wire and go through `invoke`.
`hybridir` computes embeddings locally via `mayberemote.Embedder`, independent of the chat backend.

**Goal:** make Librarian work against LM Studio (and, as a free side effect, generic
OpenAI-compatible servers) in addition to oobabooga, without regressing ooba.

**Design principle:** a *thin* backend seam, not a class hierarchy. One `backend_flavor`
value, autodetected, gating ~3 functions. Everything else writes to the common
OpenAI-compatible surface that all backends honour.

---

## 0. Backend detection (new)

Add a `backend_flavor` field to the `settings` env, determined once in `setup()` by probing:

- `GET {backend_url}/api/v0/models` succeeds → `"lmstudio"`
- else `GET {backend_url}/v1/internal/model/info` succeeds → `"oobabooga"`
- else → `"generic"`

Optionally allow an override in `config.py` (`llm_backend_flavor = None` = autodetect).
Keep detection cheap and cached in `settings`; do not re-probe per request.

---

## SETTLED — implement directly

### 1. Streaming `content` null-safety (`invoke`)

Line ~496, `chunk = delta["content"]`. Standard OpenAI streaming sends `"content": null`
(present-but-null) on the role-priming first delta and on tool-call deltas; ooba sends empty
strings, which is why this never bit us. Change to:

```python
chunk = delta.get("content") or ""
```

`get("content", "")` is **not** sufficient — it returns `None` when the key is present with a
null value, and `io.write(None)` raises `TypeError`. The `or ""` coerces {absent, null, empty}
all to `""`. Correct on ooba too.

**Add a code comment** explaining *why* it's `or ""` and not `, ""` (the present-but-null case),
so a future maintainer doesn't "simplify" it straight back into a `TypeError`. This is a
maintainer trap; the comment is load-bearing.

### 2. Streaming tool-call accumulation (`invoke`)

Current code grabs `tool_calls = delta["tool_calls"]` from the final delta, assuming the whole
tool-call object arrives in one chunk (ooba behaviour). LM Studio streams tool-call **argument
tokens incrementally** across many deltas (OpenAI semantics), keyed by `tool_calls[i].index`,
with `function.arguments` arriving as string fragments to be concatenated.

Implement an accumulator over the stream:

- Maintain `dict[int, {id, type, function: {name, arguments}}]` keyed by `index`.
- For each delta with `tool_calls`, for each entry: set `id`/`type`/`function.name` if present
  (first fragment usually carries them), and **append** `function.arguments` fragments.
- At stream end, materialise the accumulator into the `tool_calls` list in index order.

This must work for both parallel tool calls (Qwen/Mistral, multiple indices) and sequential
(GPT-OSS). Keep the existing single-chunk path working for ooba, or — cleaner — make the
accumulator a superset that also handles the "whole object in one delta" case (a single
fragment that already contains complete `arguments`). Prefer the unified accumulator.

**The exact delta shape and a reference accumulator are in the LM Studio tools doc**
(local copy `00_stuff/lmstudio_api_docs/oai_03_tools_and_function_calling.md`, "Streaming"
section; online at https://lmstudio.ai/docs/developer/openai-compat/tools#streaming): the
first fragment carries
`id`, `type`, and `function.name` with empty `arguments`; subsequent fragments carry only
`function.arguments` pieces with `id`/`name`/`type` as `None`. Their accumulator concatenates
`id`, `name`, and `arguments` with the same `(x or "")` null-guard as §1, keyed by
`tc.index`. So this is settled, not speculative — V2 below is now just a sanity-confirm that
the target model behaves as documented, not a discovery exercise.

### 3. Model picker / connection probe → standard endpoint (`list_models`, `test_connection`)

`list_models` currently hits ooba-private `/v1/internal/model/list` and reads `model_names`.
For the picker and the connection probe (order irrelevant), switch to the standard
`GET /v1/models` → `payload["data"]`, mapping `id`. ooba supports this endpoint too, so it works
on all flavors. This is *only* for listing/probing — model **identity** is handled separately
(see §4), because `/v1/models` can't tell you which model is actually loaded on LM Studio
under JIT.

### 4. Model identity for the system prompt (`setup`)

The card asserts `The LLM version is "{model}".` as a fact about the model's own identity, so a
**wrong** value is worse than an empty one (a demo audience member who asks "which model are
you?" must not be told something false — raptor contingency: low probability, but the failure
mode is the system broadcasting that it's broken).

**Add a new config item `llm_model`** (default `None` = use whatever the backend reports as
loaded). When set, it both names the desired model in requests (relevant for LM Studio JIT,
where the request's `model` field tells LM Studio which downloaded model to load on demand) and
informs identity resolution. There is currently no model picker UI in Librarian — the backend
provides a default — but this config field is the strictly-more-general replacement, and a GUI
dropdown driven off `/v1/models` is the natural follow-on whenever it earns prioritization.

(**LM Studio JIT**: just-in-time model loading. With JIT enabled, LM Studio loads a model only
when a request arrives for it, rather than keeping all configured models resident. Consequence:
`/v1/models` lists all *downloaded* models because any of them might be loaded on request, but
only zero-or-one is actually resident at any moment. The standard endpoint can't disambiguate;
the native `/api/v0/models` with `state == "loaded"` can.)

Resolve `model` by flavor:

- `oobabooga`: keep `GET /v1/internal/model/info` → `model_name` (the GGUF filename; fine —
  the model can interpret `name-size-quant.gguf` itself, and a raw filename is an acceptable
  answer).
- `lmstudio`: `GET /api/v0/models`, filter `state == "loaded"`, take its `id`. Bonus: the same
  record carries structured `quantization`, parameter count, `arch`, and `loaded_context_length`
  — assemble a richer, accurate card line (e.g. `Qwen3-32B, Q4_K_M, 128k context`) instead of a
  bare filename.
- `generic`: best-effort from `/v1/models`. If the loaded model is ambiguous (multiple entries,
  no loaded-state signal), inject the explicit string **"No model information is available"**
  rather than a guess. Never guess identity.

### 5. Context window from the backend (`setup`, plumb into budgeting later)

Capture the **loaded** context length, not the model ceiling:

- `lmstudio`: `loaded_context_length` from `/api/v0/models/{id}`.
- `oobabooga`: whatever the model-info / load reports as the active `ctx_size`. *Likely
  unreported* (verify); if so, falls through to the default.
- `generic`: unknown.

**On autodetection failure** — for any flavor that doesn't expose the loaded context — default
**conservatively to 64k tokens** (smaller than that isn't useful for discussing a scientific
fulltext, so we can assume users have at least that; 128k is more common but 64k is the safer
guess if we have to pick blind). **Log a warning** when this default is used, naming the backend
that didn't report. Budget against this captured value, never against the model's theoretical
max.

### 6. Drop vestigial request fields; keep `mode` conditionally

In `request_data`:

- Remove `name1` / `name2`. Personas are injected into message *content* by
  `chatutil.create_chat_message` (via `llm_settings.personas` — distinct, internally-consumed
  infrastructure that this removal does not touch), so the wire-level `name1`/`name2` fields
  are redundant with Raven's own handling and ignored by non-ooba backends. Clean up the
  orphaned comments at `llmclient.py:266-267` in the same edit, otherwise they become a
  contextless hanging comment block.
- Keep `mode: "instruct"` **only on the ooba path** (see verify item V1); omit it on
  `lmstudio`/`generic`, where messages → baked-in Jinja chat template is the default and the
  only behaviour.

### 7. Token counting (`token_count`)

- `oobabooga`: keep the exact `/v1/internal/token-count` endpoint.
- `lmstudio` / `generic`: three tiers, in order of preference.
  1. **Optional local tokenizer.** If `llm_tokenizer_path` (new optional config, default `None`)
     is set and loads, use a local `transformers.AutoTokenizer` for *exact* pre-send counts. The
     tokenizer is tiny and fast — for a Qwen-class model the tokenizer files (`tokenizer.json`
     plus `tokenizer_config.json`) are roughly 10-15 MB total against a multi-GB GGUF, and load
     in milliseconds. The team currently co-locates model and client, so this is usually
     available. **Caveat:** it must match the served model, or you get confidently-wrong counts.
     Cross-check cheaply — compare the local tokenizer's prediction for a sent prompt against
     the `usage.prompt_tokens` that comes back; **log a warning on large divergence** (catches a
     mismatched tokenizer for free, surfaces it to the user in the log rather than silently
     mis-budgeting).
  2. **Usage calibration.** No tokenizer files reachable (inference service on another host):
     read `usage.prompt_tokens` / `completion_tokens` from real responses — exact, server-side,
     location-independent, and inclusive of system prompt, history, RAG injection, and tool
     definitions. Maintain a per-model char→token ratio (single ratio is fine for v0; do not
     build per-content-class calibration until budgeting exists and the ratio is seen to drift).
     For forward budgeting use the ratio with a conservative safety margin, correcting from
     ground-truth `usage` each call.
  3. **Prefill-on-HEAD-switch — in v0 (exact count + cache warming).** When the user navigates
     to a different node in the chat tree, after a **~5 second idle delay** (so pure browsing
     doesn't burn prefills), send the full prompt with `max_tokens: 0` (or `1` if the backend
     rejects `0`) and read `usage.prompt_tokens` from the response. This gives an *exact*
     pre-send token count for the new HEAD's prompt, AND populates the backend's KV cache so the
     next real generation from this HEAD pays no prefill cost. Cost is one prefill, which you'd
     pay on the next generation anyway — so the net is a wash with two upsides: exact-count
     (which usage-calibration can't provide for an unsent prompt) and warm cache. Cancel the
     pending prefill if the HEAD switches again before the idle timer fires, or if the user
     starts generating directly (the prefill happens as part of the real generation anyway).
- To get `usage` while streaming, send `stream_options: {"include_usage": true}` (LM Studio /
  OpenAI require opt-in; ooba sends it unconditionally). Prefer real `usage` over the
  `n_chunks - 2` heuristic when present; keep a robustified chunk count only as last-resort
  fallback.

**Context-fill GUI indicator.** Surface the current HEAD's token usage as a percentage of the
loaded context window (§5), with **explicit honesty about precision**: render as `X%` (no
tilde) when the count is **exact** — i.e. tier-1 local tokenizer present, OR a recent `usage`
reading from the backend covers the current prompt unchanged — and as `~X%` (with tilde) when
the count is the tier-2 **calibrated estimate**. This single character of typography
distinguishes "I know" from "I estimate." Update on every: message added, HEAD switch (after
the §7-tier-3 idle-prefill returns exact usage), and response completion (real `usage` arrives).
The bare numbers alongside the percentage (`8400 / 64000`, "8% (exact)") help when a fulltext
upload pushes the user against the ceiling and the percentage alone is too coarse.

### 8. Remove the legacy tool-call instruction injection entirely

`llm_send_toolcall_instructions` / `legacy_tools_prompt` inject Raven's own tool-calling prompt
for models lacking a tool template — needed only for early-2025 DeepSeek-R1 distills. **Decision:
remove it.** Any tool-capable model newer than that carries the tool-calling instructions in its
own Jinja template, so client-side injection is at best redundant and at worst conflicting (on
LM Studio, "default tool use support" already injects a server-side tool prompt + parser for
non-native models — `00_stuff/lmstudio_api_docs/oai_03_tools_and_function_calling.md`,
"Default tool use support" — so a second client-side
injection collides; ooba does server-side injection for templated models too, so the same
collision applies there). Pre-2025 models are out of support by policy (LLM tech moves fast enough
that half-a-year-old models are effectively Bronze Age). Drop the config flag, the
`legacy_tools_prompt` builder, and the injection branch in `invoke`.

### 9. Surfacing thinking to the user (not stripping it)

**Design intent correction:** these are open models — non-empty thinking is *exposed* to the
user as a thought bubble, not discarded. The handling splits three ways:

- *think-mode disable* (§V3) controls whether the model generates a thinking trace at all
  (speed / when reasoning isn't wanted).
- *when present and non-empty* → surface to the UI as a separate channel (thought bubble).
- *empty blocks* (e.g. Gemma 4 27B's `<think></think>` emitted even when disabled, though
  Google has since added an empty-thinking-token to the template specifically to suppress these
  "ghost" channels — verify the current quants you're using) → strip.
- *prior-turn thinking* → **per-family toggle, both behaviours first-class.** The landscape as
  of 05/2026 (verified via official model docs):
  - **Gemma 4** → **strip** from resent history. Google's own docs (model card, prompt-formatting
    guide, Unsloth and Ollama mirrors) all explicitly say: "In multi-turn conversations, the
    historical model output should only include the final response. Thoughts from previous model
    turns must not be added before the next user turn begins." Definitive.
  - **Qwen 3.0** → **strip** (matches the older default).
  - **Qwen 3.6** → **preserve** (the change; its template/training expects prior thinking on
    resend).
  - **Universal exception** (the intra-turn-tool-call case): when a single model turn includes
    tool calls, the thinking *between* tool calls within that turn must be preserved for coherent
    reasoning across the tool sequence. Gemma 4's docs make this exception explicit; Qwen
    behaves the same way.

    **Current Librarian bug to fix as part of this work:** the existing `invoke` scrub of
    `history[:end_idx]` treats "current turn" as "most recent message only." But when a
    tool-call assistant message is followed by tool-result messages and then another assistant
    message (the continuation after tool returns), the original tool-call message is no longer
    the most recent — so the current scrub strips its thinking, breaking the intra-turn
    preservation contract. **The correct boundary is the most recent `role: user` message**: walk
    history backward, find the last user message, and treat everything from there onward as the
    current turn (preserve thinking universally regardless of family). Apply the family policy
    (strip-or-preserve) only to messages *before* that boundary. This unifies the
    implementation: one boundary rule, two family policies, intra-turn always preserved.

  So the default is **strip** (covers Gemma 4 + Qwen 3.0); **preserve** is the explicit
  per-family setting (Qwen 3.6 today; revisit ~2027). This is simpler than I'd framed earlier —
  one default, one exception, both first-class via the §V3 per-family table.

**UI work is smaller than it sounds**, because Librarian's `chat_controller.py` *already* has
the collapsible thought-bubble mechanism: messages carry `is_thought` per paragraph, and
`DPGCompleteChatMessage` already adds a per-message toggle button (with Ctrl+T hotkey) to show
/ hide the thinking trace. So the change is **source-of-truth**, not new UI.

**The architecture: `invoke` is the single parser; the chat_controller is purely a renderer of
typed events.** This applies uniformly to both inline-tag content and native-channel content:

- **Thinking** — `invoke` parses inline `<think>...</think>` blocks out of the model's text
  stream *and* consumes `reasoning_content` from streamed deltas when the backend separates it
  (llama.cpp / LM Studio `reasoning_format=auto` for Qwen/Gemma/GPT-OSS). Both sources emit the
  same `{"type": "reasoning", "text": "..."}` events. Inline-tag content is **stripped from
  the content stream** in the process.
- **Tool calls** — `invoke` parses inline `<tool_call>...</tool_call>` blocks out of the text
  stream *and* consumes native `tool_calls[i]` fragments from streamed deltas via the §2
  accumulator. Both sources emit the same `{"type": "tool_call", ...}` events (one per
  completed call, with full `{id, name, arguments}`). Inline-tag content is **stripped from
  the content stream** in the process.

  **Dedup against double-emission** — `invoke` must emit each actual tool call *once*,
  regardless of how many channels deliver it. Some backends (ooba as of ~Jan 2026, possibly
  fixed in latest; assume the failure mode exists) emit `<tool_call>` inline in the text stream
  **and** populate the structured `tool_calls` field at EOS. Without dedup the parser would
  produce two `tool_call` events for one underlying call. Maintain a "already emitted" set
  keyed by `id` where present (the OpenAI-native form always carries an id; the inline-tag form
  often doesn't, so assign a synthetic id at parse time). When the structured form arrives at
  EOS, check whether any previously-emitted inline-parsed call matches by `(name,
  normalized_arguments_json)`; if so, suppress the structured emission as a duplicate. If no
  content match, it's a genuinely new call — emit it. This handles ooba's double-emission
  without a backend-specific branch and preserves the legitimate "same tool called twice in one
  turn with the same args" case via id-based matching (the OpenAI native form always
  distinguishes those).
- **Text content** outside any special block emits as `{"type": "content", "text": "..."}`.

The chat_controller's streaming callback then just dispatches on event type:
`content` → text paragraph (`is_thought=False`), `reasoning` → thought paragraph
(`is_thought=True`), `tool_call` → render the tool-call sub-element with its icon (§9 above).
No regex-sniffing the text stream, no double-display worries — the inline-tag content can't
appear in the content stream because `invoke` already routed it.

**Cleanup that falls out**: `chat_controller.py:450-453` (the inline `<tool_call>` and
`<think>` text-replacement that formats them as bold markers in the displayed content) becomes
obsolete and should be removed — invoke no longer lets that text through. The `inside_think_block`
state-tracking inside chat_controller's streaming handler likewise goes away; the event type
*is* the state.

**Parser-side detail to watch**: streamed chunks may split a tag across chunks (e.g. `</thi` in
one chunk, `nk>` in the next). The invoke parser must buffer at the tag-boundary level — a
simple state machine over a small look-ahead buffer handles it. Same for `<tool_call>` JSON
that may span chunks before being parseable.

**Tool-call-fragment events are part of THIS brief, not future** — they're emitted by the
streaming accumulator from §2 (you're already restructuring tool-call parsing anyway), and
they unblock a real Librarian gap.

Today the chat_controller renders inline `<tool_call>...</tool_call>` tags as text replacement
(`chat_controller.py:450-451`) — this is the **output-side** convention some models / backends
use when emitting tool calls in the text stream rather than via the OpenAI-native `tool_calls`
field. (This is *not* the same as the legacy QwQ-32B input-side injection being removed in §8;
§8 removes how Raven *told models* to call tools — the inline-tag rendering handles what models
*emit* when they do. Independent concerns.) `role: tool` result messages also render with their
own icon as you'd expect.

**What's currently missing**: native-format `tool_calls: [{function: {name, arguments}, ...}]`
invocations in assistant messages are *not visibly rendered*. The infrastructure callbacks
(`on_tools_start`, `on_call_lowlevel_start`) already pass the data through, but it isn't wired
to a visible widget — so the user sees the assistant's text content (often empty for
tool-call-only turns), then the tool result, with no visible "→ `get_weather(city: Tokyo)`"
between. Librarian's no-magic / what-you-see-is-what-you-get design (the same principle that
shows the system prompt verbatim at session start) wants tool invocations visible.

**v0 work to close this:**
- The accumulator from §2 emits structured `tool_call` events via the typed-event stream
  already in v0 (not later); this is the data path. **The stripping of inline
  `<tool_call>...</tool_call>` text happens at the `invoke` parser layer, not at render** — see
  the parser architecture in the next subsection. Inline-tag content never reaches the content
  event stream, so the chat_controller can't possibly double-display it.
- The chat_controller renders each `tool_call` event as a styled sub-element inside its
  assistant message — analogous visual weight to the existing thinking-trace toggle, with a
  distinct icon. FontAwesome free has plenty of fitting candidates (`gear` / `cog`, `wrench`,
  `screwdriver-wrench`); `gear` pairs naturally with the cloud / thought-bubble icon used for
  thinking. CC picks.

`reasoning_format`/`reasoning_content` is the proper-channel path — worth migrating to where the
backend's reasoning parser supports the target family (Qwen, Gemma 4, GPT-OSS); keep tag-based
parsing as fallback for any family it doesn't.

**Where the typed events land in persistent storage**, and how `invoke` reassembles them for
resend, are covered in the next two sections (persistent message format + resend wrapping;
migration). The data-model decisions belong here in compat rather than in content-parts because
they're the persistent counterpart of compat's typed-event stream — same data, two
representations (stream-time and storage-time).

---

## 10. Persistent message format and resend wrapping

The typed-event stream from §9 lands in three distinct places on the persistent message,
matching OpenAI's wire format exactly (plus the llama.cpp / LM Studio `reasoning_content`
convention for the reasoning channel):

| Streaming event | Persistent storage |
|---|---|
| `content` text | accumulates into `message["content"]` (string today; the content-parts brief refactors this to a list of typed parts) |
| `reasoning` text | accumulates into `message["reasoning_content"]` (string sibling field) |
| `tool_call` (complete) | appended to `message["tool_calls"]` (list of structured OAI tool-call dicts) |

`reasoning_content` is the canonical home for reasoning — *never inside `content`*. The current
code embeds `<think>...</think>` in content (which is why chat_controller has the inline tag
parser at line 450-453); this refactor moves that out, and the migration in §11 extracts it
from old data.

**Resend wrapping — two orthogonal decisions** (this took us a moment to disentangle):

1. **Whether to wrap** (per message, when sending history back to the model): driven by family
   policy from §9 *plus* the turn-boundary check (the most-recent-`role:user`-message rule).
   - Messages *before* the most recent user message → "prior turn." Family policy decides:
     strip families (Gemma 4, Qwen 3.0) → don't wrap, omit reasoning. Preserve families
     (Qwen 3.6) → wrap and resend.
   - Messages *from* the most recent user message onward → "current turn." **Always wrap and
     resend**, regardless of family policy — universal intra-turn exception, including the
     intra-turn-tool-call case (Gemma 4's docs explicitly require thinking preserved between
     tool calls in a single turn).
2. **How to wrap** (per family): the spelling of `(open_tag, close_tag)` used when wrapping is
   triggered. *Every* family needs a spelling, because the intra-turn-tool-call case triggers
   wrapping even for strip families.

**Per-family `reasoning_wrap_input` table** (new entry alongside the existing per-family knobs
in `raven.librarian.config`):

```python
reasoning_wrap_input = {
    "qwen3":   ("<think>",   "</think>"),    # Qwen series — simple text tags
    "qwen3.6": ("<think>",   "</think>"),    # same spelling; family policy is preserve, so wraps prior turns too
    "gemma4":  ("<|channel|>thought\n", "<|channel|>"),
                                              # Documented format from Gemma's chat template uses special tokens
                                              # for the reasoning channel. CC should inspect the actual Gemma 4
                                              # chat template (in the GGUF metadata or HF tokenizer config) to
                                              # confirm the exact spelling — the markers shown here are based on
                                              # external write-ups, not direct inspection. V4 verifies the
                                              # documented format actually round-trips through raw-text input.
    # New families: add per chat-template inspection
}
```

When `invoke` is about to serialize an assistant message for resend AND the wrap decision says
yes AND `reasoning_content` is non-empty:

1. Look up the family's `(open_tag, close_tag)`.
2. Assemble the wire `content`: `f"{open_tag}{reasoning_content}{close_tag}\n\n" +
   <existing content>`.
3. Drop the structured `reasoning_content` field from the outgoing message (it's now embedded
   in `content`).

When the wrap decision says no (strip family + prior turn), `reasoning_content` is simply
omitted from the outgoing message and not wrapped at all.

**Why inline-tag wrapping rather than passing `reasoning_content` as a separate input field:**
backend support for `reasoning_content` on *input* (versus output) is inconsistent. llama.cpp /
LM Studio parse it out of model *output* well, but accepting it as an input field and
reassembling into the templated prompt depends on the chat template having an explicit reasoning
slot — not universal. Inline-tag wrapping is universal because it puts the text into the content
stream that every chat template threads through unchanged.

---

## 11. Migration of existing chat history

Existing chats have reasoning embedded inline as `<think>...</think>` in `message.content`
strings, and (for tool-role messages) the `tool_call_id` linkage stored under
`payload["generation_metadata"]["toolcall_id"]` (one word, lowercase) rather than on the message
itself. Both need normalizing to the storage format §10 specifies.

**Hook point: extend `chatutil.upgrade_datastore`** (the actual upgrade function — the docstring
reference at `chattree.py:786` previously pointed at a fictional `llmclient.upgrade`; that's now
fixed in master). The existing function handles the v0.2.2 → v0.2.3 payload-format migration
(unrevisioned → revisioned; metadata move). Adding a new stanza for the reasoning + tool_call_id
normalization is a small extension to the same pattern, called from the same load-time path.
Idempotent; safe to run repeatedly.

**Schema confirmed from real datastore samples:**

```
node:    {id, timestamp, parent, children, active_revision,
          next_free_revision, revision_names, data: {rev_id: payload}}
payload: {message, general_metadata, generation_metadata?, retrieval?}
message: {role, content, tool_calls}
  general_metadata:    {timestamp, datetime, persona}              -- always
  generation_metadata: {model, n_tokens, dt}                       -- assistant
                       {status, toolcall_id, function_name, dt}    -- tool
  retrieval:           {query, results: [{document_id, offset, score, text}]}  -- RAG-only
```

**Transformations** (each guarded for idempotency):

1. **Inline-tag reasoning extraction.** For every message: extract `<think>...</think>` blocks
   from `message["content"]` into `message["reasoning_content"]`, leaving cleanly-spaced text
   around what remains. Concrete rule:

   - **Regex** (DOTALL): `\s*<think>(.*?)</think>\s*` — captures the inner reasoning, consumes
     surrounding whitespace (including newlines) on both sides.
   - **Replacement**: a single space.
   - Then `.strip()` the result.
   - For each captured block, append its `.strip()`-ed content to `message["reasoning_content"]`,
     separated by `\n\n` if there's more than one (rare but possible).

   Worked example — old content with persona prefix and a leading think block:

   ```
   "Aria: <think>\nhmm...\n</think>\nBla..."
   ```

   becomes:

   ```python
   message["reasoning_content"] = "hmm..."
   message["content"]           = "Aria: Bla..."
   ```

   The persona prefix (`"Aria: "`) is preserved exactly because the regex consumes the
   whitespace and tag span between the prefix and the post-thinking text, replacing it with the
   single space that was already conceptually there. Same rule handles the edge cases cleanly:
   `"Aria: Let me think. <think>...</think>\nMy answer"` → `"Aria: Let me think. My answer"`;
   `"<think>only thinking</think>"` (no visible response at all) → empty content; multiple
   think blocks concatenate into reasoning_content with paragraph separators. Verified from
   the sample: 7 messages in the test datastore have `<think>` inline (many more in Juha's full
   datastore), so this step is load-bearing for old data, not optional.

2. **Tool-call inline-tag dedup.** If old `message["content"]` also contains
   `<tool_call>...</tool_call>` blocks AND the same calls are already in
   `message["tool_calls"]`, strip the inline-tag text from the content (double-represented;
   keep the structured form). If the inline tag is present but `tool_calls` is empty/missing,
   parse the JSON, populate `tool_calls`, then strip. Same dedup rule as the runtime invoke
   handles per §2.

   **`tool_calls` storage format confirmed** from a sample with a real call: list of
   structured OAI dicts (not JSON strings), e.g.:
   ```python
   {"type": "function",
    "function": {"name": "websearch", "arguments": "{\"query\": \"...\"}"},
    "id": "call_olu0kwgy",
    "index": "0"}
   ```
   Already OAI wire shape. (Note: `arguments` is itself a JSON string — OAI's serialization
   convention.) So only one shape needs handling, no JSON-string fallback path.

3. **Move `toolcall_id` to message + codebase-wide rename `toolcall_id` → `tool_call_id`.**
   OAI spec puts the linkage on the *tool response message* with the field named `tool_call_id`,
   matching the `id` field of the corresponding assistant `tool_calls[i]` entry:

   ```python
   # Assistant's invocation (already OAI-correct):
   {"role": "assistant", "tool_calls": [{"id": "call_xyz", "function": {...}, "type": "function"}]}

   # Tool response (currently in generation_metadata.toolcall_id; migration moves to message):
   {"role": "tool", "tool_call_id": "call_xyz", "content": "<result>"}
   ```

   For every tool-role message: read `payload["generation_metadata"]["toolcall_id"]`, write to
   `message["tool_call_id"]`, **delete from `generation_metadata`** (full move, single source
   of truth). The other `generation_metadata` fields (`status`, `function_name`, `dt`) belong
   with the tool *execution* and stay where they are.

   **Codebase-wide rename `toolcall_id` → `tool_call_id`.** Today Raven uses `toolcall_id`
   (one word) internally — in `llmclient.perform_tool_calls`'s parameters, in tool-response
   record attributes (`record.toolcall_id`), in `chat_controller.py`'s callback signatures
   (`on_call_lowlevel_start(toolcall_id, ...)`, `on_call_lowlevel_done`), in tests, in log
   messages. OAI wire format uses `tool_call_id`. The current divergence is an inconsistency —
   arguably a bug — and other OAI-derived names in Raven (`tool_calls`, `function`, `arguments`)
   are already used unmodified internally. Mechanical search-and-replace with attention to
   context; single canonical spelling, no translation layer anywhere.

   **Codify the convention** for the next OAI-derived field: Raven's internal naming follows
   OAI wire format spelling unless there's a specific reason not to.

4. **Robustness:** if any parse step fails (malformed tags, broken JSON), fall back to leaving
   that message's content unchanged with no extraction. Degraded but lossless — better to show
   the tags as literal text than to crash on load.

**Idempotency check** for repeat-safety: after migration, no message's `content` contains
`<think>` or `<tool_call>` substrings, every tool-role message has `tool_call_id` on `message`
(not in `generation_metadata`), and `reasoning_content` is either absent or a clean string. A
second pass through the migrator over already-migrated payloads should produce no changes.

**Stale-docstring cleanup**: update `create_chat_message`'s docstring entry for `tool_calls`
from "a list of JSON strings generated by the LLM" to "a list of structured OAI tool-call
dicts" — matches actual behavior. One-line fix, prevents the next person from chasing the same
red herring. (The companion `chattree.py:786` stale-comment fix is already done and pushed.)

---

## VERIFY ON HARDWARE FIRST, then implement

Use `lms log stream` (LM Studio) and `--verbose` (ooba) to inspect the final templated prompt
and raw stream before committing the implementation. **Test directly against the backend's HTTP
API** — via `curl` or the `openai` Python SDK pointed at the backend's URL — *not* through
Librarian, since Librarian doesn't yet work against LM Studio (chicken-and-egg). All three
checks are 10-20 minute exercises during a coffee.

**Sequencing within the CC sprint:** V1/V2/V3 are the *first task* in the session, not a
prerequisite that has to happen before CC starts. CC runs the verification scripts (below),
reports findings, then proceeds with implementation based on the actual measured behavior. This
collapses the verify-then-implement cycle into a single agentic session.

**Backend coordination — CC must ask Juha to start the appropriate server.** VRAM constraint:
Juha cannot host Qwen on both ooba and LM Studio simultaneously. The test sequence is therefore
*serial*:

1. CC asks Juha to start **ooba** with `--verbose` → run V1 → CC reports findings.
2. CC asks Juha to **stop ooba and start LM Studio** with `lms log stream` running → run V2 and
   V3 → CC reports findings.
3. CC proceeds with implementation based on the combined results.

CC must not assume both servers are running, nor try to run a test against a backend without
first asking Juha to start it.

- **V1 — ooba `mode` fallback.** Confirm whether omitting `mode` makes ooba fall back to a
  UI-configured default (possibly `chat-instruct`, which adds roleplay framing). If so, §6's
  "keep `mode: instruct` on ooba" is load-bearing; if omitting it already yields plain
  template application, `mode` can be dropped everywhere. **Protocol:** start ooba with
  `--verbose`. `curl` a minimal two-message chat (`system: "You are helpful."` + `user: "Hi"`) to
  `/v1/chat/completions` once *with* `"mode": "instruct"` in the body, once *without*. Compare
  the rendered prompts in the ooba console: identical → `mode` is redundant; structurally
  different (extra framing) → `mode: instruct` is doing work, keep on ooba path.
- **V2 — tool-call delta shape on LM Studio (sanity-confirm only).** The shape is documented
  (§2); just confirm the target model behaves as the doc
  (local copy `00_stuff/lmstudio_api_docs/oai_03_tools_and_function_calling.md`; online at
  https://lmstudio.ai/docs/developer/openai-compat/tools) says. **Protocol:**
  start LM Studio
  with `lms log stream` in a second terminal. From a quick Python script using the `openai`
  client pointed at `http://localhost:1234/v1`, send a chat completion with `stream=True`, a
  trivially-defined `get_weather(location)` tool, and the prompt "What's the weather in Tokyo?".
  Watch the streamed deltas: first chunk should carry `tool_calls[0]` with `id`/`type`/
  `function.name` and empty `function.arguments`; subsequent chunks should carry only
  `function.arguments` fragments. Bonus: ask for weather in two cities to confirm the `index`
  field separates concurrent calls. No Librarian or stub-tool needed; a tool definition that
  never actually gets called is fine — we're observing the stream shape, not executing the call.
- **V3 — think-mode passthrough, per family.** Mechanism differs by model family; reuse the
  existing per-family table in `config.py` to decide which applies.
  - *Qwen 3.5/3.6:* `chat_template_kwargs: {"enable_thinking": false}` in the request body —
    requires the backend to forward the kwarg to the template (verify LM Studio does). The old
    in-band `/no_think` soft switch is likely removed in 3.5/3.6 (verify).
  - *Gemma 4:* disabled via `<|think|>` token at system-prompt start (present = enabled, absent
    = disabled), which **Raven controls directly** — so it works on any backend regardless of
    kwarg forwarding. (Note: Google has added an empty-thinking-token to the official chat
    templates for Gemma 4 26B-A4B and 31B specifically to suppress the "ghost" empty-`<think>`
    channels that appeared even when thinking was deactivated; with current templates this may
    already be a non-issue, but the output stripper handles them as a safety net regardless.)
  The think-mode disable only decides whether the model *generates* a trace; how a generated
  trace is then handled (expose non-empty, strip empty, discard or preserve from resent history
  per family) is §9.

  **Protocol:** with `lms log stream` running and Qwen 3.6 loaded, `curl` (or `openai`-SDK) a
  request to `/v1/chat/completions` with `"chat_template_kwargs": {"enable_thinking": false}` in
  the body. In the log, inspect the rendered prompt's tail. If you see a pre-filled
  `<think>\n\n</think>\n\n` empty block after the assistant generation prompt: passthrough works.
  If the prompt ends at a bare `<|im_start|>assistant\n` without prefill: LM Studio isn't
  forwarding the kwarg — fall back to in-band injection or accept that thinking generates and
  gets handled at display. For Gemma 4: verify `<|think|>` appears in the system prompt area
  when enabled (Raven controls this directly so it should always be there).

- **V4 — Gemma 4 reasoning-tag round-trip on intra-turn tool-call continuations.** §10's
  per-family `reasoning_wrap_input` table defaults Gemma 4 to its documented special-token
  format (Gemma's chat template uses special tokens like `<|channel|>thought\n...<|channel|>`
  for the reasoning channel). But sending special tokens as **raw text input** depends on the
  backend's tokenizer recognizing them as the actual special tokens rather than tokenizing them
  character by character — which varies by backend's `add_special_tokens` handling on user
  content. Verify before relying on it.

  **Step 0 — get the actual spelling.** Inspect Gemma 4's chat template from the GGUF metadata
  (`llama-gguf-split` or similar) or the HF tokenizer config to find the exact special-token
  sequence Gemma uses for its reasoning channel. Update the table entry to match what's
  actually in the template, not what's in third-party write-ups.

  **Protocol:** with LM Studio running a Gemma 4 instance and `lms log stream` in a second
  terminal:

  1. Send a prompt that triggers a tool call. Capture the assistant message's reasoning
     content and the tool_call invocation.
  2. Send a follow-up request that includes (as history) the prior assistant message with its
     reasoning wrapped in the documented **special-token sequence** (from step 0), followed by
     a synthetic tool-result message.
  3. Inspect the rendered prompt in `lms log stream`. Three outcomes:
     - **Special tokens round-trip correctly** (template incorporates them as the reasoning
       channel) → table is correct as set in step 0. Best case, ship as-is.
     - **Special tokens get character-tokenized** (don't render as the actual special tokens
       in the tokenized prompt) → backend isn't recognizing them in raw text input. Fall back
       to plain `<think>...</think>` text tags (what `reasoning_format=auto` typically emits
       on the output side, so the model has *some* exposure to that form even if not its
       primary training format). Update the table entry.
     - **Neither works** → fall back to sending `reasoning_content` as a sibling field on the
       outgoing message and let the backend's template handle it (if it does). Worst case:
       document Gemma's intra-turn-tool-call case as not-yet-fully-working in v0 and revisit
       when Gemma's input support stabilizes.

  Five-minute measurement, resolves the last open question in the migration / wrap-table work.
  Can run in the same LM Studio session as V2/V3 — just swap the loaded model from Qwen 3.6 to
  Gemma 4 between sets.

---

## OUT OF SCOPE (separate workstreams)

- **`continue_` on generic backends.** ooba's explicit flag stays and keeps working on ooba.
  Do not replicate on LM Studio/generic. Adopt LM Studio's own default — `max_tokens` unset/`-1`,
  trust EOS, rely on the cancel button — which eliminates the only real use case (recovering a
  promising truncated response). The residual use case (extending an already-complete response)
  is useless: the model just emits EOS again.

  **UI work this requires** (Librarian GUI, not `llmclient`): at setup, capture a
  `backend_supports_continue` capability flag (true for `oobabooga`, false for `lmstudio` /
  `generic`). The Continue button is **per-message** (each rendered assistant message gets one,
  per `chat_controller.py:642–654`). **AND the capability flag into the build-time
  `continue_enabled` expression at line 642** —

  ```python
  continue_enabled = ((node_id is not None)
                      and (node_id not in greeting_node_ids)
                      and settings.backend_supports_continue)
  ```

  — rather than into the external is-latest disable logic (around line 1141ff). Rationale:
  is-latest is a *positional* check that depends on the message's position in the chat history;
  the capability flag is *global* across the whole history. Mixing them would muddle two
  orthogonal concerns. Each check belongs where its information naturally lives — the global
  flag at build time, the positional one externally afterward.

  Theme infrastructure is already in place — line 652 binds `disablable_widget_theme` (the
  default gray theme matching the white-icon-on-gray styling) to the button. The remaining
  work is the tooltip variant: "Ask the AI to continue this response [Ctrl+U]" when enabled,
  "The connected LLM backend does not support message continuation" when disabled by the
  capability flag. (The implicit positional disable for non-latest messages doesn't need its
  own tooltip — that one's obvious from context.)
- **MCP client.** Replacing hardcoded tools with an MCP client is its own project. Two paths
  (backend-hosted MCP for quick experiments vs. client-side MCP feeding `tool_entrypoints` for
  product consistency) — decide separately.

---

## Acceptance

- Librarian connects, lists models, names the loaded model correctly (or says "No model
  information is available" rather than guessing), and holds a streamed chat — including a
  streamed tool call — against both ooba and LM Studio.
- Native-format tool-call invocations in assistant messages are *visibly rendered* in chat
  history (function name + arguments) as a styled sub-element with its own icon, not silently
  swallowed between an empty-content assistant message and the subsequent tool-result message.
  Whether the call arrived via inline `<tool_call>` tags or via the native `tool_calls` field
  (or, as on some ooba versions, via *both* simultaneously), the user sees exactly one
  rendering of it — `invoke` dedups same-call double-emissions at the parser layer and strips
  inline forms from the content stream.
- The chat_controller does not regex-sniff the content stream for `<think>` or `<tool_call>`
  tags; all tag parsing happens in `invoke`, and chat_controller dispatches purely on
  typed-event type.
- The Continue button is enabled only when (it's the latest assistant message) AND (the backend
  supports continue); tooltips distinguish the two disable reasons.
- The context-fill GUI indicator shows `X%` for exact counts and `~X%` for calibrated estimates,
  updating on message add, HEAD switch (after idle-prefill), and response completion.
- No regression on ooba (diff the final prompt before/after the `name1`/`name2` removal; should
  be byte-identical).
- Token budgeting reads real `usage` where available and degrades to calibrated estimate
  elsewhere.
- Prior-turn thinking handling: messages before the most recent user turn are scrubbed per the
  family policy (strip for Gemma 4 / Qwen 3.0, preserve for Qwen 3.6); messages from the most
  recent user turn onward are universally preserved, including thinking between tool calls
  within the same turn.
- Reasoning persists as `message.reasoning_content` (string sibling field); no `<think>` tags
  remain in `message.content` after a message has been parsed by `invoke`.
- Tool-role messages carry `tool_call_id` directly on `message` (matching OAI spec); no
  `generation_metadata.toolcall_id` remains after migration.
- Codebase-wide `toolcall_id` → `tool_call_id` rename is complete: no occurrences of
  `toolcall_id` (one word) remain in source, tests, or log strings.
- Existing chat history loads cleanly under the new code: reasoning extraction, tool-call
  inline-tag dedup, and `tool_call_id` move all happen at load time via `chatutil.upgrade_datastore`;
  a second load on the migrated file is idempotent (produces no changes).
- Stale-docstring fix applied: `create_chat_message`'s `tool_calls` docstring says "list of
  structured OAI tool-call dicts." (The companion `chattree.py:786` fix is already in.)
