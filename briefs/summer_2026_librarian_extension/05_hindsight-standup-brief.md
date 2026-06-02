# Hindsight playground standup

Not a CC brief — a sysadmin checklist for the pair-session, written assuming you're not deep in
Docker, plus a captured-from-design-discussion section on integration architecture so the
design points don't evaporate before implementation.

Goal: stand up Hindsight cloud-free (only the local LLM endpoint touches anything outside the
container), wire it into Librarian via two paths (agentic via MCP, autosearch via direct
integration), and record the provenance so chats stay debuggable.

**SearXNG was originally Thread 3a here**; dropped from this brief because `npacker/web-tools`
covers the standalone-LM-Studio webfetch use case and the phase-1 webfetch brief covers
Librarian's needs. SearXNG no longer earns its place in either frontend; leaving it out of the
standup queue rather than carrying dead infrastructure.

Docker shape each time: clone repo / grab the compose file → edit a config file → `docker
compose up -d` → check logs → hit the local endpoint to verify. The traps are in the config
edits, not the Docker mechanics.

---

## 1. Hindsight Docker standup

**What it is:** an agent memory layer (memory banks; world/experience facts consolidated into
observations; TEMPR retrieval = semantic + BM25 + graph + temporal; mission/directives/
disposition config that shapes `reflect`). Open source, `vectorize-io/hindsight`. Deployable
via Docker Compose, Helm, or pip.

### The two things that matter for a cloud-free standup

1. **It's multi-service.** Expect more than one container (API server + storage + whatever
   backs the graph/vector/temporal strategies). Use their compose as-is; don't try to slim it
   on the first pass.
2. **Point its models at your local inference, or it'll want the cloud.** Hindsight needs an
   LLM (for `reflect` / observation consolidation) and an embedder, configured independently
   via env vars per Hindsight's provider matrix:

   - **LLM** → LM Studio (or llama.cpp) on `http://localhost:1234/v1`. Set
     `HINDSIGHT_API_LLM_PROVIDER=openai` and `HINDSIGHT_API_LLM_BASE_URL=...`.
   - **Embeddings** → **Raven-server's new OAI-compatible endpoint** (see §1a below). Set
     `HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai`,
     `HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL=http://localhost:<raven-port>/v1`,
     `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=<HF identifier of the model Raven-server loaded>`.
     Raven already loads an embedding model for Visualizer + Librarian RAG; routing Hindsight
     to the same one is the right call here for a hard reason: **the 4090 doesn't have
     headroom to load a second embedding model alongside the chat LLM in LM Studio.** A
     unified embedder isn't optional in this hardware envelope.

   The exact env-var names live in Hindsight's docs — read these three before the session,
   don't guess:
   - `developer/installation` (the Docker Compose path)
   - `developer/configuration` and `developer/models` (the env-var matrix)
   - `developer/services` (what each container is, so the logs make sense)

---

## 1a. Prerequisite: OAI-compatible `/v1/embeddings` endpoint on Raven-server

Raven-server currently exposes embeddings only via `POST /api/embeddings/compute` with a
Raven-native shape (`{"text": ..., "model": ...}`) — historical, matches SillyTavern's
`SillyTavern-extras` format. The legacy endpoint stays in place for backward compatibility;
**add a new OAI-compatible `/v1/embeddings` endpoint alongside it**. Both endpoints route to
the same underlying `embed_sentences` in `raven/server/modules/embeddings.py`; the new
endpoint is purely a translation layer.

This unblocks Hindsight (which needs OAI shape via its `openai` embeddings provider) and any
other future external consumer that speaks the OAI embeddings standard. Cleaner than asking
each consumer to learn Raven's bespoke shape.

**OAI embeddings spec** — the shape Hindsight and other OAI-compatible clients send:

```json
POST /v1/embeddings
Content-Type: application/json

{
  "model": "<model role or HF repo identifier>",
  "input": "the text to embed",          // OR an array of strings for batch
  "encoding_format": "float"             // optional, default float
}
```

Response:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.023, -0.009, ...], "index": 0}
  ],
  "model": "<echo of requested model>",
  "usage": {"prompt_tokens": 8, "total_tokens": 8}
}
```

**Translation work** (≈50 lines, adjacent to the existing `api_embeddings_compute` in
`raven/server/app.py`):

- Request: `input` (string or list) → pass through to `embed_sentences` as `text`.
- Request: `model` — accept either a Raven role name (`"default"`) or the HF repo identifier
  of any loaded model (e.g., `"Snowflake/snowflake-arctic-embed-l"`). Same flexibility as
  the legacy endpoint, just speaking OAI's wire shape. The OAI spec treats `model` as an
  opaque identifier and doesn't constrain its format; clients send whatever the user
  configured. **Lenient handling**: if the requested name matches neither a role nor a
  loaded model, serve the default-loaded model and log a **warning-level** mismatch message.
  Matches llama.cpp's permissive convention; minimizes configuration-error friction (a
  mis-spelled model name in Hindsight's env var shouldn't break recall) while still
  surfacing the mismatch loudly enough that a user with a typo or stale config sees it.
- Response: wrap each embedding vector in `{"object": "embedding", "embedding": [...],
  "index": i}` and assemble the list.
- Response: `usage` field — minimal naive token count (whitespace split, or character count
  divided by ~4) is fine for v0. Hindsight uses this for cost tracking with hosted providers;
  for self-hosted it's informational, doesn't affect behavior. Accurate token counting would
  need access to the embedding model's tokenizer, which is a nice-to-have, not load-bearing.

**Also expose `GET /v1/models`** alongside the embeddings endpoint — OAI clients sometimes
query it to discover available identifiers. Returns the list of loaded embedding models
formatted per OAI's models-list shape (each entry: `{"id": "<HF identifier>", "object":
"model", ...}`). Cheap addition, prevents surprises for any client that does name discovery.

**Embedding model: nomic-embed v1.5 pair, as part of the planned nomic migration.** This
brief's work makes the migration load-bearing — Hindsight will consume whatever Raven-server
serves, so the migration ships with the standup rather than waiting for Visualizer's importer
rework.

Concrete plan:

- **Single "default" role** serving both text and image via the paired nomic v1.5 models
  (replaces the previous separate "default" and "qa" roles, which existed because snowflake
  and mpnet were architecturally different models for different purposes; nomic-text-v1.5
  covers both via task prefixes, so the role distinction is now redundant).
- **Hard migration** of `embedding_models` config to the new schema. No back-compat
  acceptance of the old bare-string form. Small user base, minimal customization in the
  wild, cleaner schema worth the forced rewrite.

Config shape (post-migration):

```python
embedding_models = {
    "default": {
        "text_model": "nomic-ai/nomic-embed-text-v1.5",
        # image_model: optional; if set, must share a latent space with text_model
        # (Nomic's text-v1.5 and vision-v1.5 are trained for this). Enables unified
        # text+image search via the embed_images() entry point.
        "image_model": "nomic-ai/nomic-embed-vision-v1.5",
        # task_prefixes: per-task strings the embedder prepends to input text before
        # tokenization, for models that use prefix-based asymmetric retrieval. Keys are
        # logical task names ("query", "document", ...) as passed by callers via the
        # task parameter. If task_prefixes is absent or empty, OR if a caller requests
        # a task not listed here, no prefix is applied. Image embedding ignores
        # task_prefixes — images don't use the asymmetric-retrieval prefix convention.
        "task_prefixes": {
            "query": "search_query: ",
            "document": "search_document: ",
        },
    },
    # Example: a symmetric-only text-only model
    # "legacy_symmetric": {
    #     "text_model": "sentence-transformers/all-mpnet-base-v2",
    # },
    # Example: a vision-only role (unusual but supported)
    # "image_only_legacy": {
    #     "image_model": "some-vision-model",
    # },
}
```

Net VRAM impact on the 3070 Ti (where Raven's non-LLM ML runs — see hardware note below):
~460 MB (text + vision) vs current ~870 MB (snowflake + mpnet). Migration saves ~410 MB
*and* unblocks multimodal Visualizer search *and* unblocks Hindsight in one move.

### Hardware note — VRAM budget actually matters here

The reasoning above hinges on real VRAM constraints on the primary dev machine (`maia`),
which the brief should record so model-choice decisions aren't made against the wrong mental
model:

- **4090 (24 GB)** — *dedicated to the LLM*. Filled completely by a 30B-class model at INT4.
  No headroom for anything else.
- **3070 Ti (8 GB)** — *everything else*. Shared with the system's graphics output. Realistic
  budget for ML: 6-7 GB after baseline display usage. Resident consumers include the avatar
  (THA3 ~1-2 GB, latency-critical so it stays GPU-resident), STT (Whisper, 1-3 GB depending
  on model), TTS, classifier, translator, plus embedders.

So embedder VRAM is genuinely tight, not generous. The ~410 MB the nomic v1.5 migration
saves is meaningful in this envelope; adding v2-moe later would consume ~14% of the total
budget — non-trivial against the other ML competing for it.

- **Low-VRAM "on-the-road" mode** (`raven-server --config raven.server.config_lowvram`) —
  fits everything in 8 GB by keeping only the avatar GPU-resident (it's the latency/throughput-
  sensitive one for real-time video); all other ML, including embedders, runs on CPU. Slow
  but works for testing and demos. A small 4B-class LLM at INT4 runs alongside on the same
  GPU for self-contained operation. Quality is "stupid as a boot" (Finnish saying) for
  capability-critical work; adequate for demos and testing where the LLM isn't the
  load-bearing component.

  4B-class failure profile worth planning for — not uniformly bad, but qualitatively
  different from 30B-class:

    - **Factual recall is unreliable.** Confabulates confidently rather than recognizing "I
      don't know" — ask which stars compose the Pleiades, get a plausible-sounding wrong
      answer each time. Architectural, not training-data-quality: parameter mass stores
      facts.
    - **Tool-use meta-cognition is shakier.** Knowing *when* to invoke a tool is itself
      capability-gated. 30B-class generally fares well; 4B-class often answers from priors
      when it should be searching.
    - **Output structural consistency is mixed.** Occasional missing `</think>` close tags,
      malformed JSON in tool calls, partial parameter blocks.

  Defenses against the structural-output failures, and their limits:

    - The compat brief's §9 parsing machinery handles missing close tags so they don't crash
      the chat — but the *previous* version of Librarian's code path had the same crash-
      safety and still broke chat formatting visibly when `</think>` was dropped. **"Doesn't
      crash" and "displays correctly" are different bars**; §9's defenses need explicit
      verification at the latter against actual 4B-class output, not just unit tests for
      "no exception thrown."
    - Tool-call JSON malformation is harder to defend against in the parser; surfaces as
      visible tool-call failures. Recoverable by rerolling.

- **`electra` (the other dev machine)** has a 16 GB internal GPU, looser budget. No
  specialized config yet; likely just runs a slightly larger LLM with the same Raven-server
  config as `maia`'s primary setup.

**Implication for the multilingual upgrade path**: adding v2-moe as a "multilingual" role
isn't a free extension on `maia` — it'd consume a meaningful slice of the 3070 Ti budget,
potentially squeezing other ML. Worth a deliberate evaluation against what else is loaded,
not an assumed "we'll just add it later." On `electra` the extra ~950 MB is easier to absorb.

### Public API: two entry points, role-orthogonal modality

The role names the shared latent space; the *modality* (text vs image) picks which model
within that role serves a given input. The caller knows what it's holding (`str` vs
`PIL.Image`) and calls the appropriate function:

```python
embed_sentences(text: str | list[str], role: str = "default", task: str | None = None) -> ...
embed_images(image: PIL.Image | list[PIL.Image], role: str = "default") -> ...
```

The embedder module reads the role's config; based on which function was called, it picks
`text_model` or `image_model` and dispatches. Missing modality model for a role raises a
clear error ("role 'default' has no image_model configured"). `task` is a text-only concept;
`embed_images` doesn't accept it.

### Surfaces carrying the `task` parameter

Cross-endpoint shape, all referring to text embedding:

- **Internal `embed_sentences(..., task=...)`** — new parameter, logical task name.
- **Legacy `/api/embeddings/compute`** — gains optional `task` field. No back-compat issues
  here since we're hard-migrating; absence still means symmetric.
- **OAI `/v1/embeddings`** — gains optional `task` field as a Raven-specific extension. The
  OAI spec doesn't forbid extra fields (OpenAI itself adds non-standard ones like
  `dimensions` for matryoshka models); absence gives spec-following no-prefix behavior, so
  strict OAI clients work as before. Clients that opt into asymmetric retrieval (potentially
  a future Hindsight wrapper) send `task` and get model-correct prefixing.
- **mayberemote dispatch** — threads `task` through unchanged; per-model prefix logic lives
  in the embedder's local invocation, so remote dispatch just passes the parameter.

This preserves the role abstraction's point: RAG code says "I want a query embedding"
(semantic intent), not "prepend `search_query:` to this string" (implementation detail).
When Raven swaps embedders later, callers don't change; neither does the embedder module —
only `embedding_models` config does.

### Wire format for image embedding — `multipart/form-data` per Raven convention

Raven's existing binary-upload endpoints uniformly use `multipart/form-data` with two named
fields: `file` (the binary) and `json` (a JSON file attachment with request parameters). The
pattern is established across `/api/avatar/load`, `/api/imagefx/process`,
`/api/imagefx/upscale`, `/api/stt/transcribe`; matched helper pair in `raven.common.netutil`:
`pack_parameters_into_json_file_attachment(parameters)` for the client side,
`unpack_parameters_from_json_file_attachment(stream)` for the server side.

New endpoint follows that convention:

```
POST /api/embeddings/image
Content-Type: multipart/form-data

Form fields:
  file: <image binary>            # PNG, JPEG, etc.
  json: <JSON file attachment>    # {"role": "default"}
```

Sits at the same path depth as the existing `/api/embeddings/info` and
`/api/embeddings/compute`; the verb/noun asymmetry between `compute` (legacy
SillyTavern-extras name, kept for back-compat) and `image` (new, noun-based following the
`info` precedent) is mild and reads cleanly as "(compute) image embedding."

Response shape mirrors the existing text endpoint conventions
(`{"embedding": [...], "dimension": N}`), single vector per request.

**No OAI image endpoint.** Image embedding isn't an OAI standard (yet); adding it as an
extension to `/v1/embeddings` would be more "creative reuse of the spec" than "harmless
optional field." Image embedding stays on the Raven-native API surface only. OAI clients
like Hindsight that don't need image embedding never touch it; Raven-aware clients (the
Visualizer, content-parts phase 3 code) use the Raven endpoint directly.

**Hindsight env vars** set against this:
- `HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai`
- `HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL=http://localhost:<raven-port>/v1`
- `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL="default"` (or `"nomic-ai/nomic-embed-text-v1.5"`;
  both work via the lenient model-field handling in §1a)
- `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY="sk-local"` — Raven-server has no auth, but most
  OAI client libraries reject empty/missing API keys before sending the request. Any
  non-empty `sk-`-prefixed dummy value satisfies the client; the server ignores it.

**Known limitation for Hindsight + Nomic asymmetric retrieval**: Hindsight's `openai`
embeddings provider treats embeddings as opaque text passthrough; it doesn't send Raven's
`task` extension field (and doesn't even know to prepend prefixes manually). So Hindsight
gets symmetric embedding by default. Three options when this surfaces as a real
recall-quality issue:

- **v0 acceptance** (recommended): use symmetric embedding from Hindsight's side. Loses
  asymmetric-retrieval's small quality lift, but Hindsight's TEMPR retrieval architecture
  (semantic + BM25 + graph + temporal) is doing most of the work; the prefix benefit is
  marginal in context.
- **Small Hindsight-side wrapper**: inject `task="query"` for recall calls and
  `task="document"` for retain calls. No Raven-side changes needed (the `task` extension is
  ready to receive). Adds a small layer between Hindsight and Raven; reasonable if v0
  empirics show recall quality is unsatisfying.

(A third option — having Hindsight load its own embedder alongside Raven's — was considered
and dropped, since it conflicts with Raven's central design principle of loading each ML
model only once; the VRAM cost on `maia`'s shared 3070 Ti is genuinely incompatible with
duplicating embedders.)

v0 acceptance is the right call until empirical evidence shows recall quality is unsatisfying.

**Multilinguality parked for now**: Librarian is currently English-only, so v1.5's
English-focus is fine. If Finnish (or other non-English) memory recall becomes a real need
later, the cleanest addition is a separate "multilingual" role serving
`nomic-ai/nomic-embed-text-v2-moe`. The ~950 MB cost would consume ~14% of `maia`'s 3070 Ti
budget — tight but evaluatable against what else is loaded; easier to absorb on `electra`'s
16 GB GPU. Hindsight reconfigures by changing one env var; no other code changes. **Watch
for a v2 vision model**: when one ships paired to text-v2, the multilingual path becomes the
clean upgrade across the board, and could justify revisiting the budget allocation.

**Sequencing within the standup**: this Raven-server change happens *before* the Hindsight
container is configured to point at it. If you want maximum failure isolation: stand up
Hindsight first with its built-in `local` embedder (zero external dependencies — Hindsight
downloads BAAI/bge-small-en-v1.5 on first run, verify it works end-to-end), then add the
Raven-server endpoint and switch Hindsight's config to use it. Each step independently
verifiable; a failure at the switch leaves the working "Hindsight + local embedder"
configuration as a fallback.

**Worth knowing about Hindsight's misconfiguration behavior**: if `EMBEDDINGS_OPENAI_*` env
vars are mistyped, Hindsight silently falls back to *OpenAI's default cloud endpoint* with
`text-embedding-3-small`, not to its local embedder. The auth-error against the cloud
endpoint is loud (401), so the failure is visible rather than silent. Note also the
asymmetric env-var naming convention: `HINDSIGHT_API_EMBEDDINGS_{PROVIDER}_*` requires the
provider segment, while LLM vars use `HINDSIGHT_API_LLM_*` without one. Easy to typo;
loudly diagnosable.

### Verify

Follow their Quick Start (`developer/api/quickstart`) — a `retain()` then `recall()` round-trip
against the local HTTP API confirms storage + retrieval, and a `reflect()` call confirms the
LLM wiring (this is where a misconfigured model endpoint will surface).

### Two integration paths

Hindsight ships a **local MCP server** (`sdks/integrations/local-mcp`,
`developer/mcp-server`). That's the seam for both playground paths:

- **Now (quick play):** point LM Studio at Hindsight's local MCP server (backend-hosted MCP)
  and watch the model retain/recall through it. Zero Raven changes — pure experiment to
  verify Hindsight earns its place before investing in the proper integration.
  **Useful empirical baseline**: this configuration gives the model *agentic recall only*
  (no autosearch path — LM Studio isn't running through Librarian's context assembler). You
  get to observe the failure modes autosearch is designed to mitigate (forgotten-to-check,
  latency from lookup turns, the "I need to remember to ask it to remember" friction). When
  autosearch lands later, the contrast against this baseline is informative for whether the
  hybrid actually pays off.
- **Later (product):** the client-side MCP client (phase 4 brief) consumes the same Hindsight
  MCP server for the agentic recall path; Librarian's context assembler also calls Hindsight
  directly for autosearch (this brief, §2). Both paths share the backend; different roles.

### Aria-relevant aside

Hindsight's per-bank **mission / directives / disposition** (disposition as skepticism /
literalism / empathy on a 1–5 scale) is a structured handle for shaping the bank's
*reasoning style* when it's doing reasoning work. **Scope worth being precise about**:
disposition affects `reflect` (Hindsight's agentic-loop reasoning operation), not `recall`
(retrieval) or `retain` (storage). Makes sense in hindsight — disposition is "how this
bank thinks," which only matters when it's actually thinking.

For Librarian's current usage pattern (retain memories during conversation, recall them via
autosearch or agentic MCP) disposition is essentially inert — the bank is being used as a
retrieval store, not as a reasoning agent. Disposition becomes load-bearing only if/when
the model invokes `reflect` via the MCP path (a "think deep through memory" tool call), at
which point the trait settings shape the agentic-loop reasoning.

So for the Qwen-personality thread specifically, disposition is a useful surface to
experiment with **if the experiment involves reflect-style operations** — "ask the bank a
hard question, see how its reasoning style depends on disposition." It's less useful as a
general persona-shaping dial for ordinary chat (where the model is reading memories, not
asking the bank to reason).

**Important constraint**: disposition (and presumably mission / directives) **must be set
at bank creation; changing them requires wiping the bank.** So this isn't a live-experimental
dial you can twiddle mid-conversation to see what changes. It's a per-persona-instance
deliberate commitment — pick the values you mean when creating the bank, and treat changes
as "I'm starting fresh with a different persona configuration."

The unit of experiment is "a fresh bank, run with reflect operations, observe how reasoning
style varies with disposition" rather than "twiddle the slider and compare adjacent turns."
Costlier per data point, cleaner per data point (the persona is internally consistent
across the bank's whole history). Different shape of experimental cost; not strictly
better or worse, just different.

---

## 2. Integration architecture — two-track recall

The Hindsight integration is **not just exposing MCP tools to the model**. That's one of two
necessary paths. The other is **proactive autosearch**, which surfaces relevant memories into
context automatically without the model spending an agent turn on it.

### Why both

Agentic-only recall has real costs even though local-model token costs aren't dollars:

- **Latency cost** — every "should I look something up?" decision is a model turn the user
  waits through. Multiplies if the model makes the wrong call and the user reprompts.
- **Context-fill cost** — tool-call invocation + response wrapping eats token budget even
  when the looked-up information could have been surfaced for free as context.
- **Forgotten-to-check failure mode** — models don't always realize relevant memory exists
  until told to look. Autosearch sidesteps the omission entirely.

The split mirrors human cognition: **autosearch ≈ peripheral awareness** (relevant things
surface automatically as you think about a topic); **agentic recall ≈ deliberate lookup**
("wait, what did Juha say about THA3 perf two months ago"). Both are needed; both are cheap
when each handles its proper role.

### Composition with existing infrastructure

The lorebook brief already established a **candidate-injection interface**: multiple sources
emit candidates `{content, source_tag, priority_hint, dedup_key}` into a context-assembly
stage that merges + dedups + injects. **Hindsight autosearch is one more source feeding into
that same assembler.** No new architecture; same shape as lorebook, glossary, HybridIR. The
assembler doesn't know or care which backend produced any given candidate.

Per-user-turn flow:

```
user message arrives
  ↓
concurrent candidate gathering (all sources query via bgtask.TaskManager):
  - lorebook keyword scan (deterministic, when triggered)
  - glossary (deterministic, on demand)
  - HybridIR sweep on docs DB (ranked retrieval)
  - Hindsight autosearch (ranked retrieval)  ← new
  ↓
candidates → context assembler (tiered: deterministic + ranked + budget)
  ↓
assembled prompt → model
  ↓
[model may emit memory-related MCP tool calls for targeted recall]
  ↓
agentic recall path via MCP client (phase 4 brief) if invoked
```

**Concurrency mechanism**: `raven.common.bgtask.TaskManager` (which wraps
`ThreadPoolExecutor`) — already used in `chat_controller` and `hybridir`. The candidate
sources are I/O-bound (HTTP to embedder + Hindsight, file/DB reads for lorebook + glossary),
so OS threads give real overlap-of-wait even under Python's GIL. Not *parallel* in the
CPU-bound sense, but genuinely *concurrent* for the I/O-bound case that matters here.

**Two architectural decisions for implementation time** (don't pre-decide here, but the
considerations are worth recording so the moment of decision is informed):

1. **Owner: `scaffold` (the agent harness), not `chat_controller`.**

    - **Layering reason.** Context assembly is "produce response from input" — the
      harness's job — not GUI orchestration. Retrieval logic in `chat_controller` would
      couple GUI to inference internals (knowing about retrieval sources, candidate
      assembly, etc.); under `scaffold`'s ownership the GUI just calls
      `scaffold.run_turn(...)` and renders the result.
    - **Headless harness use is on the roadmap** (batch processing of difficult tasks
      beyond what simple `llmclient`-style scripting handles), which sharpens this from
      "cleaner layering" to "actively required for a planned use case." Future surfaces
      (CLI runner, batch eval, headless use) inherit retrieval without re-orchestrating.
    - **Dependency direction constraint.** Anything `scaffold` owns must be usable without
      GUI imports — no DearPyGUI, no `chat_controller`, no `gui_shutdown`. Dependencies
      flow one way (GUI knows scaffold; scaffold doesn't know GUI). Lifecycle hooks like
      MCP shutdown (phase-4 §5) follow the inverse pattern: `scaffold` exposes
      `shutdown()`, the GUI registers it with `gui_shutdown`, scaffold imports nothing
      GUI-side.
    - **Task-manager ownership: scaffold always creates its own task manager(s)** — they
      carve scaffold-internal cancellation boundaries that callers shouldn't model. What
      scaffold can accept is an **executor** (e.g., a `ThreadPoolExecutor` instance) that
      its task managers *bind to* for the underlying thread pool. This separates semantic
      grouping (always owned by scaffold) from resource sharing (composable). Same pattern
      as threads sharing an address space while running independent control flows.
    - **Result delivery via existing CPS-via-callbacks pattern.** `ai_turn`'s caller passes
      continuations (`on_content`, `on_tool_call`, `on_done`, etc.) that scaffold invokes
      when each piece of data arrives. Caller decides what to do — GUI renders to DPG, TUI
      prints to stdout, batch eval closes over local state to assemble a Result object.
      No new "Result type" needed in scaffold; batch users write thin closures that build
      whatever output shape they want. (If the callback parameter list ever grows past
      5-6 entries, the natural evolution is a `TurnCallbacks` dataclass with no-op
      defaults — same CPS shape, just packaged.)
    - **`minichat` is the existing canary** for "scaffold doesn't leak GUI dependencies."
      Minimal TUI chat client (terminal-based, not DearPyGUI) that already consumes
      scaffold without the desktop-app layer; if scaffold accidentally grows an import
      from DPG / `chat_controller` / `gui_shutdown` / etc., `minichat` breaks at import
      time, providing CI-checkable enforcement. **Verify `minichat` continues to start
      and function through phase-4 and phase-5 changes** — cheap to run, immediate fail
      signal. The same canary will protect headless batch-eval use later (identical
      constraint).

2. **Cancellation grouping: most likely share the AI-turn manager, but decide deliberately.**

    - **Why deliberate matters.** Task managers carve cancellation boundaries on purpose;
      `chat_controller` already has separate `task_manager` and `ai_turn_task_manager`
      for exactly this reason.
    - **Default answer**: when the user cancels the AI turn, retrieval should be cancelled
      too. Retrieval gathers context for *this* turn; abandoned turn means wasted
      retrieval. Argues for retrieval sharing the AI-turn manager.
    - **When you'd want a separate manager**: keeping retrieved memories from being
      thrown away when a turn is cancelled mid-stream — e.g., recoverable for the next
      turn. Not a current need; flagging the case in case it surfaces empirically.

### Context assembler tiering

Two tiers, distinct roles:

- **Tier 1 — deterministic / triggered**: lorebook entries (when keywords matched), glossary
  (when requested), system context. **Always in**, counted against budget but not ranked
  against retrieval. These are "the user/system said this is relevant, include it."
- **Tier 2 — ranked retrieval**: HybridIR + Hindsight autosearch results. **Compete for
  remaining budget** after tier 1 is allocated. These are "the system guesses this is
  relevant, include the best of it."

### Ranking strategy across retrieval sources

The cross-source combination question is genuinely empirical (per-source score
calibration is hard). **Starting point: RRF (Reciprocal Rank Fusion) with optional per-source
floors.** RRF is rank-based, needs no calibration across sources; the optional per-source
floor (e.g., "drop Hindsight results below similarity 0.7") filters each source's noise
*before* RRF, so cross-source comparison stays clean.

Roman-engineer ordering: ship the RRF baseline, measure, refine. If pollution from one
source becomes a problem, the per-source floor tunes that source's contribution independently
without re-calibrating others. If self-tuning ever earns its complexity later, user-feedback
signals (was the surfaced memory useful?) could update floors automatically.

### Query shape for Hindsight autosearch

Two approaches, both worth implementing in that order:

**A. Multi-query (start here — the karvalakkimalli):**

Send Hindsight the **N most recent user messages** as separate parallel queries. Hindsight
ranks across all of them and dedups in its native retrieval layer. No summarization step in
Librarian, no model in the loop, no LLM-induced latency.

Hindsight is specifically write-seldom-query-very-often by design, so parallel queries are
its sweet spot. Each individual query is short; embedding N short strings is well under a
second.

Loses narrative coherence ("the conversation is about Y in the context of Z" never gets
composed explicitly) but the retrieval signal is the union of "what relates to message_k for
each k," which is often what you actually want.

**B. Summarize-then-query (add only if A is noisy in practice):**

Compose a short narrative summary of recent turns, send as one rich query. Pros: captures
topic-of-conversation, not just literal message content. Cons: adds a summarization step.

If we go this route, **karvalakkimalli the summarizer too** — don't use the main LLM (KV
cache miss before and after; multi-second latency hit). Options:

- **Extractive from existing primitives** (~30 lines on top of `nlptools.count_frequencies`
  and `embed_sentences`): pick top sentences by token-importance score + embedding
  centrality. No new model dependency, deterministic, fast. Probably enough for "compose a
  query for retrieval."
- **Small abstractive summarizer model** (new dep): better narrative coherence, modest
  latency cost (~100-500ms even for sub-1B models). Add only if extractive isn't enough.

There was a summarizer wired up in the past — possibly in `natlang`, possibly in the
Visualizer importer (abstractive ML model tested earlier and either removed or rejected).
Worth `git log --diff-filter=D -- raven/server/modules/` archaeology when this becomes
relevant; the existing primitives plus that history give a quick path to a working
implementation, or to a deliberate "build thrice and cherrypick" if you want to
re-explore the choice with fresh eyes.

**C. Bonus — spaCy keyword extraction as cheap query enhancement:**

`raven.common.nlptools` already has `count_frequencies` / `extract_word_counts`. Pull noun
phrases from recent turns and add them as additional query terms alongside the user
messages. Cheaper than summarizing, surfaces topic signal somewhat, zero LLM/model in the
loop. Can compose with either A or B.

### Risks and what handles them

- **Latency** — extra query before every model turn. Mitigation: parallel with other
  candidate sources. Hindsight is designed for this. Likely negligible vs. model latency.
- **Token cost** — even short memories add up over conversations. Mitigation: per-source
  similarity threshold; cap on count; recency weighting handled by Hindsight itself.
- **Pollution** — autosearch could inject misleading or irrelevant memories. Mitigations:
  per-source floors (higher than agentic recall, since autosearch is less discriminating);
  user-facing inspection (the GUI affordance below); easy disable per-chat or globally.
- **Recency** — capable agentic memory backends including Hindsight should handle this
  semi-automatically via temporal weighting in their retrieval scoring. Trust the backend,
  verify empirically with a few stale-memory scenarios when Hindsight is standing up.
- **Hindsight API shape**: three core operations, with a clear division of labor that
  composes naturally with the two-track recall architecture.

    - **`retain`** — write path, stores memories.
    - **`recall`** — read path, semantic search via TEMPR (semantic + BM25 + graph +
      temporal). This is what autosearch uses.
    - **`reflect`** — agentic-loop reasoning. Runs an autonomous agent on the same LLM
      instance as Librarian's chat, so every `reflect` call incurs KV-cache miss in both
      directions (chat history → reflect's prompts → back to chat). Unworkable on every
      user turn; the natural home is as an **MCP tool** surfaced via the phase-4 MCP
      client. The model invokes it when it recognizes the need for "think deep through
      memory"; the KV-cache cost is absorbed by the flow break that any MCP tool call
      already implies, so deep reasoning during the break is marginal-free.

    Architecture symmetry that falls out: routine memory retrieval → cheap `recall`
    (autosearch + agentic via MCP); expensive thoughtful queries → `reflect` (agentic via
    MCP only). The model picks the right one.

  - **Knobs on `recall` worth tuning for autosearch**:

    - `budget` — traversal thoroughness; start with `"low"` for autosearch (fast/cheap),
      reserve `"high"` for agentic-path queries.
    - `max_tokens` — response cap; composes with autosearch's budget allocation.
    - `types` — fact-type filtering (`"world"`, etc.).
    - Tag filtering when banks are tagged.
    - `include_chunks` flag for source material.

  - **Single bank for now.** Mental model: one bank ≈ one persona. Librarian's use case
    is single user or team on common stuff, no knowledge-wall requirement; multi-bank
    queries are client-side responsibility (single bank per request anyway). Revisit if a
    domain-isolation need shows up empirically.

  - **Disposition traits only affect `reflect`**, not `recall` or `retain`. Makes sense in
    hindsight (the disposition shapes how the bank *reasons*, which only matters when it's
    actually doing reasoning, i.e., in `reflect`'s loop). The Aria aside in §1 reflects
    this narrower scope.

### Provenance storage — unified context_assembly metadata

Following the `sidecars` pattern from content-parts (record the *why* alongside the *what*):
record what the assembler did this turn as a sibling field on the message's payload, so
debug and inspection are first-class.

**Locked in: `payload.context_assembly`, replacing the legacy `payload.retrieval`.** The
legacy field carries existing RAG hits today; it gets superseded by a broader
`context_assembly` field holding everything the assembler did this turn across all sources.
Auto-migrate existing chattree data via `chatutil.upgrade_datastore` (same hook content-parts
uses for its V4 migration): on load, rewrite each payload's `retrieval` field into the new
`context_assembly` shape with `source: "rag"` per entry, drop the old field. Atomic
conversion (no transitional shim accepting both shapes) — same discipline as the embeddings
config migration: small user base, minimal customization in the wild, cleaner schema worth
the forced rewrite.

The unified schema: **one container for "what the assembler put into context this turn,"
across all sources**, with `source: "hindsight" | "rag" | "lorebook" | "glossary" | ...` as
a categorical field per entry — same shape `sidecars.source` uses for image provenance
pathways. Adding a new source later means writing entries with a new `source` value; no
schema change.

Sketch of the entry shape:

```python
payload["context_assembly"] = {
    "queries": [...],               # multi-query: N items; summarized: [summary]
    "summary": "...",               # optional, only if summary-then-query path used
    "injected": [                   # what made it into context
        {"id": "...", "source": "hindsight", "score": 0.87, "snippet": "..."},
        {"id": "...", "source": "rag", "score": 0.82, "snippet": "..."},
        {"id": "...", "source": "lorebook", "trigger": "...", "snippet": "..."},
        ...
    ],
    "rejected_below_threshold": [...]  # for debug; can be hidden in normal UI view
}
```

Sibling of `general_metadata`, `generation_metadata`, `message` at payload level (NOT nested
inside `general_metadata`) — matches the structure of the existing `retrieval` field it
replaces. Per-user-turn-message. Optional (absent on turns where assembler didn't fire,
e.g., empty messages routed straight to ai_turn).

### Full-prompt reconstructability

System prompts change per Librarian boot (config edits, persona swaps, scaffold updates),
so the actual system prompt for a past turn isn't recoverable from anything currently
stored. Record it as another payload-level sibling:

```python
payload["system_prompt"] = "<full text active at generation time>"
```

The other components of the full assembled prompt (message history, persona prefixes,
reasoning_content wrapping per family per compat §10) are reconstructable from existing
chattree data. So: storing **`payload.system_prompt` + `payload.context_assembly`** is
sufficient to fully reconstruct any past turn's assembled wire-format prompt.

Post-migration, payload-level fields are: `message`, `general_metadata`,
`generation_metadata` (optional, AI turns), `context_assembly` (optional), `system_prompt`
(optional). Five categories, flat at payload level — each a coherent provenance category,
no nesting that's not load-bearing.

Storage cost for system_prompt: typical system prompt is 200-2000 tokens, stored per-turn
for simplicity. Could be deduplicated across consecutive turns from the same boot session
(store once per "session block," reference by ID) if it ever bloats — but that's complexity
for marginal savings; per-turn is fine until it isn't.

### GUI inspection affordance

A **"Show full prompt for this turn"** button (likely in the same Tools surface that
content-parts §6 already implies — open docs DB dir, open chat datastore dir, clean & save,
and now this) reconstructs and displays the actual wire-format prompt that went to the model
at generation time.

Reconstruction uses `invoke`'s existing prompt-assembly code in "inspect mode" rather than
"send mode" — single source of truth for "what gets sent" since reconstruction shares the
function with actual sending. The function takes the stored `system_prompt` + the stored
`context_assembly` + the message history up to the turn, applies wire-format wrapping, and
returns the assembled string instead of POSTing it. Renders into a scrollable view in the
GUI.

Composes with the **per-message "why did this memory surface?" affordance** — same data,
different presentation. Click on a turn's tool result or surfaced memory → highlight which
entry in `injected[]` it came from, show its `source`, `score`, and trigger info.

For a research codebase specifically, this provenance is valuable beyond debug: "what
context did the model see when generating this reply" is part of reproducing past chats and
understanding what conclusions rested on what evidence. Worth the modest storage cost.

### Backend portability — don't over-commit to Hindsight specifically

Memory-backend ecosystem is genuinely active (Hindsight, mem0, Letta/MemGPT, Cognee, Zep,
others surfacing every few months); the right backend at year +1 may not be the one at
year 0. The integration architecture above is mostly portable already, but the discipline
needs naming so it stays portable as v0 lands.

**What's already naturally portable:**

- **The agentic path (MCP)** is backend-agnostic by construction — MCP is a *protocol*, not
  a Hindsight thing. Any memory backend that exposes MCP tools plugs into the phase-4 MCP
  client unchanged. Switching backends just means switching which MCP server endpoint is
  configured.
- **The context assembler** doesn't know or care which backend produced a candidate — it
  dedups + ranks + injects based on the candidate's tagged `source`, treating all backends
  uniformly. The `source: "hindsight"` value is just a string label.
- **The provenance schema** (`context_assembly` storage) carries `source` as a categorical
  field, designed to enumerate cleanly: tomorrow's `source: "mem0"` or `source: "letta"`
  costs zero migration.

**Where coupling can creep in — the autosearch path:**

The autosearch query layer is the one place Librarian calls memory-backend APIs directly
(rather than via MCP). Worth defining a small **`MemoryBackend` protocol** in
`raven.librarian` or `raven.common` that current code targets, with Hindsight as the first
implementation. Sketch:

```python
class MemoryBackend(Protocol):
    def query(self, queries: list[str], *,
              top_k: int, floor: float) -> list[Candidate]:
        """Return ranked candidates; backend handles its own scoring/recency."""
    def store(self, content: str, metadata: dict) -> str:
        """Persist a memory; return backend-specific id."""
    # (any other operations the assembler genuinely needs;
    #  keep this surface small — most memory operations should go through MCP, not here)
```

Hindsight-specific implementation lives in `raven.librarian.memory_backends.hindsight`;
config picks the backend by name; the autosearch layer reaches for `MemoryBackend` not
`HindsightClient`. Cost: a thin adapter (one file, ~50 lines, mostly translating Hindsight's
TEMPR retrieval surface to the protocol's `query()` shape). Payoff: switching to mem0 / Letta
/ Cognee later is *implement-the-protocol*, not *rewrite-the-integration*.

**What NOT to abstract:**

- *Hindsight's mission/directives/disposition knobs* are Hindsight-specific persona-shaping;
  abstracting these would erase what makes Hindsight interesting for the Aria experiments.
  Use them directly when Hindsight is the configured backend; ignore (or no-op) when not.
- *Hindsight-specific config fields* (URL, auth, bank IDs) stay in Hindsight's config block;
  the protocol abstracts call shapes, not configuration.

**The discipline in one sentence**: phase-5 work is "stand up Hindsight" + "implement the
MemoryBackend protocol with Hindsight as first implementation," not "wire Hindsight directly
into the assembler."

---

## 3. Snowball control

Suggested order, each step independently useful and falling back cleanly on failure:

1. **Hindsight via Docker Compose + local models** — `local` embedder (Hindsight downloads
   `BAAI/bge-small-en-v1.5` on first run), LM Studio for the LLM. Verify retain/recall
   round-trip over HTTP. Done = working Hindsight container, end-to-end memory operations,
   nothing leaves the machine. This is the prerequisite for everything that follows.

2. **Hindsight ↔ LM Studio over Hindsight's local MCP server** — quick backend-hosted-MCP
   path. The model talks directly to Hindsight as an MCP tool; zero Raven changes. **Test
   first, commit later**: this gives agentic-only recall (no autosearch path — LM Studio
   isn't running through Librarian's context assembler), which is the empirical baseline
   for what autosearch should improve over. The experience here decides whether Hindsight
   earns the proper integration work.

3. **Decision gate**: does Hindsight earn its place? If yes, proceed to step 4. If no,
   stop here — nothing committed beyond a Docker container that can be torn down. The
   architectural reasoning (autosearch + agentic two-track, scaffold ownership, provenance
   schema) is preserved as design context for a future memory-backend even if Hindsight
   specifically isn't the one.

4. **Migrate Raven-server embeddings to nomic v1.5 pair AND add OAI-compatible
   `/v1/embeddings` endpoint** (§1a). Independent of the rest of the integration except
   that integration is what triggers it. The nomic migration is independently valuable
   (Visualizer multimodal, ~410 MB VRAM savings, content-parts phase 3 readiness), but
   the standup brief's scope means it ships when the Hindsight integration is going
   forward. Verify via `curl` against `/v1/embeddings` and via existing Visualizer / RAG
   codepaths still working post-migration. Done = Raven-server speaks OAI embeddings on
   the nomic v1.5 pair.

5. **Reconfigure Hindsight to use Raven-server's endpoint** — switch
   `EMBEDDINGS_PROVIDER=local` → `EMBEDDINGS_PROVIDER=openai`, point `BASE_URL` at
   Raven-server, set `EMBEDDINGS_OPENAI_MODEL="default"`, set the dummy API key per §1a.
   Restart Hindsight. Verify retain/recall still works against the new embedder. Done =
   single canonical embedder serving Raven + Hindsight, no duplicate VRAM, no duplicate
   model maintenance.

After step 5, the phase-4 MCP client (already implemented by this point) becomes
reachable against Hindsight's MCP endpoint for the proper agentic path; this brief's §2
(autosearch path) is the remaining surface to implement.

Steps 1-3 don't touch Raven internals at all. Steps 4-5 are conditional on the step-3
decision. Failure isolation: any step can fail without disturbing the previous step's
working configuration, and the decision gate at step 3 prevents committing to migration
work before validating the integration's value.

Steps 1-4 don't touch Raven internals beyond the small `/v1/embeddings` endpoint addition,
so none of this blocks any other brief. Failure isolation: any step can fail without
disturbing the previous step's working configuration.
