# Brief: content-parts message representation (refactor)

**Why this exists:** three forces hit Librarian's message representation at once.

1. **MCP tool results** return *content blocks* (text, image, embedded resource …), not strings.
2. **VLM input** is now table-stakes for modern open-weight models (ooba already exposes image
   input over its OpenAI-compat API; LM Studio does the same for VLM models). Users will want to
   paste / attach images.
3. **Future structured tool outputs** (PDFs returning text + page-anchored links, citations with
   span pointers, etc.) won't fit a single string either.

All three are the same problem: a message whose content isn't a single string but a list of typed
parts. Current `perform_tool_calls` accepts text only; current message construction assumes string
`content`. This brief replaces that with the OpenAI multimodal content-parts model as the
*internal* representation, getting wire-format alignment for free.

**Sequencing:** depends on the compat brief (compat introduces the typed-event stream from
`invoke` and the sibling-field data model — `reasoning_content`, `tool_calls`, `tool_call_id`
— that this brief builds on). MCP client depends on this (its tool results need somewhere
typed to land); VLM input is the immediate second consumer here. So the chain is
compat → content-parts → MCP, and content-parts also unblocks image input on the same landing.

---


## 1. The core decision

**Adopt OpenAI's multimodal `content` schema as the internal representation.** Don't invent a
Raven-internal model and translate at the wire — just use OpenAI's shape directly. Three things
fall out of this:

- The wire format already matches; no serialization layer.
- The OpenAI spec is the documentation.
- MCP content blocks map onto it with near-trivial conversion (the formats are deliberately
  similar).

A message's `content` becomes a **list of typed parts**, never a bare string. Even text-only
messages are `content = [{"type": "text", "text": "..."}]`. One representation everywhere; no
"sometimes-string, sometimes-list" branching at every call site.

**Data model: what's in `content`, what's sibling.** The sibling-field part of the message data
model — `reasoning_content` (string), `tool_calls` (list), `tool_call_id` (on tool messages) —
**is specified by the compat brief**, §10 (persistent message format). It's there rather than
here because the compat brief introduces the typed-event stream from `invoke` (`{content,
reasoning, tool_call}`), and the storage locations are the persistent counterpart of those
events. Same data, two representations; one brief, one place.

This brief adds *one piece* to that data model: **the `content` field becomes a list of typed
parts** rather than a string. Everything else on the message stays where compat puts it.

| Field | Where specified | What this brief changes |
|---|---|---|
| `role` | OAI standard | unchanged |
| `content` | OAI standard, here | **string → list of typed parts (this brief)** |
| `reasoning_content` | compat §10 | unchanged |
| `tool_calls` | compat §10 | unchanged |
| `tool_call_id` | compat §10 | unchanged |

**Why `content` becomes a parts list but `reasoning_content` stays a string.** The asymmetry is
intentional, not an oversight:

- **Wire-format convention.** OAI / llama.cpp / LM Studio all emit `reasoning_content` as a
  string. The compat brief leans hard on "the OAI spec is the documentation"; diverging from it
  needs an affirmative reason.
- **`content` and `reasoning_content` aren't symmetric concepts.** `content` is "everything the
  model emits as its response output" — a heterogeneous channel by design (it's where
  multimodality lands: text + image + future types). `reasoning_content` is *one specific kind*
  of output: textual thinking. Treating them with one shape would unify two things that are
  unified only nominally.
- **Zero conversion cost.** Backend emits a string, we wrap with tags as a string on resend
  (compat §10), we persist a string. A parts list would require wrap/unwrap on both sides for
  no information gain.
- **Reversible if reality changes.** If a future model emits genuinely structured artifacts in
  its thinking channel (or if the wider world heads toward neuralese / continuous-vector
  reasoning à la Coconut, at which point the wire format will be having a much bigger
  conversation than this), the migration is exactly the mechanical one we just did for
  `content`: wrap legacy strings as `[{"type": "text", "text": ...}]`, idempotent. Not painted
  into a corner.

**Rendering reasoning paragraphs** — the chat_controller injects one paragraph at the start of
each rendered message containing all of `reasoning_content`, labeled as a reasoning paragraph
(rendered as the collapsible thought bubble via the existing mechanism). Then content-parts
paragraphs. Then tool_calls as gear-icon sub-elements (compat brief §9). In practice models
emit at most one thought block per message and it's always at the start, so a single injected
reasoning paragraph matches reality; if a future model ever emits multiple distinct thought
blocks per turn (unusual, conceivable for some agentic patterns), the rendering machinery
extends to multiple reasoning paragraphs naturally without code change.

(Why per-message string for reasoning rather than per-paragraph: in practice models emit one
`<think>...</think>` block per turn, followed by content. No real need for finer granularity at
persistence level. Paragraph splitting is a render concern.)

---

## 2. Part types in v0

```python
# Text part (the everyday case)
{"type": "text", "text": "Hello world"}

# Image part (VLM input or image-bearing tool result)
{"type": "image_url",
 "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}}
# (or a fetchable HTTP URL in place of the data URL)
```

Out of scope for v0: audio parts (model support is uneven across open-weight models), MCP
"embedded resource" blocks (downgrade to a text part inlining the resource for now), and image
*output* from the model (image-generating models aren't in Librarian's target set yet).

---

## 3. Migration / backward compatibility

The change is shaped to minimize call-site churn:

- **`chatutil.create_chat_message`** is the choke point — update its helpers to emit
  content-parts internally. Existing callers passing strings get wrapped as
  `[{"type": "text", "text": <string>}]` inside the helper. Most call sites are then unchanged.
- **`llmclient.invoke`** serializes content-parts to the wire (which is already OpenAI's format
  — pass-through), and parses incoming responses back into content-parts. The role-priming and
  null-content handling already specified in the compat brief (§1) interacts here: assistant
  message content during streaming is text-part accumulation.
- **`perform_tool_calls`** is updated to return `list[part]` instead of a string. Existing
  text-only tools (`websearch`, `webfetch`) wrap their string output as a single text part —
  again, change at the tool definition, not at the loop.
- **`chattree`** persists messages with content as the parts list. Migration extends the
  existing `chatutil.upgrade_datastore` hook (compat brief §11 covers the broader migration
  framework and the reasoning / tool_call_id stanzas; this brief adds the content-parts wrap
  stanza on top). See §3b for the content-parts portion.

---

## 3b. Migrating existing chat history (content-parts portion)

The content-parts refactor adds **one** step to `chatutil.upgrade_datastore`'s migration chain:
**wrap legacy string `content` as a single-text-part list.** Idempotent.

```python
# If message["content"] is a string, wrap it:
message["content"] = [{"type": "text", "text": message["content"]}]
# If already a list, leave alone.
```

**Preserve the persona prefix.** Old assistant content begins with `"Aria: "` (the
`add_persona=True` default in `create_chat_message`). The prefix is part of what the model sees
on the wire and must survive migration — include the whole string (prefix included) in the
wrapped text part. No special handling required.

**This step runs *after* compat brief §11's migration steps** (`<think>` extraction,
`tool_call_id` move, codebase-wide rename, stale-docstring cleanups), since compat's steps
operate on the string form of `content` and only this step changes the shape. Run order is
enforced by the upgrade-stanza version sequence — compat's stanza ships first, this brief's
stanza ships after.

**Robustness:** the wrap is unconditional and trivial; no failure modes worth handling. If
content is something other than a string or a list (corruption), the migrator raises a
descriptive error rather than silently swallowing data.

---

## 4. Tool result handling

**One concept: tools return content parts.** Every tool — websearch today, MCP tools in
phase 4, future structured tools — emits a list of content parts that becomes the
`message["content"]` of the resulting tool-role message. No Raven-internal siblings, no
custom part types, no wire-strip logic. The wire form, the storage form, and the GUI's
rendering input are all the same list.

This nearly became a Raven-internal sibling pattern (a `tool_result_data` field for
GUI-only structured rendering), but unifying on parts is genuinely simpler: MCP already
does exactly this in its native protocol (phase 4 plugs in trivially), markdown-formatted
text parts give the GUI everything it needs to render structured output cleanly, and the
model reads markdown perfectly well. The only thing lost is custom-card-layout flexibility
in the GUI — and the standard markdown-rendered-text-per-part output is more than enough
for v0.

**v0 work — wire up websearch's structured mode:**

- `raven.server.modules.websearch.format_results` already returns `(preformatted_text,
  structured_results)`; Librarian currently consumes only the string half. Switch to
  consuming the structured half.
- For each search result, emit **one text part** in markdown:
  `"[Title](url)\n\nsnippet text\n"` (or similar — minor formatting decisions locked in
  during implementation). `perform_tool_calls` collects parts into the tool-role message's
  `content` list.
- **Normalize every text-bearing field** of each result (`text`, `title`, `link`) via
  `raven.common.text.normalize` — scraped HTML from Google is external untrusted content,
  same hostile-input class the webfetch brief established the utility for.

**Text-only tools that don't produce structure** (`webfetch` v0): emit a single text part
with the extracted content (normalize-applied per webfetch brief). No special-case path;
same code as websearch, just N=1.

**GUI rendering** falls out of the existing markdown renderer (already vendored and extended
with multithread robustness). Markdown link syntax gives clickable URLs; per-result part
boundaries provide visual separation between results without custom card layout. If a future
need calls for fancier visual treatment, we can either re-parse the markdown (format is
consistent enough) or extend the vendored renderer — both real options, neither needed for
v0.

**Forward-compat for MCP (phase 4) and beyond:**

- **MCP text content** → text part. Apply `raven.common.text.normalize` — same hostile-input
  rationale.
- **MCP image content** → image part stored as a sidecar file (see §8). MCP delivers images as
  either base64 data or fetchable URL; either way Librarian materializes a sidecar at receive
  time. **External URLs are never used as live image references in stored datastores** — they
  may be recorded as provenance in the sidecars entry's `url` field (see §8), but the image part itself always
  points to a `sidecar:` URL; nothing fetches the remote URL at load/render time. Record an
  `sidecars` entry with the source URL and `source: "mcp:<server_name>"` (the MCP server's
  display name from its config entry) — see §8.
- **MCP embedded resource** → text part with description + text payload; defer rich handling.
  Text payload normalized.
- **Future structured tools** (PDF extraction with page-anchored citations, code execution
  with separate stdout/stderr, etc.) → emit multiple parts naturally. Markdown handles
  whatever structure the parts need (headings, code fences, tables, links).

The phase-4 architectural payoff is real: MCP's native protocol returns parts, Raven's
tool-result pipeline expects parts, **zero translation layer**. Same goes for any future
structured tool.

---

## 5. User image input

The other half of multimodal: user can attach an image to a message.

- **GUI**: "attach image" affordance on the message composer. On selection, **copy the
  image to the datastore's sidecar directory** (see §8) and attach to the message as an
  `image_url` part with a `sidecar:<filename>` URL value (Raven-internal URL scheme; see
  §8 for the substitution rule). Record provenance in `general_metadata["sidecars"]` (see §8):
  for user-attached local files, the sidecars entry's `url` is `file:///<absolute_path>` and
  `source` is `"user_attachment"`; `fetched_at` and `content_type` are recorded automatically;
  `original_dimensions` and `original_size_bytes` are recorded if downsampling was applied.
  Multiple attachments → multiple image parts plus multiple `sidecars` entries.
- **External URLs are never used as live image references in stored datastores.** If a user
  pastes an `https://...` image URL (a future UX affordance), download immediately, store as
  a sidecar, persist the image part as a `sidecar:` URL with the source URL recorded in the
  `sidecars` entry (as provenance metadata) and `source: "paste_url"`. The image part itself
  never carries an `https://` URL; nothing in the load/render path ever fetches a remote URL.
  Rationale: (a) **reloadable without network access** (offline-resilient chats), (b)
  **robust against link rot** (the original page going away doesn't break old chats), (c)
  **privacy** (opening an old chat doesn't ping someone's server somewhere), (d) **reduced
  cyberattack surface** (no opportunity for a tracking pixel or malicious-redirect chain
  disguised as an image reference).
- **Wire serialization** (in `llmclient.invoke`, just before sending): for each `sidecar:`
  URL in the outgoing message, resolve by reading the sidecar file, base64-encoding, and
  substituting a `data:image/<format>;base64,...` URL value in the wire copy. The persisted
  message remains sidecar-referenced. The wire substitution is a small function (~20 lines)
  that lives alongside `invoke`'s other serialization logic.
- **Format constraints**: accept common formats (PNG, JPEG, WebP). For very large images,
  downsample on store via `raven.common.image.lanczos` (GPU-accelerated Lanczos — already on
  tap, very fast, very clean). Default cap: **1 megapixel**, aspect-ratio preserved. Solve
  `H × W = 2²⁰` with `r = W / H`: `new_H = sqrt(2²⁰ / r)`, `new_W = sqrt(2²⁰ × r)`. So
  4000×3000 (12 MP) → ≈ 1183×887 (1.05 MP); smaller images pass through unchanged.

  Rationale for 1 MP: right at the resolution most current VLMs natively expect (LLaVA-NeXT
  ≈ 4 tiles, Qwen-VL ≈ 1340 patches, Gemma 4 standard input 1024²). Below it, the model would
  resize down anyway; above it, same thing plus wasted token cost. PNG/JPEG at 1 MP is 150 KB
  – 1 MB on disk — light. Tunable via `image_store_max_megapixels` in config; default `1.0`,
  set to `None` to disable.
- **Original-quality preservation.** By default, when downsampling is applied, **the original
  is also kept as a sidecar** at `<hash>.original.<ext>` next to the downsampled primary. This
  makes the datastore truly self-contained (move it to another machine, originals come along)
  and preserves full-resolution data for future use (re-downsample to a different target,
  re-export at full quality, send to a future higher-resolution VLM).

  **The original is stored byte-for-byte** (the source file's exact bytes, not a decode/re-encode
  round-trip), so embedded metadata survives — EXIF, ICC profiles, and especially AI-generation
  parameters (ComfyUI / A1111 workflow data baked into PNG text chunks). The downsampled *primary*
  is a derived working copy (re-encoded, metadata naturally gone; the original format is recorded
  in `content_type` regardless). Implemented in `raven.librarian.imagestore.store_image_as_sidecar`.
  Three cases drop out naturally:
  1. **No downsampling needed** (image ≤ 1 MP at attach): primary sidecar IS the original —
     stored verbatim (so metadata is preserved even without a second file). No second file.
     Sidecars entry's `original_dimensions` absent.
  2. **Downsampled, original kept** (default): primary is downsampled, original at
     `<hash>.original.<ext>`. Sidecars entry has `original_dimensions` and `original_sidecar`.
  3. **Downsampled, original discarded** (`store_original_image=False` in config, for
     disk-constrained users): primary is downsampled, no original sidecar. Sidecars entry has
     `original_dimensions` but not `original_sidecar`.
- **Drag-and-drop / paste**: not v0 — DPG's cross-platform DnD support is limited (Windows-only
  for window-target drops; Linux/macOS upstream gaps stemming from GLFW backend coverage and
  X11/Wayland drag-protocol fragmentation). File-picker dialog is the v0 affordance. A
  filesystem-watched "drop directory" pattern (using the `open_in_file_manager` utility from
  §6 to expose its location) is a possible interim workaround for the broader Visualizer-side
  use case, not just chat-image attach — separate workstream, not in scope here.

---

## 6. GUI rendering

The chat-history widget needs to dispatch on part type for the content list AND render sibling
fields (per the data-model split above):

- **Content parts:**
  - Text part → text bubble rendered via the vendored markdown renderer (current behaviour for
    text-only messages).
  - Image part → inline thumbnail rendered via `raven.common.image.lanczos` for crisp downscale;
    click to view full-size (existing pattern from Cherrypick). Read the sidecar file on first
    render, cache the decoded image data per message so scroll doesn't re-decode on every
    redraw.
  - Multi-part messages render parts in order, vertically stacked.

  **Image provenance buttons** (context menu on the thumbnail, or small action icons on hover):
  - **Open original file** — dispatches across the three storage cases (§5): case 1 (no
    downsample) → open the primary sidecar; case 2 (downsampled, original kept) → open the
    `original_sidecar` file; case 3 (downsampled, original discarded) → try the `url` from
    the sidecars entry (open `file:///` paths in the system image viewer; open `https://` URLs in the
    browser).

    **Predictable failures → disable the button with a hover tooltip explaining why**, so
    the user sees the constraint before clicking: no source URL recorded, or source URL is a
    `data:` URL (no "source to open"). Discoverable in advance, no surprise.

    **Runtime failures → non-intrusive feedback, never a modal dialog.** If `file:///` opening
    fails (path missing because the user moved/deleted the original off-disk), surface a
    transient toast / status-bar message like "original not found at /path/to/cat.png" that
    auto-dismisses after a few seconds. `https://` failures self-handle via the browser
    (standard 404 / DNS-error page; not our problem to intercept). Modal "an error occurred"
    dialogs are exactly the wrong shape — non-fatal, ignorable, but blocks the UI; users
    learn to dismiss without reading. Toast / snackbar pattern (existing DPG affordances or
    a small helper) keeps the UI flowing.
  - **Open containing directory** — for `file:///` origins, open the parent directory in the
    user's file manager. For remote origins, open the parent URL path in the browser (e.g.,
    `https://example.com/foo.png` → `https://example.com/`). Disable / hide if the source URL
    has no meaningful parent. Same disable-with-tooltip + toast-on-runtime-failure pattern.

  Cross-platform dispatch via a small new utility in `raven.common.utils` (~10 lines: dispatch
  on `sys.platform` → `xdg-open` (Linux) / `start` (Windows) / `open` (macOS)). Genuinely
  shared infrastructure — once the utility exists, several small discoverability wins fall
  out for free, each ~5 lines:

  - **"Open docs DB directory"** — reads the RAG document directory path from
    `raven.librarian.config` and opens it. Currently buried in config; user has to read
    source code to find out where to drop documents. Should have been a button from the start.
  - **"Open chat datastore directory"** — opens the directory containing the active chat's
    JSON + sidecar directory. Useful for backup, share, inspection.
  - (More will accrete naturally; the pattern is set.)

  These aren't strictly content-parts work, but they ride along on infrastructure this brief
  builds anyway. Note: Librarian doesn't yet have a settings/preferences surface, so these
  buttons (plus the prune-and-save button in §8) need a home that doesn't currently exist —
  three buttons is probably enough to imply a small Tools / Utilities surface. Concrete
  placement is for CC and Juha to settle during implementation; brief notes the need.
- **Sibling fields (compat brief specifies this rendering, repeated here for completeness):**
  - Paragraphs with `kind: "reasoning"` → collapsible thought bubble (existing chat_controller
    mechanism, just renamed from `is_thought`-boolean to `kind`-string predicate).
  - `tool_calls` → styled sub-elements within the assistant message body, one per call, with a
    distinct icon (`gear` / `cog` per compat brief §9). Renders function name + arguments,
    visually parallel to the thought-bubble toggle.

**Bidirectional tool-call ↔ tool-response navigation links.** Now that `tool_call_id` is on
the tool-role message (per migration step 4) and assistant `tool_calls[i].id` is the canonical
identifier, the chat_controller can index both directions on render: build
`tool_call_id → (assistant_node_id, tool_call_index)` and
`tool_call_id → tool_response_node_id` maps over the current chat-tree path (HEAD lineage —
*not* the whole forest, because branched alternates may have different responses).

- On each **tool-response message bubble**: add a small icon button that scrolls the view to
  the originating assistant message and visually highlights its specific tool-call sub-element
  (the one with the matching id).
- On each **tool-call sub-element** within an assistant message: add a small icon button that
  scrolls the view to that call's response message and highlights it. If no response exists
  yet (call in flight, partial chat tree, or genuinely orphaned), disable the button with a
  tooltip explaining ("no response recorded for this call").

Icon suggestions from FontAwesome free: `arrow-up-from-bracket` (response → call, "trace
upstream"), `arrow-down-to-bracket` (call → response, "trace downstream"); or `link` with
contextual rotation; CC picks. The visual weight should be smaller than the thinking-toggle
and gear icons — navigation is supplementary, not primary action.

The point: inspecting tool output today requires the user to scroll up and down looking for
which call produced which response. With many tool calls per turn or many turns deep in a
chat, that's friction. One click each way collapses it. Same minimal-magic principle that
gave tool calls their visible rendering in the first place.

This is mostly mechanical once the data structure is in place.

---

## 7. Token budgeting for images

Images consume non-trivial tokens, varying by model and resolution. **Char→token ratio
calibrated on text-only chats will be wildly wrong the moment an image enters context**, so a
flat extension of the text calibration isn't enough. The structure:

**Per-model per-image cost in config** (new `llm_image_token_cost` table in
`raven.librarian.config`, keyed by model family, with conservative default ~1000 tokens/image
for unknown families). Each entry is either a flat token count or a formula taking
`(height, width)` pixel dimensions and returning a token count. Concrete formulas for known
families as of 05/2026:

| Family | Image cost formula |
|---|---|
| Gemma 4 | Discrete budget per image: 70 / 140 / 280 / 560 / 1120 tokens (**user-selectable at inference time**; assume the conservative max — 1120 — unless the server's setting is known via config or probe). |
| LLaVA-1.5 | Fixed 576 tokens per image (single 336×336 tile). |
| LLaVA-NeXT | 1–5 tiles by aspect ratio × 576 tokens = 576–2880 tokens. |
| Qwen-VL (Qwen2-VL, Qwen2.5-VL, etc.) | Dynamic: roughly `ceil(H/28) * ceil(W/28)` tokens, capped per model. 1024×1024 ≈ 1340 tokens. |
| Unknown family | Flat 1000 tokens (conservative placeholder). |

(Gemma 4's user-selectability *is* awkward — the visual token budget is a server-side knob, so
without probing the client genuinely can't know which value's in effect. Assume the max (1120)
by default, or expose a `gemma4_visual_token_budget` config item the user sets to match their
server. The self-correction layer below catches drift either way; the worst case under
conservative-max assumption is over-budgeting, which is the safe failure mode.)

**Self-correction via usage.** After every image-bearing call, `usage.prompt_tokens` reveals
the true cost. Subtract estimated text tokens, divide by image count → ground-truth image cost,
fed back to refine the per-model estimate. Same mechanism as the char→token ratio calibration
in compat §7.

**Prefill-on-HEAD-switch resolves exactly.** After the 5s idle prefill, `usage.prompt_tokens`
reflects the *whole* prompt including all image tokens. The context-fill indicator flips from
`~X%` (estimate) to `X%` (exact) at that point. So a draft message with an attached image,
after 5s idle, gets an exact pre-send count and a warmed cache simultaneously — multimodal
preserves the same elegance.

**`mmproj` GGUFs are not worth using for budgeting.** ~1 GB to count tokens accurately, plus
running the vision encoder forward pass for each image (CPU/GPU time per call), plus encoder
code paths differ across families. Tier-2 (formula or calibrated) plus tier-3 (prefill exact)
covers this without the dependency or runtime cost.

Don't block the refactor on getting per-image estimates exact; the layer self-corrects from
server feedback, and the `~` vs no-`~` typography in the context-fill indicator handles user
expectation honestly.

---

## 8. Persistence

**The chattree payload pattern is already established and this refactor follows it strictly.**
Verified from `raven/librarian/chatutil.py`:

```python
# Existing payload structure:
payload = {
    "message": {"role": ..., "content": ..., "tool_calls": ...},  # OAI core, wire-serializable
    "general_metadata": {timestamp, persona},                     # Raven sibling
    "generation_metadata": {...},                                 # Raven sibling (assistant/tool)
    "retrieval": {...},                                           # Raven sibling (RAG)
}
```

**The content-parts refactor extends the OAI core only — `content` becomes a list of typed
parts — leaving the payload-sibling metadata pattern unchanged.** Wire format remains
pass-through; Raven-specific extensions land in `payload[*]` sibling fields, never mixed into
`payload["message"]`. Future extensions are sibling fields, no migration risk for the OAI core.

### Image storage: sidecar files, not inline base64

**v0 stores images as sidecar files** in a directory next to the datastore JSON, referenced
from messages via a `sidecar:` URL scheme.

**Rationale** (rejecting inline base64):

- Inline base64 bloats the datastore JSON file by ~33% of the image's binary size, per image.
  A chat with a handful of images quickly becomes a multi-MB JSON, and many text editors
  (Emacs especially) bog down or refuse to edit such files.
- Inline base64 is unreadable noise during manual datastore inspection. Frontier LLMs almost
  certainly lack a base64-PNG-decoder circuit in their heads (decoding properly via the vision
  pathway requires the actual decoded pixel array, not the encoded text); the raw base64 is
  pure context-window pollution if any AI tooling ever needs to introspect the datastore.
- Inline base64 makes diffs across datastore versions useless (any image change → massive
  unreviewable blob diff).

**Directory layout**: for a datastore JSON at `path/to/chat.json`, sidecar files live in
`path/to/chat.images/` (or analogous; exact naming TBD by implementation). Files named by
content hash (`<sha256>.<ext>`) for natural deduplication — the same image attached twice
costs one file. Other naming schemes (UUID, timestamp) are fine but lose the dedup property.

**URL scheme**: in stored messages, image parts use `{"type": "image_url", "image_url":
{"url": "sidecar:<filename>"}}`. The `sidecar:` scheme is Raven-internal and signals
"resolve against the datastore's sidecar directory." On wire-send, `invoke` substitutes a
real `data:image/<format>;base64,...` URL by reading the sidecar file and base64-encoding
its bytes.

**External URLs are never used as live image references in stored datastores** — see §5
rationale; same applies whether the image came from a user paste or an MCP tool result.
URLs may appear inside `sidecars[<filename>].url` as provenance metadata, but image parts always point to
`sidecar:` URLs; nothing in the load/render path resolves a remote URL.

**Sidecar metadata via `general_metadata["sidecars"]`**: when a sidecar is created, record where it
came from in the message's `general_metadata`. Format — a dict keyed by sidecar filename,
values are strings:

```python
"general_metadata": {
    "timestamp": ...,
    "datetime":  ...,
    "persona":   ...,
    "sidecars": {
        "abc123.png": {
            "url": "https://example.com/foo.png",
            "fetched_at": "2026-05-29 14:32:11",
            "content_type": "image/png",
            "source": "mcp:filesystem",                  # categorical pathway
            "original_dimensions": [3000, 4000],         # only if downsampled
            "original_size_bytes": 12345678,             # only if downsampled
            "original_sidecar": "abc123.original.png",   # only if original kept (§5 case 2)
        },
        "def456.png": {
            "url": "file:///home/juha/Pictures/cat.png",
            "fetched_at": "2026-05-29 14:35:22",
            "content_type": "image/jpeg",
            "source": "user_attachment",
        },
    },
}
```

**Field semantics:**

- **`url`** (required) — where the image came from:
  - `https://...` / `http://...` — remote URL (MCP, paste, future fetch features)
  - `file:///<absolute_path>` — RFC 8089 standard file URL for user-attached local files
    (cross-platform; Windows is `file:///C:/Users/.../cat.png`, handled by stdlib).
- **`fetched_at`** (required) — when this sidecar was materialized into the datastore. Same
  `"YYYY-MM-DD HH:MM:SS"` format as `general_metadata["datetime"]` for consistency.
- **`content_type`** (required) — original MIME type. Useful at sidecar-read time to choose
  the wire-side `data:` URL's MIME prefix correctly (downsampling may re-encode to a
  different format; this records the original).
- **`source`** (required) — categorical pathway descriptor. v0 values: `"user_attachment"`,
  `"paste_url"` (when the paste UX lands); phase 4 adds `"mcp:<server_name>"` (the MCP
  server's display name from its config entry). Complements `url`: `url` says *where from*,
  `source` says *via which Librarian pathway*. The two together cover provenance questions
  cleanly.
- **`original_dimensions`** (optional) — `[H, W]` of the image *before* downsampling. Present
  only when downsampled. Useful for inspecting the downsample pipeline and storage-savings
  analysis.
- **`original_size_bytes`** (optional) — byte size of the original (pre-downsample) image
  as fetched. Same conditional presence as `original_dimensions`.
- **`original_sidecar`** (optional) — filename within the sidecar directory of the preserved
  original (full-resolution) copy. Present only in §5 case 2 (downsampled, original kept —
  the default). Absent in cases 1 and 3 (no downsample → primary IS the original; or
  downsampled with original discarded via `store_original_image=False`). GUI's "Open
  original" button dispatches on this field's presence.

**Privacy note**: full `file:///` paths can carry sensitive context (`/home/<username>/...`,
project directory names). If the user later shares the datastore, they may want to scrub.
Acceptable for v0 — provenance value exceeds the share-time risk in the typical research-
notebook use case Librarian is built for. If a privacy-strict mode becomes warranted later,
the natural extension is a config knob to record only the basename — and that's backward-
compatible since basename-only is still a valid filename portion of a future scheme variant.

**Future extensibility**: if even richer provenance wants in (HTTP `Last-Modified` / `ETag`
headers for re-fetch logic, MCP server version, etc.), add fields. Readers should ignore
unknown fields silently — standard JSON-schema-evolution discipline. The dict-of-dicts
shape locked in v0 means no read-path branching on "string or dict" later.

Optional field — absent if no sidecars have known source, absent on legacy messages with no
images at all. The migration adds no `sidecars` entries; readers treat absence as "unknown source."

**Datastore portability**: the datastore JSON plus the sidecar directory together form a
self-contained unit. Moving a chat between machines means copying both. Backing up a chat
backs up both. The sidecar directory should sit alongside the JSON file with a predictable
relative path so backup tooling and `cp -r` Just Work.

### Sidecar garbage collection

Mark-and-sweep, not reference-counting. Composes cleanly with chattree's existing
`prune_unreachable_nodes` GC.

**Why mark-and-sweep over eager refcount on delete:**

1. **chattree is revisioned.** Each node holds multiple revisions of its payload. Editing a
   message creates a new revision; the old revision still references the old content. Eager
   "image not referenced by latest revision → delete sidecar" is wrong — the old revision is
   still valid and the user might switch back to it via the revision picker. Doing refcount
   *correctly* means counting across all revisions of all reachable nodes, which is the same
   work as mark-and-sweep, just spread across every delete.
2. **Branching makes orphaned subtrees routine, not exceptional.** Reroll an assistant message
   → old branch is still in the forest until pruned. Sidecars referenced by the old branch
   should stay until the branch itself is pruned. Mark-and-sweep handles this uniformly: GC
   runs after `prune_unreachable_nodes`, deletes sidecars not referenced by any reachable node.
3. **Matches the existing convention.** `prune_unreachable_nodes` is an explicit pass, not
   per-op. Sidecar GC slots into the same trigger model.

**The sweep operation.** `chattree.PersistentForest.prune_unreferenced_sidecars()` (no args) marks
every reachable sidecar and sweeps the rest; `unreferenced_sidecars()` is the same computation
without deleting, for the dry-run preview. The mark phase scans two reference sites per revision:
`sidecar:` URLs in `image_url` content-parts, and `original_sidecar` entries in
`general_metadata["sidecars"]` (the case-2-preserved originals from §5, which have no content-part
of their own). Anything in the sidecar directory not referenced by any revision of any reachable
node gets deleted.

**Ownership: injected extractor, not a self-scan (design decision, implemented).** The original
draft above had `prune_unreferenced_sidecars()` reach into `node["data"]` and parse
`payload["message"]["content"]` itself. That was dropped — it makes the storage layer parse the
chat-message schema, violating chattree's founding invariant that *payloads are opaque to
chattree*. But sidecar GC inherently needs to read references out of payloads, so the two are in
tension, and something has to give. Two coherent resolutions were weighed:

- **Delegate the read.** chattree drives the revision traversal (unambiguously its job) but calls a
  `sidecar_extractor` — configured once at construction — to read the refs out of each opaque
  payload. The extractor (`imagestore.sidecar_refs_in_payload`, a pure `payload -> set[str]`) lives
  in the message-schema layer that owns the format. chattree stays opaque; GC is a self-contained
  no-arg op; the reference stays single-sourced in the payload (no drift).
- **Register the refs.** A payload revision registers its sidecars with chattree at write time,
  stored as chattree-side metadata. Rejected: it *duplicates* a reference that already lives inside
  the payload across the opacity boundary, and the two copies can drift — a missed registration
  silently GCs a live image.

We took the first (delegate). The "leak" it seems to introduce — GC of chattree-owned storage
needing an imagestore-owned function — is really chattree's opacity invariant asserting itself:
given opacity, delegating the one forbidden step is the honest move, whereas registration is a
denormalization workaround. Configuring the extractor at construction (rather than passing it per
call) lets chattree *own* the GC operation while delegating only the payload read. See the
docstrings on `PersistentForest.__init__` (`sidecar_extractor`), `prune_unreferenced_sidecars`, and
`imagestore.sidecar_refs_in_payload` for the implemented contract.

**Two triggers, one operation:**

The cleanup operation is `prune_unreachable_nodes` + `prune_unreferenced_sidecars` run as a
pair (followed by save). Both invocation paths call the same function — single source of
truth, no risk of the two pruning steps drifting out of sync between triggers.

- **Automatic on autosave-on-exit**: Librarian's current save path is autosave-triggered at
  exit (there's no manual save action), and it already calls `prune_unreachable_nodes`.
  Adding the `prune_unreferenced_sidecars` step in the same path is a one-line extension.

  *Implication for long sessions*: orphaned sidecars (and orphaned tree nodes) accumulate
  during the session and clean up at exit. Fine for typical work patterns, but worth knowing
  — long-running Librarian instances with heavy image-attach + reroll activity won't reclaim
  disk until next exit. The manual button below covers the rare "I want it gone now" case.
- **Manual cleanup button**: invokes the same full prune-and-clean (nodes + sidecars + save).
  Not just a sidecar-only cleanup — manual cleanup IS autosave-on-exit, minus the exit. The
  user gets "save current state, fully cleaned up" as an explicit affordance for long-running
  sessions.

  Includes a **dry-run preview** before commit: "would prune N unreachable nodes and M
  unreferenced sidecar files (X MB total)". Inspectable before destructive action, useful for
  troubleshooting "where did my disk space go," doubles as debugging output for the GC logic.

  **Optional thumbnail grid** for the to-be-deleted sidecar files — expandable affordance
  ("Show images" / disclosure triangle) reveals a grid of thumbnails rendered via
  `raven.common.image.lanczos` (the same rendering path used for chat history images).
  Knowing *which* M images are about to be deleted matters: orphaned subtrees may contain
  images the user has forgotten about and would want to recover, and "47 files, 312 MB" tells
  you the magnitude but not the content. Default is summary-only (counts + total size) to
  avoid overwhelming the dialog for large cleanups; one click expands to the grid. Each
  thumbnail tooltip shows the sidecar filename and (if known) the `url` from its sidecars
  entry — gives the user enough context to recognize an image they care about.

  **Recovery-to-staging affordances on the grid.** When the user looks at a soon-to-be-deleted
  image and realizes "wait, I want to keep this, I'm not sure where my original is anymore,"
  they need an escape hatch *before* committing. Otherwise the workflow forces them to cancel,
  manually navigate to the sidecar directory, identify the right file by sha256 filename
  (impossible without the GUI's help), copy it out, then re-run cleanup. Specifically:

  - **Per-thumbnail "Save a copy"** action (icon-on-hover or context-menu): copies the
    sidecar file to a configured staging directory. The sidecar stays in place until
    cleanup commits; after commit, the staged copy is the survivor. Copy semantics rather
    than move because copy is safer mid-workflow (sidecar still exists until explicit
    commit, recoverable if the user changes their mind again).
  - **"Save all to staging"** bulk action: copies all to-be-deleted sidecars to the staging
    directory. For the "I'm cleaning up a chat but want to keep all the images that were in
    it" workflow.
  - **Staging directory location**: configurable via `librarian_config.image_staging_dir`,
    default `~/.raven/staging/recovered_images/` (user-level, not per-datastore — recovered
    images are about user data, not chat-specific). When using the staging affordances,
    if any files are saved, the dialog gets an "Open staging directory" button using the
    same `open_in_file_manager` utility from §6. The user can verify what they recovered
    without leaving the dialog.
  - **Saved-file naming**: preserve the original filename when known (from the sidecars
    entry's `url` if it's `file:///<absolute_path>`, the basename portion) — falls back to
    the sha256-based sidecar filename when no better name exists (remote-fetched images
    with hash-only names). Disambiguates collisions in staging with a `(2)` suffix or
    similar.

  Tree-node preview (showing the unreachable nodes' message text) is *not* in v0 — those
  nodes are already not visible in the chat view, so they've presumably been deliberately
  abandoned. Add later if a real need surfaces.

  Modal confirmation IS appropriate for this dialog despite the earlier non-modal-feedback
  principle (§6): the rule there was about *informational* notifications shouldn't block the
  UI. *Confirming destructive action* is exactly the case where the user is invoking the
  flow and explicitly wants to be asked. Distinct UX role; same dialog primitive deployed
  for different purpose.

  Label could be e.g. "Clean up & save" or "Prune & save" — finalize during implementation.

**Where this button lives in the GUI** — Librarian doesn't yet have a settings or
preferences surface, so neither this button nor the other "open file manager" buttons added
in §6 (Open docs DB directory, Open chat datastore directory) have an existing home. Three of
them is enough to imply a small Tools / Utilities surface — concrete placement (menu item,
sidebar panel, toolbar group, etc.) is for CC and Juha to settle during implementation based
on Librarian's existing GUI shape. Brief notes the need; doesn't dictate the answer.

**What's deliberately out of scope for v0** — granular "remove this image but keep the
message" UI affordances. That operation interacts with chattree's revision model in subtle
ways (does it edit the current revision in place, create a new revision, scrub old
revisions?), and the common "I don't want this image in this chat at all" intent is already
covered by message deletion + sweep. Revisit when there's a concrete need.

### Migration of existing chat history files

Two stanzas in `chatutil.upgrade_datastore`:

1. **Compat brief §11**: extract inline `<think>` to `reasoning_content`, dedup inline-tag
   tool calls, move `toolcall_id` → `tool_call_id`, codebase-wide rename.
2. **This brief**: wrap legacy string `content` as `[{"type": "text", "text": <string>}]`
   (single text part, persona prefix preserved verbatim — see §3b).

Old datastores have no image parts, so there's no image-related migration. Idempotent on
re-run.

---

## 9. Backend capability handling

Not every OAI-compat backend supports multimodal — non-VLM models will error or silently drop
image parts. At setup time (per the compat brief's identity-aware setup), record whether the
loaded model is a VLM. Two ways to surface:

- **Hard**: if model is non-VLM and the user attaches an image, refuse with a clear message
  ("the loaded model doesn't accept image input; switch to a VLM").
- **Soft**: send anyway and let the backend reject; surface the error to the user.

Hard is friendlier and worth the small extra config plumbing. The compat brief already pulls
model identity / capability from the backend (LM Studio's `/api/v0/models/{id}` has enough
metadata; ooba's `model_info` likewise), so the VLM-capability flag falls out of work already
specified.

---

## 10. Interaction with other briefs

- **Compat brief** — **hard upstream dependency.** Compat establishes the typed-event stream
  from `invoke` (`{content, reasoning, tool_call}`), the sibling-field message data model
  (`reasoning_content`, `tool_calls`, `tool_call_id`), the per-family `reasoning_wrap_input`
  table, the gear-icon tool-call rendering convention, the `<think>` / `<tool_call>` parser-
  side stripping, and the migration of existing chat data (extraction + tool_call_id move +
  codebase-wide rename). This brief builds on all of that — it adds the `content` field's
  string → parts-list refactor, image input, image rendering, MCP tool-result typing, and
  VLM capability detection.
- **MCP client brief** depends on this; its §5 currently sketches the resolution and points
  here. Once this lands, MCP §5 reduces to "MCP content blocks map to parts per this brief."
- **Webfetch brief** — webfetch v0 returns a string; under content-parts that string is wrapped
  as a single text part. Webfetch is also the brief that **creates the `raven.common.text`
  package** (the `normalize` utility is its first inhabitant); this brief uses it for tool
  result text (websearch results, MCP text content) and trusts webfetch's package creation as
  upstream work. (`raven.common` already has `audio`, `image`, `video`, `gui`; `text` is the
  natural addition.)
- **Lorebook brief** — unaffected. Lorebook entries are text content, and the context-assembler
  note speaks of candidate injections generically. A future lorebook entry with an image is
  just a multi-part injection, no refactor.

---

## Out of scope for v0

- Audio parts (model support uneven; revisit when target models normalize on it).
- Rich MCP embedded-resource handling (downgrade to text part for now).
- Image *output* from models (image-generating LLMs aren't a target).
- Drag-and-drop / paste image input (DPG upstream blocker on Linux/macOS — see §5; file-picker
  is v0 affordance).
- Per-model precise image token estimation (use conservative placeholder; the compat brief's
  usage-calibration mechanism refines it self-correctingly).
- Custom card-style GUI rendering for tool results (markdown-rendered text parts is enough;
  if a future need arises, options include re-parsing markdown at render time or extending the
  vendored markdown renderer — both real but neither needed yet).

---

## Acceptance

- Sending a message with attached image to a VLM-capable backend: the model receives the image
  and can describe / discuss its contents end-to-end.
- **Websearch results render as a stack of markdown-rendered text parts**, one per result,
  with clickable URLs (the standard markdown renderer's link handling). On the wire, the
  model sees the same markdown text content. The structured `format_results` output is no
  longer discarded; the preformatted-string fallback path is gone.
- **Image storage**: user-attached images are copied to the datastore's sidecar directory at
  attach time (downsampled to ≤ 1 MP via `raven.common.image.lanczos` if larger, preserving
  aspect ratio; **original also preserved as a sidecar by default**, opt-out via config),
  persisted as `sidecar:<filename>` URLs with rich provenance recorded in
  `general_metadata["sidecars"]` (a dict per sidecar with `url`, `fetched_at`, `content_type`,
  `source`, and conditionally `original_dimensions`, `original_size_bytes`, `original_sidecar`);
  wire-send substitutes data URLs from the sidecar bytes. No `https://` URLs appear in saved
  datastores. Datastore JSON is human-readable (no inline base64 bloat); sidecar directory
  plus JSON is portable as a unit.
- **Image provenance UI**: image thumbnails offer "Open original" (dispatches across the
  three storage cases) and "Open containing directory" (parent dir in file manager for
  `file:///` origins; parent URL in browser for remote origins) via a small cross-platform
  utility in `raven.common.utils`.
- **Discoverability buttons ride along**: "Open docs DB directory" and "Open chat datastore
  directory" buttons exist in Librarian's UI, using the same `open_in_file_manager` utility.
  Users can find the RAG ingestion directory without grepping the source.
- **Cleanup is one operation, two triggers**: `prune_unreachable_nodes` +
  `prune_unreferenced_sidecars` are invoked as a pair, both at autosave-on-exit and via the
  manual "Clean up & save" (or similar) button. The button shows a dry-run preview ("would
  prune N nodes, M files, X MB total") before commit, with an expandable thumbnail grid of
  the to-be-deleted sidecar images and per-thumbnail / bulk **"Save a copy to staging"**
  affordances so the user can rescue images they want to keep before committing. Sidecar GC
  correctly identifies references across image_url parts AND `original_sidecar` entries in
  the sidecars metadata. Staging directory configurable, defaults under `~/.raven/staging/`.
- Bidirectional tool-call ↔ tool-response navigation works: clicking the "trace upstream" icon
  on a tool response scrolls to and highlights the originating tool-call sub-element in the
  assistant message; clicking the "trace downstream" icon on a tool-call sub-element scrolls
  to and highlights the corresponding tool response; orphaned calls disable the button with an
  explanatory tooltip.
- Existing text-only flows (current chat behaviour) are unchanged from the user's perspective:
  bubbles look the same, no double-encoding, no spurious whitespace.
- Migration of an existing chat history file (string `content`) to the new representation is
  idempotent and lossless.
- A non-VLM backend with an image attachment: clear error to the user, no silent drop.
- Image-bearing prompts are accounted for in the token budget (conservative placeholder is fine
  for v0; budget doesn't blow context).

---

## Implementation status

**Half 1 (text-only content-parts core)** — done, merged to `main` (`b4bd79b`, 2026-06-05). Content
became a typed-parts list everywhere; tool results as parts; per-part renderer; load-time migration.

**Half 2 (multimodal / images)** — in progress, scoped into checkpoints:

- **A1 — storage foundation** (`915c624`, 2026-07-16, CI green): config knobs (§5/§7 data); new
  `raven.librarian.imagestore` (`store_image_as_sidecar`, `sidecar_url_to_data_url`,
  `sidecar_refs_in_payload`, `downsample_dims`); `chattree.PersistentForest` sidecar file store +
  mark-and-sweep GC with the injected-extractor design (§8 above). Original images stored byte-for-byte
  to preserve embedded metadata. 23 tests.
- **A2 — backend wiring** (`c4c0a2a`, 2026-07-16, CI green; live-verified against a Qwen 3.6 VLM):
  §9 VLM-capability detection (`settings.model_is_vlm`, tri-state; LM Studio flags vision via the model
  record's `type == "vlm"`); §8 wire substitution — `llmclient._serialize_history_for_wire` preserves
  image parts and resolves `sidecar:`→`data:` on send; `datastore` threaded through `invoke`/`prefill`.
  5 tests.
- **B — minimal GUI end-to-end** (split, 2026-07-17; branch `feature/librarian-multimodal-gui`, pushed, CI green):
  - **B.1 composer groundwork — DONE.** The single input row became a vertical stack — multiline text field
    (Shift+Enter = newline, Enter = send; ~5 rows, resizable deferred to `TODO_DEFERRED`), a hidden staged-image
    thumbnail strip, and a button toolbar (send / mic / VU; attach button not yet added). Composer outer height
    fixed so the chat/avatar panels don't jump when the strip appears; the strip is to steal height from the text
    field. Enter-send clears the field via deactivate → clear → refocus (ImGui ignores `set_value` on the active
    input). Commits: `665ac3e` (composer), `9e06b86` (field-clear). Incidental fixes landed alongside: INDEXING
    startup-race (`83b0ada`), silent-LLM-error → rerollable message (`b90d28b`), LM Studio default (`f05a6c0`).
  - **B.2 attach mechanics — DEFERRED** pending FileDialog image-thumbnail previews (see `TODO_DEFERRED.md`). A
    filename-only picker is a poor fit for choosing images, so the attach flow waits on that FileDialog work
    (Juha's call, 2026-07-17). Remaining when resumed: §5 attach button (`fa.ICON_PAPERCLIP`) + image FileDialog;
    §9 hard-gate attach on `model_is_vlm is False` (flag reachable at `app.py` module-level `llm_settings`); byte-
    snapshot in-memory staging + thumbnail strip (dedicated `add_texture_registry`; `add_dynamic_texture`;
    GLVND deletion workaround already set at `app.py:40-43`); on-send `store_image_as_sidecar` → parts → thread
    through `chat_round`; §6 fill the render stub (`chat_controller.py:1059-1060`); §7 image-token budget. Click-
    to-expand: **v0** shows the downscaled primary; **v1** a Lanczos mip-chain zoomable viewer (cherrypick's
    machinery).
- **C — provenance & discoverability** (not started): `open_in_file_manager` utility; open-original /
  open-dir buttons; the ride-along "open docs DB dir" / "open datastore dir" buttons — these need a
  small Tools/Utilities GUI surface (placement TBD by CC + Juha, as noted in §6). "Show original" resolves
  to **three distinct affordances** (settled with Juha), not one, because they have different reliability
  guarantees: (1) **"Show original"** → the stored archival copy (`original_sidecar`, or the primary itself
  when it is the verbatim original) — the canonical one, always present offline; (2) **"Open source"** →
  the provenance `url` (fragile — file moved / URL 404), labelled as provenance, shown only when present;
  (3) **"Open containing folder"** → file-manager reveal of `datastore.sidecar_dir`.
- **D — GC UX & navigation** (not started): manual "Clean up & save" (dry-run preview + thumbnail grid
  + staging recovery); bidirectional tool-call↔response nav links. Open question: `prune_unreachable_nodes`
  currently runs only in `minichat`, not the GUI exit path — decide GUI-exit prune vs. manual-only.
