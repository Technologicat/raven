# Brief: lorebook (keyword-triggered injection) in Librarian

**What:** a SillyTavern-style "World Info" / lorebook — curated dictionary-like entries that are
injected into the prompt when their trigger keywords appear in recent context. Primary motivating
use: inject glossary definitions (and similar custom dictionaries) so the model gets the local
meaning of a term the moment it comes up, without a search.

**Why it's aria-worthy, not gold-plating:** Raven already does the *hard* retrieval case
(HybridIR: chunk + embed + semantic/keyword rank). Keyword-triggered atomic injection is the
*easy* case of the same problem class. Having solved the hard one, leaving the easy one unbuilt
would be knowledge left on the floor. So this clears the bar the project already set.

---

## Design constraints (from the field, confirmed)

Three things make the lorebook **not** just "HybridIR with a filter":

1. **Keyword-trigger only — no semantic.** A lorebook entry fires on exact term presence, not
   conceptual similarity. Deterministic and predictable by design; that determinism is a feature
   (you know exactly when a definition will appear). HybridIR's semantic half is unwanted here.
2. **Chunking disabled — atomic entries.** A definition is injected whole or not at all. Half a
   definition — especially only its *second* half — is worse than nothing. HybridIR's chunk
   pipeline is exactly what the lorebook must *not* do.
3. **Directory-scanned source with live reload.** Definitions live as drop-in files in a watched
   directory; adding/editing a file updates the lorebook without a restart. Reuse the existing
   directory-scanner **or extend it as appropriate** — if its current shape doesn't fit cleanly,
   extend the scanner rather than contorting the lorebook design to match it. Same for the watcher.

So the lorebook uses almost none of HybridIR's embed/chunk/rank machinery, but wants its
*injection / budgeting / formatting* path.

---

## HybridIR relationship — factor, don't fork (CC to choose the seam)

Two viable shapes; pick after reading HybridIR's internals:

- **(a) Generalize HybridIR** with a "keyword-only, no-chunk, exact-trigger, atomic-entry"
  retrieval mode. Cleanest if HybridIR's trigger/rank stage is already separable from its
  inject/format stage.
- **(b) Sibling component** that shares only the injection/budgeting/formatting code with
  HybridIR and brings its own trigger (keyword match over atomic entries from the directory
  scanner).

Recommendation: factor out the shared *injection path* so both HybridIR and the lorebook call it,
and give the lorebook its own trigger. Whether that lands as a mode on HybridIR or a sibling is an
implementation detail of where the seam falls — but **do not** build a parallel injection
mechanism. The shared injection path is bigger than just these two, though — see the
*Context assembler* section below, which is the real shape of "don't fork." Build the lorebook as
a *client* of that, not as something that writes into the prompt itself.

---

## Entry model (v0)

Borrow the useful parts of ST World Info, drop the rest:

- **keywords**: list of trigger terms (aliases) for one entry. Match is case-insensitive,
  word-boundary (not naive substring — "cat" must not fire on "concatenate"). Multi-word keys
  allowed.
- **content**: the text injected when triggered. Atomic; never chunked.
- **priority / order**: for deterministic ordering when several fire, and for budget eviction.

Out of scope for v0 (note as future, don't build yet): secondary/AND keys, per-entry injection
depth, and **always-on entries** (ST "constant" — inject unconditionally). v0 is keyword-triggered
only; how always-on would be tagged is left open until it's actually built (and will have to
respect the no-fields rule when it is).

**Recursive triggering is in-scope for v0** (see its own section below) — the glossary relies on
it heavily, and so does loop-detection.

---

## Trigger & injection mechanics

- **Scan window**: match keywords against recent context (last N messages — config; not the
  whole history, or stale terms keep re-firing). **Scan all message roles**, not just user turns:
  user, assistant, and tool-response messages. The assistant case is obvious; the tool-response
  case is a real one — if the model retrieves a document from a DB and that document (written by
  the user or by the AI earlier) uses glossary language, the definitions should fire on it too.
- **Injection position**: simplest defensible default is a context block ahead of the latest
  user turn. Make position configurable but ship one sane default; don't over-parameterize before
  there's a reason.
- **Budget**: cap total injected lorebook tokens; if more entries fire than fit, evict by
  priority. Reuse the token-counting calibration from the compat brief for the budget estimate.
- **Determinism**: same context + same lorebook ⇒ same injection. No ranking nondeterminism.

## Recursive triggering (v0) — with loop detection

Injected entries can themselves contain terms that trigger further entries; the glossary does
this constantly (and contains cycles). So injection is a graph traversal, not a single pass:

- **Seed**: entries triggered directly by the scan window (depth 0).
- **Expand**: scan each injected entry's *content* for further triggers, and follow the
  glossary's explicit cross-reference anchors (`[term](#term-anchor)`) where present. The anchor
  links are the cleaner signal — explicit, no false positives — so prefer them for glossary-
  sourced entries and fall back to content keyword re-scan for plain entries.
- **Loop detection**: maintain a visited set of entry IDs; never inject an entry twice. The
  glossary has A→B→A cycles, so this is mandatory, not defensive.
- **Bounds**: cap by max recursion depth *and* the token budget (below), whichever bites first.
- **Priority by depth**: depth-0 (directly triggered) outranks transitively-pulled entries for
  budget eviction — closer to the live context = more relevant. Within a depth, use entry
  priority.

This stays bounded and deterministic: a finite entry set, a visited set, a depth cap, and a
token cap. No risk of runaway expansion.

---

## Source format, import workflow & the glossary as first customer

**Two source kinds**, both feeding the lorebook:

- **Multi-entry structured files** (the glossary). Split *deterministically* — no LLM needed —
  but assume **almost nothing** about the format. The only structure relied on:
  - **A markdown header delimits an entry.** The header *level* is **configurable per source,
    not assumed** — ours is `h2` only because `h1` is taken by a "jump to letter" TOC, but another
    author's glossary may use a different level. So: a source declares its entry-header level
    (default `h2`); split on headers at that level; treat shallower headers (e.g. `h1` letter
    sections) as non-entry structure and deeper headers as part of the entry body.
  - **The header text is the entry title** (= primary trigger keyword).
  - **`[text](#anchor)` links are the recursion graph**, mapped to entries by slugifying titles
    the standard markdown way (lowercase, spaces→hyphens, strip punctuation) and matching anchors.
  - **The body is free-form.** Do *not* parse fields — don't assume `**Meaning:**`/`**Etymology:**`
    or even that fields exist. Whatever sits between one entry header and the next *is* the
    content, injected whole. The consuming LLM is smart enough to interpret arbitrary prose; the
    splitter's only job is title + body + crossrefs.

  Read `glossary.md` from a **configured local path** — the lorebook's DB uses a local copy, not
  a hardcoded repo location. The zero-maintenance pattern: symlink the live local glossary (the
  one that syncs with the repo) into the lorebook's source path, so new definitions flow in with
  no extra step. Do *not* pre-split it into derived per-entry files, which creates a sync problem
  (glossary.md is the source of truth; derived files drift). Re-parse on change via the watcher —
  and have the watcher resolve symlinks so it watches the real target, not just the link.
- **Single-entry drop-in files** in the lorebook's own docs folder — the "plonk in a new
  definition on the fly" path: one file, zero ceremony, hand-editable. Use the existing
  directory-scanner.

**The format contract is deliberately tiny** (this matters once coworkers author against it — see
the plugin section): *entries are header-delimited at a configured level; cross-refs are markdown
anchor links.* That's the whole contract. Everything else about an entry's content is free.

**Separate docs DB folder.** The lorebook gets its own datastore folder, distinct from
HybridIR's RAG datastore — different lifecycle (curated/deterministic vs corpus/semantic), and
you don't want lorebook entries polluting RAG ranking or vice versa.

**Aliases.** v0: the title is the trigger. Optionally, variants can fire too (e.g. `Fup` /
`fuppable`) via a parenthetical *in the header line* — `## Fup (fuppable)` — since that's still
just parsing the header, not the body. No `**Aliases:**`-style body field (that would violate the
no-fields rule above). Don't auto-derive morphological variants (too clever, false positives).
Aliases are entirely optional; absent them, the title alone triggers.

**Matching is smart-case** (the vim / ripgrep convention): an **all-lowercase** keyword matches
**case-insensitively**; a keyword **containing any uppercase** matches **case-sensitively**. This
single rule handles both failure modes that pure case-sensitive and pure case-insensitive each get
wrong:

- **Sentence-initial capitalization** (frequent — missing it would be a bug): a lowercase-canonical
  term like `fron` or `fuppable`, written lowercase as the keyword, still fires on its sentence-
  initial `Fron` / `Fuppable`, because all-lowercase ⇒ insensitive. No need to list the capitalized
  form as a separate alias — which is why the example above is now just `(fuppable)`, not
  `(fuppable, Fuppable)`.
- **Common-word collision**: a capitalized-canonical term like `Recall` (a Hindsight verb), `Mode`,
  or `Fact` matches only as written and does *not* fire on casual lowercase use mid-sentence,
  because any-uppercase ⇒ sensitive.

The author controls behaviour purely by the case they write the keyword in: lowercase = forgiving,
any-uppercase = exact. Word-boundary matching still applies on top ("cat" doesn't fire on
"concatenate"). Known residual, accept it: a capitalized-canonical term that's *also* a common word
still collides with that word's sentence-initial form (`Recall` the term vs. "Recall" beginning a
sentence) — an inherent case-blind ambiguity no case rule can resolve; rare, low-harm (a spurious
definition injection), and avoidable by making such a keyword multi-word if it ever proves noisy.

**No LLM in v0.** The deterministic splitter handles `glossary.md` and well-formed drop-ins, so
v0 needs no model in the import path at all. LLM-assisted splitting of genuinely *unstructured*
.md (someone adds a file that isn't formatted as entries) is a later addition, not v0.

**Live reload**: reuse or extend the existing watcher (per the same don't-contort principle);
re-scan affected sources on change so new/edited definitions take effect without restart.

Ingesting the glossary is the proof-of-use — and it does close a loop, stated honestly: the
duolect becomes live context for the local model (Aria/Qwen) running inside Librarian. Not for
the model it was co-written with — running Claude in Librarian would be third-party API usage the
claude.ai subscription doesn't cover, so the loop closes for the local substrate, not the
originating one. Which is itself rather on-theme.

---

## Context assembler (cross-cutting — bigger than the lorebook, but the lorebook must respect it)

This section is not a lorebook feature; it's the shared layer the lorebook must be *built as a
client of*, so it's not retrofitted later. It's the real form of "factor, don't fork."

**The problem:** three things now write into one finite context window — HybridIR (semantic RAG
over the corpus), the lorebook (keyword-triggered entries), and Hindsight (recalled
memories/observations). They overlap (a glossary term can be a lorebook entry *and* a Hindsight
observation *and* present in a retrieved chunk) and share one budget. If each injector assumes it
owns the window, they step on each other, duplicate content, and there's no owner for "what
actually goes in, in what order, within the budget."

**What must be decided now (cheap, and it's only the interface):** injectors do **not** write into
the prompt directly. Each *emits candidate injections* to a common assembly point. A candidate is
a small record: `{content, source_tag, priority_hint, dedup_key}` (plus optionally trigger-reason
for debugging). The lorebook produces candidates; so will HybridIR and Hindsight when they're
wired in. This interface is the load-bearing decision — it's what makes the standalone specs
*composable* instead of unimplementable-together.

**What is explicitly deferred (prototype-first):** the assembler's *intelligence*. v0 assembler is
trivial — collect candidates, dedup exact overlaps, concatenate in a fixed source-priority order
under a single shared budget. No clever arbitration, no partial dedup, no learned priorities. Ship
that, use it, and let the a-posteriori understanding from real use shape the real assembler. (We
can overthink it later.)

**When the trivial version stops sufficing:** not in normal use — 128k absorbs incidental overlap;
4k of injections is peanuts. The pressure point is **several full texts uploaded at once for
comparative discussion**, where uploaded documents, RAG, lorebook, and memory genuinely compete
for budget. That's the regime the smart assembler is *for*, and a strong hint at its eventual
priority order: explicit user-provided content (the fulltexts under discussion) outranks
incidental injections (a lorebook term that happened to fire, a tangential RAG hit). Build the
intelligence when that use case actually pushes the limit, not before.

So: fix the candidate-emitting *interface* now (the lorebook, and later HybridIR/Hindsight, are
clients of an assembler); defer the assembler's logic to a prototype-informed second pass.

---

## Relationship to the other briefs

Independent of the LM Studio compat work and the MCP client. Touches the same token-budget
calibration (shared, not duplicated). Unlike MCP (external tools) and webfetch (a native tool),
the lorebook is a *context-construction* feature — it shapes the prompt, it isn't a tool the
model calls. Different layer entirely.

Note: the LM Studio plugin (below) is a *second frontend* for the same lorebook, not an optional
extra — Juha runs both Librarian and LM Studio standalone, so the lorebook missing from either
is a real daily inconvenience. The native Librarian version is where the design is settled first;
the plugin is a port, both landing in v0/v1.

---

## LM Studio plugin (v0/v1 — second frontend, not a stretch goal)

For driving the model from LM Studio's own chat UI, ship the lorebook as an LM Studio
**prompt-preprocessor plugin**. Because Juha runs both frontends, this is a committed deliverable
(v0, v1 at the latest), not a someday-maybe. Well-supported: LM Studio plugins are TypeScript on
Node (`@lmstudio/sdk`), and a prompt preprocessor is an `async preprocess(ctl, userMessage)` hook
that fires on Send and returns a modified message — the docs ship an "inject text before each user
message" example and a `triggerWord → instructions` config example, and RAG-for-LM-Studio already
exists as a preprocessor plugin, so the pattern is proven.

**Sequencing: native first, then port.** Settle the trigger/inject/recursion logic in the Python
lorebook (faster iteration, full control), then port the *settled* logic to TS. This is about
doing the design once in the easier environment, not about the plugin being lower-priority.

Scope and constraints:

- **The source format is the shared contract; the implementation is not.** The plugin reads the
  *same* entry files / glossary and reimplements trigger + inject in TS. So keep the trigger/
  inject/recursion spec language-agnostic (keyword match → atomic content → bounded recursion →
  budget) from the start — this is now a design constraint, not a nicety, because the port is
  committed. Two engines, one format.
- **Recursion is in the plugin too.** It's the same visited-set graph traversal over the same
  in-memory entry map — it doesn't get materially harder in TS, so don't ship a degraded direct-
  trigger-only plugin. Port the recursion.
- **Injection point differs.** The preprocessor sees and returns the *user message* (it can read
  recent context via `ctl` for the scan window), so injection wraps/prepends the user turn rather
  than editing a system prompt. Adapt the scan window to read context through `ctl`.
- **Configurable source path (required for a published plugin).** Since you'll likely `lms push`
  this so LM-Studio-running coworkers can install it, the plugin must not assume *your* machine —
  it can't point at `glossary.md` in your substrate-independent checkout. Make the source
  location (glossary path and/or entries directory) a **plugin config field** via LM Studio's
  config-schematics (the same `getPluginConfig` mechanism the `triggerWord` example uses), so each
  coworker sets their own path in the plugin's settings UI. Design this in from the start; it's
  awkward to retrofit a hardcoded path into a published plugin.
- **Distribution / account.** Running the plugin locally needs **no account** — load it from the
  local plugin directory. An account is required for `lms push` to the LM Studio Hub, which you
  *will* want here so coworkers can install it. So: develop locally without an account, create one
  when you're ready to publish for the team.

---

## Acceptance

- A directory of definition files loads into a lorebook; editing/adding a file takes effect
  without restart.
- A term appearing in recent context injects its (atomic, whole) entry into the prompt; absent
  the term, nothing injects.
- Word-boundary matching: "cat" does not fire on "concatenate".
- Injected lorebook content respects a token budget; over-budget firings evict by priority.
- `glossary.md` ingests cleanly and its terms trigger correctly.
- Recursive triggering works and terminates on the glossary's cycles (visited-set; no double
  injection, no runaway).
- Scanning covers user, assistant, and tool-response messages.
- The injection path is shared with HybridIR, not a second parallel mechanism.
- The LM Studio plugin, on the same source files, produces equivalent injection to the native
  lorebook (same triggers fire, recursion included).
