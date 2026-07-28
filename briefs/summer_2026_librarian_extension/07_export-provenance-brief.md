# Brief: export provenance metadata (Article 50(2) good-faith marking)

**Why this exists:** the EU AI Act's Article 50(2) transparency obligation (guidelines adopted
2026-07-20, marking obligation biting 2026-12-02 for systems already on the market) asks
providers of generative systems to mark AI-generated output so it's *detectable as AI-generated*.
Librarian exports chatlogs and individual messages to the clipboard, and that exported text
currently carries no origin information a downstream reader — human or parser — can rely on.

Two things to be clear about up front, because they set the scope and stop us from overbuilding:

1. **This is provenance metadata, not the robust 50(2) mark.** The robust mark for text is a
   *generation-time watermark* (green-list / SynthID-Text): it acts on the logits, inside the
   sampling loop, and only whoever controls generation can apply it. Librarian samples
   un-watermarked third-party weights through an OpenAI-compat backend — there is no watermarker
   anywhere in the path, so there is nothing for us to *add* post-hoc to text we merely received.
   That burden sits upstream with the model provider, and "as far as technically feasible" plus
   downstream-system-provider status is what covers the gap. **Do not build text watermarking.**
2. **What a downstream system provider *can* do is exactly three things**: don't strip any marks
   the model does emit (N/A today — none arrive — but a one-line constraint below keeps us honest
   if that changes); attach system-level provenance to outputs; document the limitation. This
   brief is the middle one. It also lowers the bar for any *deployer* who publishes Librarian
   output and falls under 50(4), which is their obligation, not ours — sane defaults help them.

**Sequencing:** depends on content-parts (03) — this brief reads `message["role"]`,
`general_metadata["datetime"]`, and `generation_metadata["model"]` off the persisted payload,
whose shape 03/compat fix. No downstream brief depends on this one. It can land any time after
03; the December date is the only external peg, and there's no rush ahead of it.

---


## 1. The core decision

**Surface provenance that already exists in the payload, in a machine-readable schema, at the two
export surfaces we already have.** Nothing about origin needs to be *computed* — every node
payload already records it:

| Datum | Payload location | Notes |
|---|---|---|
| Origin role | `message["role"]` | `user` / `assistant` / `tool` / `system` |
| Model | `generation_metadata["model"]` | present on `assistant` nodes; the one datum not currently exported anywhere |
| Timestamp | `general_metadata["datetime"]` (ISO) | already surfaced, but as prose |
| Persona | `general_metadata["persona"]` | already surfaced (human heading) |

The gap is not data, it's *form* and *coverage*: today's exported metadata is prose bullets
(`- *Node ID*: ...`) aimed at debugging (node IDs, revision numbers), it's gated behind a
Shift/`include_metadata` toggle, and it never includes the model name. Provenance wants the
opposite defaults: machine-readable, origin-focused, on by default.

**Machine-readable means a documented, consistent, parseable schema — not a prose sentence.** A
line a parser can key on, not "this was written by an AI" in running text. For Markdown the
natural home is **YAML front matter**; that also answers the "Markdown has no metadata slot"
worry — front matter *is* the slot, and it's conventional enough that downstream tooling already
expects it.


## 2. Two emit surfaces, two jobs

The surfaces already exist. Each does a job the other can't, so both get the treatment.

**(a) Document manifest — `DPGLinearizedChatView.get_chatlog_as_markdown`.** Prepend a YAML
front-matter block ahead of the existing `# Raven-librarian chatlog` heading. This is the clean
whole-file parse: one manifest for the exported log.

```yaml
---
generator: raven-librarian
generator_version: <__version__>
exported_at: <ISO>            # already have: format_chatlog_datetime_now()
ai_generated: true            # at least one assistant message present
messages:
  - n: 0
    origin: user
  - n: 1
    origin: assistant
    model: <generation_metadata["model"]>
    generated_at: <general_metadata["datetime"]>
  - n: 2
    origin: tool
---
```

The existing prose `- *HEAD node ID*` / `- *Log generated*` header can stay as human-facing text
under the H1, or fold into the manifest — implementer's call; the manifest is the load-bearing
part. Keep the debug metadata (node IDs, revision numbers) exactly where it is, behind
`include_metadata` — it's orthogonal to provenance and serves a different reader.

**(b) Single-message copy — `DPGCompleteChatMessage.copy_message_to_clipboard_callback`.** The
document manifest doesn't travel when someone lifts one message out — and single-message copy is
*the* fragment-lift case (it exists because the DPG Markdown renderer isn't selectable). So the
copied fragment needs its own provenance.

**Use the same front-matter format here, not a second ad-hoc marker.** An earlier draft used a
one-line blockquote for this case; that was a mistake — a lone copied message is just a tiny
document, YAML front matter parses identically whether it precedes one message or fifty, and one
schema means one parser path and one thing to test. A blockquote would be a second format a
consumer has to special-case for no gain. So a single assistant/tool copy gets a front-matter
block carrying the one message's `messages: [ {n, origin, model, generated_at} ]`; the
`ai_generated` and `generator` keys carry over unchanged. (The only thing the blockquote had going
for it — surviving a copy-of-a-copy — is a vanishing case, and front matter degrades the same way
any partially-deleted provenance does: worst case the origin line just isn't there.)


## 3. Mark AI content, leave user messages clean

The regulation only cares about AI-generated content. That scoping resolves a friction in the
current code for free: `copy_message_to_clipboard_callback` deliberately omits the role heading in
regular mode so a user can copy their own earlier question straight back into the input field to
tweak and resubmit. Prepending a `---`-fenced block to *that* copy would be actively worse than the
old blockquote idea — the user now has a multi-line YAML header to delete before their question is
usable. So the format-unification decision in §2b makes this rule matter *more*, not less.

So: **emit the front-matter block for `assistant` and `tool` single-message copies; emit nothing
for `user`-role single copies.** User turns are human-authored — there's nothing to mark as
AI-generated, so the disclosure has no subject — and skipping it keeps the resubmit path clean.
Compliance intent and ergonomics point the same way; no toggle needed. (For the *full-log* export,
user messages still appear in the manifest as `origin: user` — there the manifest is one block for
the whole document and the resubmit concern doesn't arise; it's only the single-message copy of a
user turn that gets no header. `tool` messages are retrieved external content, not model-generated
prose; label them `origin: tool` truthfully — don't claim synthesis that didn't happen.)


## 4. What this is *not* — scope fence

- **Not text watermarking.** See "Why this exists." No logit-level anything; we don't control the
  sampler and the weights aren't watermarked. If a future backend *does* emit marked text, the
  constraint is only: **don't strip it** on export. Nothing to implement now beyond not doing harm.
- **Not per-span code exclusion.** 50(2) carves out source code, but Librarian's domain is
  literature analysis; marking is per-message origin, not per-span, and a message that happens to
  contain a fenced code block is still an assistant turn. No special-casing.
- **Not retroactive.** Content generated before 2026-08-02 needs no labelling; existing stored
  chatlogs don't need a migration pass. New exports carry the manifest; that's sufficient.
- **Not a legal artifact.** The manifest is a good-faith engineering measure. Whether JAMK signs
  the Code of Practice on Transparency of AI-Generated Content is the actual compliance decision,
  and it's legal's, not this brief's.


## 5. Out of scope for v0

- Plain-`.txt` export. The live export path is Markdown (`get_chatlog_as_markdown`); there's no
  txt emitter to extend. If one is added later, the equivalent is a documented header block —
  weaker (no front-matter convention) but the same idea. Not now.
- Cryptographic provenance (C2PA-style signed manifests — Coalition for Content Provenance and
  Authenticity; tamper-evident, X.509-signed manifests bound to a file, the "Content Credentials"
  you see on some AI images). Overkill for a local research tool's clipboard export, and the
  strong-verification niche it fills isn't ours. Notably, C2PA is the front-runner for what
  "machine-readable marks" concretely means under 50(2), so if JAMK ever signs the Code of
  Practice this is plausibly what "adequate marking" cashes out to — but at the *model-provider*
  level, which loops back to the core point that the robust mark lives upstream, not here. Revisit
  only if a real downstream consumer asks Raven for verifiable provenance.

(The 50(1) interaction-disclosure badge is *not* out of scope — it's an adjacent one-liner handled
in §7, since it's the same disclosure surface and the context is loaded.)


## 6. Implementation checkpoints

- [ ] **A — manifest builder.** A pure helper (chatutil-level) that takes the linearized history
      and returns the YAML front-matter string. Testable without GUI. Emits `ai_generated`,
      `messages[]` with per-node `origin` / `model` / `generated_at`.
- [ ] **B — wire into `get_chatlog_as_markdown`.** Prepend the manifest; leave the debug-metadata
      path untouched behind `include_metadata`.
- [ ] **C — single-message copy.** Reuse the checkpoint-A builder to emit a one-message
      front-matter block in `copy_message_to_clipboard_callback`; assistant/tool only, nothing for
      `user`-role copies (see §3). No separate marker format.
- [ ] **D — tests.** Round-trip: export a known small history, parse the front matter back
      (`yaml.safe_load` on the fenced block), assert origins/models line up with the payloads.
      One case each: user-only, mixed, tool-bearing, streaming node with `model is None`
      (must degrade gracefully — the existing code already tolerates a `None` node ID, match that).


## 7. Adjacent: the 50(1) interaction-disclosure label

Different article (50(1), *identity* disclosure, applies 2026-08-02, no grace period), same
disclosure surface, and the context is loaded — so fold it into this pass rather than spinning a
separate brief.

**The problem.** The current label (`app.py`, the `ai_warning_text` widget) reads: *"Response
quality and factual accuracy depend on the connected AI. Always verify important facts
independently."* That's a *quality/fallibility* disclaimer — it warns the AI might be wrong. It
never states the interlocutor *is* an AI, which is precisely what 50(1) requires. The identity
claim is only *implicit* ("depend on the connected AI" reads as a component-quality note), and
50(1)'s obviousness exception is to be read restrictively, so don't lean on the inference.

**The fix.** Make the identity statement explicit; keep the quality line (good practice, not
required):

> ⚠ You are interacting with an AI system. Response quality and factual accuracy depend on the
> connected AI — always verify important facts independently.

First clause satisfies 50(1) on the nose; second is the existing warning.

**Layout cost — this is not just a string swap.** The below-chat row is already crowded and the
new wording is longer, so it needs a reserved second line: bump `ai_warning_h` (currently `42` in
`config.py`) and let the text wrap, rather than swapping the string and letting it clip. Check the
`ai_warning_spacer` sizing math (`app.py` ~1080, and the resize handler ~1205) still lays out with
the taller block.

**Keep, deliberately:** the label is always-visible and non-disableable. That's the low-effort
compliant choice — "at the start of the first interaction" is trivially satisfied by a persistent
label, and non-disableable means it can't be configured out of compliance. Don't add a dismiss
affordance; it would only add first-interaction-timing logic for no benefit.

**Known limitation — document, don't fix here:** the label is a DPG text widget, and DPG (via Dear
ImGui, immediate-mode) exposes no OS accessibility tree, so it's near-certainly invisible to screen
readers. This is a *whole-app* property, not a disclosure-specific defect — 50(1)'s accessibility
requirement mainly guards against making the notice *specifically* less perceivable than its
surroundings (e.g. baking it into an image), and this label is the same modality as the entire UI,
so it isn't singled out. A blind user can't perceive any of the DPG interface, so the badge gap is
subsumed by the app-level situation. Posture: note the limitation in docs, same as the 50(2) side;
do not attempt to make one widget screen-reader-visible in isolation. (Real future route, big
scope, not now: Raven already has TTS — a *self-voicing* mode needs only tab-focus + focus events,
not an OS a11y tree, and would sidestep DPG entirely. Backlog, not this brief.)

- [ ] **E — label wording + layout.** Update `ai_warning_text`; bump `ai_warning_h`; verify the
      spacer math and resize handler cope with two lines. Add a one-line note to the docs recording
      the screen-reader limitation and the self-voicing backlog item.
