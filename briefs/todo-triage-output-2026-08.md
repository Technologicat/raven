# Triage decisions for `TODO_DEFERRED.md` — under construction

**Started 2026-08-10, still open.** One accumulating document rather than one per session: the dispositions
here are applied together, once the whole list has been worked through. Batches are dated where it matters.

**Consumes `investigations/todo-sweep-2026-08-10/README.md`. Report-and-apply: the dispositions here are
decided, the wording is not.** Where a row says *reword* or *decline*, write it from the evidence in the
sweep and in the code, not from the note in this table — the sweep produced better prose than its own brief
specified twice, and the same latitude applies here.

**Join key is the verbatim heading**, as in the sweep. Line numbers are not used anywhere in this file, for
the reason in §4.

## Dispositions used

| Disposition | Means |
|---|---|
| `already-done` | Shipped since filing. **Tag, do not delete** — re-home any prose worth keeping, then leave it for a single deletion pass once the whole list has been processed, so nothing goes inadvertently. Suggested mechanism: a `## Already done` holding section mirroring `## Declined`, which gives that pass one contiguous block to review. |
| `decline` | Considered and rejected. Move to `## Declined` with the reason, so it stays rejected — this is the anti-re-litigation category, not the finished-work one. |
| `supersede` | Reduce the item to a stub pointing at the brief that now owns it. Do **not** delete — see §4. |
| `rehome` | Move the prose into `investigations/`, leave a stub pointing there. |
| `reword` | Item stays; its text names something that has been renamed or is wrong. Correct it in place. |
| `merge` | Fold into the named sibling; one item survives. |
| `hold` | Decision pending elsewhere. Do not act. |

---

## 1. The nine STALE

All `already-done` except where noted — none of these was rejected, every one shipped.

**Re-sort what has already moved.** CC applied the 2026-08-05 rulings this morning and put four items in
`## Declined`. Two of them are `already-done` under this distinction and should move again: *Librarian chat
input: make it multiline* (shipped) and *Attachment + docs-DB: support office document formats* (landed
2026-07-29). The other two — `torch.compile` and the Intel Mac workaround — are genuine declines and stay.

| Heading | Disposition |
|---|---|
| OS drag-and-drop of files into DPG apps (cross-platform) | `already-done` — shipped 2026-08-10. |
| "Internet" toggle: scope `tools_enabled` to a clear security boundary in the GUI | `already-done` — landed in 0.2.9-dev. **Re-home first**: the "separate toggle for MCP tools later" aside belongs in brief 04 and would otherwise die with the item. |
| Indexing a large corpus is silent for minutes, and reads as a hang | `already-done` — 0.2.8 added the INDEXING indicator. |
| Idle throttle for Librarian | `already-done` — 0.2.8. |
| Enable HTTP response compression on raven-server | `already-done` — implemented. |
| Hybridir: cover the edit-queueing layer with tests | `already-done` — covered at unit level. |
| EU AI Act Article 50 (transparency) compliance | `already-done` — both halves shipped, scoping brief closed. |
| Tokenization is dominated by per-call overhead | `already-done` — batched since filing. Filed 2026-08-06, fixed within days. |
| Triage CLAUDE.md style conventions: global vs project-specific | **`hold`** — see §5. |

## 2. SUPERSEDED, ANSWERED, MOVED

| Heading | Disposition |
|---|---|
| Headless scaffold mode for `ai_turn` (scriptable agent layer) | `supersede` → brief 15. Citation fix in flight, §4. |
| Lazy `api.initialize` in `llmclient` and `hybridir` | `supersede` → brief 15 Part 0. Same precondition. |
| Ligature mojibake in PDF-extracted text | `supersede` → `briefs/ligature-repair-brief.md`. Already a stub as of `62b41b8`; free to close. |
| RAG: rerank retrieved chunks and inject only the best few | **`decline` + `rehome`** — the textbook case for the Declined section: measured, and it made retrieval *worse*. Neural reranking was tested against multiple corpora alongside BM25/vector arm balancing and distribution-shape detection; all of them lost, and BM25+vector+RRF stood. Rehome the design prose to `investigations/retrieval/` first — check what `REPORT.md` already covers and move only what it does not — then decline with a one-line reason pointing there. The item already carries *"Measured and rejected, 2026-08-06. Retained for the design, not as a plan."* |
| AMD GPU (ROCm) support audit | `reword` — `raven/common/image/lanczos.py`. |
| CLAUDE.md: rephrase DPG pitfall #5 to avoid Claude thinking loops | `reword` — `dpg-notes.md`. |
| Audit and slim down project CLAUDE.md | `reword` — same rename. |
| Datastore scaling: a single `data.json` won't hold years of chats | `reword` — `chat.json`, `<datastore>.sidecars/`. |
| Librarian has no periodic autosave | `reword` — same; names `chat.json` and `state.json`. |
| Reasoning traces with indented bullets mis-render | **`supersede`** → `briefs/researchers-night/markdown-block-rendering-brief.md`, which absorbs it as step 4. Its cause is *correct* — the four-space indented-code collision — and its evidence is worth re-homing rather than summarizing: the font-tag half of its grep eliminates a real confound (Raven adds those at render, so their absence in stored text is informative), while the backtick half only rules out stray markup for the traces sampled. Keep both, and keep the distinction. Still needs the `chat.json` reword if any prose survives here. |

**The four `chat.json` rewords are one mechanical sweep, not four decisions.** Grep the whole file for
`data.json` and `.images/` and fix every hit, including ones no verdict flagged — the docs store's own
`fulldocs/data.json` is *not* affected and must be left alone, which the sweep already caught the reference
checker getting wrong.

## 3. Merges

| Keep | Fold in | Why |
|---|---|---|
| No way for the user to attach a document from a URL | Attach an image from a web URL (paste-URL path) | One absent affordance, differing only in what gets fetched. Sweep flagged it. |
| Holding the chat view's scrollbar does not hold your place | the scroll-jump item (per sweep batch 5) | Same missing prerequisite: a per-frame render-thread hook in `animation.py`. |
| **Markdown ATX headings don't render in the chat view** | Fenced code block support; Markdown tables | All three `supersede` → `briefs/researchers-night/markdown-block-rendering-brief.md`. One cause, and the cause named in all three is wrong. See §6. |

## 4. Before closing anything: two citation hazards

The sweep generalized one rule — **grep the briefs for an item heading before deleting it**, because a
document that is pruned cannot be cited by one that is kept. Applying it turned up a second instance
immediately, with a different failure mode.

**Brief 15 *cited* the items it supersedes by line number**, at four places (`:1825`, `:1474`, `:1813`).
Those numbers were stale the moment the metadata pass shifted every line, and by 2026-08-10 all three
resolved to *different items* — `:1825` had landed on *Extract `raven.common` into an upstream library
("corvid")*, a real heading and the wrong one, which is the dangerous kind of stale. **Fixed in flight**;
recorded here because the shape recurs. Brief 15 carries its own reasoning — unlike the ligature brief, it
restates rather than points — so nothing is lost by closing the items themselves.

Worth stating as the general form, since this is now twice in one day: **cite by heading, never by line
number.** A heading survives insertion above it; a line number does not survive anything.

## 5. Held: the CLAUDE.md growth check

STALE is correct — the global-vs-project split is done. But the rider is that CLAUDE.md files grow without
bound and want a periodic re-check, and deleting the item discards the only record of that.

What it becomes is settled (a recurring growth check). **Where it lives is Juha's open decision**, with the
dehydration pass as the natural pairing — which would put it in the fleet's global config rather than in
Raven's tree. Do not act on this row.

## 6. The Markdown chain — now `markdown-block-rendering-brief.md`

The sweep's largest finding, and it changed the exhibit plan rather than a table row. **Written up as
`briefs/researchers-night/markdown-block-rendering-brief.md`**; what follows is the summary the four
superseded items point at. The brief is the live tracker.

**Two barriers, both in Raven's own code, and the vendored renderer is behind both.** `_render_text` wraps
every paragraph as `<font color='...'>{text}</font>`, which makes the whole thing a CommonMark paragraph
containing inline raw HTML — and a heading is a block construct, which cannot occur inside a paragraph.
`_render_text_paragraphs` splits on single newlines, so a construct spanning lines cannot form at all.
Inline formatting is unaffected by either, which is exactly what made "the renderer must not support
headings" the natural conclusion.

The renderer maps `<h1>`–`<h6>` and has `MessageEntityPre`. Only `table` is a genuine gap.

It has since grown to seven steps, two of which were absorbed from other items because step 1 made them
cheap. Read the brief rather than this summary; the outline below is only enough to know why four items now
point at one document.

**The core, and the streaming path is a non-goal:**

1. Add a colour parameter to the vendored renderer (`add_text` currently takes only `markdown_text, wrap,
   parent, pos, tag`; colour exists internally via `font_attributes.Font`, and the `add_text_bold` family
   already takes a `color` kwarg). Stop wrapping in `_render_text`. **This alone fixes headings in both
   paths** — a heading is single-line, so it survives the split; it just cannot survive the wrapper.
2. Remove the dead inline-`<think>` handling. Its own comments call it *"leftover from the pre-June-2026
   inline handling; slated for removal"*. It is the only reason the splitter exists.
3. `_render_text_paragraphs` stops splitting. Fixes fenced code and multi-line lists.
4. Tables — the one real renderer gap, now optional rather than blocking.

**Streaming is untouched, and needs to be.** `_render_text_paragraphs` is called only from
`DPGCompleteChatMessage`; streaming uses `replace_last_paragraph` and re-renders only the last paragraph per
chunk, and at turn end the streaming message is demolished and a fresh complete message built — so a full
re-render at completion already happens today. Mid-stream, a fenced block stays literal until it completes,
which is correct: a half-arrived fence has no closing delimiter and nothing correct to render.

## 7. Settled separately today: the PyPI name

`decline` the item **Decide the public name**, with the decision and its reasoning, because "why not just
`raven`?" will otherwise be asked again.

**Decision: distribution `raven-lab`; import package stays `raven`.**

- `raven` on PyPI is Sentry's legacy client — 187 releases, none yanked, still installable, and its wheel
  ships a top-level `raven/`. So the index name is squatted by a tombstone and cannot be had.
- The import collision is real but confined to shared environments. Raven installs into a venv, as ML/AI
  applications generally must — ComfyUI and ooba ship the same way — and Sentry's client is a web-application
  error reporter whose audience does not overlap. **Make the venv requirement prominent in the README**;
  that is the mitigation.
- `raven-lab` → `raven` is a qualified form of the same name, not an opaque mapping like `cv2` from
  `opencv-python`. A reader seeing `import raven` can guess where it came from. The qualifier exists only
  because the index forced it.
- *lab* because Raven is one: a repo for experimental research prototypes across AI, LLMs and HCI, currently
  applied to literature management. That reading survives a change of application area, which the
  "lab computer" framing in `briefs/design/product-identity-sketch.md` explicitly may not.

**Follow-on, not decided**: `corvid` for the eventual `raven.common` extraction is taken on PyPI too. The
same qualified-form move is available.

## 8. New item to file: `ruff` does not check indentation, and `flake8` did

Raised 2026-08-10; **not currently tracked anywhere** — no hit for `ruff`, `flake8`, `pycodestyle` or `E1xx`
in `TODO.md` or `TODO_DEFERRED.md`. File it as a new deferred item with the metadata line, text along these
lines:

> ## The `flake8` → `ruff` migration dropped indentation checking
>
> *Cluster: ? · Cost: S · Gate: — · Filed: 2026-08-10*
>
> `[tool.ruff.lint] select = ["E", "W", "F", "SIM"]` looks like it covers pycodestyle, and does not: the E1
> (indentation) family splits in two, and ruff ships neither half by default.
>
> **E101 and E111–E117 exist but are preview-gated** — the rule pages say so explicitly, and Astral
> recommends against them alongside their formatter, on the grounds that the formatter makes them redundant.
> Recoverable now: add them to `select` and pass `--preview` in CI.
>
> **E121–E131, the continuation-line family, is not implemented.** astral-sh/ruff#4666 (2023) reports E124,
> E125, E128 and E129 passing silently under ruff while pycodestyle flags them; the community PR
> astral-sh/ruff#13585 to add the remaining E12x rules describes itself as incomplete and a work in progress.
> This is the half that matters here — continuation lines aligned to the opening bracket are the house style,
> and E127/E128/E124 are precisely the rules that police it.
>
> **Options, in order of preference:**
> 1. Enable the preview E11x rules in ruff. Free, no new tool, recovers the basic-indentation half.
> 2. Run `pycodestyle --select=E12` as a second CI pass for the continuation-line half. Note that `flake8`
>    is *still* a declared dev dependency (`pyproject.toml:153`) while CI runs ruff only — a residue of the
>    migration — so this is closer to finishing the migration than to adding a tool. Prefer `pycodestyle`
>    directly over `flake8`, which is only a runner around it.
> 3. **Not** adopting a formatter. Astral recommends it; **fleet policy is a hard no** (Juha, 2026-08-10),
>    covering `ruff format` and `black` alike. The reasoning is Goodhart's: a formatter optimizes a proxy
>    (uniformity) for a goal (readability), which pays when the alternative is unbounded variance across a
>    large team with fast turnover. Solo, the variance is already bounded by one person's judgment, so the
>    proxy buys nothing and costs the cases where layout carries meaning — aligned continuation lines,
>    math-heavy comment blocks, tabular literals. The only acceptable formatter would be an in-house one.
>
> **Adopt the E12x rules for the right reason.** They are a formatter's opinion in linter clothes, worth
> having here because they happen to encode the house style — not because pycodestyle is authoritative.
> Where the style deviates deliberately, the answer is `noqa`, not a change of habit. Worth stating in the
> config comment so the reason stays visible to whoever reads it next.
>
> **Measured 2026-08-10, before any of this lands**: 149 violations across 24 files — 81 E127, 64 E128,
> 4 E126, all in the visual-indent family. Normalized by lines of code the density is **2.77 per kloc in
> `tests/` against 1.23 elsewhere, a 2.25× ratio**, which is what the drift hypothesis predicts: tests are
> where CC writes most and where a human editor's flycheck never runs. Suggestive rather than conclusive —
> test code is call-heavy and so has more continuation lines to get wrong — but it points the right way.
> Top files: `test_textfilestore.py` (31), `chat_controller.py` (29), `test_layout_math.py` (14),
> `scaffold.py` (13). Small enough for one focused pass; `autopep8` fixes E12x mechanically, though on a
> hand-aligned codebase its choices want reviewing rather than trusting.
>
> Tidy-up either way: decide whether `flake8` stays in dev dependencies, and say why.

## 8b. New item to file: a canary for linter configuration

Raised 2026-08-10, out of the `ruff` finding, and general beyond it.

> ## Assert the linter actually runs the rules we rely on
>
> *Cluster: ? · Cost: S · Gate: — · Filed: 2026-08-10*
>
> `select = ["E", "W", "F", "SIM"]` reads as covering pycodestyle's error rules and does not enable the E1
> indentation family at all. Nothing warned: no error, no skipped-rule notice. The lint passed and the
> checks were not running. A configuration that is silently narrower than it looks is indistinguishable from
> one that is working.
>
> **Guard: a fixture file carrying one deliberate violation per rule family we rely on, and a test asserting
> the linter reports every one.** Drop a rule from the config and the canary fails loudly instead of the
> codebase quietly drifting.
>
> Implementation notes:
> - The fixture must be excluded from the normal lint run, or CI fails permanently. Name it so pytest does
>   not collect it either — `lint_canary_fixture.py`, not `test_*.py`.
> - **Invoke the linter exactly as CI does**, same flags and same config file. A canary that shells out
>   differently can pass while CI's real invocation has drifted, which reproduces the original bug one level
>   up.
> - Assert on the *set* of codes reported, so a dropped rule shrinks the set and names itself in the failure.
> - Keep it minimal: one violation per rule whose loss would matter, not coverage of the rule set.
>
> The same shape recurred three times on 2026-08-10 — two sweep verdicts reached by grepping Markdown
> vocabulary against a module that switches on HTML tag names, a `line_atributes` typo that makes a grep for
> `attributes` skip one of three sibling modules, and this. **An absence is only evidence if the check could
> have found the thing.** The canary is that rule applied to tooling.

## 8c. New item to file: wake-word voice input — `RN2026`

Raised 2026-08-10. **Goes in `TODO.md`'s STT / voice section**, beside the two existing `[High]` demo-facing
entries (in-room tunable silence/autostop, and the input-language combobox), not in `TODO_DEFERRED.md`.

> **[High]** Wake-word trigger for voice input. Demo-facing (Researchers' Night, 2026-09-26).
>
> The exhibit case: a visitor speaking directly to Aria reads very differently from an operator typing on
> their behalf. Higher demo value than anything else in this section, and the only item here that changes
> who is talking rather than how well it is heard.
>
> **The architectural cost is continuous capture.** `Recorder` is `start`/`stop` on demand, and `pvrecorder`
> is a single device handle, so a wake word cannot simply open a second stream. It needs one always-on
> capture fanned out to three consumers: the VU meter, the detector, and — once armed — the recording
> buffer. `connect_vu_readout` is the existing precedent for a consumer, so the shape is established; what
> is new is that there is more than one.
>
> **`pvrecorder` itself is fine — this is about a *different* Picovoice package.** Checked 2026-08-10:
> `pvrecorder` 1.2.7 is Apache-2.0 with **no dependencies at all** — pure PCM capture, no key, no account, no
> network. Nothing has changed and nothing needs replacing. The AccessKey regime applies to Picovoice's
> *inference* engines (Porcupine, Cheetah, Leopard, Rhino, Orca).
>
> Note the tell, since the licence field does not carry it: `pvrecorder` and `pvporcupine` declare the
> *identical* Apache-2.0 classifier, and what distinguishes them is that `pvporcupine` 4.0.3 depends on
> `requests`. A wake-word engine that runs entirely on-device has no reason to need an HTTP client; that
> dependency is the activation call, visible in the package metadata. **For anything claiming to run
> locally, read `requires_dist` before the licence field.**
>
> **Engine choice, and the obvious one is wrong here.** `pvrecorder` being Picovoice's makes `pvporcupine` the
> natural technical fit — same frame conventions, designed to be fed from it. But Porcupine has required an
> AccessKey and online activation since v2.0, its free Console tier is evaluation-only, and a Picovoice user
> is allowed **one unique device**, with reports of containerized runs registering as a new device and
> locking the account out. Phoning home for activation, bound to a machine, is a single point of failure on
> exhibit night — the kind that cannot be debugged with a queue of visitors waiting. It also sits badly
> against the project's declared license expression, which is a list of actual licenses rather than a
> service agreement.
>
> `openWakeWord` is the usual open alternative — ONNX, on-device, no key. **Verify its license and its
> custom-word training path from the project itself**; the readily-found comparisons are competitors'
> marketing pages and should not be cited for it.
>
> **Intended approach: build it, using the STT already present** (Juha, 2026-08-10). A ring buffer of recent
> audio, gated by VAD or the existing energy threshold, transcribed in short windows and matched against the
> wake word. This removes both risks above — no licensing question, and no custom-model training for "Aria",
> which is a non-stock word under every engine and was the schedule risk worth sizing first.
>
> **Two interaction styles, and which suits this audience is an open question — test both** (Juha,
> 2026-08-10). They are different HCI, not one being a better implementation of the other.
>
> - **Two-phase**, the *Star Trek* form: "Aria" — cue — "what's the atomic number of hydrogen?" The cue is
>   *feedback*: the visitor knows they were heard before committing to a question, and failure is legible —
>   no cue, say it again. Also cheaper technically, since detection and transcription separate cleanly with
>   no query to reconstruct from the buffer.
> - **Single-breath**: "Aria, what's the atomic number of hydrogen?" Natural, no convention to learn, and it
>   falls out of the ring buffer at no extra cost — when a transcription *begins with* the wake word, the
>   remainder of that utterance is the query.
>
> The considerations pull opposite ways and the audience decides it. Strangers with no model of the system
> arguably need feedback more than naturalness: with single-breath there is nothing to look at until the
> answer starts, so an unsure visitor repeats themselves mid-transcription and corrupts the query they
> already gave. Against that, a convention has to be explained, which at an exhibit means signage or the
> operator saying it each time.
>
> **The confirmation channel is worth testing separately from the style, and it wants to be both** (Juha,
> 2026-08-10). An audio cue can be missed in a loud room; a visual one is missed by anyone not looking at
> the avatar, which in a lab is most people. **Redundant channels for one signal — the same argument
> `:2816` makes about ok/error flashes**, so state the reasoning once and cross-reference.
>
> **The visual side is buildable with what THA3 already has**, with three pieces of work:
>
> - **Head and eye morphs** point the character at the user. The complication is the idle animation: it
>   changes body and head angles continuously, so the morph values that mean "looking at you" move with it.
>   An earlier draft of this note argued the compensation was arithmetic: the idle animator knows its own
>   angles, so composing against them should suffice. **That assumed the transform decomposes, and THA3 is a
>   neural net, which guarantees nothing of the sort** (Juha, 2026-08-10). The eye morphs' effect may depend
>   on the head and body morph values rather than adding to them — an interaction term, not a scale factor —
>   in which case composition is wrong rather than imprecise. The animator knows the values it *sends*, not
>   what the net does with them.
>
>   **Measure the error before characterising the mapping.** The question that decides the work is not "what
>   is the mapping" but "is the naive composition's error visible" — gaze tolerance for *looking at you* is
>   generous, and a few degrees reads fine. So: implement naive composition, run the idle animation through
>   its range against a fixed gaze target, screenshot at the extremes, and look for visible drift. Minutes,
>   not a sweep. If it holds, skip the calibration. If it drifts, the sweep is warranted and you already know
>   from which direction — which tells you whether a correction function of head angle suffices or a full
>   table is needed. Pose editor plus parameter sweep plus screenshots is the instrument for that second
>   stage.
> - **animefx** can put the visible "\ | /" over the character's head. It currently fires only on emotion
>   changes, and a wake trigger is not an emotion change — so the trigger surface wants generalizing from
>   "emotion changed" to "named effect fired", which is small and pays for itself the next time any event
>   should drive an effect.
> - **Per-character opt-in is a requirement, not a nicety.** The effect looks right on Aria and wrong on the
>   researcher DT, which has no animefx configured. So the effect set belongs in character config rather
>   than being global.
>
> **Two-tier detection model — record, do not schedule.** A small model for detection with the full one
> after trigger would cut continuous-transcription cost, at the price of a second resident model. Worth
> looking at once VAD gating is in and its saving is measured; gating may make the tier unnecessary.
>
> The ring buffer earns its place under either style: with two-phase it still covers the visitor who starts
> talking before the trigger has registered.
>
> **The risk moves rather than disappearing, and it lands on the exhibit's exact condition.** Whisper
> hallucinates fluently on noise — a crowd murmur transcribes to confident text. A purpose-built KWS engine
> is trained against hard negatives precisely so it does not; a general ASR model has no such training. So
> this route trades a licensing problem for a false-accept problem, in a loud room, where the spurious
> trigger mid-answer is the failure that looks broken. Mitigations to design in from the start: require the
> match at the *start* of the utterance, require energy above threshold, require a minimum utterance length,
> and use whatever no-speech signal the model exposes.
>
> **Cost competes with what the demo needs.** Continuous Whisper on a GPU also running the LLM and THA3 is
> not free. Gating on VAD or the existing silence threshold means detection runs only while someone is
> actually speaking, which is most of the saving; a small detection-tier model with the full one after
> trigger is the next step, at the cost of a second resident model.
>
> **Interacts with the input-language item.** A Finnish/English audience means the wake word must be
> recognized under both, and Whisper's rendering of "Aria" may differ by decoding language. Worth testing
> both before the day, and another reason to design the three STT items together.
>
> **Room constraint, shared with the two items above.** An open-doors evening is loud, and *false accepts
> are worse than false rejects*: a spurious trigger mid-answer looks broken to a watching visitor, while a
> missed trigger just means saying it again. So it wants the same tune-it-in-the-room control the
> silence-threshold item specifies, and a push-to-talk fallback that can be switched to on the day without
> a restart. Design the three STT items together — they share the constraint and the GUI surface.

## 8d. Record a decision that exists nowhere: Kokoro, torchaudio, and the Python cap

Raised by CC 2026-08-10. **Not a new item** — `TODO.md:766`–`768` already carries the full analysis. What is
missing is the *decision taken on it*, which lives only in conversation.

**Decision: keep both, do not act now.** Kokoro stays as the TTS engine, torchaudio stays as a dependency,
and Raven's `requires-python` stays capped at `<3.13`.

**Why it is not urgent.** The cost of keeping Kokoro is being pinned below Python 3.13. That only becomes
forcing when 3.12 goes end-of-life, in October 2028 — two years out, during which agentic build-out
continues. Charted rather than urgent.

**The escape route named in `TODO.md:767` has decayed, and this is the part that needs writing down.**
The plan was `torchaudio.functional.forced_align` to recover word timings from synthesized audio — and the
technical objection was properly tested: `lipsync.build_phoneme_stream` already splits each word's span
linearly across its phonemes, so Raven has never used phoneme-level timings and needs no phoneme-aligning
model. That analysis stands.

What has changed is the dependency. **torchaudio stopped shipping** (checked on PyPI 2026-08-10): 2.11.0 on
2026-03-23, released the same day as torch 2.11.0, and nothing since — while torch has shipped 2.12.0
(May 13), 2.12.1 (Jun 17) and 2.13.0 (Jul 8). Because torch and torchaudio minor versions must match — the
constraint `TODO_DEFERRED.md:1585` documents — **adopting torchaudio now would pin torch to 2.11.0**. That
trades the Python cap for a torch cap, which is the worse of the two: torch pins drag CUDA, THA3, Whisper
and the embedding stack along with them.

Note the shape of the miss, because it is cheap to avoid and was not part of an otherwise thorough
feasibility analysis: the 2026-07-28 note was written four months into torchaudio's silence and asked every
question about the *technique* and none about whether the dependency was still being released.

**Whisper timestamps were considered and rejected** (Juha, 2026-08-10): too slow, and it hallucinates at the
tail. The speed objection is the decisive one — alignment sits in the interactive path, so seconds per
utterance delays speech onset on every reply, whatever the quality.

**The technique survives; only the library was the problem.** `torchaudio` supplied two things: a wav2vec2
CTC acoustic model, and `forced_align`, which is Viterbi over CTC log-probabilities against known text. The
model is available from `transformers`, which is already present transitively via `sentence_transformers`.
The alignment step is a dynamic program over a lattice — small, well-specified, and squarely in-house
territory for this codebase, which already carries a custom xdot renderer and a GPU Lanczos scaler for
similar reasons.

So the escape route to re-scope is **CTC forced alignment with the alignment step implemented rather than
imported**, not a hunt for a replacement library. Worth a probe before it is needed rather than after.

**Record the trigger, not only the date.** October 2028 is the *latest* the cap can bite, not the earliest.
Any dependency that comes to require 3.13+ brings it forward, and **torch is the likeliest candidate** — at
which point Raven is wedged between torch and Kokoro with no room to move. A date invites forgetting; a
trigger keeps the check cheap. Concretely: whenever a dependency bump is blocked by `requires-python`, that
is this decision coming due, and the forced-alignment route is what to reach for.

Write this into the existing item rather than creating a new one, so the analysis and the decision sit
together.

## 8e. New item to file: drop the torchaudio dependency — it is one function

Raised 2026-08-10, out of §8d. **Higher priority than it looks**, because torchaudio is already a hard
dependency rather than a future one, and it is silently pinning torch.

> ## Replace `torchaudio.functional.resample`, and drop torchaudio
>
> *Cluster: ? · Cost: S · Gate: 0.2.9 · Filed: 2026-08-10*
>
> **torchaudio has stopped shipping.** Last release 2.11.0 on 2026-03-23, the same day as torch 2.11.0;
> torch has since shipped 2.12.0, 2.12.1 and 2.13.0 with no counterpart.
>
> **The pin is real and invisible to the resolver.** torchaudio 2.11.0 declares *no* `requires_dist`, so the
> torch pairing is a compiled-ABI constraint rather than a declared one: pip will install it beside torch
> 2.13.0 without complaint and fail at load with a missing-symbol error — the same class of failure
> *"torch / torchaudio CUDA version alignment on fresh installs"* already documents. `pyproject.toml:77`
> already pins `torchaudio==2.11.0` exactly, so Raven is effectively held at torch 2.11.x with nothing in
> the metadata saying so.
>
> **The exposure is one function.** Production use is `raven/common/audio/resample.py`, a thin wrapper over
> `torchaudio.functional.resample`. Everything else is two tests behind `pytest.importorskip`.
>
> **The replacement is already installed.** `scipy>=1.14.0` is a dependency (`pyproject.toml:67`, for
> Visualizer's KDTree), and `scipy.signal.resample_poly` is polyphase resampling of the same family. If its
> quality is not enough, `soxr` is the dedicated alternative — small wheel, no torch — but try scipy first
> since it is free.
>
> **The one property that would be lost** is device-agnosticism: the wrapper's docstring says it follows the
> tensor's device, and scipy is CPU-only. Almost certainly not load-bearing — speech-length audio resamples
> in milliseconds on CPU, and the call sites are Whisper input at 16 kHz and TTS output around 24 kHz — but
> measure rather than assume. `raven/common/audio/tests/test_resample.py` already exists and already works
> to a tolerance (it allows ±2 samples of rounding slack), so a backend swap is testable against what is
> there.
>
> **What it buys: newer torch.** `pyproject.toml:75` pins `torch==2.11.0`, and **that pin was added on
> 2026-08-10 because of torchaudio** — it is this problem already written into the build, not pre-existing
> CUDA-index policy. torchaudio 2.11.0 is the last release and is compiled against torch 2.11.0, so torch
> cannot move while torchaudio is a dependency. Remove torchaudio and the pin has no remaining reason;
> torch 2.12 and 2.13 become available, and the `[cuda]` extra's version-alignment problem goes from three
> packages to two.
>
> *(Recorded because it was misread once already, on 2026-08-10, as a deliberate CUDA-alignment choice
> independent of torchaudio. The comment at `:72` describes the mechanism — pinned trio from the
> `pytorch-cu128` index — without saying which package forced it, which is what makes the misreading
> available. Worth a word in that comment.)*
>
> The Kokoro escape route also stops depending on a dead package.

## 8f. Per-item gates — batch 1 (exhibit-facing chat/GUI)

Ruled 2026-08-10. `Gate:` values to write into the metadata line.

| Item | Gate | Note |
|---|---|---|
| In-flight AI turn bleeds into a new chat | `RN2026` | Highest-frequency gesture of the evening |
| Librarian has no periodic autosave | `RN2026` | One long-lived process on the night |
| Render the streaming thinking trace inside a bubble from the start | `RN2026` | Part of hidden-thinking |
| Clickable chip has no hover cue | `RN2026` | Discoverability for strangers |
| Librarian doesn't check the backend has a model loaded | `RN2026` | Sent to CC with brief 15; `403fd5a` split `setup` into a backend probe + pure `configure`, so the natural home now exists |
| Chat composer scrolls sideways instead of wrapping | `RN2026` | Visitors type into it |
| Idle prefill fires even when the count is already exact | `RN2026` if the gate is as cheap as the item says, else `0.2.9` | The exact/estimate bit already exists and already drives `X%` vs `~X%`. Watch the second half — gating on "the HEAD's prompt changed since that reading" means tracking prompt identity, not HEAD identity, since injects carry a datetime and RAG matches move with the corpus |
| Librarian leaks its server-side avatar instance on abnormal exit | `0.2.9` | Won't fire in practice; on-brand robustness |
| Chat view scroll position jumps back down while writing | `merge` → *Holding the chat view's scrollbar…* | App-side faults fixed; the remainder is the ImGui fraction-vs-absolute drift, same mechanism. Confirmed distinct from the 2026-08-11 clamp-timeout fix, which does not settle it |
| Holding the chat view's scrollbar does not hold your place | `0.2.9` | Absorbs the above |
| Colorblind-safe status signaling | `0.2.9` | Real for users installing on their own machines; nothing in the demo relies on a red/green distinction |
| Modernize the Librarian system prompt / character card | `next` | Post-RN. Prompt changes need soak time, and there are more important things before the exhibit |
| TTS reads arXiv IDs digit by digit | `next` | Needs design; scope *what else* is mispronounced first |
| Help card has no room for more entries | `next` | Redesign, post-exhibit. Pair with the hotkey-discoverability audit, which touches the same card, and note brief 16 will want an entry plus a hotkey-shaped commit gesture |
| Context-window budgeting / conversation compaction | `next` | More important once longer chats become common |
| No-avatar mode with the chat tree in the panel | `next` | With road mode; brief 16 settle-item |
| Streaming thinking shows as gray for models that pre-fill `<think>` | `0.2.9`, unlinked from the think toggle | See below |
| Remove the dead inline-`<think>` handling | `supersede` → Markdown brief, step 2 | |
| Make the Librarian chat composer resizable | `—` | |
| Hotkeys assume a keyboard layout | `—` | |

**Two corrections worth keeping**, since both were reasoned wrongly first:

- **The gray-vs-blue item is not linked to the think toggle.** Three cases, not two: a backend sending `reasoning_content` as its own channel (LM Studio) never exercises tag inference at all; single-channel with both tags works; single-channel with a template-prefilled open tag is the bug — QwQ-style, and Qwen3-2507 *as served by ooba*. LM Studio is the demo backend, so this is not exhibit-relevant. And the toggle prefills into the *prompt*, after which the model generates plain content and the parser is correctly in `_PS_TEXT` — so the toggle needs no parser change. The fix is moving `chatutil.scrub`'s orphan-close recovery into the parser. Arguably `—` until ooba is run again.
- **The two character-drop items are probably one cause, in the font layer.** *Chat view drops a character mid-message* and *`dpg_markdown` intermittently drops a single letter* both eliminated text-layer explanations rigorously — datastore verbatim, `mistletoe` output intact, glyph present in the TTF per `fontTools`. But *present in the TTF* is not *present in the rasterized atlas*, and neither checked that layer. `dpg_markdown` loads fonts at runtime via `markdown_add_font_callback` per size and family (italic being a separate font), and a glyph that fails to rasterize renders blank while keeping its advance width — which predicts the observed signature exactly: right-sized gap, variant-specific, intermittent across launches as atlas packing varies. **Merge, and *start* the investigation at the atlas** without confining it there.

## 8g. Per-item gates — batch 2 (file order)

Ruled 2026-08-11.

| Item | Gate | Note |
|---|---|---|
| The DPG tests we have never run in CI | `0.2.9`, soon | **Smaller than it looks: the split already exists.** `conftest.py` marks `test_focus_semantics` `gui` and skips it unless `--run-gui`, precisely because it maps a window. **Counted 2026-08-12 — both the item and an earlier draft of this row were wrong**: there are *seven* modules and 77 tests, of which `test_focus_semantics` (5) is gui-marked, leaving **72 tests across six modules** that drive an unmapped viewport — `test_animation` (27), `test_filedrop` (17), `test_layout_math` (14), `test_utils` (7), `test_messagebox` (5), `test_fontsetup` (2). The item says 50 tests in four modules and names `test_viewport_math`, now `test_layout_math`; this row repeated both errors and missed `test_filedrop` and `test_fontsetup` entirely. **Fix the item's numbers when updating it**, or delete it by the outcome of the experiment it prescribes. So this is adding `dearpygui` to `requirements-ci.txt` and letting the existing marker do the rest. Cross-link to the linter-canary item (§8b): a skip is indistinguishable from a pass in the summary line, which is the same defect `requirements-ci.txt`'s own header documents twice (the trafilatura drift, and the `qoi` gap at 1606 tests against 1603) |
| The avatar upscaler offers bilinear and bicubic, but not Lanczos | `0.2.9`, early | Quick win. `raven.common.image.lanczos` is GPU-enabled, already a dependency, and takes `(B, C, H, W)` — the shape the bypass branch already juggles with `.unsqueeze(0)` / `[0]`. Irritatingly non-uniform to have a fully compatible engine and not wire it in |
| Move the avatar backdrop onto `image.utils.fit_cover` | `0.2.9`, early | **Group with the Lanczos item** — both are "the constellation grew a shared implementation and a call site predates it". One session closes two heads. Note there was some code-path reason it was not refactored at the time; find it before assuming it is a drop-in |
| `SmoothScrolling` commits during construction | `0.2.9` | Bad design, fix it. `self.start()` at `animation.py:701` is still the tail of `__init__`; the 2026-08-11 follow-tail fix went around it rather than into it |
| Two adopted directories ship without their licence text | `0.2.9`, soon | Plain compliance failure, and the item says it wants fixing ahead of and independently of the `pyproject.toml` work. **Partly done already**: a round of missing-LICENSE fixes landed a week or two prior. Note the FontAwesome case — all fonts ship in one folder and its licence file is not named exactly `LICENSE`, so a filename-matching audit will report a false positive. Verify what actually remains |
| Smooth scrolling in Cherrypick too | `0.2.9` | A consistency failure |
| Make the DPG reference a skill | `0.2.9` | Force multiplier for CC |
| The 8/3 pass: bare DPG margins should name themselves | `hygiene-sweep` | See the rationale below |
| Updating the vendored FontAwesome means both files | **close** | *Answered, not open.* The measurement refutes the claim it was filed against: header and shipped fonts are exactly in sync — 1969 codepoints over 1395 distinct glyph names, all 1395 named, no glyph without a constant, and the 574-codepoint surplus is pure aliasing rather than icons waiting to be exposed. Remaining action: correct `CLAUDE.md`'s "outdated version" note, then close |
| The licensing story is accurate only in a subdirectory README | `0.2.9` | Gated by the first PyPI upload, alongside the name decision |
| Revisit `recenter_window`'s degrade-instead-of-raise policy | `next` | Future-proofing. Already works correctly in Librarian, and the operator drives the demo |
| Web status panel | `next` | First scope of a larger direction — see below |
| Browse *all* attachments in the datastore | `next` | |
| GUI: hardcoded stand-ins for values DPG has no getter for | `—` | Known-hazard record. Direction, if taken up: see the shadow-state note below — this is one instance of a general pattern, not a DPG quirk |
| `replace_last_paragraph`'s `dpg.mutex()` is disabled | `—` | Cause still unknown, consequence now doubly contained: `replace_last_paragraph` is the only clamp source, and the 2026-08-11 follow-tail fix made a clamp harmless by recording where the panel actually is |
| The subtitle translator silently drops `=` | `—` | Nothing available short-term — the cause is the neural translator. Switching models is a large effort for a small gain; take the easier wins first |
| TODO.md goes stale | *partly superseded* | This round's answer is the present brief plus the `## Already done` section. What survives is the **periodic** half — see §9, the release-procedure step |
| Make the canned AI greeting optional | `next` | For the exhibit, keep a greeting: a visitor at a blank chat has nothing telling them what this is or that it is ready. An explanatory one, or the current "How can I help you today?". The philosophically correct version — blank in config means no greeting node, and a new chat points HEAD at the system prompt — lands after RN, and **should be decided together with the system-prompt rewrite**, since both are one question about asserting identity at a model that does not need it |

### Why the hygiene-sweep cluster ranks higher than it used to

Under solo human development a stylistic inconsistency sits where it is. **Under agentic development it
propagates.**

Two framings of the same phenomenon (Juha, 2026-08-11), the second sharper. First: agentic coding is like a
hyperbolic PDE — errors travel along characteristics and do not damp. Second, mechanistically: it is
**distribution matching**. The existing code is taken as evidence of the intended style; if it has drifted,
new work is matched against a wrong distribution, and the drifted estimate becomes the next generation's
evidence. Uncontrolled, it drifts arbitrarily far.

**The two are the same thing at different levels of description**, not one weaker and one sharper. An
earlier draft here claimed transport preserves amplitude while drift compounds — a category error, comparing
the continuous equation against a discrete stochastic process. The analogy was never to the PDE but to the
*scheme*: each timestep injects local truncation error, and without dissipation those accumulate. Each code
generation is a timestep, and the per-step error is the imperfect match to the intended distribution.

That also names the restoring force properly: **lint rules and hygiene sweeps are artificial viscosity.** A
non-dissipative scheme accumulates error and develops spurious oscillations; dissipation is added
deliberately, known not to be physical, because the alternative is a solution that stops meaning anything.
Same trade as a style rule — not *true*, but a damping term that keeps the trajectory bounded.

Two consequences, which hold under either description:

- **Local review cannot detect it, by construction.** Each step is within tolerance of the state it was drawn
  from — that is what makes it drift rather than a defect. Diff review compares against the immediately
  preceding state, so it is structurally blind. Detection requires comparison against a *fixed reference*.
- **Fixed references are the restoring force, and that reranks them.** `CLAUDE.md` conventions and lint rules
  act continuously; the hygiene sweep acts episodically. Losing pycodestyle's E12x removed the only
  *mechanical* restoring force on continuation-line style — which is why §8's ruff gap and this cluster are
  one concern rather than two.

**Today's measurement is that prediction confirmed**: 2.25× E12x violation density in `tests/` against the
rest of the tree (2.77 vs 1.23 per kloc), which is the code the model writes most and the human reviews in
Emacs least. Drift concentrated where the restoring force was weakest, with an unaffected baseline in the
same repo to compare against.

That inverts the usual ranking of hygiene work, which is normally deferred precisely because it is local and
cosmetic. It is neither, now. Record this as the cluster's rationale.

### Shadow state: the general pattern behind the no-getter items

Stated generally (Juha, 2026-08-11), not as a DPG note. **When a stateful third-party API accepts writes but
exposes no reads, state cannot be recovered by asking — so become the sole writer and record what went in.**
A registry, and all writes funnelled through it.

Raven already has a worked example, including the failure mode. `commanded_y_scroll` is exactly this: a box
recording every scroll position written, because DPG's reported position cannot distinguish "we moved it"
from "the reader moved it". And the 2026-08-11 follow-tail bug was that shadow drifting from reality — the
record held a value the panel never took, because DPG clamped the write and the animation timed out waiting
for an equality that could never hold.

Three properties the pattern carries:

- **Record what was observed, not what was requested**, wherever the result is observable. That is precisely
  what the follow-tail fix changed, and it is the difference between a shadow that tracks reality and one
  that tracks intent.
- **It requires sole writership.** If anything else mutates the state, the shadow drifts — and a drifted
  shadow is worse than none, since confidently wrong beats honestly unknown and this gives the former. So
  all writes go through the registry, and a direct call to the underlying setter is a bug.
- **Measured defaults are version-fragile.** Initial state is not observable either, so it comes from
  measurement — and a measured default can become silently wrong on a library upgrade with nothing
  reporting it. That is the linter-canary problem again (§8b), and it wants the same answer: assert the
  measured value still holds.

### The web status panel is the first scope of something larger

Raised 2026-08-11. If the engine is live, a JS frontend to the whole stack follows: Librarian usable from a
phone on the same LAN, Visualizer as a web interface. Also where the world is — desktop apps are the
minority, and the mainstream expectation is browser-accessible or it does not exist. Adjacent to the parked
JS avatar client.

The status panel is the right first scope: read-only, no auth, smallest surface.

**One licensing consequence, and it is narrower than an earlier draft of this section claimed.**

`AGPL-3.0-only` covers raven-server — including the avatar, but *not* the postprocessor and upscaler — and
the pose editor (Juha, 2026-08-11).

**Corrected**: a web frontend does *not* inherit AGPL by calling the server. A client communicating at arm's
length over HTTP is a separate work, which is why Librarian — a Python client of the same server — does not
inherit it today, and a JS client is in the same position. Nothing about the frontend's licence follows from
this.

What changes is on the server side, and it is about *users* rather than scope. AGPL §13's source-offer
obligation runs to those interacting with the program remotely over a network. Today raven-server has one
user, who has the source, so the obligation is satisfied invisibly. Exposed on a LAN for other people, those
people become remote users and the offer has to be made for real — true regardless of what the frontend is
written in, and a property of raven-server having been AGPL all along rather than of the frontend existing.

**Not a gate on the work.** The deployment reality (Juha, 2026-08-11): remote users are either the same
person as the local user, or lab colleagues in a lab deployment. Nothing is served to external users over the
internet, and colleagues have repo access anyway — so the obligation is satisfied before it is made.

Do it regardless: a visible source offer from the server or the web UI, linking to the public repo. Cheap,
correct, and good advertising. Relates to the licensing item above.

## 8h. New item to file: open an existing chat datastore

Raised 2026-08-11. **Gate: `next`.**

> ## Librarian: open a chat datastore other than the configured default
>
> *Cluster: ? · Cost: ? · Gate: next · Filed: 2026-08-11*
>
> Librarian loads one datastore, fixed at `librarian_config.llm_datastore_file`. There is no way to open
> another from inside the app.
>
> **The motivation is brief 15.** With the scripting surface, a run can generate a datastore on a specific
> topic — scripted questions, the model's replies, in Raven's native format. Librarian already reads that
> format, so opening one turns it into the **results browser for scripted experiments**: branching, thinking
> traces and tool calls all rendered by the UI that already knows how, instead of read by squinting at JSON.
> That is a capability rather than a convenience, and it is the reason to do this rather than the
> general-purpose "open a file" argument.
>
> **Scripted datastores are self-contained, and the existing type distinction is what makes them so**
> (settled by Juha, 2026-08-11). Brief 15 gained in-memory attachment support so a scripted call can attach a
> full-text PDF, which raised the question of whether a generated datastore could end up referencing bytes
> that were never written. It cannot, and no new mode was needed:
>
> - A plain **`Forest`** carries *in-memory* sidecars (added 2026-08-11), so attachments work identically
>   within a run — only their lifetime differs. Nothing dangles, because nothing outlives the process and
>   there is no artifact to open later. A URL fetched into an in-memory store is lost on exit, which is the
>   caller's choice rather than a defect.
> - Persistence is opted into by handing the scripting layer a **`PersistentForest`**, which already carries
>   a sidecar directory and the machinery to fill it.
>
> So the sidecar concept is uniform across both and only the backing store differs, which means the
> attachment path does not branch on forest type — the same move the `Forest` / `PersistentForest` split
> already makes.
>
> *Interaction to keep in mind*: the per-document pass (proposed brief 17) over a large corpus would hold
> every document's bytes for the life of the run under in-memory sidecars. Two exits, and the second is the
> better default — persist to a `PersistentForest` when the run is worth keeping and resumable, or **reset
> between documents**, which is the shape `pdf2bib` already had. The reset is not only a memory fix: a map
> stage processes documents independently, so a fresh forest per item also gives isolation, with bounded
> memory falling out of the correct semantics rather than being arranged for.
>
> **Sidecars are a cache, not the record.** Scientific repeatability rests on the user's own files plus the
> script: an in-memory run reproduces by supplying the same files again, at the cost of reprocessing them.
> The content-hashed sidecar naming (`<hash>.original.<ext>`, `config.py:325`) means materialization is
> deterministic, so re-running writes identical sidecars rather than a second copy — dedup and reproducible
> output fall out of the naming rather than needing anything added.
>
> **The datastore claim is per-datastore, and read-only opens do not take one** (settled 2026-08-12). The
> interprocess claim added on 2026-08-12 — which stops a second app silently discarding the first's work,
> and releases on process end including a crash, so there is no stale lock to clean up — is keyed to the
> datastore rather than the app. So two Librarians on *different* datastores coexist, which is what makes
> this item workable at all. A read-only open writes nothing and therefore claims nothing: inspecting a
> generated run must not lock out the process still writing it, which is exactly the case a scripted
> experiment plus a Librarian window creates.
>
> **It is not just swapping a path.** The sidecars directory is derived from the datastore's name
> (`<datastore>.sidecars/`), so attachments move with it; HEAD, the chat view and any open graph view all
> need rebuilding; and the current datastore has to be persisted before the swap.
>
> **Both modes wanted** (Juha, 2026-08-11) — they have different uses. A generated datastore is a research
> artifact, and opening it read-write means a stray keystroke mutates a measurement, which is the concern
> `investigations/README.md` states about apparatus staying as it was run. But opening one to continue a
> scripted conversation by hand is a real use too.
>
> So the questions are how the mode is chosen and how it is enforced:
>
> - **Enforce at the datastore layer, not the GUI.** Disabling the send button is not read-only — autosave,
>   HEAD moves, sidecar writes and the cleanup flow all have their own paths. One gate low down in
>   `PersistentForest` is what holds; disabled controls are the affordance on top of it.
> - **Let the artifact carry its own default.** A flag in the datastore, set when a scripted run generates
>   one, means it opens safe regardless of who opens it or whether they remember — overridable at open time
>   for a deliberate edit. Better than an open-time-only choice, where the protection depends on recalling
>   what kind of file this is.
>
>   **Not a change to brief 15**, which is in flight and whose scope has already grown once. Recorded here
>   as a requirement on the producer side, to be picked up when this item is built: `raven.librarian.agent`
>   will need to set the read-only flag on the datastores it writes, with the caller able to choose and
>   **`True` as the default** — scripted output is a measurement unless someone says otherwise.
> - **The mode must be visible while open**, not confirmed once at open time. An invisible read-only either
>   gives false confidence or makes one hesitate to type in a datastore that is in fact writable.
>
> Read-only is also the cheaper path — no autosave, no persisted HEAD moves, no attachment writes — so the
> safer default is the smaller implementation, which is not usually how that goes.
>
> **Interacts with autosave** (`RN2026`). Once periodic autosave exists it will write to whatever is open,
> so it has to know whether the current datastore is writable. Cheaper to design that in than to retrofit
> it.
>
> **A new use site for `FileDialog`**, so it inherits whatever comes of *FileDialog: reduce per-use-site
> boilerplate*. A recent-datastores list follows naturally once swapping works, and both the graph view and
> *browse all attachments* are scoped to "the datastore" — they inherit whichever is open, with no extra
> work if the swap rebuilds cleanly.

## 8i. New item to file: agent skills for Librarian

Raised 2026-08-11. **Gate: `next`** — design work deliberately postponed; this records the idea and what was
already established about it.

> ## Agent skills for Librarian (natural-language workflows over the document database)
>
> *Cluster: ? · Cost: ? · Gate: next · Filed: 2026-08-11*
>
> The Agent Skills open standard (https://agentskills.io) — originally Anthropic's, now adopted across a
> range of agent products. A skill is a folder with a `SKILL.md` carrying metadata (`name`, `description`)
> plus instructions, optionally bundling scripts, references and templates.
>
> **Why it belongs in a librarian, given that Librarian is deliberately not a generic agent harness**: MCP
> and skills solve different problems, and the hype conflates them. **MCP is capability supply** — a plugin
> surface giving the agent tools. **Skills are procedure over whatever tools exist** — natural-language
> scripting on top of the building blocks. For a digital librarian that means the user can describe custom
> workflows over their own document database, without either party writing code.
>
> **Librarian already has stages one and two, under another name.** The standard loads skills by progressive
> disclosure: *discovery* keeps only names and descriptions resident, *activation* pulls the full `SKILL.md`
> when a task matches, *execution* follows the instructions. The first two are what the lorebook (brief 05)
> does — a small always-resident index, full text injected on match. So the mechanism largely exists and the
> question is what sits on top of it.
>
> **Stage three is where the line falls, and it is a line already drawn.** Executing a skill's bundled
> scripts would be third-party code running in-process, which is precisely the user-plugin system brief 15
> rules out ("MCP is tool *supply*: an external process offering capabilities inward. User plugins would be
> app *extension*: third-party code inside the process. Not wanted."). So the shape for Librarian is
> **instructions-only skills** — `SKILL.md` and reference material, no `scripts/`. A skill that needs a
> capability asks for it as an MCP tool, which is the division the two standards already imply.
>
> **Correction to an earlier draft of this note**, which said the shape for Librarian is instructions-only
> with no `scripts/`. That rested on brief 15's line "no plugin system, no workflow DSL, no orchestration
> layer" — see §8j, where the boundary is unpicked. The corrected position: **instructions-only by default,
> scripts behind an explicit opt-in**, treated like the trust decision that MCP servers and `tools_enabled`
> already are. No *GUI* plugin surface, which is the part actually ruled out and which the standard does not
> ask for.
>
> Open questions for the design session: how skills and lorebook entries relate (one mechanism with two
> front-ends, or two separate things); whether a skill can restrict itself to a corpus scope; and whether
> declining to run bundled scripts makes Raven's implementation non-conforming or merely a subset — worth
> checking the specification rather than assuming.

## 8j. Scope boundary: what Librarian is not, restated

Raised 2026-08-11, unpicking a line in brief 15. **Not an item — a note for whoever reads that brief.**

Brief 15 says Librarian is "not a generic agent harness, and it must not become one: no plugin system, no
workflow DSL, no orchestration layer". **The first clause is Juha's; the three prohibitions were Claude's
addition**, and the agent-skills idea is the first thing to test them. Checked 2026-08-11: that line is the
only place in the tree where the boundary is stated, so there is no pattern of the design bumping into it —
one line, four days old, first contact. Each of the three needs restating:

- **Plugin system.** *Partly wrong.* GUI plugins are off the table for now; other kinds are an open
  question, and a clean solution would be admissible. What is genuinely not planned is a plugin
  *ecosystem* — no centralized Raven plugin repository, no dynamic website; the scope is local-to-user or
  local-to-lab, and users sharing plugins on GitHub is fine. The security surface is the one Raven already
  has: MCP servers run arbitrary code, `webfetch` reaches the network, and `tools_enabled` already gates a
  security boundary in the GUI. Third-party code is a risk an informed user may want to opt into.
- **Workflow DSL.** *Stands, with a correction to the alternative.* The fewer idiosyncratic DSLs the better
  — but the answer is not "no scripting", it is **Python**, which technical users and AIs already know. A
  Turing-complete security hole, acknowledged as such; the same trust decision as `scripts/`, not a design
  objection.
- **Orchestration layer.** *Overreach, and possibly moot.* There is a class of problems multi-agent solves,
  and synthesizing across a collection of papers is plausibly one: a clean context per paper before the
  synthesis step would likely improve results and save context in the main chat.

  **But that case is already scoped, and not as orchestration.**
  `briefs/design/corpus-interrogation-sketch.md:122` has it as
  `summaries = map(summarize, docs); overview = synthesize(summaries)` — and records that `summarize`
  (1–3 sentence summaries with progress, ETA and caching over a whole dataset) is *shipped code that is
  currently switched off*, sitting in the importer. What is missing is `synthesize`. The sketch's own
  framing: this changes the size of the job from building a map-reduce engine to three smaller things —
  lift `summarize` into the library, add the reduce, let both run against a scope.

  The clean-context-per-paper property is what a map stage *is*. Naming it multi-agent orchestration would
  have imported an architecture for a problem that already has a smaller shape. Note also that brief 17's
  proposed per-document pass names this sketch's map stage as one of its three users, so the pieces are
  converging rather than accumulating.

  **But the general solution is the better fit, and it may not cost more** (Juha, 2026-08-11 —
  six impossible things before breakfast). The difference is what the per-document step *is*: map-reduce
  makes it a pure function, doc → summary, no decisions. Multi-agent makes it a *turn* — the sub-task can
  search the corpus, fetch a reference, follow a citation, notice the paper does not address the question.
  "Summarize this paper" is fine as a pure function; "what does this say about X, and does it contradict Y"
  is not. And by the aria-worthy criterion the pure-function version is knowledge left on the floor: Raven
  already handles the harder case, so refusing a sub-task the capability the main chat has is a visible
  lapse.

  Claude's claim that multi-agent is the larger job was wrong for this codebase, because the pieces already
  compose: `ai_turn` is the agent, a `Forest` is the scoped context, brief 15's scripting surface is the
  driver, and reset-between-documents (settled 2026-08-11) is the isolation. A sub-agent is a fresh forest,
  a scoped prompt, one `user_turn` + `ai_turn`, and a result. **Map-reduce is then the degenerate case** —
  a map whose agent makes no tool calls — rather than a cheaper alternative.

  Three things settled in passing, worth keeping:

  - **Not necessarily an in-memory forest.** An earlier draft said in-memory `Forest` is the clean context,
    which quietly frames a sub-run as a tool call with disposable internals. A sub-run's transcript is
    exactly what a researcher wants to inspect afterwards. Keeping it is probably not much more than a
    `PersistentForest` with a sub-agent-scoped sidecar directory.
  - **The main agent should be able to spawn sub-runs too**, via a builtin tool — not only the scripting
    surface. That is a capability decision rather than plumbing: scripting-only means the *user* decides to
    decompose, a builtin tool means the *model* can. Interacts with the tool-budget work, since a sub-run is
    one tool call costing many turns.
  - **Sub-runs are sequential**, Raven being a local system. That dissolves most of the UX difficulty an
    earlier draft claimed: twenty concurrent runs would need a novel display, while one at a time with a
    queue behind it is the shape of the existing INDEXING indicator. Transcripts arrive in order, too.

  What genuinely remains is UX and storage detail, and opening it is a design session of its own. Not now;
  not off the table.

## 8k. Per-item gates — batch 3 (file order, from ~`:1435`)

Ruled 2026-08-11.

| Item | Gate | Note |
|---|---|---|
| torch / torchaudio CUDA version alignment | `—`, re-evaluate after §8e | The documented `libcudart.so.13` failure *is* the torchaudio wheel, so removal likely closes it, leaving a torchvision note. **Two cautions on §8e, both Juha's**: (1) *removal may not be free later* — current use is `resample.py` only, called from `stt.py:169` for Whisper input, and nothing touches lipsync; but the forced-alignment route for phonemes-from-synthesized-audio would have used `torchaudio.functional.forced_align`. §8d already re-scopes that to acoustic model from `transformers` plus an in-house alignment step, so the bridge is buildable rather than importable — a real cost to state. (2) *"dead" was overclaimed* — a few months of silence is not conclusive. The sharper signal is that torchaudio **missed three consecutive torch releases** after a history of same-day pairing. Suggestive; hiatus still live |
| Untested but test-worthy modules in `raven.common` | `0.2.9` | Decent coverage matters. Same session as the DPG-tests-in-CI switch |
| Remaining server modules without a `MaybeRemote` | `next`, **scoped down** | Skip the avatar and it is much lighter |
| Consolidate remaining numpy/tensor/DPG image conversions | `0.2.9` if it fits | Third member of the `:389` / `:709` group — try the three as one session |
| Preload cache: 16MP image optimization | `0.2.9` | Concrete need: a large set of 16MP photos from ECCOMAS 2026. Gate on whether it actually affects speed |
| raven-cherrypick: low FPS with large images | investigate in `0.2.9` | Performance issues matter; investigate without promising a fix in that cycle |
| Audit typing: abstract params, concrete returns | `hygiene-sweep` | |
| Audit toolbar buttons for WidgetFlash acknowledgment | `hygiene-sweep` | |
| Faster PNG decoder | `next` | Wanted eventually |
| raven-cherrypick: export image sequence (QOI→PNG batch) | `next` | |
| Avatar settings editor: custom postprocessor chain ordering | `next` | **GUI work only** — the item says so: `crt-display.md` §0 establishes the backend has always supported multiple instances at arbitrary positions (`render_into` applies the chain positionally, `_priority` only sorts settings panels, `name` keys caches apart). What it unlocks is `scanlines` and `crt` coexisting at different diegetic layers, so more than convenience |
| Client-local avatar animator (licensing-bounded) | `next`, deprioritized, **plus a wording fix to the item itself** | The AGPL comes from THA3's detour through *SillyTavern-Extras*, not from Raven-side extensions. The **licensing documentation is already correct** — `raven/avatar/README.md:331` says the AGPL exists to comply with ST-Extras' original licence, and the `-only` analysis in the licensing item makes the inheritance load-bearing (the election has to come from upstream; Raven has no standing to relicense others' contributions). What misleads is *this item's own opening*: "THA3 upstream is actually MIT — so the AGPL tax comes from Raven-side extensions, not the model itself" is true about the model and invites the wrong conclusion about the source, since ST-Extras is a third possibility it does not name. **Reword that sentence.** Much of the animator — including the idle-animation logic — is Juha's; the server skeleton and the first animator version are other ST-Extras authors'. A full provenance audit is possible in CC if it becomes worth doing. **Consequence for the "fully-BSD Raven" goal the item states**: that is not a relicensing decision available unilaterally, it is a rewrite of the parts Juha did not author — which is what makes deprioritizing straightforwardly right rather than a judgement call. **What has already been relicensed, has been**: anything unilaterally relicensable is done, the postprocessor in particular. What remains derived is the *service skeleton* — the THA3 + distilBERT wiring and the `x-multipart-replace` video-streaming approach — built on and partly rewritten, but from ST-Extras. Note that "partly rewritten but derived" is the hardest position of the three: wholly inherited is clear and wholly rewritten is clear, while a substantially reworked file has no bright line where derivation ends. So the rewrite's *endpoint* is unfalsifiable without someone qualified saying it is now a different program, which is a stronger reason to defer than the effort estimate alone. Note OSS licences bind on *distribution*, so private experimentation is unconstrained — though that is a lot of building for something that could not be pushed to GitHub |
| Split `raven.common.nlptools` per backend | `—` | No opinion yet; worth keeping tracked |
| Uniform load-on-demand for server modules | `—` | Same |
| MPS (Apple Silicon) device synchronization | `—` | Importance unclear — worth establishing before scheduling |
| pillow-simd | `—` | No opinion yet |
| raven-cherrypick: further reduce idle CPU/GPU load | `0.2.9` | Already 80% → 20% of one core (2026-04-05); the remainder is the floor cost of `render_dearpygui_frame()` at ~12fps, ImGui resubmitting the whole UI each call. **Try the adaptive sleep ramp** — 80ms → 500ms over ~5s idle, snapping back on input — since it is small. Skipping the render call stays out of scope, the item flagging it risky because event processing is tied to it. **Add a second line of investigation** (Juha, 2026-08-11): all thumbnails are currently kept in VRAM, so a directory with many images builds a very large texture atlas, which may itself be slowing DPG down. That is a different mechanism from the idle-frame cost and may dominate on large directories — worth measuring before assuming the ramp is the whole story, and it plausibly bears on *raven-cherrypick: low FPS with large images* too |
| Extract `raven.common` into "corvid" | `next` or `—` | Not urgent. **Name settled: `corvid-lab`**, matching `raven-lab` — `corvid` is taken on PyPI, and the qualified form keeps the family |
| Robust public API auditing tool | `—` | Not important now, and **do it properly or not at all** (Juha): if built, build the general tool. Claude's suggestion to scope it down to brief 15's needs is declined. Note `unpythonic` is the only project in the fleet using macros, so mcpyrate's full difficulty applies to one consumer |
| Audit and slim down project `CLAUDE.md` | periodic | Pass done recently; this is recurring rather than open |

## 8l. New item to file: batch runs do not survive a backend that goes away mid-run

Raised by CC 2026-08-12, out of the model-loaded check (§8f). **Gate: `next`** — but see the convergence
note, which is the reason not to solve it standalone.

> ## Batch tools: LLM reconnect mid-run
>
> *Cluster: ? · Gate: next · Filed: 2026-08-12*
>
> The model-loaded work made `raven-pdf2bib` and `raven-importer` stop at *start time* on both failure
> states — unreachable, and reachable-with-no-model. The second was the one that most needed it: the backend
> answers, so nothing looks wrong until every extraction comes back empty several hundred documents in.
>
> **What remains, recorded as deliberately out of that brief's scope**: a backend that goes away *mid-run*.
> Every remaining document fails. Wording for all three states now lives in
> `llmclient.describe_backend_status`, printed by four frontends (two batch tools, `minichat`, the GUI
> tooltip), so the reporting side is already centralized — what is missing is the recovery behaviour.
>
> **Do not solve this standalone — it is brief 17's territory.** The three questions CC left open are that
> brief's scope exactly: how long to wait before giving up, whether to resume or restart, and what to do
> with the documents already written. `raven-pdf2bib` is already named as one of brief 17's three users,
> with the same symptom on record — eight hand-written retry loops around `perform_throwaway_task`, no
> caching and no resume, so a crash at document 2400 restarts from zero. Building reconnect on its own would
> put a second retry mechanism beside the one brief 17 exists to unify.
>
> **This is now the fourth pointer at that brief** (pdf2bib's retry loops, `rag_live_corpus`'s persistence
> layer, the corpus-interrogation sketch's map stage, and this), which is worth weighing when deciding
> whether 17 gets written. Note also the sibling item *A crash during ingest loses the whole run, however
> long it was* — same class, different trigger: long batch runs with no durability. A resume mechanism
> answers both.

## 8m. A triage heuristic: old priority labels that encoded *cost* are stale

Raised 2026-08-12, from a worked instance.

`TODO.md:548` — *"**[Low]** Add lockfile so `raven-minichat` and `raven-librarian` can't run simultaneously
(prevents losing changes made in one app). Quick CC session."* Shipped 2026-08-12. Its actual severity was
silent loss of an in-progress chat: each app holds the whole datastore in memory and writes it back on exit,
so whichever closed last discarded everything the other had done. Two Librarians did it to each other too.

**Why it was rated `[Low]`**: the trigger is rare, and fixing it by hand would have cost more than it was
worth. **Neither of those is a statement about value.** A single priority label collapses value and cost, and
once collapsed the two are indistinguishable — so a `[Low]` may mean *low value* or *not worth the effort*,
with nothing in the notation saying which.

The cost side has since moved by a large factor. So: **an old low rating that encoded value still stands; one
that encoded effort is stale.** For each remaining item, ask which axis the rating came from. This one was
cost — rare trigger, fiddly interprocess work — while the value side was never low.

**Where this bites hardest is the audits and sweeps**, which were deferred when a fleet-wide mechanical pass
meant a day of tedium and are now an afternoon. The `hygiene-sweep` cluster is the clearest case: its cost
fell *and* its value rose, since distribution drift (§8g) makes stylistic inconsistency propagate rather than
sit still. Both axes moved the same way on the same items, and the recorded label reflects neither.

Applies to the ~72 items not yet triaged, and is worth a second look at anything already marked `—` on
effort grounds.

## 8n. New item to file: audit what is left in the wheel

Raised 2026-08-12. **Gate: `0.2.9`**, gating on the first PyPI upload alongside the name and licensing
items. **The main cause is already fixed** — an exclusion for `00_workfiles` was added the same day — so
what remains is the audit of the rest and the standing check.

> ## Audit what the built wheel actually contains
>
> *Cluster: ? · Cost: S · Gate: 0.2.9 · Filed: 2026-08-12*
>
> **The 83% case is fixed** (2026-08-12): an exclusion for `00_workfiles` was added after the measurement
> below. What follows is the record of why, and what has *not* been checked.
>
> **Measured 2026-08-12, before the fix**: `raven/` is 132.9 MB, of which **110.9 MB (83%) is two `00_workfiles/`
> directories** — 78 MB under `raven/avatar/assets/characters/`, 29 MB under `raven/icons/`. Everything else
> is ~22 MB. Largest single file is a 15 MB character `.xcf`; there are a dozen more above 2 MB.
>
> These are editing masters — GIMP `.xcf`, source SVG, camera originals. They belong in git, since
> regenerating an asset starts there, and the icon-provenance work of 2026-08-05 depended on having them.
> They have no business in an installed wheel: nobody `pip install`s Raven in order to edit the source
> artwork.
>
> **The fix was one line.** `pyproject.toml:248` already read
> `excludes = ["**/tests", "**/__pycache__", "raven/vendor/tha3/models"]`, and the comment above it records
> the same class of problem — PDM's `includes` does not consult `.gitignore`, so without that entry the
> build swept in 860 MB of `.pt` files. The `00_workfiles/` naming is uniform, so `**/00_workfiles` covers
> both directories.
>
> **Result, measured 2026-08-12 after the exclusion: the wheel goes 107 MB → 14.4 MB, 7.5× smaller.**
> 95 files, 117 MB excluded, largest single file 14 MB. That also explains a note already in the same
> `pyproject.toml` block: the "107 MB after" figure recorded on 2026-08-03 was almost entirely these files.
>
> *(Two sets of figures appear above and they reconcile rather than conflict: the pre-fix measurements were
> `du` block counts against the uncompressed source tree, these are file bytes and a compressed wheel. The
> 15 MB / 14 MB largest file is one file measured two ways.)*
>
> **The audit was done the same day (CC).** Breakdown of the 14.4 MB: `raven/avatar/assets` 8.1 MB (over
> half), `raven/vendor/anime4k` 3.24 MB, fonts 2.17 MB, Python 3.78 MB. Of the anime4k share, 2.18 MB is
> `.glsl` kernel source the PyTorch port extracts from — load-bearing. Fonts and Python are fine.
>
> **Two candidates remain, ~4 MB together, and they are different kinds of decision.**
>
> - **`raven/vendor/anime4k/images/6486130.png`, 1.04 MB — a rule, like the workfiles.** Checked
>   2026-08-12: the only reference is `anime4k.py:441`, inside the module's `__main__` demo block, next to a
>   `wget` comment giving the URL it came from. No Raven code path loads it. **The confirmation step belongs
>   in the audit**: check nobody uses that demo block as a manual benchmark before excluding — a demo whose
>   own comment says how to re-download the input is otherwise a clear cut.
> - **Backdrops, 5.3 MB for three — settled: ship them at full size** (Juha, 2026-08-12). Measured:
>   `cyberspace.png` 1920×1080 at 2.89 MB; `anime-plains.png` and `study.png` 1344×768 at 1.46 and 1.20 MB.
>
>   *An earlier draft here argued these were oversized, reasoning that the avatar canvas is 512×512 so the
>   pixels are discarded at load. That was backwards.* The 512×512 is THA3's *character* input; the backdrop
>   fills the **window**, which is 1080p or larger, and the pipeline upscales after posing. So 1920×1080 is
>   correctly sized, not wasteful.
>
>   The two 1344×768 backdrops are technically under-sized for a 1080p display and get upscaled to fit, but
>   **they look fine in practice** — that resolution is simply what the generator produced.
>
>   **And full resolution is required, not merely harmless**: the backdrop blur is a *user toggle* in the
>   settings editor. Were it always on, these could ship pre-blurred and much smaller — a blurred image
>   compresses far better, and any upscaling softness would be hidden by the filter that made it small. The
>   optionality is what forces shipping the source at full fidelity, since a user who turns blur off gets
>   whatever detail is actually there. Generalizes: **any optional downstream transform means shipping the
>   untransformed source.**
>
> **One dangling reference created by the fix**, worth knowing rather than fixing: `raven/avatar/README.md`
> points at `animefx.svg` under `00_workfiles` as a drawing guide. It is a link for a repo reader rather than
> a file the package opens, so it works where it is actually read and dangles only for someone reading that
> README out of `site-packages`. Trivial to make absolute if it matters.
>
> **The convention is what makes the glob safe**, and it is now in `CLAUDE.md`: originals are never runtime
> assets, and code that reads one is a bug. That is what lets `**/00_workfiles` cover directories nobody has
> created yet.
>
> Suggests a standing check at release time: the release skill could compare wheel size against the previous
> release and flag a large jump. Same shape as the linter canary (§8b) — the failure is silent, and a build
> that quietly grew by 100 MB looks exactly like one that did not.

## 8o. Per-item gates — batch 4 (file order, ~`:1865`–`:1956`)

Ruled 2026-08-12. Four of the ten items in this range were already dispositioned (`:1877` and `:1970`
`already-done`; `:1984` and `:1998` superseded by brief 15).

| Item | Gate | Note |
|---|---|---|
| Easy install with a chosen CUDA version (and a sensible CPU default) | `0.2.9`, **re-scope after §8e** | This is the parent of the two torch items already touched, and §8e changes its shape. Its central complaint is a worked example of what §8e removes: *"a plain `pdm install` quietly upgraded `torchaudio` from 2.10.0+cu128 to 2.11.0 (the latter wants CUDA 13 and silently broke imports on the CUDA-12.8 machine)"*. Its stated remedy — torchaudio pinned into the CUDA dep set rather than free-floating — is moot if torchaudio goes. **Merge `:1585` into this**; they are the same problem recorded twice. What survives §8e is the genuinely separate half and the real content: the CPU-default path, where a bare `pdm install` must still yield a working `import torch`, with `-G cuda12` / `-G cuda13` overriding the base pins |
| webfetch "approve denied host" button relocates in brief 03 | `0.2.9` | Should have closed already: brief 03 landed, the relocation did not, and `chat_controller.py:1090` still carries the `role == "tool"` branch. The sweep flagged it CONFIRMED with the note that *its precondition landed without the follow-up, so it is more live than when filed*. **Keep that observation** — it is what a conditional deferral looks like when nothing watches for the condition, and it is the `Bitten:` field's use case |
| `raven.papers` user manual | `0.2.9` | Small-ish; ask CC for a draft. **Schedule with the README correctness sweep** — out-of-date docs scare away potential users, which is a stronger reason than completeness. Existing material to fold in: `raven-arxiv-search`'s instructions live in the separate `arxiv-api-search` project's README, and other fragments are scattered in Raven's main `README.md` |
| Convert startup `print()`s to `logger.info()` | `hygiene-sweep` | The item supplies its own finding method: the `--log` smoke-test `stderr - logfile` diff. Judgement per call — user-facing tool output stays `print` (e.g. `raven-check-cuda`'s markers), app status is promoted. Vendored prints left alone unless the file is already being touched |
| Hybridir: BM25 backend migration for larger corpora | `—` | `bm25s` rebuilds the whole keyword index on every commit — IDF changes make it non-incremental by design — which is not nice, but it is sub-second at ~1k documents and only pinches at 10k–100k. Premature optimization; the Tantivy plan is written down for when it is not. **Note for §8m**: this `—` encodes *value at current scale*, not effort, so the stale-label heuristic does not apply and it stays parked correctly |
| webfetch: batch-approve several denied hosts at once | `—` | v1 follow-up, and **blocked on the relocation above**: batch approval attaches to wherever the approve button ends up, so building it first means building it twice |

## 8p. Per-item gates — batch 5 (file order, ~`:2439`–`:2739`)

Ruled 2026-08-12. Most of the `:2033`–`:2369` range was already dispositioned — the Markdown cluster to the
block-rendering brief, plus the composer, auto-RAG, scroll-jump and character-drop items.

| Item | Gate | Note |
|---|---|---|
| Reconsider the webfetch allowlist default: ship deny-by-default? | **`RN2026`** for the demo config, **`0.2.9`** for the shipped default | Two decisions, and only the first is forced by the date. `webfetch_allowlist` defaults to `None`, so the gate at `if allowlist is not None:` is skipped entirely and the model may fetch any public URL — the item calls this a larger exposure than the opt-in `webfetch_trust_search_results` flag, since with allow-all the model can already follow any link a poisoned search result feeds it, which is the injection→exfiltration vector the allowlist exists to bound. A curated baseline already sits in the same config (`webfetch_default_allowlist`: DOI, arXiv, major publishers). **For the night: set an explicit allowlist**, a config line. **Separately: decide the shipped default**, which has been open since June and which the demo is merely the forcing function for |
| Fleet-wide: shared two-phase DPG shutdown helper + audit | `0.2.9` | The abnormal-exit cluster's structural item, and the best-evidenced item in the file: `raven-librarian` and `raven-avatar-settings-editor` both got teardown wrong **independently**. That is §8g's distribution drift from another direction — the correct shape exists in `raven.cherrypick.app`, and the others hand-rolled variants that do not match it. Copy-paste boilerplate with a subtle correctness condition is exactly what drifts. The item states the shape precisely: cancel-only in the exit callback (a `wait=True` there deadlocks anything parked in `dpg.split_frame`), blocking drain and teardown in the render-loop `finally` before `destroy_context()`, and drive both phases yourself rather than trusting DPG to have run the callback. Pairs with the avatar-leak item — both are abnormal-exit correctness |
| Add built-in calculator and weather LLM tools | **split** | **Weather: use the openmeteo MCP server**, not an in-tree tool (Juha, 2026-08-12) — it already exists in the dotclaude setup, and building it in-tree while MCP is landing anyway is duplicated effort. Drop that half. **Calculator: build it**, with `simpleeval` — an AST walk restricting the allowed function set plus size limits, scoped to *expressions* rather than statements. About a page of code. Worth more than its size suggests at an exhibit: a model doing arithmetic badly in front of an audience is a visible failure, and this is the one built-in that removes it |
| Cherrypick: zoom-in doesn't upgrade already-cached preload neighbors | `0.2.9` if time, else `next` | Arguably a UX bug rather than an optimization |
| Fleet audit: every hotkey discoverable in a tooltip + help card | `next` | Pairs with the help-card redesign, which touches the same surface |
| webfetch local (client-side) mode | `—` | The item argues for deferral itself. Licensing splits the two tools: `websearch` is AGPL (a port of the SillyTavern-Selenium extension) and must stay server-side, while `webfetch` is Raven's own and could be BSD — except its Tier-2 fallback borrows websearch's Selenium driver (`_fetch_tier2` → `websearch.get_driver()`) |
| Parse Gemma's inline tool-call spelling if a raw-passthrough backend needs it | `—`, on hold | Conditional on a backend that does not yet exist in the setup: LM Studio, the live-verified Gemma 4 backend, delivers structured `tool_calls`, so there is nothing to parse inline. Same class as the gray-thinking item — an ooba-only concern, and ooba is untested and not installed |

## 8q. Status advances and two new items (CC, 2026-08-12)

Six of these were already dispositioned here and only need their status moved; two are corrections; two are
new.

**Now `already-done` (were `supersede` or open):**

| Item | Was | Now |
|---|---|---|
| `TODO.md`: *[Low] Add lockfile so `raven-minichat` and `raven-librarian` can't run simultaneously* | §8m's worked instance | Shipped as `raven.common.datastorelock`. **Note it lives in `TODO.md`**, so the `## Already done` section and the deletion pass must cover both files |
| *Headless scaffold mode for `ai_turn`* | §2 `supersede` → brief 15 | Brief 15 landed as `raven.librarian.agent`; brief archived to `briefs/researchers-night/done/` |
| *Lazy `api.initialize` in `llmclient` and `hybridir`* | §2 `supersede` → brief 15 Part 0 | Done — `test_scaffold.py`'s `importorskip` is gone and its comment explains why. That was the item's whole rationale |

**Amendments to items that stay open:**

- ***Make the canned AI greeting optional*** (§8g, `next`). Its first bullet — the one flagged *"needs a real
  fix, not a length tweak"* — is fixed: greeting classification now checks `role == "assistant"` as well as
  position. Still true: the `chat_controller` assert, `minichat`'s `len(node_id_history) < 4`,
  `chatutil.factory_reset_datastore`, `appstate._refresh_greeting`. **Its line numbers have all drifted —
  re-cite by name**, which is the rule from §4.
- ***The DPG tests we have never run in CI*** (§8g). Two stale details, corrected in that row above; it also
  prescribes exactly the experiment now being run, so update or delete it by the outcome.
- ***Modernize the Librarian system prompt / character card*** (§8f, `next`). Add a pointer to
  `briefs/researchers-night/done/15_headless-agent-driver-brief.md`, last section: the argument that
  `setup_interaction_style` is three different kinds of thing and cannot move as a unit.

**Already recorded here, no action:** mid-run backend recovery is §8l; the wheel audit is §8n, including the
two remaining candidates and the resolution finding.

**New: chat panel doesn't scroll to the end after a rebuild.** *May not survive the day* — under
investigation at the time of writing. File only if it persists.

**New — but much weaker than first stated: `chattree.get_all_root_nodes` is an O(n) scan.**

> *Cluster: ? · Cost: S · Gate: `—` · Filed: 2026-08-12*
>
> `chattree.get_all_root_nodes` scans the whole forest for nodes whose parent is `None`. Deliberate, and the
> docstring says so — *"We don't keep track of these separately; this is done by an O(n) linear scan over the
> whole forest."*
>
> **The path that mattered is already fixed** (Juha, 2026-08-12; CC's note omitted this).
> `chat_controller._scan_for_root_nodes` memoizes it, and `_get_all_system_prompt_node_ids` filters the
> cached list against `datastore.nodes` before returning. Both halves of the reasoning are in the
> docstrings: safe to cache because roots are only ever *created* while app state loads, and the filter is
> required because a card that is not in use can be deleted from the GUI, after which `get_children` raises
> on a node that is gone. Without the memo it *"would otherwise run once per chat message widget created,
> over the whole datastore"* — that was the hot path, and it is closed.
>
> What remains is the underlying method, still O(n), called from `appstate` (×2, at startup), `minichat`,
> `app.py`, and `chattree.get_siblings`.
>
> **The `get_siblings` path is live after all, and the reasoning went round twice.** Claimed, then withdrawn
> as unchecked, then reinstated (Juha, 2026-08-12): **multi-root support landed the same day**, and
> enumerating a root's siblings — the character cards — is its natural consequence. If the graph view shows
> the root level, every such lookup is a full-forest scan. See the brief 16 amendment below, which is where
> that decision now sits.
>
> Gate accordingly: `—` while the graph view does not show roots, `0.2.9` if it does.
>
> Shape of the answer: an index of roots maintained by `create_node` / `delete_node`. Roots are few while
> the scan is over every node, so the index is small and the saving grows with the datastore. Same class as
> *Datastore scaling: a single `chat.json` won't hold years of chats*, and worth cross-linking on that basis.

## 8r. Amendments to brief 16 (chat graph view)

Brief 16 is committed at `briefs/researchers-night/16_chat-graph-view-brief.md`, written 2026-08-04 and not
yet built. Two things have moved under it, and one of its own claims is factually wrong. **Apply these as
edits to the brief**, not as new items.

### 1. Correct the factual error: roots are not sessions

The brief says (`:155`): *"The accumulated forest is what is large, and scoping to the tree containing HEAD
plus windowed siblings is what handles it."* An earlier passage makes the same assumption — that an
open-house evening accumulates dozens of roots.

**Wrong.** `chat_controller.py:219` states what a root is: *"There are as many as there are distinct system
prompts the datastore has seen: `appstate` keeps one root per variety of card, so a chat written under an
older card is rooted at its own."* An evening at one system prompt produces **one** root and dozens of
branches beneath the greeting. The forest is never wide at the root level; the width is one level down, at
the first user message, which the brief already identifies correctly as doubling as the recent-chats list.

So the scoping decision may still be right, but the stated reason is not the reason. **Rewrite that
sentence**: depth limiting is about branch depth and the wide session level, not about root count.

### 2. Multi-root support landed 2026-08-12, after the brief was written

Roots are now first-class, so showing the root level is the natural consequence — and the brief's
out-of-scope entry (`:309`, *"A forest view across all roots. The windowed wide level covers what the demo
needs"*) was decided against a world where roots were effectively singular.

**What the root level is, and is not.** A root is a distinct *system prompt text*; the character's name,
avatar and voice live in `config.py`. So the root set is largely **version history of one character's card**,
not a character selector. Two consequences:

- **Clicking a root does not switch character.** The app would keep rendering the configured avatar and
  voice while the chat sits under a different system prompt. That mismatch needs a decision, and it is the
  main reason not to expose roots casually.
- **What it does give is reachability of chats written under older cards** — precisely the "access old
  chats" gap the brief says it closes. Those chats are in the datastore and unreachable from the GUI today;
  that was the substance behind the removed tech-demo disclaimer. **Archaeology rather than switching**, and
  worth having on those terms.

**The graph has two wide levels doing different jobs**, and the brief designed for only one: cards at the
root, sessions one level down. Whether the root level gets the same windowed treatment, a different
affordance, or stays out of scope is now an open question rather than a settled one.

### 3. Consequent cost: `get_siblings` on a root is a full-forest scan

`chattree.get_siblings` calls `get_all_root_nodes`, which is O(n) over every node (see the item above). If
the graph view shows the root level, every root-sibling lookup pays that. **This decision determines that
item's gate** — `—` if roots stay out of scope, `0.2.9` if they come in.

Note the memoization that already exists is at the *`chat_controller`* layer
(`_scan_for_root_nodes` + a live-node filter), not in `chattree`, so a graph view calling `chattree` directly
would not inherit it.

### 4. Add to the settle-list

> 7. **Whether the graph view shows the root level at all**, and if so how — given that roots are card
>    versions rather than characters, that clicking one creates an avatar/voice mismatch against
>    `config.py`, and that it is the only route to chats written under older cards. This decision also
>    fixes whether `get_all_root_nodes` needs an index.

## 8s. Per-item gates — batch 6 (file order, ~`:2759`–`:3377`)

Ruled 2026-08-12.

| Item | Gate | Note |
|---|---|---|
| Version the chat datastore file, so migrations can be skipped once applied | `0.2.9` | Cheapest load-time win left. `appstate.backfill_sidecar_metadata` walks every revision of every node **at every load**, because nothing distinguishes a migrated datastore from a fresh one — so every migration runs unconditionally, forever, at a cost that grows with history. The item also works out the design trap: a lone root node holding the version breaks twice, since `prune_unreachable_nodes` collects it and `_get_system_prompt_node_id` reads roots as system prompts. Cross-link with *Datastore scaling* and the root-scan item: all three are load-time cost proportional to total history |
| Expose the docs-DB source files behind a reply's RAG citations | `next`, **wanted this year** | Provenance is already tracked per turn (the payload's `retrieval` field records the query and the snippets the AI saw) and the open-file machinery exists (`common_utils.open_file` / `open_in_file_manager`, built for image attachments). What is missing is the GUI affordance and where it lives |
| Consolidate the flash palette into named constants | `hygiene-sweep`, **take it first** | The clearest visible instance of drift in the file: `WidgetFlash`'s own default ok-green is re-hardcoded in `flash_button`, and the text-green recurs in ~7 places across `vumeter`, `visualizer/app`, `visualizer/annotation`, `xdot_viewer/app`. Drift already present rather than inferred |
| Visualizer's importer should read the document database, not just `.bib` | `next` | Pairs with brief 11 |
| Let the AI drive the constellation's own views (tools, then voice) | `next`, **this year if possible** | Pairs with Visualizer/Librarian integration rather than standing alone |
| Datastore scaling: a single `chat.json` won't hold years of chats | `next` | Real, not urgent. `PersistentForest.save` rewrites the entire file every time, so load and save grow linearly and a corrupted write risks the whole history at once |
| Document ingestion (5 items) | `next` **as a group** | Already clustered `document-ingestion`; brief once rather than five times. Members: *format parity* (docs DB and attachments should accept the same set — a split is arbitrary from outside); *spreadsheets* (deliberately excluded from the office-formats work: tabular content means "the text of this file" is ill-defined, and row-major reading chunks badly for retrieval); *OCR and SVG `<text>`* (the image→text cell, wanted for three distinct reasons); *SVG figures* (hand-authored figures are commonly SVG, the native form of exactly the figures an author would want read — not exotic for this audience); *page images* (extraction is text-layer only, so equations, plots, diagrams and tables-as-figures are lost). **`:3122` is the one that bites on this project's own terms**: figure- and equation-heavy literature extracts to prose that omits the argument, in exactly the corpus Raven exists to read |
| Upgrade oobabooga and re-check Raven's ooba support | `—` until after RN, **possibly after 0.2.9 too** | New backend variables are the last thing wanted in September. Note 0.2.9 is already carrying a lot and has not been reconciled against the open briefs or the preference list. **This item unblocks two others**: the gray-thinking item and the Gemma inline tool-call item are both ooba-only and currently untestable, since the install is stale and absent from the 16 GB machine |
| VLM reranking of mixed-modality search results (post-Nomic) | `—`, and **explicitly not struck by the reranking result** | The item already defends itself (2026-08-06) against being killed by association with *RAG: rerank retrieved chunks*, and the defence stands: a VLM does something different in kind from a cross-encoder. Add a `See also:` so a later triage does not pattern-match it away. **Re-test after the Nomic switch** — the empirical result is specific to Raven's current hybridir, where reranking was worse than useless, and a new embedder changes the conditions |

**A note on the "which is why" in the reranking argument.** The empirical result is solid and repeatedly
replicated: two models (22.7M and 278M parameters, different training data), three placements, three corpora,
k up to 200 — no configuration helped. **The offered mechanism is a post-hoc explanation, and stays one.**
`REPORT.md:44–49` says *"fusing two cheap independent signals beats one expensive model's opinion — RRF's
value is evidence diversity, and collapsing it is the cost"*, and the placement gradient (fused-rerank worst,
single-arm-then-fuse recovers most of the loss but still loses to plain fusion) is *consistent* with it. It is
not a test. The test would need a second independent cheap signal, and none was found — CC proposed IDF, which
on inspection is too correlated with BM25 to add anything. A correlated signal adding nothing is weak
confirmation at best.

**The genuinely strange part, and the better lead** (Juha, 2026-08-12): reranking did not merely fail to help
— **the rank of the best-matching documents became *worse***, with both models, in every test. The mechanism
of reranking is retrieve `M >> k`, rerank, cut at `k`; the point is to pull good documents *up* into the
cut. A relevance-trained model with nothing to add should be roughly neutral, not harmful. The diversity
story predicts *not better*; it does not predict *worse*. Something is mis-scoring.

Candidate worth putting ahead of the two proposed here (pool size and fused input — **both already tested**,
so neither is the answer): **domain mismatch in what cross-encoders are trained on.** They learn short web
queries against short passages; Raven feeds whole conversational messages against scientific prose chunks.
Scoring out-of-distribution pairs can be anti-correlated rather than merely uninformative. Testable with
short focused queries against the same corpus — and the report already establishes that query form matters
here, since rambling questions retrieve at half the MRR of focused ones.

The practical resolution stands either way and is already shipped: a slightly larger `k` at which to cut, and
no reranker in the path. Re-test after the Nomic switch; the result is specific to the current embedder.

## 8t. Per-item gates — batch 7, the tail (`:3407`–`:3987`)

Ruled 2026-08-12. **This completes the pass over `TODO_DEFERRED.md`.**

| Item | Gate | Note |
|---|---|---|
| The docs DB stores each document's full text *and* its chunks | `0.2.9`, **resolve soon** | Not hypothetical at current corpus sizes: ~12k hydrogen abstracts already ingested, ~2500 one-page ECCOMAS 2024 conference abstracts, and an arXiv AI fulltext set of 1200+ full papers. Measured bloat with a hard number: `fulldocs/data.json` holds a per-document `"text"` field *and* a `"chunks"` list whose entries carry their own text — overlapping slices, so the chunks alone exceed 100% of the source, with the full copy stored beside them. **48 MB of source documents produce a 124 MB `data.json`, about 2.6×**, and the file is rewritten whole on every save. Same family as the datastore-versioning and datastore-scaling items. **Check what actually consumes `"text"` before removing it** — the fix looks like one field, and that is exactly when it is not |
| `pdm.lock` is gitignored, against the fleet policy for applications | `0.2.9` **if it resolves cleanly, else `next`** | Fleet policy: libraries do not commit a lockfile, applications do, and Raven is an application. The `.gitignore` entry is an inconsistency nobody decided on rather than a documented exception. **Same blocker as *Easy install with a chosen CUDA version***: what a lock does with the `pytorch-cu128` index and the matched `+cu128` set. Do them in one session, and note §8e's torchaudio removal simplifies both. **Do not let this become a large must-resolve item inside 0.2.9** — if the lock/index interaction turns out thorny, move it to `next` rather than expanding the cycle |
| A fetched web page is budgeted as a user attachment, not as a speculative fetch | `0.2.9` | 0.2.8 stores a long `webfetch` result as an attachment sidecar, which puts it under `fit_attachments_to_context` — the *user attachment* budget, deliberately carrying no per-document ceiling on the reasoning that an attachment is the user saying *read this*. A fetched page is the opposite case: the model's speculation. `docs_fetch_max_fraction_of_context` exists and its own comment says so; it is simply not applied here |
| HTML pages whose content is produced by running them | `next` | With the `document-ingestion` cluster. **Licensing constraint to carry into that brief**: if it needs a browser driver, the only one in the tree is `websearch`'s Selenium driver, which is AGPL — so this would have to be server-side, exactly as the *webfetch local mode* item records for `_fetch_tier2`. That is a structural constraint on where the feature can live, not a detail |
| Rendering LaTeX equations in the chat log | **`0.2.9`** | Upgraded from `next`: the Markdown renderer work turned out far smaller than assumed — tables are close to the only missing feature, syntax highlighting being off the roadmap — so it will be worked on soon and this rides with it. Wanted this year; can wait until after RN. Scoping already done: display equations via `matplotlib.mathtext` (no LaTeX install, RGBA straight to a DPG texture), inline math merging with the emoji item as "inline images in text flow", multi-line derivations falling back to a fenced code box |
| A crash during ingest loses the whole run | `next`, **with the per-document pass — see §8u** | The measured version of the durability gap, and the strongest single argument for that brief. The delayed-commit coalescer defers a commit one second after each finished read, so on a large corpus it never fires until reads *stop*: **~40 minutes on the 1268-PDF fulltext corpus (2026-08-06)** with every extracted document pending in memory and nothing on disk. A crash, kill or power loss anywhere in that window discards all of it. Not a hypothetical window — a measured one |
| Semantic grouping in the sidecar cleanup preview | `—` | Gated on Nomic. Today there is nothing to group *by*: sidecars are content-addressed, so the set is globally unordered, filenames are hashes, and the only other handle is a per-file provenance URL that may be absent |
| **NEW (CC, 2026-08-12): `chat_controller` is not importable without spaCy** | `0.2.9` | Same anti-pattern as the just-completed *Lazy `api.initialize` in `llmclient` and `hybridir`*, one layer up: `chat_controller` reaches the full ML stack through the avatar client, so its tests can only run on a dev machine. **Now that the GUI tests run in CI, this is the only thing keeping the controller's pure datastore helpers out of it** — and those helpers are exactly the code the graph view and the open-datastore work will build on. Same fix shape: move the side-effecting import behind the seam that already exists |
| Source code in the document database wants its own tokenizer | `—` | **The item's value is its analysis, not its scheduling.** Adding `.py` to `llm_docs_exts` looks like a one-line change and is not: `HybridIR._tokenize` lowercases, lemmatizes, drops English stopwords and keeps only `token.is_alpha`, so keywords, operators, digits and underscored identifiers are largely destroyed and the keyword arm would index almost nothing useful. The failure is the quiet kind — retrieval simply does not find code — which is why the item exists in this form |

## 8u. The per-document LLM pass ("brief 17") — six users, no brief, and its only description is archived

Asked 2026-08-12 ("what was brief 17?"), which is itself the finding: **the number was reserved for a brief
nobody wrote, numbering was then abandoned, and the reference has been dangling since.**

**What it is**: a per-document LLM pass with **retry, cache, resume and progress**. Its only description is
`briefs/researchers-night/done/15_headless-agent-driver-brief.md:946`, with a second mention at `:362`
recording that it gained two more prospective users during 15's implementation.

**Six users now**, none of which knew about the others when they surfaced:

1. `raven-pdf2bib` — eight `perform_throwaway_task` call sites, each wrapped in its own hand-written retry
   loop, the same six lines eight times in one 1058-line file. No caching, no resume: a crash at document
   2400 restarts from zero.
2. `rag_live_corpus`'s persistence layer — a `PersistentForest` per sample plus a JSONL ledger, worth
   lifting wholesale; those runs take an hour and the machines reboot.
3. `briefs/design/corpus-interrogation-sketch.md`'s map stage — and note `summarize` is *shipped code that is
   currently switched off*, so the map half already exists.
4. Mid-run LLM backend recovery for batch tools (§8l) — CC's three deferred questions are this brief's scope
   exactly: how long to wait, resume or restart, what to do with documents already written.
5. *A crash during ingest loses the whole run* — the measured version: ~40 minutes on the 1268-PDF corpus
   with everything pending in memory.
6. Two further prospective users found during brief 15's implementation.

**Written 2026-08-12 as `per-document-llm-pass-brief.md`** (unnumbered, RN working set) — so the dangling
number is retired and the content has a live home. The notes below record why, and stay as the rationale.

**Three problems, now addressed:**

- **The description lives in an archived brief.** `done/` is a historical record and is not retconned, per
  the standing rule — so this content needs a live home regardless of whether the brief is written.
- **The dangling number.** Either write it as 17, or drop the number and refer to it by name everywhere.
  Since numbering was abandoned after 15, the second is more consistent — and the working-set briefs since
  (`markdown-block-rendering`, `wake-word-voice-input`, `ligature-repair`) are all unnumbered.
- **Six independent pointers is the signal to write it.** Each arrived from a different direction, none
  knowing about the others — the pattern that says a shared primitive is missing rather than six features
  being wanted. Two of the six now have measured costs (40 minutes of pending work; a restart from document
  2400), which is more evidence than most briefs start with.

## 9. The deletion pass — last, and deliberately so

**Do not run this until the whole list has been processed.** Everything tagged `already-done` accumulates in
the `## Already done` section and stays there through every triage session; only once the last session has
finished does that section get reviewed as a block and removed.

The ordering is the point. An item tagged in session one may turn out, in session four, to be the only place
some piece of prose lived — the sweep already found one instance of exactly that (the ligature item was a
live brief's cited source, and nothing in the item's own text said so). Reviewing them all at once, at the
end, is what makes a missed re-homing recoverable instead of a `git log` archaeology exercise.

Two checks before deleting anything, both learned today:

1. **Grep the briefs for the item's heading.** A document that is pruned cannot be cited by one that is
   kept. This is the sweep's own generalization, and it found its first instance the same day it was written
   down.
2. **Grep for the heading in `TODO.md`, `investigations/`, and `CLAUDE.md` too.** The one instance found so
   far was a brief, but nothing makes briefs special — any kept document can cite a pruned one.

Attach this to the *last* triage session rather than to a date, since it is gated on completion rather than
on the calendar.

## 10. Not covered

107 CONFIRMED, of which this session dispositions the ones the sweep flagged as needing more than ranking.
The remainder are genuine ranking work and are the next sessions' subject. The first batch's 25% stale rate
did not hold — 9 of 125 overall — so those sessions are not ratification.
