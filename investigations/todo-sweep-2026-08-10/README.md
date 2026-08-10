# TODO_DEFERRED verification sweep, 2026-08-10

**Part C of `briefs/todo-sweep-2026-08-10/todo-mechanical-pass.md`. Report only — nothing here was applied
to `TODO_DEFERRED.md`.** The triage discussion consumes this file; heading text is verbatim, so it joins
against the deferred file and the cluster map.

**Status: partial. 70 of 130 items carry a verdict.** The remainder are unchecked, not confirmed — see
"Coverage" below, which says plainly what that means for the discussion.

It is a directory rather than the single file the brief named, because it came with a script, and this repo
keeps an artifact with what produced it.

## Scripts

| Script | What it answers |
|---|---|
| `check_references.py` | Which items name a file or symbol that no longer exists anywhere in the tree. Cheap evidence for MOVED and STALE, gathered for all 130 items at once. Re-runnable; run it again before the discussion pass. |

An earlier version of that script resolved bare basenames against the repo root and reported 43 items as
broken. Nearly all were false — `config.py` and `app.py` exist in plenty of places, just not at the root.
Resolving by basename across tracked files, and letting a filename satisfy a symbol reference, brought it to
11. **Treat a first-run number from a checker like this as a bug report about the checker.**

## Verdicts

Five verdicts, per the brief: CONFIRMED (claim holds), STALE (claim false — fixed, landed or removed), MOVED
(true, but the code named has relocated), SUPERSEDED (true, but a brief now owns it), UNCHECKABLE (needs a
running app, a live backend, or a human eye).

| Heading | Verdict | Evidence |
|---|---|---|
| OS drag-and-drop of files into DPG apps (cross-platform) | **STALE** | Shipped 2026-08-10 as `raven/common/gui/filedrop.py`, wired into all six GUI apps and live-tested in each. |
| "Internet" toggle: scope `tools_enabled` to a clear security boundary in the GUI | **STALE** | Landed in 0.2.9-dev: `scaffold.ai_turn` takes `internet_enabled`, and `llmclient.maybe_tool_names_for_turn` computes an allowed-tool-name list from `network_tool_names` / `document_tool_names`. The item asks for exactly this. Its "separate toggle for MCP tools later" aside belongs to brief 04. |
| Indexing a large corpus is silent for minutes, and reads as a hang | **STALE** | 0.2.8 added the INDEXING indicator with per-document progress (`librarian/app.py:848`). |
| Idle throttle for Librarian | **STALE** | 0.2.8 added it; `librarian/app.py:225`, `IDLE_SLEEP_S = 0.08` (~12 fps idle). |
| Headless scaffold mode for `ai_turn` (scriptable agent layer) | **SUPERSEDED** | This *is* brief 15, `briefs/researchers-night/15_headless-agent-driver-brief.md`. |
| Lazy `api.initialize` in `llmclient` and `hybridir` | **SUPERSEDED** | Brief 15 Part 0 owns it, and cites this entry by line number. |
| webfetch "approve denied host" button relocates in brief 03 | **CONFIRMED** | Brief 03 is closed, but `chat_controller.py:1090` still carries the `role == "tool"` branch and the approve button the item says to relocate. Its precondition landed without the follow-up, so it is *more* live than when filed. The 2026-08-05 ruling would have declined this conditionally; the condition is not met. |
| Same file formats in the docs DB and in chat attachments | **CONFIRMED** | Two halves, and the item says so. Office formats landed (`librarian_config.llm_docs_exts` now equals `docextract.supported_extensions()` exactly — 13 formats, verified equal). The images half is untouched and blocked on the Nomic embedder. |
| The DPG tests we have never run in CI | **CONFIRMED** | `.github/workflows/requirements-ci.txt` still has no `dearpygui`, so the `importorskip` fires on every CI run. |
| The docs DB stores each document's full text *and* its chunks, both in the JSON | **CONFIRMED** | `hybridir.py:410` still writes `fulldocs/data.json`. The `data.json` in this item is the *docs* store, untouched by the chat-datastore rename — the reference checker flagged it, and was wrong. |
| AMD GPU (ROCm) support audit | **MOVED** | Names `raven/common/lanczos.py`; it is `raven/common/image/lanczos.py`. |
| CLAUDE.md: rephrase DPG pitfall #5 to avoid Claude thinking loops | **MOVED** | Names `dpg-threading-notes.md`; it is `dpg-notes.md`. |
| Audit and slim down project CLAUDE.md | **MOVED** | Same rename. |
| Datastore scaling: a single `data.json` (+ flat sidecar dir) won't hold years of chats | **MOVED** | The concern stands; the names changed in 0.2.9-dev — `data.json` → `chat.json`, `<datastore>.images/` → `<datastore>.sidecars/`. |
| Librarian has no periodic autosave: an abnormal exit loses the whole session's chat | **MOVED** | Same rename; the item names `data.json` and `state.json`. |
| Reasoning traces with indented bullets mis-render | **MOVED** | Uses `data.json` as an example path; same rename. |
| Enable HTTP response compression on raven-server | **STALE** | Implemented: `server/app.py:59` imports `flask_compress`, `:90` calls `Compress(app)`. |
| Two adopted directories ship without their licence text | **CONFIRMED (halved)** | It names two. `raven/vendor/DearPyGui_Markdown/LICENSE` now exists (MIT, IvanNazaruk) — added 2026-08-03, the same day the item was filed, so the item was stale on arrival for that half. `raven/vendor/IconsFontAwesome6.py` still carries no licence or copyright header, and the item already worked out what it should say (Font Awesome under its MIT clause, generator under zlib). |
| `pdm.lock` is gitignored, against the fleet policy for applications | **CONFIRMED** | `.gitignore:9` lists it; the file exists (340 KB) and `git ls-files` does not know it. |
| Migrate the remaining `dpg.split_frame()` sites to the guarded `guiutils.split_frame` | **CONFIRMED** | 33 bare `dpg.split_frame(` call sites outside `utils.py` and the vendor tree; 3 files use the guarded form. Barely started. |
| Remaining server modules without a MaybeRemote | **CONFIRMED** | 12 server modules; `mayberemote.py` has 9 service classes. `avatar`, `imagefx`, `webfetch` and `websearch` have none. |
| Remove the dead inline-`<think>` handling in the chat renderer | **CONFIRMED** | Still there, and the code says so itself: `chat_controller.py:1305` "that path is dead — leftover from the pre-June-2026 inline handling, not yet removed", and `:1518` "slated for removal". |
| Convert startup `print()`s to `logger.info()` where appropriate | **CONFIRMED** | 180 `print(` calls in `raven/`, excluding the vendor tree, tests, and the CLI tools where printing is the output. "Where appropriate" is still a judgment call per site. |
| Audit fleet for dict constants that should be `frozendict` | **CONFIRMED** | `frozendict` appears in exactly one module (`client/avatar_renderer.py`), so the audit has not been done. |
| Version the chat datastore file, so migrations can be skipped once applied | **CONFIRMED** | No version field in `chattree.py`; the 0.2.9-dev rename migration runs by probing for the legacy name, which is the situation the item is about. |
| Move the avatar backdrop onto `image.utils.fit_cover` | **CONFIRMED** | `client/avatar_renderer.py` has no reference to `fit_cover`. |
| Librarian doesn't check that the LLM backend has a model loaded | **CONFIRMED** | Nothing anywhere handles the empty-model case; `NO_MODEL_INFO` is only a display label for the model-info readout. |
| Consolidate the flash palette into named constants | **CONFIRMED** | Flash colours are still literals at their call sites; the only named `FLASH_COLOR` in the tree is a test fixture. |
| `replace_last_paragraph`'s `dpg.mutex()` is disabled because it hangs the app | **CONFIRMED** | `chat_controller.py:518-519` still carries the commented-out `with dpg.mutex():` and its TODO. |
| Idle prefill fires even when the HEAD's token count is already exact | **CONFIRMED** | `_context_prefill_entrypoint` bails on cancellation, shutdown, an in-flight generation, and a moved HEAD — but has no known-exact check, which is the one the item asks for. (A neighbouring guard *does* clear a pending prefill when a real turn starts, which is a different case and easy to mistake for this one.) |

### Batch 2 — items settleable by "does this exist yet?"

| Heading | Verdict | Evidence |
|---|---|---|
| Hybridir: cover the edit-queueing layer with tests | **STALE** | `test_hybridir.py:384` onward covers `_pend_edit` collapse at unit level, which is the layer the item names. |
| Smooth scrolling in Cherrypick too, now that Librarian has it | **CONFIRMED** | `SmoothScrolling` is used by Librarian and Visualizer; `raven/cherrypick/` does not reference it. |
| The avatar upscaler offers bilinear and bicubic, but not Lanczos | **CONFIRMED** | `upscaler.py:36` accepts exactly `low`, `high`, `bilinear`, `bicubic`. |
| RAG: rerank retrieved chunks and inject only the best few | **CONFIRMED** | No occurrence of "rerank" anywhere outside tests. |
| Split `raven.common.nlptools` per backend | **CONFIRMED** | Still one module, `raven/common/nlptools.py`. |
| Untested but test-worthy modules in `raven.common` | **CONFIRMED** | Five remain genuinely untested: `docstring_utils`, `hfutil`, `gui/helpcard`, `gui/vumeter`, `gui/widgetfinder`. (`gui/fontsetup` also matches a naive check, but is exercised through `guiutils.bootup` by `test_fontsetup.py`.) The two the item's prose names as pending, `text/normalize` and `text/speakable`, are covered. **This row was wrong twice before it was right — see below.** |
| Faster PNG decoder | **CONFIRMED** | `image/codec.py` has a turbojpeg fast path for JPEG only; no `fpng`/`fpnge`/`spng`. |
| raven-cherrypick: export image sequence (QOI→PNG batch) | **CONFIRMED** | No export path in `cherrypick/app.py`. |
| Avatar settings editor: custom postprocessor chain ordering | **CONFIRMED** | `strip_postprocessor_chain_for_gui` documents itself as "Fixed render order"; the GUI imposes the order rather than exposing it. |
| raven.papers user manual | **CONFIRMED** | No documentation files under `raven/papers/`. |
| webfetch: batch-approve several denied hosts at once | **CONFIRMED** | `approve_host_for_session(host: str)` takes one host, and the button calls it once. |
| scaffold: collect `ai_turn`'s callbacks into a single bundle object | **CONFIRMED** | The signature still threads the callbacks individually. |
| Add built-in calculator and weather LLM tools | **CONFIRMED** | Neither exists; the only "calculator" in the tree is a running-average docstring. |
| Reconsider the webfetch allowlist default: ship deny-by-default? | **CONFIRMED** | Ships `webfetch_allowlist = None`, which the config documents as leaving the whole constrain-mode dormant — i.e. allow-by-default. Still a decision, not a defect. |
| Rendering LaTeX equations in the chat log | **CONFIRMED** | The only LaTeX handling is the BibTeX importer's accent decoding. |
| Text out of images, so figures work without a vision model (OCR, SVG `<text>`) | **CONFIRMED** | No OCR anywhere; `docextract` says so itself ("OCR for those is a separate, later concern") and the GUI tells the user to run `ocrmypdf` first. |
| Vector figures in the docs DB and attachments (`.svg`) | **CONFIRMED** | `docextract` has no SVG path. |
| Read documents as page images, for figure- and math-heavy sources | **CONFIRMED** | No page-image ingestion path. |
| A no-avatar mode, with the chat tree in the panel the avatar vacates | **CONFIRMED** | No such mode. |
| A crash during ingest loses the whole run, however long it was | **CONFIRMED** | The delayed-commit coalescer is still in place (`hybridir.py:1651`, "Schedule delayed commit after each add"), so the pending-edit window the item measured still exists. The sweep first read `indexer.py:154` ("the commit is per-document and the index auto-persists… re-running resumes") as contradicting it. **It does not** — resolved by Juha, 2026-08-10: the CLI and GUI paths are the same, and the comment is right that *any commit that lands* leaves a valid index. The two describe different things, commit **validity** versus commit **latency**. The real mechanism is contention: on a bulk fulltext ingest the extraction workers (pypdf) starve the commit workers, so the *first* commit does not run until very late. Nothing is wrong with a landed commit; the problem is that none lands for ~40 minutes. |

### Batch 3 — more "does this exist yet?", and none of it did

Twenty items, twenty CONFIRMED. Worth noting as a result rather than a boring batch: batches 1 and 2 were
picked for looking closeable and yielded six STALE; this one was picked purely for being *queryable*, and
yielded none. The stale items are not distributed evenly through the file — they cluster in what recent work
touched, which is the argument for checking rather than sampling.

| Heading | Verdict | Evidence |
|---|---|---|
| Make the DPG reference a skill, so it fires when it is needed | **CONFIRMED** | `~/.claude/skills/` holds seven skills; none is about DPG. |
| Revisit `recenter_window`'s degrade-instead-of-raise policy | **CONFIRMED** | Still `required=False` at `utils.py:581`. The item asks for a decision, not a fix. |
| GUI: hardcoded stand-ins for values DPG has no getter for | **CONFIRMED** | `DPG_WINDOW_PADDING = 8`, `DPG_FRAME_PADDING_Y = 3`, `DPG_SCROLLBAR_SIZE = 14` — still constants read off the default theme. |
| Web status panel: check on a long job without being at the machine | **CONFIRMED** | Nothing of the kind in the server. |
| Browse *all* attachments in the datastore, not just the orphaned ones | **CONFIRMED** | The only attachment-enumeration path is the orphan cleanup preview. |
| TTS reads arXiv IDs digit by digit | **CONFIRMED** | `text/speakable.py` has no arXiv handling. |
| FileDialog: smart-case the Find (search) field | **CONFIRMED** | No smart-case logic in `fdialog.py`. |
| FileDialog: image thumbnail previews (Lanczos'd) | **CONFIRMED** | No thumbnail path. |
| FileDialog: multi-extension filter as one labelled item | **CONFIRMED** | `file_filter` is a single extension string; grouped filters do not exist. This is the constraint `raven-librarian`'s attach dialog works around by offering `.*`. |
| Uniform load-on-demand for Raven-server modules | **CONFIRMED** | No lazy loading. (A naive grep for "lazy" hits only the pangram in the API examples — "the quick brown fox jumps over the lazy dog".) |
| pillow-simd for faster PIL image processing | **CONFIRMED** | Not in `pyproject.toml`. |
| Robust public API auditing tool | **CONFIRMED** | No such tool in `raven/tools/`. |
| Fenced code block support in the Markdown renderer | **CONFIRMED** | The vendored renderer has no fenced-code handling at all. |
| Markdown tables don't render in the chat view | **CONFIRMED** | Likewise no table handling. |
| Context-window budgeting and conversation compaction (Librarian) | **CONFIRMED** | No compaction path exists. |
| Fleet-wide: shared two-phase DPG shutdown helper + audit | **CONFIRMED** | No shared helper in `common/gui/`; each app still spells out its own teardown. |
| Parse Gemma's inline tool-call spelling | **CONFIRMED** | Gemma appears only incidentally (its ghost `<think></think>`, its image-tiling scheme); no inline tool-call parsing. |
| Make the canned AI greeting optional | **CONFIRMED** | `llm_greeting` is a plain string with no way to switch it off. |
| MPS (Apple Silicon) device synchronization | **CONFIRMED** | `torch.cuda.synchronize` at four sites in `cherrypick/preload.py` with no MPS equivalent. MPS *detection* exists in `deviceinfo.py`, which is what makes this easy to mis-call — the item is about synchronization. (A naive `grep mps` also matches the tail of "timesta**mps**".) |
| Let the AI drive the constellation's own views (tools, and then voice) | **CONFIRMED** | The tool set is the six shipped in 0.2.9-dev — two network, three document, plus `get_current_time`. None touches a view. |

## What the reference checker flagged, and what it was worth

Eleven items name something that no longer resolves. Six became the MOVED verdicts above. The other five are
false positives worth recording so the next run does not re-litigate them:

- `librarian/cleanup.py`, `librarian/app.py` — real files, written repo-relative-ish rather than as full paths.
- `raven/common/gui/dpgstyle.py` — a file the item *proposes creating*, not one it claims exists.
- `icons.yml` — upstream FontAwesome's file, not ours.

## Coverage, stated plainly

**70 of 130 items have a verdict. 60 do not.** They were selected, not sampled: the ones today's and
0.2.8's work plausibly closed, everything the reference checker flagged, and then three batches chosen for
being settleable by a query rather than by reading. So the STALE rate here says nothing about the rest of
the file, and **an item's absence from this table is not evidence that it holds.**

The selection bias runs the other way from what you might expect, and batch 3 sharpened the point. Batches 1
and 2 were picked partly *because* the items looked closeable, and produced six STALE. Batch 3 was picked
purely for being queryable, and produced **none**. So the stale items are not spread evenly through the
file: they cluster in whatever recent work touched. A bulk "the old ones are probably stale" move in the
triage would be backwards — of the six STALE found so far, five sit in areas 0.2.8 or 0.2.9 changed.

Tally so far: 56 CONFIRMED (one of them halved), 6 MOVED, 6 STALE, 2 SUPERSEDED.

### What is left, and what it will take

Of the 60 unchecked, roughly:

- **~15 are settleable the same way** — "does this exist yet", one query each.
- **~25 need the item and the code read together.** No shortcut; this is where the remaining time goes.
  The ones deferred so far, and why:
  - *Audit unnamed lambdas* and *Adopt dotted import style* — the counts (214 lambdas, 1052
    `from X import Y`) mean nothing without knowing which instances the item considers wrong.
  - *Attach an image from a web URL* — staged attachments already carry a `provenance_url`, so part of it
    may exist.
  - *EU AI Act Article 50 compliance* — a disclosure **does** exist (`librarian/app.py:1167` cites Article
    50(1); `chatutil.py:306` emits AI-generated front-matter on export). Whether that satisfies the item is
    a question about the item's scope, not about the code.
  - *The 8/3 pass: bare DPG margins should name themselves* — the named constants exist and have 21 call
    sites, so this is partly done; the question is how many bare literals remain.
  - *The subtitle translator silently drops `=`*, *Librarian's help card has no room to describe
    attachments*, *torch / torchaudio CUDA version alignment*, *webfetch local (client-side) mode*,
    *Markdown ATX headings don't render*.
- **~20 are UNCHECKABLE from the tree** — rendering bugs in the Markdown widget, the turn-sequencing race,
  "reads as a hang", colourblind-safety. These want a running app or an answer from memory.

## The untested-modules row, and two ways of getting it wrong

Worth recording, because both errors are the kind a checker produces confidently.

**First attempt** looked for a file named `test_<module>.py` and reported eight untested modules, including
`layout_math`. Wrong: `layout_math.py` was tested all along by `test_viewport_math.py`, named after the
module's *previous* name. A test whose name has gone stale is invisible to a name-based check, and reads as
a coverage gap.

**Second attempt** searched test files for `import.*<module>`, and reported twenty-six untested modules. Also
wrong, and more embarrassingly: the pattern requires `import` to come *before* the module name, so it misses
every `from raven.common.gui.layout_math import ...` — which is the usual spelling. It flagged `numutils`,
`smoothvalue` and the whole `xdotwidget` package, all thoroughly tested.

**Third attempt** — does any test file mention the module name at all — gives five, which survives being
checked by hand.

The general lesson is the one from the reference checker earlier: **a checker's first number is a claim about
the checker.** Both wrong answers were plausible, neither raised an error, and the first one nearly went into
this report as a finding about `CLAUDE.md` being inaccurate. It was the *test filename* that was stale, not
the documentation.

That stale name is now fixed, along with three others, and the naming rule is written down in `CLAUDE.md`
under "Naming and placing a test module" so the next rename does not leave the same trap.

The brief predicted a meaningful STALE rate and explained why it is expected rather than embarrassing: many
of these were filed during a period when the standing instruction was never to get sidetracked, so things
were filed rather than fixed, and some were then fixed without the item being closed. Four of the five STALE
findings above are exactly that shape.

What is left is the per-item work: read the claim, find the code it rests on, check it. The unchecked 114
skew old — the March–June items are where the yield should be, and none of those were touched here beyond
what the reference checker saw.

## Re-running

`python investigations/todo-sweep-2026-08-10/check_references.py` regenerates the mechanical half against
whatever the tree looks like then. The verdict table above is hand-made and does not regenerate; treat it as
of 2026-08-10 and re-check anything acted on much later.
