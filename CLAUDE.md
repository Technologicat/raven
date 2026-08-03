# Raven - CLAUDE.md

## Project Overview
Local research assistant constellation. Privacy-first, 100% local.

**Components:**
- **Visualizer** (`raven/visualizer/`): BibTeX topic analysis, semantic clustering, keyword extraction. The original app. See `raven/visualizer/CLAUDE.md` for architecture.
- **Librarian** (`raven/librarian/`): LLM chat frontend with tree-structured branching history, hybrid RAG, tool-calling, message attachments (images on a VLM + text/PDF documents on any model, stored as content-addressed sidecars), avatar integration. See `raven/librarian/CLAUDE.md` for architecture.
- **Server** (`raven/server/`): Web API for GPU-bound ML models. Primary inference endpoint.
- **Client** (`raven/client/`): Python bindings for Server API.
- **Avatar** (`raven/avatar/`): AI-animated anime character (THA3 engine, lipsync, cel animations). Some avatar-related code (video postprocessor, colorspace) lives in Common for licensing reasons.
- **Common** (`raven/common/`): Shared utilities (video processing, audio, GUI widgets, networking, document text extraction — `docextract`: plain text + PDF via pypdf, the single extraction backend for both RAG ingestion and chat attachments). Mostly BSD, but **not uniformly** — `common/gui/xdotwidget` is LGPL-3.0-or-later (derived from xdottir, and through it from Jose Fonseca's `xdot.py`) and `common/video/upscaler` is MIT (matching Anime4K). Server and Avatar pose editor are AGPL-3.0. The full picture is in `TODO_DEFERRED.md`; `raven/vendor/README.md` covers the adopted tree.
- **Papers** (`raven/papers/`): Academic paper tools — arXiv search/download, bibliography converters (WoS, CSV, PDF, BibTeX burst).
- **Tools** (`raven/tools/`): Miscellaneous CLI utilities (CUDA check, audio device listing, image format conversion, dehyphenation).

## Where the non-source material lives

Four trees, sorted by what a document *is* rather than what it is about. Each has its own README; this is the
index.

- **`briefs/`** — prose. `design/` for sketches (direction clear, mechanism not), `summer_2026_librarian_extension/`
  for the current sprint's numbered implementation briefs, `done/` for closed ones, `reference/` for material we
  consult but did not produce (the EU AI Act summary, the DPG keycode table, an archived style snapshot).
- **`investigations/`** — things we measured, profiled or reproduced. **One directory per investigation, holding
  its write-up, its scripts and its data together**, because a measurement whose apparatus lives in another tree
  is not reproducible in practice however carefully it was written. Currently `context-injects`, `retrieval`,
  `tool_budget`, `vram`, `tha3-performance`, `anime4k-performance`, `dpg-focus`.
- **`TODO.md`** for planned work, **`TODO_DEFERRED.md`** for things noticed mid-task and set aside.
- **`dpg-notes.md`**, **`raven-style-guide.md`** — at the root because they are consulted constantly.

Two conventions worth knowing before adding to any of them:

- **Keep an artifact with what produced it.** This is why `investigations/` exists, and it applies wherever the
  artifact lives — a completed brief with apparatus becomes a directory too (`briefs/done/dpg-markdown-bullet/`
  is a write-up plus its reproduction script). A **shared** instrument is pointed at by path, not copied into
  every bundle that used it.
- **Record the link in the bundle's README**, naming each script and what it answers. The doc↔script connection
  was not recorded consistently in the past, and recovering it meant asking git what landed in the same commit
  as each script. Writing it down is what stops that recurring.

## Build and Development

Uses PDM with `pdm-backend`. **Python 3.11–3.12** (see `pyproject.toml`: `requires-python = "<3.13,>=3.11"`). Optional CUDA extras via `pdm install -G cuda`.

### Why the 3.12 upper cap

The cap comes from `kokoro` (Kokoro TTS) and its phonemizer `misaki`, which currently require `<3.13`. Raven's own code and every other dependency (`mcpyrate`, `unpythonic`, `torch`, `Pillow`, `numpy`, …) already support Python 3.13 and 3.14. The plan to lift the cap has two branches:

- **(a)** Kokoro/Misaki upstream expand their supported Python range — in which case we just bump `requires-python` and widen the CI matrix.
- **(b)** If those projects look dead after a reasonable wait, we vendor both. Kokoro is the TTS engine, Misaki is its English phonemizer; together they're self-contained enough to be absorbed into `raven/vendor/` alongside `tha3/`, `DearPyGui_Markdown/`, etc.

Until one of those branches lands, **don't add `3.13`/`3.14` to the CI matrix** — it would fail at dependency resolution time. The test CI currently works around this by using `pip install -e . --no-deps` and hand-picking a minimal dependency subset for the test suite, which avoids pulling in kokoro/misaki at all. That's how the test matrix can stay lightweight even though kokoro lives in the full `[project] dependencies`.

### Working-tree state: `config.py` files are edited in place

Raven is configured via in-place edits to tracked `config.py` files — paths, model choices, hardware-specific tweaks. On any dev machine, expect some subset of the following to show up as `M` in `git status` as the **normal steady state**, not as a pending change that needs committing:

- `raven/client/config.py`
- `raven/librarian/config.py`
- `raven/visualizer/config.py`

The specific files and the specific contents differ between dev machines; the pattern is the same everywhere — at least some config.py somewhere carries local overrides.

**Implication for `git add`**: add specific files by name. **Never** `git add -A`, `git add .`, or `git add raven/`. If a commit you're working on touches one of these files coincidentally (e.g. a refactor sweeps through them), check with me before staging — there may be an unrelated local override mixed in that shouldn't be part of the commit.

Version is defined in `raven/__init__.py` (`__version__`), read by PDM via `[tool.pdm.version]` in `pyproject.toml`. Tag format: `vX.Y.Z`.

```bash
pdm install              # creates .venv/ and installs deps
pdm use --venv in-project
```

Prefix commands with `pdm run` if the venv is not active.

Entry points defined in `pyproject.toml` under `[project.scripts]` — main apps are `raven-visualizer`, `raven-librarian`, `raven-server`, `raven-importer`, `raven-minichat`, `raven-xdot-viewer`, `raven-cherrypick`, `raven-conference-timer`, `raven-avatar-pose-editor`, `raven-avatar-settings-editor`.

### Running Tests

```bash
pytest                   # runs all tests (currently minimal coverage)
```

### Linting

```bash
ruff check <changed .py files>   # primary linter (config in pyproject.toml)
```

Legacy `flake8rc` also present (used by Emacs flycheck, not by CI or CC).

### Workflow Rules

1. **Lint after every code change**: `ruff check <changed .py files>`. Do this before review, testing, or committing. Catches unused imports and dead names early.

### CHANGELOG layout: group by component

Raven ships many separate user-facing apps, so within each of **Added** / **Changed** / **Fixed**, entries are grouped under an italic component header and the per-entry `*Raven-<app>*:` prefix is dropped — the header carries it. Entries then read as continuations of the header, so they start lowercase.

Component order is fixed, so a reader learns where to look: *Raven-librarian*, *Raven-visualizer*, *Raven-server*, *Raven-avatar*, *Raven-cherrypick*, *Raven-arxiv-download*, then *Constellation-wide* for anything cross-cutting (install, device strings, CLI options shared by every app, client-side HTTP behavior). Omit a component that has no entries in that section. An entry spanning two tools goes under the primary one and names the other inline ("with `raven-wos2bib`: …").

**File a new entry into its group when you write it.** The failure this prevents: 0.2.8 accumulated 58 flat entries — 24 of them opening with `*Raven-librarian*:` — before anyone noticed the prefix was a heading doing prose duty, and regrouping after the fact is a large, error-prone reshuffle that has to be verified entry by entry.

This is Raven-local, not fleet-wide: elsewhere in the fleet a project *is* the component, so a header would be noise. Wording rules (density, nesting, users-not-commits, "was it broken in the last tagged release?") are fleet-wide and live in the `changelog` skill.

### Live GUI testing on a shared desktop

Raven's apps are DPG, so verifying GUI work means running them — and the agent and the human are on the *same X session*. Keyboard focus is therefore a shared, single-holder resource: a window that maps or gets activated takes focus away from wherever the human is typing, and their next keystrokes land in the app instead of their editor or terminal. (Observed the obvious way: a launched Librarian window swallowed a half-typed message and its Enter, which sent an empty chat turn.)

- **Screenshots need no focus** — `import -window <id> shot.png` captures an unfocused window fine. Prefer a screenshot-only check; it is never intrusive. `wmctrl -l` or `xdotool search --name <title>` gets the window ID.
- **Synthetic input does need focus.** GLFW ignores the `XSendEvent`-based `xdotool key --window <id>`, so driving the GUI means really activating the window. When doing that: **say so first**, keep the burst short, and **restore focus afterwards** — capture `PREV=$(xdotool getactivewindow)` *before* launching or activating, and `xdotool windowactivate --sync "$PREV"` when done.
- **Announce a launch even without input injection**, because the mapping window itself steals focus.
- **When tuning placement or sizing, render the candidates side by side** into one image rather than asking about them one at a time. The eye ranks a comparison; it can't rank a sequence, so serial single-shot proposals cost a restart per candidate.
- **Get click coordinates from `xwininfo -id <wid>`, not from `xdotool getwindowgeometry`.** A screenshot taken with `import -window <id>` is in *client-area* coordinates, so `screen = xwininfo "Absolute upper-left X/Y" + the coordinate read off the screenshot`.

  `xdotool getwindowgeometry` reports something else. Measured across every decorated window on the desktop, its `Position` equals `xwininfo`'s **Absolute + Relative upper-left** — the client's offset inside its window-manager frame, counted twice. The error is therefore exactly the decoration size (32 px for a plain title bar; 10 px + 40 px on a Librarian window), which is roughly one toolbutton — enough to land on empty panel, where a click silently does nothing rather than failing loudly. The one window that matched `xwininfo` was the unreparented desktop, which has no frame to double-count. (The arithmetic is measured; that reparenting is the mechanism inside xdotool is inferred from it, not read from its source.)
- **Confirm a click actually landed, don't assume it.** Put a sentinel in the clipboard first (`printf SENTINEL | xclip -selection clipboard`) so an unchanged clipboard is distinguishable from a successful copy of stale content — otherwise a missed click reads as a pass against whatever the previous step left there.
- **Never `pkill -f raven-<app>`.** The pattern matches the agent's own shell command line, so it kills the invoking shell (exit 144) and usually leaves the app running. Select real PIDs instead: `pgrep -af raven-librarian | awk '$2 ~ /python/ {print $1}' | xargs -r kill`.
- Needs `xdotool` and `xclip` installed (clipboard round-trip: press the hotkey, then `xclip -o -selection clipboard`). This is how a clipboard-export feature gets verified end-to-end rather than only through unit tests.

### DPG Pitfalls

**Before editing any DPG code, read `dpg-notes.md` first** (project root) — the full DPG reference: threading model, callback dispatch, `split_frame` mechanics, texture upload ordering, keyboard input / keycode traps, window sizing gotchas, diagnosing background-task races. "DPG code" = anything importing `dearpygui`, the render loop, key/mouse handlers, or texture / `split_frame` work. The pitfalls listed below are an index, not a substitute for the full notes. **When you discover a new DPG gotcha, record it in `dpg-notes.md`** (and add a one-line pointer below if it's pitfall-grade).

1. **DPG threading — push work to background threads aggressively.** Unlike most GUI toolkits, DPG allows all operations from background threads: creating/deleting items, setting values, creating OpenGL textures. Resist the "standard GUI toolkit" instinct to marshal everything to the main thread — doing work on background threads simplifies code and reduces GUI stutter, especially when the heavy lifting is non-Python (C/CUDA) and can release the GIL.
2. **`dpg.split_frame()` — not in the render loop thread.** `split_frame()` waits for the render loop to complete one frame. Safe to call from background threads, DPG event callbacks, and frame callbacks (DPG dispatches these on a separate thread). **Deadlocks** if called from code that runs synchronously in the render loop — i.e. anything in the `while dpg.is_dearpygui_running(): dpg.render_dearpygui_frame()` loop body (e.g. animation frame updaters), or before the render loop starts (startup code). Common use: call from a background thread after creating textures, to ensure DPG processes them before the next render.
   - **The restriction is enforced, not lifted.** Use `raven.common.gui.utils.split_frame(operation=..., required=...)` in preference to the bare `dpg` call wherever the calling thread isn't obvious from two lines of context. It cannot wait in the render loop either — nothing can — but it *detects* that it was called there and reports it instead of hanging: `RuntimeError` when waiting is load-bearing (`required=True`, the default), or a warning and a stale-geometry fallback when the wait only improves the result (`required=False`). The trade is a silent hang for a named failure, not a lifted constraint. One predicate (`guiutils.is_render_thread`) covers both this pitfall and pitfall 4, since startup runs on the main thread too. Rationale and per-site policy in `dpg-notes.md`.
3. **`dpg.set_frame_callback(N, cb)` — one callback per frame number.** Only one callback can be registered for any given frame N. A second `set_frame_callback(N, ...)` silently overwrites the first. If you need multiple actions at the same frame, combine them into a single callback, or use different frame numbers.
4. **Defer startup work that may show error dialogs to a frame callback.** The modal messagebox uses `split_frame`, which deadlocks before the render loop is running. If startup code (e.g. loading a file from a CLI argument) may need to show an error dialog, defer it to `dpg.set_frame_callback(N, ...)` so the render loop is active. This is a standard Raven pattern — see `raven.avatar.settings_editor.app` and `raven.xdot_viewer.app`.
5. **DPG widget IDs must be unique — violating this crashes the process, not raises an exception.** Combined with Python's lazy garbage collection, explicit `dpg.delete_item(...)` does not guarantee the ID is free for reuse: the old widget may still be in DPG's registry for some unbounded time after the delete call. Raven's defensive pattern for any widget that gets dynamically recreated (tooltip groups, info-panel content, per-entry groups, etc.) is **version-counted tags**: every rebuild increments a monotonic counter, and every tag created during that rebuild embeds the counter (e.g. `f"cluster_{cid}_item_{data_idx}_annotation_title_build{build_number}"`). Even if the old widgets aren't collected yet, the new tags won't collide. The counter increments on *every* build attempt, including cancelled ones, so a cancelled build's partial widgets can't collide with the next build either. For the top-level "current vs. previous" swap (where the slot itself has a stable identity), track the current widget *ID* in a module-level Python variable rather than relying on an alias rebind — `dpg.set_item_alias(new_item, existing_alias)` does not reliably rebind after the aliased item is deleted.
6. **When rebinding an alias across a swap, delete the old item by widget ID, not by alias string.** The working pattern is: hold the current widget ID in a Python variable, call `dpg.delete_item(old_id)`, then `dpg.set_item_alias(new_id, alias_str)`. Calling `dpg.delete_item(alias_str)` instead appears to leave the alias→id mapping partially dirty, so the subsequent `set_item_alias` lands in an inconsistent state and later lookups by that alias return `0` (→ `configure_item(0, ...)` raises `SystemError: Item not found: 0`). This is observable even on DPG versions that fixed the older manual-alias-cleanup bug (hoffstadt/DearPyGui#1350). See `raven.visualizer.info_panel`'s content swap (app.py `_update_info_panel`) and `raven.visualizer.annotation`'s `_current_group` handling for the working pattern.
7. **Focus is not the caret, and `focus_item` cannot focus a child window.** ImGui auto-focuses the first navigable item of a window, so a text field reports `is_item_focused` True within a few frames of startup with nobody having touched it — a global hotkey handler gated on that silently swallows every key it delegates to the field. Gate on `dpg.is_item_active` instead, which is True only while the field owns the caret. Separately, `dpg.focus_item` on a *child window* does not focus it: focus lands on the enclosing window's first navigable item and is **activated**, so "park focus on the scroll panel" hands the caret to the composer instead. Park on a real widget; a focused button is safe (DPG leaves ImGui's keyboard-nav activation off, so it ignores Space/Enter). See `dpg-notes.md` "Keyboard input".
8. **Keyboard input has two non-obvious traps.** (a) *Stale key constants*: some `dpg.mvKey_*` values are pre-2.0 codes that no longer match what a handler receives in `app_data` — Page Up arrives as `517` not `mvKey_Prior` (266), Page Down `518` not `mvKey_Next` (267), plus LWin/RWin and Quote/Colon/Plus/Tilde. Comparing against the constant silently never matches; compare against the literal code. (b) *Same-frame dispatch is by keycode, not press order*: a keyless key-press handler fires once per key pressed that frame in ascending keycode order, so two near-simultaneous keys — where one handler mutates state another reads — interact as if the lower-keycode key came first (e.g. cherrypick's fast `C`+`Right` tagged the *next* image until navigation was deferred a frame). See `dpg-notes.md` "Keyboard input"; full table + reproduction in `briefs/reference/dpg-keycodes.md`.

## Architecture

### Server/Client Split
All ML inference in `raven/server/modules/` when Server is running:
- `tts.py` - Kokoro TTS with phoneme timestamps (needed for lipsync)
- `stt.py` - Whisper speech recognition
- `embeddings.py` - Sentence embeddings (currently snowflake-arctic; Nomic-embed-text v1.5 + vision v1.5 migration pending, bundled with Visualizer importer rework)
- `translate.py` - Neural machine translation
- `classify.py` - Sentiment/emotion classification, to control avatar's facial expression
- `sanitize.py` - Text cleanup (dehyphenation etc.)
- `natlang.py` - spaCy NLP analysis
- `websearch.py` - Web search tool for LLM
- `avatar.py`, `avatarutil.py`, `imagefx.py` - Avatar rendering pipeline

Client apps call Server via `raven/client/api.py`. Server can run on a different machine (trusted network only — no encryption). When Server isn't running, Visualizer's importer uses the `MaybeRemoteService` pattern to load models in-process, making the Visualizer deployable standalone.

### The Raven Way: three-layer module organization for ML-bearing subsystems

Each subsystem that has both a local (in-process) and remote (HTTP) mode follows the same three-layer pattern:

1. **`raven.common.<subsystem>`** — the actual implementation, pure library code, runs on whichever machine calls it. Framed as "explicit local mode", but the framing is incidental: this is where the work happens regardless of which process is doing it.
2. **`raven.server.modules.<subsystem>`** — the server-side subsystem module, delegating to `raven.common.<subsystem>`. Defines request handlers but not the routes themselves — routes and Flask plumbing live in `raven.server.app`, which wires each `modules.<subsystem>` handler onto its `/api/<subsystem>/...` URL. On the server, "local" means "server-side" — the server loads the same common-layer module the client would have loaded.
3. **`raven.client.api.<subsystem>`** — explicit remote mode. Client functions that make HTTP calls to the server. Mirrors the server's API surface one-for-one. In practice most subsystems are *inlined* directly into `raven.client.api` (they're small — a handful of request-sending functions). Only `tts` got large enough to warrant its own `raven.client.tts` module, re-exported through `raven.client.api`. Whether we should split the others out for symmetry with `raven.server.modules.*` is an open design question; inlined is the current reality.
4. **`raven.client.mayberemote.<Subsystem>`** — transparent remote/local mode. A class per subsystem; in remote mode it delegates to `raven.client.api.*`, in local mode it delegates to `raven.common.<subsystem>.*`. Callers don't need to know which mode is active.

Concrete example — `speech.tts`:

| Layer | Module | Role |
|---|---|---|
| Common (impl) | `raven.common.audio.speech.tts` | `prepare` / `prepare_cached` (TTSResult), `prepare_encoded_cached` (EncodedTTSResult); `encode`, `decode`, `synthesize`, `finalize_metadata` |
| Server module | `raven.server.modules.tts` | request handlers; uses common `synthesize_iter`, `audio_codec.encode` |
| Server app | `raven.server.app` | registers `/api/tts/...` routes onto the handlers |
| Client remote | `raven.client.tts`, re-exported via `raven.client.api` | `tts_prepare` / `tts_prepare_cached` (EncodedTTSResult), `tts_prepare_decoded_cached` (TTSResult), `tts_list_voices`, `tts_speak`, … → HTTP |
| Client mayberemote | `raven.client.mayberemote.TTS` | pure 2×2 dispatch, no cache state of its own; delegates to the cached bottom functions per (location, shape) |

**Caching strategy** (used if a subsystem needs it — currently only `tts`; other subsystems like `nlp`, `stt`, `embeddings` don't cache because their inputs are essentially never repeated in a session). When a subsystem has two natural output shapes (e.g. raw vs. encoded for TTS), caching lives in the bottom layers, not in mayberemote. Each of `common` and `client.remote` exposes:

- The "natural" cached shape for that side — `TTSResult` in common (local synthesizes float natively), `EncodedTTSResult` in client.remote (server returns encoded over the wire).
- The other shape, composed on top via `encode` / `decode`, also cached.

Mayberemote's `synthesize(format=...)` is then pure 2×2 dispatch — it picks one of the four cached bottom functions by `(location, shape)`. No cache state in the mayberemote class itself. This keeps the cache next to the engine (natural single-source-of-truth) while still giving the mayberemote caller the same "call it twice, second one is free" guarantee regardless of mode.

Same shape applies to `nlp` (`nlptools` ↔ `natlang`), `stt`, `embeddings`, `sanitize`, etc. — cross-check `raven.client.mayberemote` for the current set.

**Implications:**
- New ML work goes in `raven.common.<subsystem>` first. The server module and mayberemote wrapper come after and are thin shims.
- Playback / audio output stays in `raven.client.*` even when synthesis is local — the user is on the client machine, audio hardware is local by definition.
- `raven.client.tts.tts_prepare` and friends are **not** obsolete when `MaybeRemote.TTS` exists. They remain the explicit-remote path, used by `MaybeRemote` itself and by any app that wants to force remote mode.
- Data conversion at the boundary: in-process uses dataclasses (`TTSResult`, `WordTiming`), HTTP wire uses JSON-friendly dicts. Converter functions (`decode`/`encode`, `finalize_metadata`) live in the common layer — neither "local" nor "remote", they're shape conversions.
- Engine-agnostic data shapes live in their own module, separate from the engine wrapper. For TTS, `WordTiming`, `TTSSegment`, `TTSResult`, `EncodedTTSResult` are in `raven.common.audio.speech.datatypes`; only `TTSPipeline` (which holds a `kokoro.KPipeline`) stays in `raven.common.audio.speech.tts`. This lets consumers that only need the shapes (e.g. `lipsync`) import them without dragging in Kokoro/PyAV/huggingface_hub.

### Common Subsystems
- `raven/common/video/` - Postprocessor, upscaler (PyTorch Anime4K), colorspace conversions, cel compositor
- `raven/common/audio/` - Player, recorder, codec (PyAV streaming)
- `raven/common/gui/` - Custom DearPyGui widgets (VU meter, GUI animation framework, messagebox)

### Vendored / adopted dependencies (`raven/vendor/`)

**`raven/vendor/` is *adopted* code — effectively ours to fix and extend, not pristine upstream snapshots.**
Each of these has already diverged from upstream with Raven-specific robustifications and features (see notes
below). So when you hit a bug *in* vendored code, fix it like any other Raven code (with the usual care for a
foreign-API layer — match the wrapped library's conventions). We may upstream a given change later, or not;
either way, treat the in-tree copy as the source of truth. Don't reach for "it's vendored, leave it alone."

- `tha3/` - Talking Head Anime 3 neural network (avatar animation). Switched `no_grad` → `inference_mode` in the
  hot paths for a few-percent speedup.
- `DearPyGui_Markdown/` - MD renderer, substantially robustified for Raven's background-threaded rendering
  (most call sites guarded with `guiutils.nonexistent_ok` / `does_item_exist` against DPG's lazy GC). Known
  remaining issue: the persistent render worker thread (`CallInNextFrame._worker`) doesn't participate in app
  shutdown — it keeps calling DPG (incl. `split_frame`) during teardown, which can segfault on a mid-boot close
  while a URL-heavy message is mid-render. Tracked in `TODO_DEFERRED.md` (fleet shutdown item).
- `file_dialog/` - File dialog, extended (sortable, animated OK button, click twice when overwriting).
- `anime4k/` - PyTorch port of Anime4K upscaler (extracts kernels from GLSL), slightly cleaned up.
- `kokoro_fastapi/` - Streaming audio writer for TTS over network.
- `IconsFontAwesome6.py` - Icon font (note: outdated version).

## Code Style
All new and modified code must follow `raven-style-guide.md` (in the project root). **Read the full guide before implementing a new app.** The summary below covers the most commonly needed conventions.

- Impure functional, Lispy (closures, `unpythonic` patterns)
- `unpythonic` pure-Python features are fair game. Currently used: `env` (namespace), `Timer` (benchmarking), `@call` (scoping), `box`/`unbox`, `sym`, `dyn`. Other features welcome where they improve clarity. **Do not** use the macro layer (`unpythonic.syntax`) or features that primarily serve as macro backends (e.g. `let` bindings — these are readable only through the macro surface syntax).
- OOP where appropriate (GUI components, stateful objects)
- Config via Python modules (`config.py` files, not YAML/JSON)
- Type hints on all new and modified functions (public and internal). Existing untyped code can be left as-is unless you're already editing it. Use the modern spelling — `X | None` (not `Optional[X]`), `list[X]`/`dict[K, V]` (not `typing.List`/`Dict`); the codebase is mid-migration. Full guidance in `raven-style-guide.md` under "Type hints".
- `__all__`: all public symbols must be listed in `__all__` (PEP 8). Whether locally defined or re-exported, doesn't matter. This allows star-importing a module in a REPL to bring in its public API only.
- Imports: prefer `import module` + `module.func()` (dotted style) over `from module import func`. Makes it clear at the call site where a function comes from. For modules with ambiguous names, use an alias: `from ..common.gui import utils as guiutils`, `from ..server import config as server_config`.
- Naming: don't repeat the module name in function names. With dotted imports, `lanczos.resize()` reads better than `lanczos.lanczos_resize()`. The module provides the namespace.
- Docstrings: use raw backtick names (`` `func_name` ``), not RST cross-reference markup (`:meth:`, `:func:`). The codebase is read as raw code, not via Sphinx. Single space after sentence-ending period (European convention), not double.
- Log messages: prefix with the function name (or `ClassName.method_name` for methods), e.g. ``logger.warning("TriageManager.scan: ...")``. Python's logging already shows the module name, but not the function/method name.
  - Background tasks: include the instance name — ``logger.info(f"speak_task: instance {task_env.task_name}: message")``. This groups log output from the same task instance when multiple run concurrently.
  - Classes with multiple instances: include instance identification — a natural name attribute (e.g. ``instance '{self.base_dir.name}'``) or ``instance 0x{id(self):x}`` as fallback. Not needed for obvious singletons (e.g. GUI app classes).
  - Exceptions: use ``{type(exc)}: {exc}`` in log messages, not bare ``{exc}``. The type name is cheap insurance against uninformative `str()` output.
- Timers: use the right clock for the job. ``time.perf_counter()``/``perf_counter_ns()`` for benchmarks (highest resolution, monotonic). ``time.monotonic()``/``monotonic_ns()`` for elapsed time in app code (animation, polling, timeouts — immune to NTP adjustments). ``time.time()``/``time_ns()`` only for wall-clock timestamps that need epoch identity (chat message timestamps, persistent records).
- License DRY: the project-level `LICENSE.md` is the single source of truth (2-clause BSD). Don't repeat the license in individual module docstrings unless a module has a *different* license from the project default (e.g. AGPL for Server and Avatar pose editor).
- Blank lines in code are paragraph breaks — insert when the topic changes, not mechanically (e.g. not "always before `return`").
- Properties: define as `def get_x(...) ... def set_x(...) ... x = property(fget=..., fset=..., doc=...)` instead of the `@property`/`@x.setter` decorator syntax.
- DPG string tags: any line that mentions a DPG string tag must carry a ``# tag`` comment (for greppability across the codebase). The only exception is a line that already passes ``tag=...`` as a keyword argument — the word "tag" is right there in the parameter name, so the comment would be redundant. This applies to any API that takes a DPG tag/alias: ``dpg.add_*``, ``dpg.hide_item``, ``dpg.show_item``, ``dpg.set_value``, ``dpg.set_item_pos``, ``dpg.get_item_rect_size``, ``dpg.does_item_exist``, ``guiutils.wait_for_resize``, etc. If the line already has a trailing comment, keep both: ``dpg.show_item("foo_window")  # tag  # existing note``.
- **Changing Raven's own library code is fine when it yields a better design.** `raven.common.*` and friends are first-party: if a caller needs a shape the library does not offer, prefer improving the library over working around it at the call site. Much of the caution that would apply in a corporate multi-team setting — freeze the interface, add an adapter, coordinate with owners — has no counterpart here; this is a solo-maintained project, and every consumer is in the same tree and greppable. The vendored-code note below says the same thing about `raven/vendor/`, and first-party code is the easier case, not the harder one. Not a licence to churn: the test is whether the design comes out cleaner, not whether the change is possible.
- Contract-style preconditions/postconditions would be useful, but mostly not implemented yet

## Key Patterns

### DearPyGui App Structure
See `dpg-notes.md` "Raven DPG app structure" section for layout patterns, startup sequence, background work, thread safety, DPG item management, and texture handling.

### Avatar Lipsync
TTS (Kokoro) provides timestamped phonemes → mapped to mouth morphs → THA3 animator. Audio playback occurs on the client side.
This coupling limits TTS engine choices (most don't expose timestamped phoneme data).

## Current State

### Well-structured (target style)
- `raven/librarian/` - Clean module separation (~14,600 lines across 15 modules, measured 2026-08-03). Note it has outgrown the per-module guideline below in several places — `chat_controller.py` is 2769 lines and `llmclient.py` 2181 — without losing the layering, which is the property that made it the target style. Size is a smell here, not a verdict. See `raven/librarian/CLAUDE.md` for the layer map.

### Needs refactoring

Target ~700 lines per module as a guideline, not a hard limit — some modules can be longer when appropriate (e.g. lots of simple related code).

- `raven/visualizer/app.py` - 1912 lines. The split into `info_panel`, `selection`, `plotter`, `annotation`, `word_cloud`, `entry_renderer` and `app_state` has landed; what remains is ordinary size rather than a god object. See `raven/visualizer/CLAUDE.md` for the module map.
- `raven/visualizer/info_panel.py` - 1518 lines, the largest of the extracted modules; a candidate for further splitting, but not urgent.
- `raven/visualizer/importer.py` - 1260 lines, pipeline architecture, lower priority but could benefit from stage separation

### Test coverage

68 test modules as of 2026-08-03, ~1600 tests. Library and utility code is broadly covered; what is
untested is the GUI layer and the Visualizer.

- **`common/`** — numutils, smoothvalue, utils, bgtask, deviceinfo, docextract, logsetup, netutil, nlptools, readcsv, running_average, stringmaps, text_normalize, text_speakable; `audio/` (codec, resample, utils) and `audio/speech/` (tts, stt, lipsync, and a TTS→STT round trip); `image/` (codec, lanczos, utils); `video/` (colorspace, compositor, postprocessor, upscaler); `gui/` (animation, messagebox, utils, viewport_math, and all of `xdotwidget/`).
- **`librarian/`** — chattree, chatutil, hybridir, appstate, scaffold, llmclient, cleanup, imagestore, sidecarstore, textfilestore.
- **Elsewhere** — `client/` (api, mayberemote), `papers/*`, `cherrypick/*`, `server/webfetch`, `xdot_viewer/dot_utils`.

What is **not** covered:

- **Visualizer has zero tests.** Still the biggest gap, and the refactor that motivated writing them
  landed without them — so what they would pin now is the new module boundaries rather than a rewrite
  in flight.
- **The DPG frontends**: librarian `app`, `chat_controller`, `cleanup_dialog`, and every Visualizer GUI
  module. **Not because DPG resists testing** — it runs without a mapped window, and
  `common/gui/tests/` (messagebox, animation, utils — 18 tests) already drives a real context with an
  unmapped viewport. See `dpg-notes.md`, "Testing DPG code". The barrier is that nobody has written them
  for the large frontend modules, which is a different and more tractable problem than "untestable".
  - **Caveat: those 18 tests never run in CI.** `dearpygui` is not in `requirements-ci.txt`, so their
    module-level `importorskip` fires on every run and they execute only on a dev machine. Whether the
    toolkit can initialize on a headless runner is untested — see `TODO_DEFERRED.md`.
  - Splitting an operation from its dialog is what makes the operation testable at all; `cleanup.py` /
    `cleanup_dialog.py` is the worked example, and its module docstring explains why.
- **`librarian/minichat`** — the readline REPL, and the odd one out: no DPG anywhere in it, so none of the
  above applies. It is a terminal app with the same backend as the GUI, which makes it the *cheapest*
  frontend to test rather than the hardest. Untested because nobody has, not because anything is in the way.
- `config.py` modules, which are configuration-as-code and carry local overrides anyway.

## Upstream warning noise in `pytest raven/`

The pytest summary normally shows a handful of `DeprecationWarning`/`UserWarning` captures. They look alarming but are **all upstream** and not fixable from raven's side. Catalogued here so we don't re-investigate each time. (This subsection is temporary; eventually factor it out to a dedicated `.md`.)

- **`DeprecationWarning: builtin type SwigPyPacked/SwigPyObject has no __module__ attribute`** — from `sentencepiece`, whose Python wrapper is SWIG-generated. Verify with `find .venv -name "*.so" | xargs -I{} sh -c 'if strings "{}" 2>/dev/null | grep -q swigvarlink; then echo "{}"; fi'`. Python 3.12+ warns when built-in types don't set `__module__`; SWIG's generated helper types (`SwigPyPacked`, `SwigPyObject`, `swigvarlink`) pre-date that convention. Upstream fix has to happen in the SWIG project itself; every SWIG-wrapped library inherits the warning. `sentencepiece` is a transitive dep via NLP tokenizers (`transformers`, `kokoro`'s phonemizer chain).

- **`DeprecationWarning: torch.jit.script is deprecated`** — from `transformers` (HuggingFace). Many of its model files use `@torch.jit.script` as a decorator at module load: `deberta`, `deberta_v2`, `gpt_bigcode`, `zoedepth`, `sew_d`, `vits`, `sam3_video`, … When raven's tests import `sentence-transformers` (via `raven.librarian.hybridir` for embeddings), transformers eagerly loads these model modules and the decorators fire. Verify with `grep -rn "@torch.jit.script" .venv/lib/python3.12/site-packages/transformers/`. Upstream fix waits on HuggingFace migrating these decorators to `torch.compile`/`torch.export`. Raven's own code no longer calls `torch.jit.script`.

- **`UserWarning: pkg_resources is deprecated as an API`** — from `pygame` 2.6.1 (currently the latest on PyPI). Its `pkgdata.py` still imports from `pkg_resources`. Upstream fix waits on a pygame release that stops using it. Pinning `Setuptools<81` would silence it but isn't worth the collateral; just wait for the next pygame.

### Fixed locally (for reference)

- **`RuntimeWarning: divide by zero encountered in divide` — `raven/common/numutils.py:psi()`**: the mollifier helper computes `np.exp(-1.0 / x**m) * (x > 0.0)` and relies on the `(x > 0.0)` mask to zero the divide-by-zero. A previous attempt used `warnings.filterwarnings(..., module="__main__")` which silently failed (numpy emits the warning from its own internal module, not `__main__`). Correct fix: `with np.errstate(divide='ignore', invalid='ignore'):` — numpy's own mechanism for suppressing float-error warnings within a dynamic extent.

## LLM Backend
Uses text-generation-webui with OpenAI-compatible API.
Recommended model: Qwen3-VL-30B-A3B (24GB+ VRAM) or Qwen3-VL-4B (8GB VRAM).

## Known Issues / TODOs
- Visualizer: the `app.py` split has landed (see `raven/visualizer/CLAUDE.md` for the module map). What remains is ordinary tidying — `info_panel.py` at 1518 lines is the next split candidate, and `importer.py` could use stage separation — not a god-object rescue
- Visualizer has zero tests (the librarian gaps this used to list — `scaffold`, `appstate`, `llmclient` — are all covered now)
- DearPyGui_Markdown URL highlight bug (threading-related, untracked)
- FontAwesome version outdated
- Hindsight integration pending (PDM dependency conflicts; likely separate container with optional backend, keeping BM25+vector backend as primary)
- TTS engine expansion limited by phoneme timestamp requirement
- Many `# TODO: DRY duplicate definitions for labels` scattered through Visualizer `app.py`
- Annotation tooltip help section rebuilt every time (could be static with show/hide)
- `_update_info_panel` race condition: current item highlight sometimes doesn't update immediately after selection change
- Search match scrolling race condition: hammering the button can error out (`info_panel.py:670`/`685` — the code moved there in the refactor; the old `app.py:2978` pointer was past EOF. Not re-verified since the move, so it may or may not survive)
- XDot viewer: GraphViz `--concentrate` produces near-miss edge endpoints (0.02–0.09 graph units off) at edge split/merge points, visible as small gaps at high zoom. This is a GraphViz precision issue in the xdot data, not a rendering bug.
