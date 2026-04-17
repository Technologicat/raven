# Raven - CLAUDE.md

## Project Overview
Local research assistant constellation. Privacy-first, 100% local.

**Components:**
- **Visualizer** (`raven/visualizer/`): BibTeX topic analysis, semantic clustering, keyword extraction. The original app. See `raven/visualizer/CLAUDE.md` for architecture.
- **Librarian** (`raven/librarian/`): LLM chat frontend with tree-structured branching history, hybrid RAG, tool-calling, avatar integration. See `raven/librarian/CLAUDE.md` for architecture.
- **Server** (`raven/server/`): Web API for GPU-bound ML models. Primary inference endpoint.
- **Client** (`raven/client/`): Python bindings for Server API.
- **Avatar** (`raven/avatar/`): AI-animated anime character (THA3 engine, lipsync, cel animations). Some avatar-related code (video postprocessor, colorspace) lives in Common for licensing reasons.
- **Common** (`raven/common/`): Shared utilities (video processing, audio, GUI widgets, networking). BSD-licensed; Server and Avatar pose editor are AGPL.
- **Papers** (`raven/papers/`): Academic paper tools — arXiv search/download, bibliography converters (WoS, CSV, PDF, BibTeX burst).
- **Tools** (`raven/tools/`): Miscellaneous CLI utilities (CUDA check, audio device listing, image format conversion, dehyphenation).

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

### DPG Pitfalls

See `dpg-notes.md` (project root) for the full DPG reference — threading model, callback dispatch, `split_frame` mechanics, texture upload ordering, window sizing gotchas. The notes below are the key pitfalls distilled from that reference.

1. **DPG threading — push work to background threads aggressively.** Unlike most GUI toolkits, DPG allows all operations from background threads: creating/deleting items, setting values, creating OpenGL textures. Resist the "standard GUI toolkit" instinct to marshal everything to the main thread — doing work on background threads simplifies code and reduces GUI stutter, especially when the heavy lifting is non-Python (C/CUDA) and can release the GIL.
2. **`dpg.split_frame()` — not in the render loop thread.** `split_frame()` waits for the render loop to complete one frame. Safe to call from background threads, DPG event callbacks, and frame callbacks (DPG dispatches these on a separate thread). **Deadlocks** if called from code that runs synchronously in the render loop — i.e. anything in the `while dpg.is_dearpygui_running(): dpg.render_dearpygui_frame()` loop body (e.g. animation frame updaters), or before the render loop starts (startup code). Common use: call from a background thread after creating textures, to ensure DPG processes them before the next render.
3. **`dpg.set_frame_callback(N, cb)` — one callback per frame number.** Only one callback can be registered for any given frame N. A second `set_frame_callback(N, ...)` silently overwrites the first. If you need multiple actions at the same frame, combine them into a single callback, or use different frame numbers.
4. **Defer startup work that may show error dialogs to a frame callback.** The modal messagebox uses `split_frame`, which deadlocks before the render loop is running. If startup code (e.g. loading a file from a CLI argument) may need to show an error dialog, defer it to `dpg.set_frame_callback(N, ...)` so the render loop is active. This is a standard Raven pattern — see `raven.avatar.settings_editor.app` and `raven.xdot_viewer.app`.

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

### Common Subsystems
- `raven/common/video/` - Postprocessor, upscaler (PyTorch Anime4K), colorspace conversions, cel compositor
- `raven/common/audio/` - Player, recorder, codec (PyAV streaming)
- `raven/common/gui/` - Custom DearPyGui widgets (VU meter, GUI animation framework, messagebox)

### Vendored Dependencies
- `tha3/` - Talking Head Anime 3 neural network (avatar animation)
- `DearPyGui_Markdown/` - MD renderer (robustified for background threads, has one remaining URL highlight bug)
- `file_dialog/` - File dialog, extended (sortable, animated OK button, click twice when overwriting)
- `anime4k/` - PyTorch port of Anime4K upscaler (extracts kernels from GLSL), slightly cleaned up
- `kokoro_fastapi/` - Streaming audio writer for TTS over network
- `IconsFontAwesome6.py` - Icon font (note: outdated version)

## Code Style
All new and modified code must follow `raven-style-guide.md` (in the project root). **Read the full guide before implementing a new app.** The summary below covers the most commonly needed conventions.

- Impure functional, Lispy (closures, `unpythonic` patterns)
- `unpythonic` pure-Python features are fair game. Currently used: `env` (namespace), `Timer` (benchmarking), `@call` (scoping), `box`/`unbox`, `sym`, `dyn`. Other features welcome where they improve clarity. **Do not** use the macro layer (`unpythonic.syntax`) or features that primarily serve as macro backends (e.g. `let` bindings — these are readable only through the macro surface syntax).
- OOP where appropriate (GUI components, stateful objects)
- Config via Python modules (`config.py` files, not YAML/JSON)
- Type hints on all new and modified functions (public and internal). Existing untyped code can be left as-is unless you're already editing it.
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
- Contract-style preconditions/postconditions would be useful, but mostly not implemented yet

## Key Patterns

### DearPyGui App Structure
See `dpg-notes.md` "Raven DPG app structure" section for layout patterns, startup sequence, background work, thread safety, DPG item management, and texture handling.

### Avatar Lipsync
TTS (Kokoro) provides timestamped phonemes → mapped to mouth morphs → THA3 animator. Audio playback occurs on the client side.
This coupling limits TTS engine choices (most don't expose timestamped phoneme data).

## Current State

### Well-structured (target style)
- `raven/librarian/` - Clean module separation (~8000 lines across 10 modules)

### Needs refactoring
- `raven/visualizer/app.py` - 4427 lines, monolithic, needs splitting into modules with clear responsibilities (see `raven/visualizer/CLAUDE.md`). Target ~700 lines per module as a guideline, not a hard limit — some modules can be longer when appropriate (e.g. lots of simple related code).
- `raven/visualizer/importer.py` - 1286 lines, pipeline architecture, lower priority but could benefit from stage separation

### Test coverage
- Library/utility code is reasonably covered: `common/` (numutils, smoothvalue, utils, image/lanczos, image/utils, video/{colorspace,compositor,postprocessor,upscaler}), `common/gui/{viewport_math,xdotwidget/*}`, `client/api`, `papers/*`, `cherrypick/*`, `xdot_viewer/dot_utils`, librarian (`chattree`, `chatutil`, `hybridir`, `appstate`, `scaffold`).
- Untested librarian layer: `llmclient` (HTTP/SSE-bound, awkward to fake).
- Visualizer has **zero tests**.
- **Priority: a first pass of tests on Visualizer** (especially before refactoring).

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
- Visualizer refactoring needed (see `raven/visualizer/CLAUDE.md` for plan)
- Visualizer has zero tests; librarian `scaffold`/`appstate`/`llmclient` untested
- DearPyGui_Markdown URL highlight bug (threading-related, untracked)
- FontAwesome version outdated
- Hindsight integration pending (PDM dependency conflicts; likely separate container with optional backend, keeping BM25+vector backend as primary)
- TTS engine expansion limited by phoneme timestamp requirement
- DPG 2.0.0: Page Up/Down key constants changed (mysterious 517/518, `app.py:4316`)
- Many `# TODO: DRY duplicate definitions for labels` scattered through Visualizer `app.py`
- Annotation tooltip help section rebuilt every time (could be static with show/hide)
- `_update_info_panel` race condition: current item highlight sometimes doesn't update immediately after selection change
- Search match scrolling race condition: hammering the button can error out (`app.py:2978`)
- XDot viewer: GraphViz `--concentrate` produces near-miss edge endpoints (0.02–0.09 graph units off) at edge split/merge points, visible as small gaps at high zoom. This is a GraphViz precision issue in the xdot data, not a rendering bug.
