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
- **Tools** (`raven/tools/`): CLI utilities (format converters, system checks).

## Build and Development

Uses PDM with `pdm-backend`. Python 3.11–3.12 (3.13/3.14 blocked by unpythonic/mcpyrate compatibility; see `pyproject.toml`). Optional CUDA extras via `pdm install -G cuda`.

```bash
pdm install              # creates .venv/ and installs deps
pdm use --venv in-project
source .venv/bin/activate
```

Entry points defined in `pyproject.toml` under `[project.scripts]` — main apps are `raven-visualizer`, `raven-librarian`, `raven-server`, `raven-importer`, `raven-minichat`, `raven-xdot-viewer`, `raven-avatar-pose-editor`, `raven-avatar-settings-editor`.

### Running Tests

```bash
pytest                   # runs all tests (currently minimal coverage)
```

### Linting

```bash
flake8                   # lint check
```

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
All new and modified code must follow `raven-style-guide.md` (in the project root).

- Impure functional, Lispy (closures, `unpythonic` patterns)
- `unpythonic` pure-Python features are fair game. Currently used: `env` (namespace), `Timer` (benchmarking), `@call` (scoping), `box`/`unbox`, `sym`, `dyn`. Other features welcome where they improve clarity. **Do not** use the macro layer (`unpythonic.syntax`) or features that primarily serve as macro backends (e.g. `let` bindings — these are readable only through the macro surface syntax).
- OOP where appropriate (GUI components, stateful objects)
- Config via Python modules (`config.py` files, not YAML/JSON)
- Type hints encouraged but not enforced everywhere
- Contract-style preconditions/postconditions would be useful, but mostly not implemented yet

## Key Patterns

### DearPyGui App Structure (Librarian as Reference)
- **Layout**: App-specific. Both Librarian and Visualizer use two-column layouts, but this isn't a general requirement. All in a single `main_window`.
- **Resize**: `resize_gui()` callback recalculates sizes. Debounced via background task for expensive updates.
- **Themes**: Named themes for button variants, pulsating indicators. Created at module level.
- **Fonts**: Default + icon fonts (FontAwesome), loaded at startup.
- **Animations**: `PulsatingColor` (cyclic) and `ButtonFlash` (one-shot) via `raven.common.gui.animation` global `animator` singleton.
- **Hotkeys**: Registered via `dpg.add_key_*_handler` in a handler registry.
- **Help card**: Each GUI app should have a help card (built with `raven.common.gui.helpcard`). Currently present in Librarian, Visualizer, and raven-xdot-viewer; some apps are still missing theirs.
- **Background work**: All async ops (LLM, avatar, RAG) run in background threads via `raven.common.bgtask`. `TaskManager` represents a set of related tasks sharing a `ThreadPoolExecutor`; the whole set can be cancelled via `.clear()`. Several task managers can share one executor. Debouncing via `ManagedTask` (OOP) or `make_managed_task` (functional) — use whichever is clearer. **Warning**: `dpg.split_frame()` is safe from background threads but **must not** be called from the main thread — it will hang the GUI loop.
- **DPG error handling**: DPG raises either `SystemError` (older versions) or `Exception` (newer) for "item not found" errors, with no proper exception subclass. The `nonexistent_ok()` context manager in `raven.common.gui.utils` suppresses these via string matching on the exception chain (EAFP pattern, avoids TOCTTOU). Has `.errored` attribute to check whether the block errored out.

### Startup Sequence (Librarian as Reference)
1. DPG init (context, fonts, themes, viewport)
2. Connect to raven-server (`raven.client.api`)
3. Connect to LLM backend, if needed by this app (`llmclient.setup()`)
4. Load persistent state (app-specific `appstate` implementation)
5. Load domain-specific backends (e.g. RAG: `hybridir.setup()`)
6. Build GUI layout
7. Create controller(s)
8. Initial view render
9. Start DPG event loop

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
- Minimal: only `raven/client/tests/test_api.py` and `raven/librarian/tests/test_hybridir.py`
- Visualizer has **zero tests**
- **Priority: expand test coverage** (especially before refactoring Visualizer)

## LLM Backend
Uses text-generation-webui with OpenAI-compatible API.
Recommended model: Qwen3-VL-30B-A3B (24GB+ VRAM) or Qwen3-VL-4B (8GB VRAM).

## Known Issues / TODOs
- Visualizer refactoring needed (see `raven/visualizer/CLAUDE.md` for plan)
- Test coverage very low (Visualizer has none)
- DearPyGui_Markdown URL highlight bug (threading-related, untracked)
- FontAwesome version outdated
- Hindsight integration pending (PDM dependency conflicts; likely separate container with optional backend, keeping BM25+vector backend as primary)
- TTS engine expansion limited by phoneme timestamp requirement
- DPG 2.0.0: Page Up/Down key constants changed (mysterious 517/518, `app.py:4316`)
- Many `# TODO: DRY duplicate definitions for labels` scattered through Visualizer `app.py`
- Annotation tooltip help section rebuilt every time (could be static with show/hide)
- `_update_info_panel` race condition: current item highlight sometimes doesn't update immediately after selection change
- Search match scrolling race condition: hammering the button can error out (`app.py:2978`)
