# Deferred TODOs

## Tier 1 REPL tests for `raven.librarian.minichat`

Implement in-process REPL tests for `raven-minichat` using the `scripted_repl` context manager pattern established in a 2026-04-15 cross-project session. The canonical reference implementation lives at `mcpyrate/test/test_126_repl.py` (mcpyrate commit `0fee81b`) — the helper is ~50 lines, inline, and the pattern is: monkey-patch `builtins.input` with a fake that pops scripted lines and raises `EOFError` at end-of-script; swap `sys.stdout` / `sys.stderr` for `io.StringIO` captures; state changes inside `try`, restoration and `.getvalue()` materialization inside `finally` so the helper is atomic from the caller's perspective.

**Why `builtins.input` rather than `sys.stdin`**: `input()` reads through `PyOS_Readline` at the C level (file descriptor 0), not through Python-level `sys.stdin`. Replacing `sys.stdin` silently does nothing and the REPL hangs waiting for real keyboard input. This is the single biggest gotcha in REPL testing and the reason `raven-minichat` has no tests today.

**Important framing (scope of tier 1)**: this is a *REPL-logic test*, not a terminal-UX test.  Monkey-patching `builtins.input` replaces the entire `input()` pathway before `PyOS_Readline` is ever called, so readline's line editor, history, and tab completion (as rendered to the user via key events) will be **0% covered**, not partially covered.  Line-editing regressions, history-recall bugs, and SIGINT-during-readline issues would pass tier 1 silently.  If those ever matter for `minichat`, a tier 2 (subprocess + `pexpect` driving a real pty) is the principled fix — but that's a separate design problem and probably not worth it unless a real regression bites.  For the planned tier 1, this framing just means: don't write assertions that expect tab completion or up-arrow history to work during the test — they won't run.

**Where to put the tests**: new file `raven/librarian/tests/test_minichat.py`. Raven uses pytest (unlike mcpyrate/unpythonic which have their own runners), so the `scripted_repl` helper is used as a plain context manager inside pytest test functions — no adaptation to pytest fixtures is strictly needed, though it could be promoted to a fixture if multiple tests grow to share state.

**Mocking required** — minichat has real dependencies that must be stubbed:

1. **LLM backend (`llmclient`)**: tests must not depend on a live text-generation-webui. Inject a fake `llmclient.chat_completion` (and friends) that returns deterministic canned responses. Either via dependency injection (if the code already supports it — check) or via `unittest.mock.patch` on `raven.librarian.llmclient`.
2. **Paths in `librarian_config`**: `llmclient_userdata_dir`, `llm_database_dir`, `llm_docs_dir` all point at real filesystem locations. Tests should redirect them to `pytest`'s `tmp_path` fixture so they don't touch the user's home directory or collide between runs.
3. **`datastore`** (chat-node datastore): minichat persists chat history. Tests should either use an in-memory variant if available, or let it write to `tmp_path` and wipe between tests.
4. **`retriever`** (RAG / hybridir): if enabled, this loads documents and indexes. Simplest: start tests with `!docs False` in the scripted input, or mock the retriever to return empty results for every query.
5. **`readline`**: already handled by the three-tier hybrid fallback landed in this same 2026-04-15 session. Tests inherit the graceful-degrade behaviour automatically — no special handling needed.

**Tests to start with** (5–7 is a good starting coverage):

- `test_help_command` — script: `["!help"]`. Expect help text in stdout (tab-completion list, special commands).
- `test_simple_chat` — script: `["Hello"]`. With the LLM mock returning `"Hi there"`, expect both the user input echo and the canned response to appear in captured stdout.
- `test_special_command_docs_toggle` — script: `["!docs True", "!docs False"]`. Expect the docs-enabled state to toggle, visible in status messages.
- `test_model_commands` — script: `["!model", "!models"]` (with `llmclient.list_models` mocked to return a fixed list). Expect both to print.
- `test_syntax_of_chat_commands` — script: `["!clear", "!history"]`. Verify clear starts a new chat, history shows the empty state.
- `test_clean_exit` — script: `[]` (empty → EOFError on first input → clean exit). Verify no traceback leaks to stderr.
- *Stretch*: `test_chat_history_persistence` — submit two messages, verify both appear in `!history`.

**Watch out for**:

- **`atexit.register(persist)`** — minichat registers a chat-database pruning handler at atexit. Under tests, this may fire at pytest teardown and touch real filesystem state. Consider unregistering it manually after each test, or arranging the `datastore` mock so `persist()` is a no-op.
- **app_state dict** — minichat keeps its state in a module-level `app_state` dict that persists across invocations within a process. If tests share a process, reset `app_state` between tests via a pytest fixture.
- **Prompt colour codes** — minichat uses a colorizer for the input prompt. The `scripted_repl` helper's `fake_input` echoes whatever prompt string it receives; tests should probably assert on substrings that skip the ANSI escape sequences (e.g., assert `"!help" in output` rather than asserting on the exact prompt rendering).

**Why deferred**: the helper pattern is straightforward, but the per-test mocking surface (LLM, datastore, retriever, paths, atexit) deserves focused design attention rather than being squeezed in at the end of an already-long session. A fresh CC session in raven can pick this up cold using this entry plus the `mcpyrate/test/test_126_repl.py` reference.

**Related work** landed in the same 2026-04-15 session:

- `mcpyrate/test/test_126_repl.py` — canonical tier-1 `scripted_repl` implementation.
- `mcpyrate` TODO_DEFERRED D5 — tier 2 (subprocess+pty) notes; we might never need it.
- `mcpyrate` TODO_DEFERRED D6 — macro-import test isolation problem (not relevant to minichat, which doesn't import macros at runtime).
- `unpythonic` TODO_DEFERRED D10 — tier 2 for `unpythonic.net` client/server.
- `unpythonic/net/tests/test_client.py` — the sibling of this entry, landed 2026-04-15: tier 1 for `unpythonic.net.client/server`, including a two-REPL-in-one-process variant of the `scripted_repl` helper (needed there because the server *also* calls `builtins.input` internally, so a global monkey-patch would hijack both ends — unpythonic instead uses a private `_input` seam on `client._connect`).

Discovered during the cross-project interactive-REPL testing-strategy design (2026-04-15).


## torch.compile for the postprocessor

`torch.compile()` on THA3 was investigated (2026-04-09) and yields only ~6% speedup (20.3ms → 19.0ms on 3070 Ti) at the cost of 37s compilation startup. Not worth it for THA3 — the model is already lean with separable convolutions + FP16. Also hangs in the server (works in standalone; cause unresolved — possibly Triton subprocess interaction with waitress/threads).

The postprocessor (`raven.common.video.postprocessor`) might benefit more from compilation (20–60 kernel launches per frame, more fusible elementwise ops). Worth investigating separately. See `briefs/tha3-performance-audit.md`.

Discovered during THA3 performance optimization work (2026-04-09).

## MPS (Apple Silicon) device synchronization

`torch.cuda.synchronize()` calls throughout the codebase (preload cache, imageview mip loading) only handle CUDA/ROCm. Apple MPS (`torch.device("mps")`) needs `torch.mps.synchronize()` instead. Audit all `torch.cuda.synchronize` call sites and add MPS equivalents. Consider a `deviceinfo.synchronize(device)` helper.

Discovered during raven-cherrypick compare mode review (2026-03-30).

## Rename disablable_button_theme → disablable_widget_theme

The theme now covers both `mvButton` and `mvCombo` disabled states (extended during cherrypick compare mode work). The name "button" is misleading. Rename across all Raven apps: the DPG tag `"disablable_button_theme"`, the `themes_and_fonts` attribute, and all `bind_item_theme` / `ButtonFlash.original_theme` references. Similarly for the red and blue variants.

Discovered during raven-cherrypick compare mode toolbar work (2026-03-31).

## Audit unnamed lambdas

Unnamed lambdas produce unhelpful `<lambda>` in stack traces. Audit all Raven apps for unnamed lambdas and name them using either `unpythonic.namelambda` or by hoisting to a `def`. Start with raven-cherrypick and raven-xdot-viewer.

Discovered during raven-cherrypick compare mode review (2026-03-30).

## AMD GPU (ROCm) support audit

ROCm presents as `"cuda"` in PyTorch, so our Lanczos kernel and `deviceinfo` validation should already work on AMD GPUs. However, the rest of the codebase needs auditing:

- All custom Torch code (postprocessor filters, Anime4K upscaler, avatar renderer) — likely fine, but verify.
- Third-party ML libraries: `transformers`, `sentence-transformers`, Flair, Whisper, Kokoro TTS — check ROCm compatibility status for each.
- THA3 (vendored) — uses standard `nn.Module`, probably fine.

Discovered while implementing `raven/common/lanczos.py`.

## pillow-simd for faster PIL image processing

`pillow-simd` is a drop-in Pillow replacement with SIMD-optimized processing (resize, convert, transpose, etc.). Doesn't accelerate format decoders (libjpeg, libpng), but Raven has real PIL `.resize()` calls that would benefit:

- `raven/client/avatar_renderer.py` — 4 resize calls (backdrop, frame resizing)
- `raven/vendor/tha3/util.py` — Lanczos resize for character images

Limitations: x86-only (no ARM/Mac M-series), may lag behind Pillow releases. Needs `pip uninstall pillow && pip install pillow-simd`.

Discovered during raven-cherrypick loader pipeline design.


## Consolidate remaining numpy/tensor/DPG image conversions

`raven/common/image/utils.py` provides canonical `np_to_tensor`, `tensor_to_np`, `tensor_to_dpg_flat`. The `imagefx.py` conversions have been migrated. Remaining sites have intentional differences that make direct replacement impractical:

- `raven/server/modules/avatarutil.py` — involves sRGB ↔ linear colorspace conversion (domain-specific preprocessing, not just axis reordering)
- `raven/avatar/pose_editor/app.py` — pure numpy `.ravel()` for DPG, too simple to benefit from abstraction
- `raven/client/avatar_renderer.py` — pure numpy `/ 255` + `.ravel()` for DPG (3 sites), same
- `raven/vendor/tha3/util.py` — vendored, with custom scale/offset normalization (THA3-specific)

The remaining gain would be single-source-of-truth, not code reduction. Revisit if the avatar pipeline is ever refactored.

Discovered during raven-cherrypick imageutil extraction.

## Adopt dotted import style in remaining modules

Raven style is `from ..common.gui import utils as guiutils` + `guiutils.func()`, not
`from ..common.gui.utils import func` + bare `func()`. The dotted style makes it clear
at the call site where a function comes from. Modules with ambiguous names get an alias
(e.g. `guiutils`, `server_config`, `client_config`).

Cherrypick and xdot_viewer migrated (session 6). The xdotwidget internal
sibling imports (Node, Edge, etc.) are fine as-is — tightly coupled types.

Remaining: audit other Raven apps (Librarian, Visualizer, Server) if desired.

Discovered during raven-cherrypick imageview review.

## Triage CLAUDE.md style conventions: global vs project-specific

Many code style conventions currently in Raven's `CLAUDE.md` apply to all of Juha's projects (import style, naming, docstrings, log format, license DRY, sentence spacing). These should be moved to `~/.claude/CLAUDE.md` so they're picked up everywhere. Review each entry and split accordingly.

Discovered during raven-cherrypick development.

## Robust public API auditing tool

A tool that checks all public symbols are listed in `__all__` (PEP 8 compliance). The simple AST approach works for straightforward modules but misses re-exports, macro-generated symbols, and barrel `__init__.py` patterns. See mcpyrate's troubleshooting docs for the full complexity: https://github.com/Technologicat/mcpyrate/blob/master/doc/troubleshooting.md#how-to-list-the-whole-public-api-and-only-the-public-api

Could be a useful addition to pyan3 (static call graph generator already understands Python module structure).

Discovered during raven-cherrypick development.

## Faster PNG decoder

PIL's PNG decode via libpng is slow (~59 ms for a 1 MP image). Unlike JPEG (where turbojpeg provides scaled decode), libpng has no equivalent fast path. Options to investigate:
- `cv2.imread` — uses libpng but OpenCV's memory handling may be faster
- `fpng` / `fpnge` — fast PNG codecs, but Python bindings may not exist
- `spng` — simpler PNG library, sometimes faster than libpng
- For thumbnails specifically, could decode at reduced bit depth or skip interlacing

Discovered during raven-cherrypick test drive.

## Preload cache: 16MP image optimization

With 16MP images (4624×3472), each cached mipchain is ~342MB as flat arrays. The current 1500MB budget fits only ~4 images, causing most preloads to be dropped after doing the full GPU work (wasted ~530ms each, with GPU contention degrading frame times to ~90ms).

Three improvements needed:
1. **Cap preload mip resolution** — skip the full-res mip in preload (only needed at 1:1 zoom, rare during triage). At 0.5× max, per-image cost drops from 342MB to ~85MB → 17 images fit.
2. **Check budget before submitting** — currently the budget check is in `_on_task_done` (after all the work). Reject early in `schedule()` to avoid wasted GPU work and contention.
3. **Move decode to background thread** — `decode_image` (50-110ms) still runs on the main thread for cache misses. Add `set_image_path`/`set_image_bytes` to ImageView (sans-IO style), folding decode into the bg mip task.
4. **Profile ~300ms mipgen anomaly on non-sequential navigation** — `lanczos.mipchain` takes ~300ms wall-clock on click-after-scroll, but 0ms on End key. Both are cache misses, same image size, same allocator state, no thumbnail or preload contention (confirmed by cancelling all background work). The function contains only async CUDA kernel launches (F.pad, F.conv2d) — nothing that should block. Needs py-spy (GIL analysis) or nsight (CUDA timeline) to identify what's actually blocking. Might be cuDNN autotuning, CUDA memory allocator fragmentation, or something else entirely.
5. **Wait for preload CUDA completion on cancel** — `cancel_pending()` is cooperative (sets flag, doesn't wait). Cancelled preload tasks may still be mid-CUDA-operation (Lanczos mipchain, tensor transfers). The bg_mip_task's `cuda.synchronize` then blocks on both its own work AND the lingering preload ops. Observed: `mipgen=508ms` for 1024×1024 (should be ~1ms) after a far jump. Consider `cuda.synchronize` before starting the bg_mip_task, or use CUDA streams to isolate preload vs display work.

Discovered during raven-cherrypick preload performance session.

## raven-cherrypick: export image sequence (QOI→PNG batch conversion)

raven-cherrypick is effectively an image viewer with QOI support, which is rare. This makes it ideal for previewing avatar recordings frame-by-frame. Integrate `raven-qoi2png` CLI functionality so that raven-cherrypick can export avatar recordings for external consumption (e.g. as a PNG image sequence for OpenShot or other video editors).

Discovered during raven-cherrypick preload performance session.



## pygame pkg_resources deprecation warning

pygame 2.6.1 emits a deprecation warning: `pkg_resources is deprecated as an API` (from `pygame/pkgdata.py`). Functional but noisy. Check if a newer pygame version fixes this, or if pygame has moved to `importlib.resources`.

Discovered during smoke-testing on new machine (2026-03-25).

## raven-cherrypick: further reduce idle CPU/GPU load

Idle throttle (2026-04-05) reduced CPU load from ~80% to ~20% of one core by sleeping ~80ms between frames when nothing needs updating. The remaining ~20% is the floor cost of `render_dearpygui_frame()` at ~12fps — ImGui resubmits the entire UI each call. Further reduction options: adaptive sleep ramp (80ms → 500ms over ~5s idle, snap back on input), or skipping `render_dearpygui_frame()` entirely (risky — event processing is tied to the render call).

Originally discovered during raven-cherrypick session 5 (2026-03-19).

## Idle throttle for Librarian

Librarian has an avatar idle auto-off, and a no-avatar mode is under consideration. When the avatar is off and no LLM generation is in progress, the GUI is mostly static — same pattern as cherrypick/xdot-viewer. Busy sources: avatar rendering, LLM streaming, RAG indexing, pulsating color animations (audit which are always-on vs. conditional), recent user input. The existing cherrypick/xdot-viewer pattern (`_is_busy()` + sleep) should port directly.

Discovered during idle throttle discussion (2026-04-05).

## Idle throttle for Avatar pose editor

Low priority — the pose editor is very rarely used. But if the pattern is cheap to add (same `_is_busy()` + sleep), could be worth it. The GUI is fully static when no slider is being dragged. Busy sources: avatar preview rendering (always-on while visible?), recent user input.

Discovered during idle throttle discussion (2026-04-05).


## raven-cherrypick: low FPS with large images

With large images (e.g. 4247×891, 5203×1313), steady-state FPS drops to 10–15 (66ms/frame) compared to ~30 FPS for 1MP images. DPG metrics show the bottleneck is in presentation/rendering, not input routing. Likely causes:

- Large `draw_image` textures are expensive to blit every frame.
- Texture pool growth — `_release_texture` accumulates pooled dynamic textures; DPG scans all registered textures O(n) per frame even when not drawn.
- The double `split_frame()` workaround (needed because DPG doesn't guarantee texture upload before rendering within a single frame) adds ~16ms latency per mip during loading, but shouldn't affect steady-state FPS.

Related: the existing "investigate GPU/CPU load at idle" item. Both may benefit from frame-skip when nothing changes, and/or pool trimming to reduce registered texture count.

Discovered during raven-cherrypick deadlock/flash fix session (2026-03-28).

## CLAUDE.md: rephrase DPG pitfall #5 to avoid Claude thinking loops

DPG pitfall #5 (callback thread deadlock pattern) was temporarily removed from CLAUDE.md because it causes Claude Opus and Sonnet to hang when analyzing cherrypick concurrency code. The model reads the complex three-way deadlock description, then enters an unproductive reasoning loop — consistently stalls at ~250–300 output tokens across multiple retries and effort levels.

The information is correct and important (confirmed by C++ source analysis — see `dpg-threading-notes.md`). Needs rephrasing in a way that conveys the same constraints without the chain-of-reasoning structure that triggers the loop. The original text is recoverable from git history.

Discovered during raven-cherrypick debugging session (2026-03-28).

## Audit and slim down project CLAUDE.md

Raven's CLAUDE.md is growing long, which increases token cost per conversation and may contribute to reasoning issues (see pitfall #5 incident above). Audit for:

- Material that could move to **project-specific skills** (e.g. "how to set up a new Raven DPG app" — the DPG app structure, startup sequence, and key patterns sections are reference material, not per-conversation instructions).
- Material already covered by **sub-project CLAUDE.md files** (Visualizer and Librarian have their own — check for redundancy).
- Material that belongs in the **global `~/.claude/CLAUDE.md`** (see existing deferred item "Triage CLAUDE.md style conventions").
- Sections that are **too detailed for instructions** and would be better as standalone reference docs (like `dpg-threading-notes.md`).

Goal: CLAUDE.md should be concise instructions and constraints, not an encyclopedia. Reference material goes in separate files that can be read on demand.

Discovered during raven-cherrypick debugging session (2026-03-28). Reinforced 2026-04-06: instruction volume caused Claude to lint a .md file despite existing memory saying not to.

## Audit typing: abstract parameter types, concrete return types

Raven convention: parameters should use abstract types from `collections.abc` (`Mapping`, `Sequence`, `Iterable`) for widest-possible-accepted semantics. Return types should use concrete lowercase builtins (`tuple[int, int]`, `list[int]`, `dict[str, int]`) — PEP 585, Python 3.9+. The capitalized `typing` forms (`Dict`, `List`, `Tuple`) are deprecated aliases for the builtins and offer no extra width — avoid them. Audit existing type hints across the codebase for consistency.

Discovered during raven-cherrypick compare mode planning (2026-03-30).

## Audit toolbar buttons for ButtonFlash acknowledgment

Check existing toolbar buttons in raven-cherrypick and raven-xdot-viewer for whether their actions should flash green on activation (Raven's convention for acknowledging a click or hotkey press). Other Raven apps (Librarian, Visualizer) already use `ButtonFlash` consistently — cherrypick and xdot-viewer may be missing it.

Discovered during raven-cherrypick compare mode planning (2026-03-30).

## Extract `raven.common` into an upstream library ("corvid")

Raven's `common/` package has grown into a general-purpose DPG toolkit: GUI widgets (file dialog, markdown, helpcard, xdot widget, animation framework, VU meter), video/audio processing, networking utils, bgtask infrastructure. This creates a gravitational well — new apps land in Raven because the batteries are there, even when they have nothing to do with NLP/ML.

Extracting `raven.common` (and the vendored DPG extensions) into a standalone library would:
- Let pyan-gui and other non-Raven DPG apps use the toolkit without vendoring
- Move the general DPG notes (`dpg-notes.md`) upstream with the code they document
- Reduce Raven to domain apps (Visualizer, Librarian, Server, Avatar) + ML-specific code
- Clarify the dependency direction: corvid → DPG, Raven → corvid + ML

Short-term: vendor the xdot widget into pyan for pyan-gui. Long-term: extract properly.

Discovered during tooltip feature session (2026-04-03).

## raven-xdot-viewer: FPS drops on dense graphs

With ~84 nodes and defines-edges enabled (pyan3 output of 3 files, `--uses --defines`), FPS drops to 5–10 when moving the mouse, and also when the cursor is outside the viewport. Reproduces reliably; uses-only graph of the same nodes has fine FPS. The `_refresh_hover` early-return path sets `_needs_render = True` unconditionally when mouse is outside the widget, triggering a full re-render every frame. For dense graphs (many edges), this is expensive. Investigate guarding re-renders to only when the hover highlight actually changes.

Discovered during tooltip feature development (2026-04-03).

## Add F1 help card to Avatar pose editor

The pose editor is the last GUI app missing an F1 help card. Follow the same pattern as the settings editor (add helpcard import, hotkey_info tuple, HelpWindow instance, F1 in keyboard handler, help button in toolbar). Check what hotkeys exist in the pose editor to populate the help card content. F11 fullscreen is not needed for this app.

Discovered during F1/F11 consistency pass (2026-04-06).

## Avatar settings editor: custom postprocessor chain ordering

The settings editor currently presents filters in a fixed priority order, with at most one copy of each filter. With the desaturate/monochrome_display and noise/analog_vhs_noise splits, the signal pipeline model is becoming richer — users may want to reorder filters or have multiple instances. The GUI needs drag-and-drop chain building: add/remove filters, reorder freely, support multiple instances of the same filter (with independent `name` keys). Currently, `strip_postprocessor_chain_for_gui` enforces fixed ordering and single instances.

Discovered during postprocessor chain ordering redesign (2026-04-09).

## Avatar settings editor: dynamic parameter help from docstrings

The postprocessor filter parameters (noise `channel`, `ntsc_chroma`, `double_size`, etc.) have detailed docstrings, but the settings editor GUI shows only bare parameter names and value dropdowns — no descriptions. Add a dynamically generated info button (ⓘ or similar) next to each parameter that extracts the relevant section from the filter's docstring, converts RST markup to Markdown, and displays it via `dpg_markdown`. This would make the growing number of filter options self-documenting in the GUI.

Discovered during NTSC VHS noise development (2026-04-08).

## raven-server: CUDA sanity check at startup

raven-server boots without complaint even when NVRTC is broken (missing `libnvrtc-builtins.so`). The error only surfaces later when something triggers JIT compilation. Server startup should probe CUDA early (e.g. a trivial JIT-compiled kernel or `torch.cuda.is_available()` + an NVRTC smoke test) and log a clear warning/error if the environment is misconfigured.

Discovered during demo prep (2026-03-26).

## raven.papers user manual

The `raven.papers` tool collection has grown to the point where it deserves its own user manual, like Visualizer, Librarian and Server already have.

There are existing usage instructions for `raven-arxiv-search` in the README of the separate `arxiv-api-search` project, which the tool was created from. These should be included in the manual.

For the others, some instructions are scattered in Raven's main `README.md`.

Some instructions don't yet exist, and need to be written.

