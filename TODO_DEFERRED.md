# Deferred TODOs

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

## Timer function audit

Raven uses a mix of `time.time_ns()`, `time.monotonic()`, and `time.perf_counter()` across the codebase. Each has different guarantees:

- `perf_counter_ns`: highest resolution, monotonic — correct for benchmarks
- `monotonic_ns`: monotonic — correct for elapsed time in app code (animation, polling)
- `time_ns`: NOT monotonic (NTP jumps) — only correct for wall-clock timestamps, not durations

Audit all duration-measuring call sites and verify the correct timer type is used. The xdot widget session notes mention a `time.time_ns()` with `//` truncation bug — there may be more subtle issues.

Discovered during raven-cherrypick loader bench review.

## Consolidate numpy/tensor/DPG image conversions

`raven/common/imageutil.py` now provides canonical conversion functions (`np_to_tensor`, `tensor_to_np`, `tensor_to_dpg_flat`). Several existing modules have their own inline versions of these conversions, with slight variations (some `.detach()`, some don't; some clamp, some don't):

- `raven/server/modules/avatarutil.py` — lines 269, 290, 309
- `raven/server/modules/imagefx.py` — lines 144, 150, 152, 191, 217, 219
- `raven/server/modules/avatar.py` — line 1681, 1686
- `raven/avatar/pose_editor/app.py` — line 1052, 1054
- `raven/client/avatar_renderer.py` — lines 229, 301, 468, 566, 576
- `raven/vendor/tha3/util.py` — line 92 (vendored, lower priority)

Migrate these to use the common functions where possible.

Discovered during raven-cherrypick imageutil extraction.

## Migrate xdot widget to shared viewport math utilities

`raven/common/gui/utils.py` now has `screen_to_content`, `content_to_screen`, and `zoom_keep_point` — the same formulas that `raven/common/gui/xdotwidget/viewport.py` implements inline. The xdot viewport should be refactored to use the shared functions.

Discovered during raven-cherrypick imageview implementation.

## Adopt dotted import style in cherrypick and xdot modules

Raven style is `from ..common.gui import utils as guiutils` + `guiutils.func()`, not
`from ..common.gui.utils import func` + bare `func()`. The dotted style makes it clear
at the call site where a function comes from. Modules with ambiguous names get an alias
(e.g. `guiutils`, `server_config`, `client_config`).

Files to migrate:
- `raven/cherrypick/loader.py` — uses from-imports for imageutils and lanczos
- `raven/cherrypick/triage.py` — from-imports IMAGE_EXTENSIONS
- `raven/cherrypick/tests/test_loader.py` — from-imports for test utilities
- `raven/common/gui/xdotwidget/` — check existing style, align if needed

Discovered during raven-cherrypick imageview review.

## Remove module-name prefix from function names in new modules

With dotted imports, `lanczos.resize()` reads better than `lanczos.lanczos_resize()`.
Rename in `raven/common/image/lanczos.py`:
- `lanczos_resize` → `resize`
- `lanczos_mipchain` → `mipchain`

Update all call sites: `imageview.py`, `loader.py`, `image/utils.py`, test files, benchmarks.
Also check `raven/common/image/utils.py` and any xdot widget functions for the same pattern.

Discovered during raven-cherrypick imageview review.

## Triage CLAUDE.md style conventions: global vs project-specific

Many code style conventions currently in Raven's `CLAUDE.md` apply to all of Juha's projects (import style, naming, docstrings, log format, license DRY, sentence spacing). These should be moved to `~/.claude/CLAUDE.md` so they're picked up everywhere. Review each entry and split accordingly.

Discovered during raven-cherrypick development.

## Robust public API auditing tool

A tool that checks all public symbols are listed in `__all__` (PEP 8 compliance). The simple AST approach works for straightforward modules but misses re-exports, macro-generated symbols, and barrel `__init__.py` patterns. See mcpyrate's troubleshooting docs for the full complexity: https://github.com/Technologicat/mcpyrate/blob/master/doc/troubleshooting.md#how-to-list-the-whole-public-api-and-only-the-public-api

Could be a useful addition to pyan3 (static call graph generator already understands Python module structure).

Discovered during raven-cherrypick development.

## Refactor toolbar separator helpers into guiutils

Librarian has `add_separator_for_horizontal_toolbar` (drawlist-based vertical line) and Visualizer has one for vertical toolbars. Both should be refactored into `raven.common.gui.utils`. raven-cherrypick currently uses `dpg.add_separator()` which causes visual artifacts in horizontal groups.

Could refactor as classes (cleaner than the current FP closures for stateful DPG widgets).

Discovered during raven-cherrypick test drive.

## VHS noise loading placeholder

When switching images, the texture briefly shows a stride-mismatch glitch. For PNGs (fast decode) this looks like a brief cyberpunk glitch; for large JPEGs it lingers and looks broken. Idea: intentionally fill the texture with VHS-style noise as a loading placeholder, wait one frame for it to stabilize, then swap in the real data. The noise reads as "loading" rather than "broken."

The video postprocessor (`raven/common/video/postprocessor.py`) already has VHS noise generation code (grain, horizontal bands). Extract a `generate_noise_frame(width, height, style="vhs")` utility into `raven/common/image/` or `raven/common/video/` for reuse.

Discovered during raven-cherrypick test drive.

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

Discovered during raven-cherrypick preload performance session.

## Move SmoothValue to raven.common.gui

`SmoothValue` (framerate-independent exponential decay animation) is currently defined inside `raven/common/gui/xdotwidget/viewport.py` but is a general-purpose GUI utility. Move it to `raven/common/gui/` as its own module (e.g. `smoothvalue.py`) so that any DPG widget can use it for animated transitions.

Discovered during raven-cherrypick imageview implementation.
