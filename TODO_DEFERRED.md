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
