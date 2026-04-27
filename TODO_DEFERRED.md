# Deferred TODOs

## Audit fleet for dict constants that should be `frozendict`

Several modules across Raven hold module-level dict constants that are used as immutable defaults or lookup tables, relying on "don't mutate this" by convention. `unpythonic.frozendict` (already a Raven dep) enforces it with teeth and costs nothing extra; Python 3.15 will also ship a stdlib `frozendict`.

Worth a pass across the fleet to find these and convert them. Low-risk (any call site that was mutating a shared default was a bug anyway), non-urgent.

Discovered during avatar-client-crop brief review (2026-04-20).

## `/api/embeddings/info` endpoint

Parallel to the `/api/{stt,tts}/info` endpoints. Should expose at least model name and embedding dimension; add other engine metadata as use cases appear. Discovered during STT/TTS info-endpoint work (2026-04-21).

## Split `raven.common.nlptools` per backend (reduce import weight)

`raven.common.nlptools` is a hub module: it imports `torch`, `transformers`, `sentence_transformers`, `flair`, `dehyphen`, and `spacy`. All five ML-engine loaders (spaCy, classifier, dehyphenator, embedder, translator) live in it, so importing the module drags the entire ML stack into any process that touches it.

`raven.client.api` currently imports `nlptools` purely to reach `deserialize_spacy_docs` for the natlang response reconstruction. If a *lighter* client module ever wants to reconstruct spaCy Docs from wire data without pulling transformers/flair/dehyphen, a clean way to do it would be: extract each backend into its own module (e.g. `raven.common.spacy_wire` for the spaCy serialize/deserialize pair, parallel modules for each of classifier, dehyphenator, embedder, translator), and leave `nlptools` as a thin aggregator that re-exports them.

Why this is deferred: `api.py` already imports torch, qoi, spaCy, etc. for its other endpoints — so `nlptools` riding along costs `api.py` nothing extra *today*. The split pays off only when some caller wants a minimal "just reconstruct a Doc from JSON" importable, which no one currently needs. Also ties into the companion item "Lazy `api.initialize` in `llmclient`" — that one is about letting `llmclient` be imported in minimal-deps CI without triggering the full chain; slimming `nlptools` becomes relevant only once that parent effort is on the table.

For symmetry, if we ever start splitting, we should do all five backends, not just spaCy.

Discovered during natlang wire-format migration (2026-04-21). Original framing lived in the now-resolved "Language-neutral wire format for the natlang (spaCy) endpoint" item, superseded by this follow-up.

## Enable HTTP response compression on raven-server

The natlang wire-format migration (JSON via `Doc.to_json()` instead of DocBin) lost DocBin's vocab-sharing optimization — categorical strings (POS tags, dep labels, lemmas) now appear once per token rather than once per batch. gzip/deflate recovers most of the loss because those are exactly the patterns dictionary-based compression eats for breakfast; natively we're probably 1.5×–2.5× bigger uncompressed, within 10–20% after gzip.

Raven-server uses Flask/waitress without response compression currently. Adding `Flask-Compress` or a waitress pre-filter is ~one line. Other endpoints would benefit too (imagefx JSON metadata, the server's HTML index page).

Not urgent — Raven's trusted-LAN-or-localhost deployment means bandwidth isn't the bottleneck for typical payloads (KB-range, not MB-range). Revisit if profiling shows wire time becoming a meaningful fraction of end-to-end latency, or when a JS client on a WAN-ish link enters the picture.

Discovered during natlang wire-format migration (2026-04-21).

## Remaining server modules without a MaybeRemote

With `Classifier`, `Translator`, `Postprocessor`, `Upscaler` landed (2026-04-22), the following server modules still don't participate in the MaybeRemote pattern:

- `avatar`, `avatarutil` — licensing-constrained (see "Client-local avatar animator" below). Also the rendering pipeline is tied to real-time animation driver state that's server-local; a client-local path would be effectively a parallel rewrite, not a wrapper.
- `websearch` — AGPL-constrained (~90 % from SillyTavern-extras, rest ported from SillyTavern-selenium's JS version — see the licensing item below). Also heavy to run locally: Selenium + headless browser.

These are both intentional omissions, not TODO gaps. Kept as a navigational note so future readers can see the coverage at a glance.

## Client-local avatar animator (licensing-bounded)

The avatar animator currently lives only in `raven.server.modules.avatar` under AGPL. THA3 upstream (the underlying ML model, vendored in `raven/vendor/tha3/`) is actually MIT — so the AGPL tax comes from Raven-side extensions, not the model itself.

A client-local animator would be valuable even though the server one stays:

- It extends the "server-optional" story (the goal behind the existing MaybeRemote pattern) to the avatar: a Raven app running standalone could still show the avatar, without requiring the server to be running.
- It enables a **fully-BSD Raven distribution** — simpler to configure for single-app users, and avoids the "license: it's complicated" friction that tends to drive people away from otherwise-perfectly-serviceable software.
- It skips the QOI encode/decode + loopback-socket round-trip. This *may* be a meaningful latency contributor even on localhost setups — needs measuring before being used as justification. On a non-localhost server setup, the user has put the server elsewhere for a reason (shared GPU across machines, a specific box with the VRAM, etc.), so "skip the network" isn't really the escape for those cases — a client-local animator helps only standalone / localhost use.

**Per-module authorship provenance on the server side** (for scoping what can / can't be unilaterally relicensed):

- `raven.server.app` — the Flask application proper. Has external contributors from the SillyTavern-extras era (and possibly earlier). Shared authorship.
- `raven.server.modules.websearch` — ~90 % from SillyTavern-extras, then patched by porting later modifications from SillyTavern-selenium (JS) into Python. AGPL lineage; not the user's code to relicense.
- `raven.server.modules.avatar` — mostly user-authored *now*, but with a tangled lineage:
  - The very original avatar module (in SillyTavern-extras, pre-user) was based on example code from THA3 (MIT).
  - Just before the user's first commit on it, it had a `result_feed`, a rudimentary `Animator` class (including its Discordian class docstring, which was a keeper), and little else.
  - Everything that makes the current avatar actually work at 10+ FPS — the optimised animator, sway animation, breathing, blinking, the postprocessor, the addon cel machinery, animefx, morph overriding — is user-authored. Some of this landed already during the SillyTavern-extras era (inside the AGPL project, but authored by the user and therefore his to relicense), the rest was added during Raven's development.
  - Clean-room scoping therefore isn't just "Raven-era = clean": the user's authored contributions pre-date the ST-extras → Raven conversion. The actual scoping would be per-line `git blame` on the final post-deprecation ST-extras snapshot to identify user-authored lines there, combined with all of Raven-era avatar work. Non-user-authored lines from the ST-extras era are the only ones that must not be reused.
- `raven.server.modules.classify` — thin shim over `raven.common.nlptools`. User-authored.
- `raven.server.modules.embeddings` — thin shim over `raven.common.nlptools`. User-authored.
- `raven.server.modules.imagefx` — new in Raven, 100 % user-authored.
- `raven.server.modules.natlang` — new in Raven, 100 % user-authored.
- `raven.server.modules.sanitize` — new in Raven, 100 % user-authored.
- `raven.server.modules.translate` — new in Raven, 100 % user-authored.
- `raven.server.modules.tts`, `raven.server.modules.stt` — new in Raven; the ST-extras versions were discarded when the final ST-extras was converted into the first Raven-Server. 100 % user-authored.

So the AGPL tax on the server side is concentrated in three places: the Flask app scaffolding (`server.app`), `websearch`, and the pre-rewrite foundation of `avatar`. Everything else is user-authored and could in principle be relicensed BSD — but the server as a whole still ships AGPL because of those three.

**Licensing distinction the open question really hinges on:**

- **Web-API RPC between processes** (the current MaybeRemote pattern): two separate products, networked, already clearly fine. No combined-work question.
- **In-process linking** (importing the AGPL module from a BSD caller): analogous to linking object files. *This* is the real AGPL question — does the combined work inherit AGPL, and can AGPL make the BSD caller's effective requirements *stricter* than BSD's?
  - GPL-family consensus: yes, a combined work inherits the strictest applicable license, which is AGPL's intent (prevent embrace-and-extinguish).
  - BSD's permissive posture doesn't change that — BSD code can be used *in* a stricter work, it just can't be *stripped of* attribution.
  - Practical consequence: if a BSD client module imports AGPL server code into the same process, the resulting program is effectively AGPL for distribution purposes.
  - Takeaway: the server stays server-only (RPC boundary, clean); an in-process client-local animator has to be a separate implementation that doesn't import the AGPL module.

**What a client-local animator would require:**

- A clean-room implementation built on MIT-licensed THA3 plus the user's own BSD-licensable contributions (the bulk of the current animator — see authorship breakdown above).
- Per-line `git blame` on the final post-deprecation SillyTavern-extras avatar module to classify each line by author. User-authored lines (from either the ST-extras era or Raven) are reusable; non-user lines from the ST-extras era are the only ones that must not be copied. The Raven-era delta is entirely user-authored and reusable wholesale.
- No code copy-paste from the non-user-authored ST-era portions — even small fragments with shared authorship would re-infect.
- The server-side animator keeps its tangled lineage and stays AGPL; the BSD client-local animator lives in `raven.common.avatar` (new) or similar, with a cleaner, from-scratch scaffolding around THA3.

The server animator is not going away — it remains indefinitely useful, especially once a JavaScript avatar client exists. A BSD client-local animator is purely additive.

When a client-local animator lands, `raven-avatar-pose-editor` should gain a mayberemote mode as well: it currently loads THA3 in-process, which collides with other local GPU consumers (observed 2026-04-24 — CUDA OOM on a 3070 Ti with Qwen + one THA3 instance already resident). Remote mode would let the pose editor run against a separate server process, or share a single THA3 instance with the live animator on the same box.

No action until the user decides whether to pursue the clean-room path. Discovered during speech-extract-to-common discussion (2026-04-17).

## Untested but test-worthy modules in `raven.common`

Cross-referencing `raven/common/**/*.py` against existing `tests/` dirs, the following have non-trivial algorithmic content but no tests:

- `raven/common/bgtask.py` — background task queue / lifecycle primitives. Pure orchestration; testable with fake tasks.
- `raven/common/gui/layout_math.py` — coordinate / packing math for DPG layouts. Pure functions despite living under `gui/`; testable the same way `viewport_math` and the xdotwidget math modules are.
- `raven/common/hfutil.py` — HuggingFace model installer. Side-effectful but the path-computation / repo-name-parsing parts are testable with tmpdir + monkeypatched `snapshot_download`.
- `raven/common/deviceinfo.py` — GPU detection, dual-GPU ordering. Small surface area; the logic that matters (device counting, visibility filtering, user-facing string formatting) can be tested with a monkeypatched `torch.cuda`.

The following are deliberately untested and should stay that way (consistent with the "test the algorithm layer, not GUI code" principle recorded in memory):

- `raven/common/audio/{player,recorder}.py` — audio hardware I/O.
- `raven/common/gui/{animation,fontsetup,helpcard,messagebox,utils,vumeter,widgetfinder}.py` — DPG glue.
- `raven/common/gui/xdotwidget/{widget,renderer,highlight,constants}.py` — rendering / DPG-bound.

Priority if picking one up: `bgtask` (most likely to harbour concurrency bugs; test-time cost is low), then `layout_math` (easy win), then `hfutil`/`deviceinfo` (requires monkeypatching but small).

Discovered during speech-extract-to-common discussion (2026-04-17).

## torch / torchaudio CUDA version alignment on fresh installs

`torchaudio>=2.4.0` was added as a direct dep alongside the existing `torch>=2.4.0`. Bare `pip install torchaudio` on a machine with `torch==2.10.0+cu128` fetched `torchaudio==2.11.0` from PyPI, which is built against CUDA 13 and fails to load (`libcudart.so.13: cannot open shared object file`). Workaround used on the dev box: `pip install "torchaudio==2.10.0" --index-url https://download.pytorch.org/whl/cu128`.

This is a broader torch-ecosystem packaging issue (torch/torchvision/torchaudio minor versions must match, and PyPI's default wheels track the latest CUDA while most installed torch is older). Not fixable from within raven's `pyproject.toml` without pinning a specific torch build — which would create its own problems across Linux/Mac/Windows and CPU-only/CUDA users.

Follow-up options to consider:

- Document the issue in `README.md` / install instructions: if `pip install` from PyPI pulls a torchaudio that fails to load, install it from `https://download.pytorch.org/whl/<your-cuda-or-cpu>` matching the installed torch minor.
- Check whether PDM respects PyTorch's index-url convention if we add it to `[[tool.pdm.source]]` — might auto-resolve correctly on fresh installs.
- Revisit once torchvision is pinned somewhere (same class of problem).

No code change; this is a documentation / install-experience issue. Discovered during speech-extract-to-common step 2 (2026-04-17).

## Lazy `api.initialize` in `llmclient` and `hybridir` (would unblock `test_scaffold` in minimal CI)

`raven/librarian/llmclient.py` calls `api.initialize(...)` at module top (lines 55–58). This means `from raven.librarian import llmclient` both (a) requires the full `raven.client.api` import chain to succeed (qoi, spaCy, Kokoro TTS, …), and (b) runs the initialization side effect. As a result, `scaffold` — which imports `llmclient` at module level — is not importable in environments without the full dep stack.

The same anti-pattern also lives in `raven/librarian/hybridir.py` (same line-range).

Concrete cost observed 2026-04-17: `test_scaffold.py` has to `pytest.importorskip("raven.librarian.scaffold")` at the top, so the scaffold tests skip entirely in the CI minimal-deps job (matching the existing pattern for `test_api.py` and `test_hybridir.py`). Scaffold coverage is visible only in dev environments — not a regression, just a cap on what CI can report.

Refactor sketch:

- Move `api.initialize(...)` out of the module body into a lazy setup function. The natural home in `llmclient` is probably `llmclient.setup`, which app startup already calls; `hybridir` has an analogous setup path.
- Audit `llmclient`'s / `hybridir`'s module-top imports for other side effects; move to lazy/TYPE_CHECKING where possible. `scaffold.py` now uses `TYPE_CHECKING` for its `hybridir` import, which is a good model.
- Verify no other module relies on `api.initialize` being called as a side effect of importing `llmclient` / `hybridir`.

Once done, remove the `pytest.importorskip` from `test_scaffold.py`; the scaffold tests then contribute to CI coverage too (~90% of scaffold.py's 119 statements).

Fleet status as of 2026-04-24: `raven/visualizer/importer.py` used to have the same pattern but was cleaned up — `api.initialize(...)` now lives in that module's `main()` (for the `raven-importer` CLI) and in `raven/visualizer/app.py` (for the GUI). Same shape as `librarian/app.py` already uses. Use as a reference when tackling `llmclient` / `hybridir`.

Discovered during scaffold/appstate test work (2026-04-17).


## torch.compile for the postprocessor

`torch.compile()` on THA3 was investigated (2026-04-09) and yields only ~6% speedup (20.3ms → 19.0ms on 3070 Ti) at the cost of 37s compilation startup. Not worth it for THA3 — the model is already lean with separable convolutions + FP16. Also hangs in the server (works in standalone; cause unresolved — possibly Triton subprocess interaction with waitress/threads).

The postprocessor (`raven.common.video.postprocessor`) might benefit more from compilation (20–60 kernel launches per frame, more fusible elementwise ops). Worth investigating separately. See `briefs/tha3-performance-audit.md`.

Discovered during THA3 performance optimization work (2026-04-09).

## MPS (Apple Silicon) device synchronization

`torch.cuda.synchronize()` calls throughout the codebase (preload cache, imageview mip loading) only handle CUDA/ROCm. Apple MPS (`torch.device("mps")`) needs `torch.mps.synchronize()` instead. Audit all `torch.cuda.synchronize` call sites and add MPS equivalents. Consider a `deviceinfo.synchronize(device)` helper.

Discovered during raven-cherrypick compare mode review (2026-03-30).

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

## Idle throttle for avatar settings editor

The settings editor's whole point is to view the live avatar while tweaking settings, so at first glance it looks unthrottle-able. But it has a pause mode that switches the avatar video off — and *that* is the state where, for consistency with the rest of the constellation (cherrypick, xdot-viewer, pose editor, and Librarian once landed), it should conserve CPU/GPU. Same `_is_busy()` + sleep pattern as the others; the busy predicate gates on `dpg_avatar_renderer.animator_running` so unpaused playback keeps full fps.

Discovered during Librarian idle throttle implementation (2026-04-27).


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

## Avatar settings editor: custom postprocessor chain ordering

The settings editor currently presents filters in a fixed priority order, with at most one copy of each filter. With the desaturate/monochrome_display and noise/analog_vhs_noise splits, the signal pipeline model is becoming richer — users may want to reorder filters or have multiple instances. The GUI needs drag-and-drop chain building: add/remove filters, reorder freely, support multiple instances of the same filter (with independent `name` keys). Currently, `strip_postprocessor_chain_for_gui` enforces fixed ordering and single instances.

Discovered during postprocessor chain ordering redesign (2026-04-09).

## raven-server: CUDA sanity check at startup

raven-server boots without complaint even when NVRTC is broken (missing `libnvrtc-builtins.so`). The error only surfaces later when something triggers JIT compilation. Server startup should probe CUDA early (e.g. a trivial JIT-compiled kernel or `torch.cuda.is_available()` + an NVRTC smoke test) and log a clear warning/error if the environment is misconfigured.

Discovered during demo prep (2026-03-26).

## raven.papers user manual

The `raven.papers` tool collection has grown to the point where it deserves its own user manual, like Visualizer, Librarian and Server already have.

There are existing usage instructions for `raven-arxiv-search` in the README of the separate `arxiv-api-search` project, which the tool was created from. These should be included in the manual.

For the others, some instructions are scattered in Raven's main `README.md`.

Some instructions don't yet exist, and need to be written.

## Hybridir: cover the edit-queueing layer with tests

`raven/librarian/tests/test_hybridir.py` has 18 tests but they all target the post-commit query side — corpus is added once, committed, queried. The edit-queueing layer (`_pend_edit` dedup, update/delete paths, the add-then-update-same-doc race) is untested. A latent shape-mismatch bug in `_pend_edit` survived this gap until it was triggered by dropping ~200 .bib files into the docs dir at once (watchdog burst → multiple concurrent `scheduled_add` / `scheduled_update` tasks).

Concrete coverage to add:

- `update()` on an existing document (internally delete + add).
- `delete()` on an existing document.
- Dedup behavior: queue add then update same `document_id` before commit; queue delete then add same id; queue add for two docs and update one; etc.
- Idempotency of `commit()` on empty queue.
- `is_indexing()` reference-counting under threaded concurrent `commit()` calls — mock the slow inner work, have two threads enter, verify `is_indexing` stays True throughout and goes False only when both have exited (also covers same-thread re-entry under the existing `datastore_lock` RLock).
- BM25 + semantic search is becoming the de-facto standard hybrid retrieval shape, so the layer is worth investing in regardless.

The watchdog-driven flow (tmpdir + `Path.touch` / `unlink` to drive `HybridIRFileSystemEventHandler`) crosses into bgtask scheduling and is harder to make deterministic — separate, lower-priority follow-up.

Discovered during DOCS-indexing-indicator smoke test (2026-04-27).


## Hybridir: BM25 backend migration for larger corpora

`bm25s` rebuilds the entire keyword index on every commit (full corpus → full reindex; IDF changes mean it can't be incremental in this design). Sub-second on ~1k small documents, so a non-issue today. Will start to pinch around the 10k–100k mark.

The standard fix is the **segmented index** model: each batch of writes lands in a small immutable segment with deletes-as-tombstones, IDF is computed across segments at query time (or partial-pre-aggregated), and a background merge thread occasionally consolidates segments to keep the count bounded. Writes become O(batch); reindex cost is amortized through merges. This is what Lucene / Elasticsearch / Solr / Tantivy all do.

For Raven, the natural migration target is **Tantivy** via the `tantivy-py` Python bindings — a Rust port of the Lucene model, MIT-licensed, no JVM, decent Python ergonomics. Would replace `bm25s` end-to-end. The semantic side (ChromaDB) and the hybrid-fusion logic above stay the same.

Plan when this becomes necessary:

- Audit `_rebuild_keyword_search_index` and `_keyword_retriever` callsites to extract the BM25-specific surface from `HybridIR`.
- Introduce a thin keyword-index abstraction (add / delete / search) so the backend swap touches one module.
- Migrate index storage on first run; existing `bm25s` indices on disk get rebuilt into Tantivy form.

Approximate alternatives if we want to stay on `bm25s`: rebuild on a schedule (every Nth commit, or every M seconds of accumulated edits) rather than on every commit — relevance drifts a tiny bit between rebuilds in exchange for cheaper writes. Cheaper than a full backend swap; doesn't help asymptotically.

Discovered during cancellable-commit work (2026-04-27).

