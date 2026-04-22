# Changelog

**0.2.7** (in progress):

**Added**:

- New submodule: `raven.papers` — consolidates all paper and bibliography tools.
  - New tool: `raven-arxiv-search`.
    - Usage: `raven-arxiv-search query.txt -o sometopic.bib`
    - Search arXiv with boolean expressions (AND/OR/ANDNOT, quoted phrases, parenthesized grouping), export results as BibTeX.
    - This was originally a standalone tool, [`arxiv-api-search`](https://github.com/Technologicat/arxiv-api-search).
  - New CLI option: `raven-arxiv-download --from-bib sometopic.bib -o papers/`.
    - Does as it says on the tin.
    - Takes the arXiv ID for each BibTeX entry from the `eprint` field.
      - Metadata records returned by arXiv's API (e.g. via `raven-arxiv-search`) include that field.
      - Additionally, if the `archiveprefix` field is present, it is checked that its value is `arxiv` before attempting to download the paper.
  - Internal changes:
    - Relocated from `raven.tools`: `raven-arxiv2id`, `raven-arxiv-download`, `raven-burstbib`, `raven-csv2bib`, `raven-pdf2bib`, `raven-wos2bib`. CLI command names unchanged.
    - Shared `RateLimiter` (thread-safe, tqdm progress bar) — extracted from the arXiv downloader, now also used by the search tool.
    - Shared `bibtex_escape`/`bibtex_unescape` — single source of truth, replacing duplicate definitions in `csv2bib` and `wos2bib`.
    - Consolidated arXiv ID handling: `identifiers.strip_version()` replaces three separate implementations.
    - New dependency: `feedparser>=6.0`.
- *Raven-cherrypick*: new "mark winner" action (Ctrl+Shift+C, or Ctrl+Shift+click cherry button). Marks the current image as cherry and all other selected images as lemon — one keystroke to commit a compare-mode choice.
- HTTP API: new `/api/stt/info` and `/api/tts/info` endpoints return the currently loaded model name and the model's native sample rate. Lets clients avoid hardcoding values that drift when the server config changes.
- *Raven-avatar*: client-side crop.
  - New crop panel in the settings editor — drag a rectangle on a viewport overlay, debounced push to the server, live preview. Works on the rendered avatar, independent of upscaler and postprocessor.
  - Server-side: crop now happens *before* upscale in the pipeline (previously after), so the upscaler only processes the cropped region.
  - Avatar renderer now sizes its texture reactively from decoded frame dimensions (previously fixed-size, mismatched when the server changed resolution mid-session).
  - New server-side telemetry: `X-Server-Stats` response header with per-request server time, so the client can display an end-to-end latency breakdown.
- *Raven-avatar*: avatar apps (settings editor, pose editor) now show the Raven version in the viewport title.
- *Raven-avatar*: settings editor now has per-parameter help as Markdown tooltips, sourced from each filter's docstring. An info button next to every postprocessor parameter shows that parameter's description, rendered via `dpg_markdown`. Filter-level info buttons show each filter's preamble. New helper module `raven.common.docstring_utils` parses Raven-style docstrings (`` `name`: description``) into summary + per-parameter sections.
- *Raven-avatar*: pose editor F1 help card with a prose section explaining the posing workflow. Hotkey table + two-column layout matching the settings editor / xdot viewer style.

**Changed**:

- *Raven-avatar* performance improvements:
  - ~4–5% faster avatar rendering via `torch.inference_mode`, cached `affine_grid` base grids in the THA3 engine, and zero-copy pose tensor expansion. Pure inference paths across the render pipeline (avatar, postprocessor, upscaler, pose editor) now use `inference_mode` instead of `no_grad`.
  - Chromatic aberration filter optimized, now 2.2× faster (batched grid_sample and GaussianBlur calls), cached grids, in-place alpha averaging.
    - Default postprocessor chain now 38–67% faster depending on resolution.
  - Auto-sized GaussianBlur kernels based on sigma (previously hardcoded at maximum). Saves blur cost at typical settings (e.g. CA at sigma=1.0: kernel 5 instead of 13).
  - Anime4K upscaler — eliminated unnecessary tensor clone, in-place output clamp.
  - New upscaler quality options: `bilinear` and `bicubic` bypass Anime4K entirely for compute-constrained GPUs.
    - This trades off image quality for ~18× faster upscaling - which may upgrade an avatar from 20 FPS to 25 FPS.
    - Quality difference is unnoticeable with the postprocessor enabled (with the default chain).
    - Main difference between Anime4K and `bicubic` is in details with thin lines, such as the rims of a character's glasses.
  - `bicubic` is now the default upscaler. The quality is good enough with the default smoke and mirrors enabled (and can now get 25 FPS at 1024x1024 on a laptop RTX 3070 Ti).

- *Video processing* (`raven.common.video`):
  - `chroma_subsample` filter to simulate a lo-fi video look.
    - Reduces chrominance (color) resolution while keeping luminance (brightness) at full resolution. Real video systems use this to improve compression, because human vision isn't as sensitive to color as it is to brightness.

- *Speech TTS wire format* default: MP3 → FLAC. FLAC is lossless, so the remote path now produces bit-identical audio to the local path; MP3's historical advantage (smaller files over the wire) doesn't matter on the trusted local network Raven targets. MP3 was originally chosen because Kokoro-FastAPI couldn't produce FLAC reliably; Raven no longer routes through Kokoro-FastAPI. The OpenAI-compatible `/v1/audio/speech` endpoint stays on MP3 (SillyTavern-facing; OpenAI spec defaults to MP3). Callers can still request any PyAV-supported format explicitly via `format=`.

- *Speech STT wire format* (client → server): MP3 → FLAC, for symmetry with the TTS direction and for the same reason — lossless on a trusted LAN beats lossy. `raven.client.api.stt_transcribe_array` now encodes the audio as FLAC before upload; the server continues to auto-detect the container format via PyAV, so no server-side change was needed.

- New module `raven.common.audio.resample` — device-agnostic sample-rate conversion (torchaudio-backed). Works on numpy arrays and torch tensors; three quality presets (`"default"`, `"kaiser_fast"`, `"kaiser_best"`) matching librosa's naming.
  - New dependency: `torchaudio>=2.4.0`.

- *TTS/STT* plumbing improvements:
  - New module `raven.common.audio.speech.stt` — Whisper wrapper callable in-process (no Flask).
    - `raven.server.modules.stt` is now a thin wrapper that decodes the audio container and forwards to the common layer.
  - New module `raven.common.audio.speech.tts` — Kokoro wrapper callable in-process (no Flask).
    - Two-layer API: `synthesize_iter` yields per-segment `TTSSegment` with already-absolute word timestamps, `synthesize` is the concatenating wrapper returning a single `TTSResult`.
    - `raven.server.modules.tts` is now a thin wrapper that casts float→s16 at the transport boundary, URL-encodes Unicode phonemes for HTTP headers, and handles Flask response construction.
  - New module `raven.common.audio.speech.lipsync` — engine-agnostic lipsync and subtitle driver.
    - Pure time-slicing for phoneme and word tracks, plus a callback-driven tick loop (`drive(on_tick, clock, tick_seconds)`).
    - Consumers compose tracks inside their own `on_tick` closure, calling `phoneme_at(stream, t)` / `word_at(timings, t)` as needed — lets the same loop drive avatar morphs, per-phoneme subtitles, word-level captions, or any combination.
    - No dependency on Kokoro or any other TTS engine.
  - New module `raven.common.audio.speech.playback` — synchronous audio playback + optional lipsync drive.
    - `play_encoded` and `play_encoded_with_lipsync` factored out of `raven.client.tts`; callers wrap in their own task manager for fire-and-forget.
    - Pure: takes the `player` as an explicit argument; the avatar-driving closure is caller-supplied.
  - `raven.client.tts` gains `play_encoded_with_avatar_lipsync` — the avatar-specific Raven wrapper that builds the mouth-morph closure and handles the server-side `avatar_modify_overrides` cleanup. Used by both `api.tts_speak_lipsynced` and `MaybeRemote.TTS.speak_lipsynced` local mode.

- `raven.client.mayberemote` new services, mirroring the existing `Dehyphenator` / `Embedder` / `NLP` pattern.
  - `TTS` and `STT`. Apps can now use speech locally when the server is down (or skip the round-trip entirely for latency), with a uniform API across modes.
    - `STT.transcribe` auto-resamples mismatched input.
    - `TTS.synthesize(format=...)` is shape-agnostic: no argument returns float32 `TTSResult` in both modes; `format="flac"`/`"mp3"`/… returns `EncodedTTSResult` ready for playback or storage.
      - Caching lives in the bottom layers (one source of truth per (location, shape)), so the mayberemote dispatcher has no cache state of its own.
    - `TTS.speak` / `TTS.speak_lipsynced`, mirroring `raven.client.api.tts_speak*`. Local-mode TTS; and local playback + remote avatar (for lipsynced).
      - `prep` accepts either `TTSResult` or `EncodedTTSResult` — encoded to FLAC internally as needed.
    - Stop / query playback via the player, not TTS: `raven.common.audio.player.instance.stop()` / `.is_playing()`. One call surface works for all three API paths (explicit-local, explicit-remote, maybe-remote), since the audio hardware is always client-local regardless of where synthesis happens.
    - `DPGAvatarController` now routes synthesis + playback through `MaybeRemote.TTS` (instead of the explicit-remote `api.tts_prepare_cached` / `api.tts_speak_lipsynced`). Stop goes through the player as above. Reads `tts_allow_local` / `tts_model_name` / `tts_lang_code` from `raven.client.config` and the device from `client_config.devices["tts"]`; flipping `tts_allow_local = True` gives the app standalone TTS capability (Kokoro loaded in-process when the server is unreachable).
  - New `raven.client.config.devices` — same shape and convention as `devices` in `raven.{librarian,visualizer}.config`. Validated by `raven.common.deviceinfo.validate` during `api.initialize` (CUDA → CPU fallback, `device_name` injection). Currently holds the `tts` record; more services join as their `<svc>_allow_local` paths gain real use.
  - `Classifier` (text sentiment), `Translator` (machine translation), `Postprocessor` and `Upscaler` (imagefx).
    - Each dispatches to the corresponding Raven-server module in remote mode and to a local in-process instance in local mode, with identical call surfaces.
    - `Translator` takes a `spacy_model_name` for local-mode sentence chunking.
    - `Upscaler` caches local `_LocalUpscaler` instances per `(width, height, preset, quality)` config, since Anime4K model choice depends on preset/quality and the constructor loads real weights.
  - With these, now every server module that isn't license-constrained (avatar, websearch) is reachable via `MaybeRemote` - and the same functionality is transparently available in-process in local mode.

- *TTS warmup* gains a common-layer implementation and routes through `MaybeRemote.TTS.warmup(voice)`, matching the three-layer shape of `synthesize` / `speak`. New `raven.common.audio.speech.tts.warmup(pipeline, voice)` runs the throwaway synthesis in-process; `raven.client.tts.tts_warmup(voice)` stays as the explicit-remote path; `MaybeRemote.TTS.warmup` dispatches. Raven-librarian now warms up via its avatar controller's TTS dispatcher, so standalone runs (`tts_allow_local=True`) warm the local pipeline instead of hitting the server.

- *Audio player / recorder singletons* lifted out of `raven.client` into `raven.common.audio`. `Player` / `Recorder` live next to their own classes now, not inside the remote-API config namespace. Apps that need audio call `raven.common.audio.initialize(player=..., recorder=...)` — each side accepts `True` (defaults), a kwargs dict, or `False` to skip. `raven.client.api.initialize` no longer touches audio; apps that don't use audio (e.g. `raven-importer`, `raven-dehyphenate`) therefore skip the pygame/pvrecorder init entirely. Downstream consumers read `raven.common.audio.player.instance` / `recorder.instance` (or `.require()` for fail-fast when uninitialized).
  - `raven.client.api.tts_stop` / `tts_speaking` and `MaybeRemote.TTS.stop` / `.is_speaking` removed; use `raven.common.audio.player.instance.stop()` / `.is_playing()` directly. Rationale: the three API paths (explicit-local / explicit-remote / maybe-remote) all share the same local audio hardware, so one call surface works for all of them.
  - `api.initialize` signature lost its `tts_playback_audio_device` / `stt_capture_audio_device` arguments — those belong to `raven.common.audio.initialize` now.

- *Image codec*: new module `raven.common.image.codec` with `encode` / `decode` — the unified image I/O layer. Parallel to `raven.common.audio.codec`.
  - Lifts the previously-duplicated decode logic out of `raven.server.modules.imagefx` (AGPL, this module 100% by @Technologicat) and `raven.common.image.utils` (BSD) into a single BSD-licensed home.
  - `decode` accepts bytes, binary streams, or filesystem paths interchangeably, and returns natural channel count (no forced RGBA).
  - Callers that need a guaranteed 4-channel output use the new `raven.common.image.utils.ensure_rgba` helper.
  - `IMAGE_EXTENSIONS` moved here from `image.utils`.

- *XDot viewer*: dense graphs no longer burn CPU while the cursor is outside the widget. The hover-refresh path was unconditionally marking the frame dirty every tick, which defeated the idle throttle; now the flag is only raised when hover state actually changes. Visible on graphs with many edges — idle FPS drops back to the background rate instead of pegging at the redraw rate.

- *Raven-avatar*: the "data eyes" effect fadeout moved from the client to the server. The client now sends one `start_data_eyes` / `stop_data_eyes` command; the server's animator cycles the cels and drives the fadeout like the other animation drivers. New HTTP endpoints `/api/avatar/start_data_eyes` and `/api/avatar/stop_data_eyes`, and a new animator setting `data_eyes_fadeout_duration` (default 0.75 s) alongside the existing `data_eyes_fps`. The former `avatar_modify_overrides({"data1": ...})` fade stream (~45 HTTP calls per fadeout at 0.75 s) is gone.

- *Raven-server* / *natlang*: `/api/natlang/analyze` now returns language-neutral JSON instead of a spaCy `DocBin` binary blob. Each response item is `{"lang": ..., "doc": <spaCy Doc.to_json()>}`, with optional `"vectors"` when the new `with_vectors` request flag is set. Per-item `lang` makes the wire format naturally multilingual-ready (for future server configurations loading multiple pipelines — e.g. English plus Finnish). Python clients remain unaffected at the API surface — `raven.client.api.natlang_analyze` continues to return `list[Doc]` — but non-Python clients (a future JS avatar frontend) can now consume the endpoint directly. Trade-off: the DocBin vocab-sharing optimization is gone, so repeated categorical strings (POS tags, dep labels, lemmas) appear once per token rather than once per batch; invisible in practice given Raven's LAN-only deployment (KB-range payloads on localhost or trusted LAN). `with_vectors=True` round-trips `doc.tensor` as base64 float32, giving `MaybeRemote.NLP` callers identical feature parity across local and remote modes.

- *Common utilities*: minimum `unpythonic` dependency bumped to 2.1.0. `environ_override`, `maybe_open`, `UnionFilter`, and `si_prefix` graduated to `unpythonic` in that release — Raven's local copies have been removed; the names now come from `unpythonic`.
  - Visible side effect: SI-prefixed numbers in log messages (bitrate, byte-rate, pixel-rate strings in the avatar renderer and audio codec) now use correct SI casing — lowercase `k` for kilo (previously uppercase `K`, which is the symbol for kelvin). `si_prefix` also gained binary (base-1024) mode, sub-unity prefixes (`m`, `µ`, ...), and correct handling of negative and zero values.

**Fixed**:

- *Raven-minichat*:
  - `raven-minichat` no longer crashes on MS Windows. Previously, the command would fail at startup with `ImportError: No module named 'readline'` because Python's stdlib `readline` module is POSIX-only. The fix is a three-tier hybrid load: try stdlib `readline` first (Linux/macOS), fall back to `pyreadline3` (a drop-in Windows replacement; `pip install pyreadline3` to get the full experience), and finally degrade gracefully to plain `input()` if neither is available — the chat loop still works, you just lose command history, tab completion, and persistent cross-session history. When running in the degraded mode, a startup notice explains what's missing and how to restore it.

- *Numerical utilities* (`raven.common.numutils`):
  - `psi()` (mollifier helper, also used via `nonanalytic_smooth_transition()`) no longer emits a stray `RuntimeWarning: divide by zero encountered in divide` when evaluated at `x = 0`. The function is correct — it uses the standard "compute-then-mask" idiom `np.exp(-1.0 / x**m) * (x > 0.0)` where `-1/0 = -inf`, `exp(-inf) = 0`, and the mask zeros the result — but numpy was still emitting the warning from the division step. A previous suppression attempt used `warnings.filterwarnings(..., module="__main__")`, which silently failed in practice (numpy emits the warning from its own internal module, not `__main__`). Replaced with `np.errstate(divide='ignore', invalid='ignore')`, numpy's own mechanism for suppressing float-error warnings within a dynamic extent.

- *NLP tools* (`raven.common.nlptools`):
  - `count_frequencies(..., lemmatize=False)` no longer crashes with `TypeError: 'int' object is not callable`. Latent bug: the non-lemmatize branch called `.lower()` on a spaCy `Token`, whose `.lower` attribute is the orth hash (an integer), not a method. The default `lemmatize=True` path rebinds the loop variable to `token.lemma_` (a `str`) before calling `.lower()`, so the default masked the bug. Every caller threading `lemmatize=False` through crashed.

- *CSV parsing* (`raven.common.readcsv`):
  - `parse_csv` with autodetected headers no longer silently drops the first data row. Latent bug: after sniffing the header via `next(reader)`, the code rebuilt a fresh `csv.reader` at the current (post-header) file position and then advanced another row via `reader.__next__()`, so the main parse loop effectively started at row 3. A `header + N`-row file returned `N - 1` rows; a `header + 1`-row file silently returned `[]`.

- *Audio codec* (`raven.common.audio.codec`):
  - `decode` no longer crashes on FLAC (and any other container that reports `duration=None` via pyav). A log-info line divided `None / av.time_base` on first frame, raising `TypeError`. Affected any caller decoding FLAC from a `BytesIO`.

- *arXiv tools* (`raven.papers`):
  - `raven-arxiv2id` (and other tools using arXiv ID extraction from filenames): fix detection of IDs embedded between underscores, letters, or hyphens in filenames. Previously, filenames like `Smith_2301.12345_notes.pdf` silently failed to match. Also adds support for 4-digit new-style IDs (2007–2014 era, e.g. `0704.0001`) and old-style IDs with subject class prefix (pre-2007, e.g. `hep-th/0601001`).

- *BibTeX tools* (`raven.papers`):
  - Fix `bibtex_escape`: unmatched `{` in source text (e.g. WoS abstracts) produced unbalanced braces that broke bibtexparser parsing. The old approach doubled braces (`{` → `{{`); now uses proper LaTeX escapes (`\{`, `\}`).
  - Add missing `#` and `$` escaping — both are BibTeX/LaTeX specials that could cause parse or render errors in downstream tools.
  - `pdf2bib` now applies `bibtex_escape` to all field values (literal, LLM-extracted, and function-generated). Previously, LLM output was written unescaped.
  - `requests` and `tqdm` added as explicit dependencies (were used directly but only present as transitive deps).
  - `raven-csv2bib` now converts **all** input files when given more than one. Previously, entries from all but the last file on the command line were silently dropped — the aggregation loop collected rows into an accumulator that a later loop never read, so only the last file's entries made it into the output.

- *Video processing* (`raven.common.video`):
  - Fix filter cache invalidation on resolution change. Filters using texture caches now check their own tensor dimensions instead of relying on the video frame dimensions, preventing stale data when the image resolution changes mid-session.

- *Raven-avatar*:
  - Settings editor: avatar panel now resizes with the window (especially noticeable when going fullscreen). Previously the rightmost column expanded uselessly; now the postprocessor column stays at its default width and extra space goes to the avatar panel.
  - Fix settings editor crash when loading filters with `!ignore` parameters (e.g. anything with a `name`). The canonize and generate paths now skip these, matching the GUI build path.
  - Pose editor: FPS counter no longer shows near-zero on the first frame (warmup fix), and an idle throttle brings CPU use down when the editor is not being interacted with. The window now also fits on 1080p displays without overflow.

- *Text file I/O* (Windows correctness): every text-mode `open()` in Raven now specifies `encoding="utf-8"` explicitly. On Linux/macOS the system default is UTF-8, so this was latent; on MS Windows Python's default is the ANSI code page (cp1252 in Western locales), silently corrupting non-ASCII content — emotion names with Unicode symbols, API keys with high-bit bytes, BibTeX entries with accented author names, audio timing reports with paths containing ä/ö. Affected `raven.common.readcsv`, the avatar pose / settings editors, `raven.server` (API key file, animator settings, emotion defaults), and `raven.papers.pdf2bib`. Also future-proofs for Python 3.15, which will start warning on a missing `encoding=` (PEP 597).


---

**0.2.6** (9 April 2026):

**Added**:

- New GUI app: *Raven-cherrypick*.
  - An image triage tool for quickly sorting a folder of images into cherries (keepers), lemons (rejects), and neutral.
  - Start with `raven-cherrypick some/path/to/images/`. If no path given, defaults to CWD.
  - GPU-accelerated Lanczos scaling with mipmapped progressive loading and preload cache for instant image switching.
  - No on-disk thumbnail cache, no metadata files. Image state is encoded by directory path (`base/cherries`, `base/lemons`).
  - Easy two-hand operation: arrows navigate; X=lemon, C=cherry, V=clear mark. Ctrl+click in grid view for multi-select.
  - Filter view: show only cherries, lemons or neutral, or show all (G / Shift+G to cycle).
  - Jump to next cherry/lemon/neutral with B/N/M.
  - Zoom/pan preserved when switching between images with the same dimensions — useful for comparing variations of the same shot.
  - Compare mode: select 2–9 images and press Enter to cycle through them automatically. Adjustable speed, pause/resume, zoom while cycling. Press a digit key to pick a winner and exit.
  - Status bar: current position, image dimensions and approximate aspect ratio, selection count.
  - F11 fullscreen mode and F1 help card available.

- New CLI tool: *Raven-conference-timer*.
  - A large-font countdown timer for conference presentations.
  - Start with `raven-conference-timer 15:00` (or bare minutes: `raven-conference-timer 15`).
  - Auto-sizes the window to fit the countdown text. `--size N` sets font size in pixels (default 500).
  - Color changes at configurable thresholds: white → yellow → red → pulsating glow when expired.
    - `--yellow` and `--red` set the thresholds (default 5:00 and 2:00).
  - Hotkeys: Space to pause/resume, F11 for fullscreen, F1 for help card, Esc to exit.

- *Raven-xdot-viewer*:
  - GUI: combobox to choose which GraphViz layout engine to use, re-rendering the current graph with the chosen engine.
  - Tooltip support for node annotations (from GraphViz `tooltip` attribute; e.g. Pyan3 2.4.0+ generates these).
  - Dashed and dotted edge rendering.
  - Error dialog for failed graph loads.
  - Idle framerate throttle (reduced GPU usage when not animating).

- *Raven-avatar*:
  - F1 help card for the avatar settings editor.

- *Video processing* (`raven.common.video`):
  - New filter: VHS head switching noise (horizontal distortion bands at frame bottom). The most iconic VHS artifact.
  - New noise mode: VHS, with PAL and NTSC modes. NTSC comes with 4:2:0 chroma subsampling for a more authentic analog look.
  - Bloom filter: added `sigma` parameter for controlling glow width. Recommended values: 7.0 for dreamy early 2000s anime glow, 1.6 for modern tighter glow.

- *Common libraries*:
  - New module: `raven.common.image` (image utilities and GPU-accelerated loader pipeline).
  - Extracted `SmoothValue`/`SmoothInt` into `raven.common.smoothvalue` (shared across xdot viewer, cherrypick, and future apps).
  - `PyTurboJPEG` dependency added for fast JPEG decoding. Requires the `turbojpeg` system-level library (on Debian-based Linux: `sudo apt install libturbojpeg`).

**Changed**:

- *Video processing* (`raven.common.video`):
  - There are now two noise stages: `noise` (sensor/film grain, early in the chain) and `analog_vhs_noise` (VHS tape noise, later). This better models the physical signal path.
    - If you have custom chains that use `noise`, check whether you need both stages.
  - Split `desaturate` into `desaturate` (retouching stage) and `monochrome_display` (display output stage), allowing separate control over the artistic and output desaturation.
  - Renamed the `translucency` filter to `translucent_display` for consistency with the new `monochrome_display`, and moved it late in the chain because it models a scifi display device.
    - If you have custom avatar postprocessor chains (in `raven.server.config` or custom JSON presets), rename the filter in your chain. The bundled presets have been updated.

- *Dependencies*:
  - Bump `mcpyrate` to 4.0.0.
  - Bump `unpythonic` to 2.0.0.
  - Widen Python support to `<3.15`.
    - But narrow `requires-python` to `<3.13` for `kokoro`/`misaki` compatibility.

**Fixed**:

- Compatibility: detect "Item not found" across different Python/DPG versions, needed in GUI code.

- *Raven-xdot-viewer*:
  - Fix dark-mode text contrast on colored node fills (text now adapts based on perceived luminance).
  - Fix graph area too small on 1080p displays.
  - Fix `--size` flag for fonts smaller than the default 120px.


---

**0.2.5** (3 March 2026):

**Added**:

- New GUI app: *Raven-xdot-viewer*.
  - This is a utility app for viewing GraphViz graphs (`.dot`, `.gv`, `.xdot`).
  - Start with `raven-xdot-viewer`.
  - In a future version, this technology will be deployed in Librarian for navigating the nonlinear chat history.

- Tools:
  - New command-line tool: *Raven-csv2bib*.
    - This converts comma-separated values (`.csv`) to BibTeX.
    - The first row of the `.csv` file must consist of column headers. Fields with these names will be populated in the BibTeX output.
      - For use with *Raven-visualizer*, the fields *Author*, *Year*, *Title* are required, and the field *Abstract* is optional.
        - If your dataset has no meaningful text descriptions beyond an item title, you can omit the whole *Abstract* column.
        - But if you have text descriptions, including them should improve the accuracy of the semantic map, by making it easier for Raven to detect which items are semantically similar.
      - Arbitrary other fields can be included and will be transcribed into the output BibTeX.
    - Author names use BibTeX format.
      - If an item has multiple authors, separate them with the lowercase literal word "and".
      - Each author name can have up to four parts (first, von, jr., last).
      - Each author name must be in one of three formats:
        - First von Last ("First Last" if no "von" part)
        - von Last, First ("Last, First" if no "von" part)
        - von Last, Jr., First
      - For more details, see: https://www.bibtex.com/f/author-field/

**Changed**:

- Bump minimum **Python** version to **3.11**.
  - **Upgrading to Raven 0.2.5 requires a fresh reinstall**.
    - To do this, delete the `.venv` hidden subdirectory inside your top-level Raven directory, then `pdm install`.
    - It should still remember if you had CUDA enabled, and if so, automatically install the NVIDIA packages.
  - Raven requires [`av`](http://pyav.org/docs/stable/) for its audio handling.
    - Particularly, the audio encoder for transporting TTS audio over the network from *Raven-server* to *Raven-librarian* needs `av`.
    - Recent versions of `av` may require installing some upgrades.
      - FFMPEG 7 is now required. Older versions (4, 5, 6) are no longer supported.
        - To see which version you have, `ffmpeg -version` (note only one dash).
        - For Ubuntu 22.04 based systems (e.g. Linux Mint 21.3), there is a PPA; for how to add it, see e.g. [here](https://blog.programster.org/install-ffmpeg-7-on-ubuntu-22).
      - To be able to build `av`:
        - Beside `ffmpeg` itself, you will need the various related `lib*-dev` packages from the same repository. You can use the `synaptic` GUI to locate them easily (filter the view by *Origin*).
        - From the distro's default repositories, you'll need `clang`.
        - You'll also need `cython`. For this, `pip install cython` (in the environment where you're running `pdm`) should work.
  - The Python upgrade was needed to support the [Hindsight](https://hindsight.vectorize.io/) AI agent memory system that will be later used by *Raven-librarian*.

---

**0.2.4** (16 December 2025):

**Added**:

- Tools:
  - New command-line tool: *Raven-arxiv-download*.
    - This takes arXiv paper IDs from the command line (e.g. 2511.22570, 2411.17075v5, cond-mat/0207270, math/0501001v2), and downloads the corresponding PDFs.
      - Main use case is: people have recommended a bunch of interesting arXiv papers, hence you have dozens of URLs or arXiv IDs on your phone, and you'd like to download the fulltexts on your PC.
      - The files are automatically named based on the metadata record queried from the arXiv API.
      - You can download either the latest version of each paper (default) or a specific version (just specify e.g. "v2" at the end of the ID, in the usual arXiv notation).
      - Duplicates are not downloaded.
        - The tool checks the specified output directory (default: current working directory) whether there is already a PDF file with a matching (versioned) arXiv ID in the filename.
      - The tool automatically respects the one-request-per-three-seconds-of-wall-time limit of the arXiv API TOS.
    - For instructions, see the [visualizer user manual](raven/visualizer/README.md).

  - New command-line tool: *Raven-burstbib*.
    - This takes a BibTex `.bib` file, and splits it into many `.bib` files, each containing one individual entry from the input file.
      - The files are automatically named based on the slug (BibTeX item ID), omitting any characters that are not valid in a filename.
      - If the output file already exists, the tool appends "_2", "_3", ... to the filename until it finds a filename that is not in use.
    - This is convenient to turn a huge BibTeX file (full of scientific abstracts) into individual documents for *Raven-librarian*'s document database, so that you can synthesize information over them with your LLM.
      - Run `raven-burstbib -o my_topic my_references.bib` and then copy/move the `my_topic` subdirectory to `~/.config/raven/llmclient/documents/`.
      - Raven-librarian will pick them up at next start, or immediately (if already running).
        - Note that Librarian's search indexing may take a while. For progress messages, see the terminal window from which you started *Raven-librarian*.
    - The current implementation of `raven-burstbib` is hacky. It doesn't actually parse the file properly, but only splits at BibTeX item headers.
      - It will handle invalid slugs properly, but other types of invalid input may cause crashes or unexpected behavior. It does work if your input file is valid BibTeX. :)
      - If you encounter a bug, please [open an issue](https://github.com/Technologicat/raven/issues).

  - New command-line tool: *Raven-dehyphenate*.
    - This uses the `dehyphen` Python package to sanitize text bro-ken by hyp-he-na-tion.
    - This is useful for `pdftotext` outputs, and for text obtained from PDF files by OCR (such as with `ocrmypdf --force-ocr input.pdf output.pdf`).
    - Raven-server's `sanitize` module is used automatically, if the server is reachable and the module is loaded on the server; else the dehyphenator model is loaded locally.

- *Raven-visualizer*:
  - Importer: New keyword detection mode "llm".
    - This uses the LLM backend configured for *Librarian*.
      - When this mode is used, the LLM backend must be running when *Visualizer* (or the command-line tool `raven-importer`) is started.
    - To initialize the task, this uses the same system prompt and AI character as *Librarian* uses for its chat.
      - This gives results consistent with what *Librarian* would say, because the LLM operations are handled by the same AI simulacrum.
      - See `raven.librarian.config`.
    - The AI analyzes the titles and abstracts for each cluster (separately), and suggests keywords. These keywords are recorded as the cluster keywords for *Visualizer*.

- *Raven-librarian*:
  - New feature: STT (speech to text, speech recognition). Talk to the AI using your mic!
    - To start speaking to the AI, click the mic button next to the chat text entry field (hotkey Ctrl+Shift+Enter).
      - The mic starts glowing red, to indicate that Librarian is listening. The VU meter (audio input level) next to the mic button becomes active.
      - To stop speaking, and send the spoken message to the AI, click the mic button again, or wait until the recorder detects silence and stops automatically.
        - The gray line on the VU meter is the silence threshold level.
      - The mic stops glowing (and returns to its default white).
    - Librarian then runs the recorded audio through a locally hosted [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) speech recognizer, which lives in the `stt` module of Raven-server.
    - The transcribed text is then sent to the AI, just as if it was typed as text to the chat text entry field.
    - For now, this feature has certain limitations:
      - The recorder autostop settings are hardcoded: 1.5s of input audio signal level under -40.00 dBFS.
        - This should work under most circumstances, but if you are not in "most circumstances", you'll have to stop recording by clicking the mic button again.
      - It is not possible to edit the transcribed text before sending.
    - To choose your mic device, see `raven.client.config`.
      - By default, Librarian picks the first available NON-monitoring audio capture device, in the order listed by the command-line tool `raven-check-audio-devices`.
      - The default should work on laptops, and in general, most systems that have just one audio input device.
      - The help card (F1) shows which mic device is active. It is also printed to the client log at app startup.
  - Very rudimentary chat branch navigation added.
    - In the linearized chat view, each message has buttons for next/previous sibling, jump 10 siblings, jump to first/last sibling.
      - Hotkeys apply to the last message displayed in the view.
    - It's easier to show than explain (try it out yourself!); but when switching siblings in the linearized chat view:
      - The sibling node switched to becomes the candidate HEAD.
      - If the candidate HEAD has any child nodes (i.e. chat continuations):
        - The child node with the most recent payload (according to payload timestamp) is chosen.
        - That child node becomes the new candidate HEAD, and the process repeats.
      - Once no more child nodes are found (i.e. the candidate HEAD is a leaf node), the candidate HEAD becomes the final new HEAD.
      - The linearized chat view scrolls to the sibling node that was switched to, regardless of where the final HEAD is.
    - Contrast this with the branch button, which sets the chat HEAD to the given node, without scanning the subtree for continuations.
      - Just like in `git`, branching is cheap. Branching only sets the HEAD pointer; no data is copied.
      - If you branch, but then change your mind, click the "Show chat continuation" button on the last message (hotkey Ctrl+Down).
        This rescans the chat continuation just like when switching siblings.
  - The LLM system prompt, the AI's character card, persona names (AI and user), and the AI's greeting message can now be customized in `raven.librarian.config`.
    - Changes take effect when Librarian is restarted.
    - Limitation: for now, only one AI character icon is loaded. If you switch characters, old chats will show the current character's icon (the persona name is stored in the chat database, but the avatar and icon paths are not).

**Changed**:

- *Raven-visualizer*:
  - Configurable plotter colors (background, grid, colormap). Loaded from `raven.visualizer.config` at app startup.
  - Configurable word cloud colors (background, colormap). Loaded from `raven.visualizer.config` at app startup.
  - The section headings in the BibTeX import dialog are now clickable, and perform the same function as the icon buttons.
  - *Visualizer*'s importer now automatically uses *Raven-server* for embeddings and NLP if it is running.
    - If the server is not running, the AI models are loaded locally (in the client process) as before.
    - There is no visible difference from the user's perspective (other than saving some VRAM, if also *Librarian* is running at the same time).
  - The importer now sanitizes abstracts using the `sanitize` module of Raven-server. This feature is on by default.
    - This affects only new BibTeX imports. Existing datasets are not modified.
    - The feature can be turned off in `raven.visualizer.config`. See the `dehyphenate` setting.
    - For each abstract, all paragraphs are sent together for processing. This may cause paragraphs to run together, if an abstract contains multiple paragraphs,
      but is often the only way if the input text is REALLY broken and contains newlines at arbitrary places. It was felt this is preferable, because scientific
      abstracts are often just one 200-word paragraph.
    - Raven-server's `sanitize` module is used automatically, if the server is reachable and the module is loaded on the server; else the dehyphenator model is loaded locally.

- *Raven-librarian*:
  - The document database now uses *Raven-server* for embeddings and NLP.
    - This saves some VRAM, by avoiding loading another copy of the same models in the client process.
    - This also makes the `raven.librarian.hybridir` information retrieval backend fully client-server, allowing the AI components for this too to run on another machine.
    - Because *Librarian* requires *Raven-server* for other purposes, too, *Librarian* will not start if the server is not running.
  - The document database now ingests `.bib` files, too.
    - This allows using the `raven-burstbib` command-line tool to mass-feed abstracts into Librarian's document database.
      - The tool takes a `.bib` file and splits it into individual files, one per entry. Hence each entry becomes a separate document in Librarian's document database.
  - The app now recovers if `state.json` is missing or corrupt.
  - Many small UI improvements, for example:
    - Window resizing implemented.
    - Collapsible thinking traces.
    - Interrupt/continue.
    - Avatar idle off.
      - Configurable, optional. See `avatar_config.idle_off_timeout` in `raven.librarian.config`. Seconds as float, or `None` to disable.
      - This saves some GPU compute by switching off the avatar video after the AI avatar is idle for a while.
      - The avatar video switches back on when:
        - The AI starts processing (writing new message, continuing existing message, rerolling existing message).
        - The chat view is re-rendered (e.g. by switching chat branches, or resizing the window).
        - The AI starts speaking (Ctrl+S, send last message to TTS).
    - Help card added.
    - TTS audio playback device setup in `raven.client.config`:
      - For configuration symmetry reasons, `None` now means "use the first available playback device as listed by `raven-check-audio-devices`", not the system's default playback device.
      - The new special value "system-default" uses the system's default playback device, so that the playback goes to the same device as from other apps. (This is what `None` did before.)
      - The default configuration has been changed to use "system-default", so that behavior should remain the same.

- Tools:
  - *Raven-pdf2bib*: Overhauled. See updated instructions in the [visualizer user manual](raven/visualizer/README.md).
    - To initialize each LLM task, this uses the same system prompt and AI character as *Librarian* uses for its chat.
      - This gives results consistent with what *Librarian* would say, because the LLM operations are handled by the same AI simulacrum.
      - See `raven.librarian.config`.


**Fixed**:

- *Raven-visualizer*:
  - Fix bug: "reset zoom" missed some datapoints (in a "select visible", hotkey F9), if they were exactly at the edges of the data bounding box.
    - Note that also loading a dataset resets the zoom, so the bug also affected the initial view upon loading a dataset.
    - Workaround for previous versions: after a "reset zoom", zoom out by one mouse wheel click before using "select visible".
  - Fix bug: wrong dtype in the embedder loader's CPU fallback.
    - The CPU fallback loader now always uses float32.
    - Workaround for previous versions: when working without a GPU, configure the embedder explicitly to use dtype `torch.float32`. See `raven.visualizer.config` and `raven.server.config`.
  - Fix UI bug: the plotter axes no longer light up when the mouse hovers on them.
    - The axes are not clickable, so the highlight was spurious.
    - This was broken when we upgraded to DearPyGUI 2.0, where the plotter changed to introduce that hover-highlight by default. Now we disable the highlight by theming the plotter.

- *Raven-avatar*:
  - Fix bug: Also the background image is now hidden while the avatar is paused.

- *Raven-librarian*:
  - Fix bug: The avatar's subtitle now re-positions itself correctly when the GUI is resized while the avatar is speaking.


---

**0.2.3** (7 October 2025):

**Added**:

- Prototype of *Raven-librarian*, a scientific LLM frontend GUI app.
  - Features an animated AI avatar with TTS and auto-translated subtitles, document database (plain text files for now), and tool-calling support (websearch for now).
    - Document database uses hybrid search (BM25 for keyword search, ChromaDB for semantic search).
  - In this prototype, chats are saved, but going back to previous chats is not yet possible because the GUI for that has not yet been developed.
  - When the *Documents* checkbox in the GUI is ON, the document database is autosearched, using the user's latest message to the AI as the query.
    - If, additionally, the *Speculation* checkbox is OFF, the LLM is bypassed when there is no match in the document database.
  - Websearch is enabled when the *Tools* checkbox in the GUI is ON.
  - Requires both *Raven-server* and the LLM backend (oobabooga/text-generation-webui, with `--api`) to be running.
  - For configuring Raven-librarian, for now, see `raven.librarian.config`.
    - The default location for the document database is `~/.config/raven/llmclient/documents`.
      - Librarian monitors this directory automatically, and also scans for offline changes at app startup.
      - Put `.txt` files there; they are search-indexed automatically. Replace files; the index is updated automatically. Remove files; they are removed from the index automatically.
      - If you need to force a manual index rebuild: make sure Librarian is not running, then delete `~/.config/raven/llmclient/rag_index`. It will be rebuilt at app startup.

- *Raven-avatar* now has a "data eyes" effect, for use as an LLM tool access indicator in Librarian.
  - Cel animation, up to 4 frames. Can be tested in `raven-avatar-settings-editor`.

- Speech video recording in `raven-avatar-settings-editor`.
  - Output goes in the `rec/` subdirectory.
  - TTS speech is saved as `.mp3` files, one per sentence.
  - Avatar video (avatar only, no background) is saved as as individual frames as `.qoi`.
    - For converting the video frames into a usable format, see the `raven-qoi2png` tool.
  - A speech timings list is saved as `.txt`.
  - These can be used to piece together a speech video in a video editor such as *OpenShot*.


**Changed**:

- *Raven-visualizer*'s importer now uses both the title and the abstract to cluster the inputs.
  - This requires *Snowflake-Arctic* or better as the embedding model; the older *mpnet* model tends to lead everything to become one cluster if the abstracts are used for clustering.
  - Old datasets must be imported again for the changes to take effect, because the embeddings are computed at import time.
    - Old embeddings caches must be deleted before re-importing!
    - For example, when `mydata.bib` is imported, Raven's importer produces:
      - `mydata_embeddings_cache.npz`: The vector embeddings. Delete this cache file!
      - `mydata_nlp_cache.pickle`: The natural language processing results, used for keyword detection. This cache is not affected by this change, so no need to delete.

- *Raven-server* now hosts all AI components, including embeddings and spaCy NLP.
  - spaCy NLP is only available for Python-based clients running the same versions of Python and spaCy, because it communicates in spaCy's internal format.


---

**0.2.2** (13 August 2025):

**Added**:

- First complete tech demo of *Raven-avatar*.
  - See the GUI apps `raven-avatar-settings-editor` (completely new postprocessor settings GUI) and `raven-avatar-pose-editor` (ported from the old THA3 pose editor).
  - The settings editor requires *Raven-server* to be running.

- *Raven-avatar* now has cel-blending and animefx support.


---

**0.2.1** (18 June 2025):

Otherwise the same as 0.2.0 (17 June 2025), but with the TODO cleaned up. Documenting both here.

**Added**:

- *Raven-server*: to provide an animated AI avatar, and to eventually host all AI components.
  - This is a web API server, initially ported and stripped from the discontinued *SillyTavern-extras*.
    - AGPL license! Affects the `raven.server` and `raven.avatar.pose_editor` directories.
    - All other components of *Raven* remain BSD-licensed. This includes `raven.avatar.settings_editor`.
  - Pose editor ported from wxPython to DPG, to match the rest of the *Raven* constellation, and to require only one GUI toolkit.
  - Added in d42d52356d61d290d4d9a1e5ffd0e1b6e0843c61, 22 May 2025. The entrypoint has since moved to `raven.server.app`.
  - Avatar has new features compared to the old Talkinghead:
    - Lipsync, to new TTS module based on Kokoro-82M.
    - Anime4k upscaling (super-resolution).


---


**0.1.x** and older

The project was started in December 2024.

No changelog was maintained.

These versions included only *Raven-visualizer*.
