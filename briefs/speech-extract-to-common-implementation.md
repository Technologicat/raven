# Speech extraction — implementation overview

**Status**: steps 1–5 complete + review-pass polish, step 6 (manual smoke) pending.
**Reading companion to**: `briefs/speech-extract-to-common.md` (the design brief).
**Purpose**: a guided tour of what actually landed in the code, so the diffs can be read with context.

This document is the reading guide, not the authoritative record — git log is. Use this to decide which diffs to read carefully and which to skim.

## TL;DR

Kokoro (TTS) and Whisper (STT) are no longer tied to the Flask server. Both can be called in-process from any Raven app, and the server modules are now thin transport wrappers. `raven.client.mayberemote` gained `TTS` and `STT` services so apps can transparently use the server when available or run locally otherwise. `raven.client.tts.tts_prepare` was rewritten to return an `EncodedTTSResult` dataclass at the wire boundary, with `WordTiming` dataclasses everywhere in-process; the dict-of-dicts legacy format is gone.

| Location | What | Size |
|---|---|---|
| `raven/common/audio/resample.py` | New — torchaudio-backed, device-agnostic resampler with quality presets | ~105 lines |
| `raven/common/audio/speech/stt.py` | New — in-process Whisper, auto-resamples mismatched input | ~195 lines |
| `raven/common/audio/speech/tts.py` | New — in-process Kokoro; two-layer synthesize API, finalize_metadata, prepare, decode | ~500 lines |
| `raven/common/audio/speech/tests/*` | New — smoke tests + in-process round-trip + bench tool | 5 files |
| `raven/client/mayberemote.py` | Added `TTS` + `STT` classes | ~130 lines added |
| `raven/client/tts.py` | `tts_prepare` → `EncodedTTSResult`; `tts_speak_lipsynced` migrated to attribute access; inline cleanup/diphthong tables removed | -45 lines net |
| `raven/client/api.py` | `stt_transcribe_array` accepts float; Whisper prompting pointer updated | tiny |
| `raven/client/avatar_controller.py` | `prep.audio_bytes` attribute access; empty-check replaces None-check | tiny |
| `raven/server/modules/stt.py` | Rewritten as thin wrapper | 183 → 75 lines |
| `raven/server/modules/tts.py` | Rewritten as thin wrapper | 218 → 172 lines |
| `raven/server/app.py` | Whisper prompting guidance moved *out* of `api_stt_transcribe` (now a transport-only doc) | -67 lines |

External HTTP API unchanged — same endpoints, same JSON wire shapes. The existing `test_api.py::TestStt::test_tts_stt_roundtrip` integration test passes untouched. Lipsync wire format (`x-word-timestamps` header) byte-for-byte unchanged.

## Commit sequence

A focused chain of commits, ordered so each stands alone and bisect has clean targets. `TODO_DEFERRED` commits are housekeeping; skim last.

```
# Design + first two layers
5f0606e  briefs/speech-extract-to-common: design corrections after review
40e7a81  common.audio: add device-agnostic resample helper (torchaudio)
482134c  common.audio.speech.stt: extract Whisper engine out of server module
64eeea4  common.audio.speech.tts: extract Kokoro engine + in-process round-trip test
84912ee  bench_tts_nondeterminism: save the Kokoro measurement as a tool
8d24f71  test_tts: correct the Kokoro nondeterminism comment

# MaybeRemote additions
e8d684f  mayberemote: add TTS and STT services (step 5 of speech extraction)

# Review-pass polish (multiple rounds)
6d687ef  speech refactor: review corrections (step 5 polish)
716148b  tts.WordTiming: don't speculate about when Kokoro returns None timestamps
c7fcc98  tts: rename KOKORO_SAMPLE_RATE to SAMPLE_RATE
c178ae2  tts.load_tts_pipeline: document espeak-ng prerequisite
ac8afa8  stt: document the [-1, 1] float range for audio inputs
fffa127  stt_transcribe_array: accept float input; drop cast from MaybeRemote
3068a58  common.audio.speech: tighten module docstrings per review style
f091d7d  stt.STTModel: reword sample_rate docstring
6986189  stt: move Whisper prompting guidance to the engine docstring

# tts_prepare generalization (the "WordTiming end-to-end" migration)
8671cfd  speech.tts: add dipthong_vowel_to_ipa table + expand_phoneme_diphthongs helper
cc7c5b8  client.tts: migrate to common-layer WordTiming at the wire boundary
6299d6b  mayberemote.TTS.synthesize: route remote path through api.tts_prepare
74b0132  speech.tts: add EncodedTTSResult + finalize_metadata + prepare
623563e  speech TTS: migrate prep dict → EncodedTTSResult dataclass end-to-end
e9f9b85  MaybeRemote.TTS.synthesize: full symmetry with the rest of the fleet

# TODO_DEFERRED / meta
b0108b3 c99a06f 32c2656 8a4eee9 4ef9a4d 84a1196 a5fd9c7 80f54e5 6c4a18f fcccaaf c0db5a9 1b0bb60
f8657a4  briefs: implementation overview for the speech extraction (this document)
```

## Module-by-module tour

### `raven/common/audio/resample.py` (new)

**Design call**: torchaudio over scipy. torchaudio adds a new dep (~20 MB wheel), and operates on torch tensors on any device — so a future GPU-flow-through path (keeping audio on the device between TTS and audio output) uses the same call with no code change. scipy would have worked fine today (audio is CPU-bound all the way through Kokoro→Whisper) but would have locked us out of GPU resampling forever.

**API shape**: `resample(audio, from_rate, to_rate, quality="default")` with a constrained `TypeVar` — `numpy in → numpy out`, `tensor in → tensor out on the same device`. `AudioT` and `Quality` are in `__all__` so callers can annotate their own code.

**Quality presets**: `"default"` (Hann window, width 6, speech-grade), `"kaiser_fast"` / `"kaiser_best"` (librosa naming). Baked in from the start so the helper can serve music/HQ audio later.

**Caveat captured in TODO_DEFERRED**: `torchaudio` on a fresh install can pull a CUDA-mismatched version from PyPI. Install via the pytorch index-url matching the installed torch CUDA build. Not fixable from `pyproject.toml` alone.

### `raven/common/audio/speech/stt.py` (new)

**`STTModel` @dataclass**: holds `(model, processor, device, dtype, sample_rate)`. `sample_rate` mirrors `processor.feature_extractor.sampling_rate` — inspectable metadata; `transcribe` auto-resamples so callers don't need to pre-convert.

**`load_stt_model(model_name, device_string, dtype)`**: memoized on `(model_name, device_string, str(dtype))`, same pattern as `nlptools._spacy_pipelines`.

**`transcribe(stt_model, audio, sample_rate, prompt=None, language=None, progress_callback=None)`**:

- **Auto-resamples** to Whisper's native rate via `raven.common.audio.resample.resample` if sample rates don't match. (Convenience lives in common, not MaybeRemote — MaybeRemote is a pure delegator.)
- `progress_callback` is a plain `(current, total)` callback; common layer adapts HF's batch-tensor `monitor_progress` convention internally.
- Full Whisper prompting guidance (pattern examples, 224-token limit, "not an instruction-following model") lives in the docstring. The HTTP endpoint at `raven.server.app.api_stt_transcribe` points to this — source of truth is at the engine layer.
- Preserves the original "feature length < 3000 means short-form" path (`transformers` issue #30740 workaround from the pre-refactor code).

### `raven/common/audio/speech/tts.py` (new)

This one is the largest module and the centre of the whole refactor. Contents:

**Dataclasses** — `TTSPipeline`, `WordTiming`, `TTSSegment`, `TTSResult`, `EncodedTTSResult`. `TTSResult` has `audio: np.ndarray` (raw float32). `EncodedTTSResult` is its sibling — "a `TTSResult` where the audio is encoded (to a file format) instead of raw" — `audio_bytes: bytes` + `audio_format: str`. Same `sample_rate` / `duration` / `word_metadata` fields on both.

**Two-layer synthesize API**:

- `synthesize_iter(pipeline, voice, text, ...)` yields per-segment `TTSSegment` with already-absolute word timestamps. The `t0` accumulator lives inside the iterator so consumers of either API entry point see absolute timings — a future streaming consumer doesn't have to re-implement accumulation.
- `synthesize(pipeline, voice, text, ...)` is the thin concatenating wrapper: collect segments, flatten audio, flatten metadata, return one `TTSResult`. Raw engine output; no post-processing.

**Audio dtype**: float32 in [-1, 1], Kokoro's native output. The `.cpu().numpy()` happens once per segment inside `synthesize_iter`, at the engine→common boundary. The server wrapper casts to s16 right before `audio_codec.encode` at the transport boundary.

**Post-processing helpers**:

- `clean_timestamps(timings, for_lipsync=True)` — two-mode filter. Dedup of consecutive same-start_time entries (a Kokoro-FastAPI-era bug guard) always applies. Single-char-token drop is lipsync-specific and opt-outable.
- `expand_phoneme_diphthongs(timings)` — Misaki's single-letter shorthand (A, I, W, Y, O, Q) → canonical IPA. Functional update via `dataclasses.replace`; no mutation.
- `finalize_metadata(timings)` — **single source of truth** for the lipsync-ready post-processing sequence. Composes `clean_timestamps(for_lipsync=True)` + `expand_phoneme_diphthongs`. Both the in-process `prepare` path and the HTTP wire path in `raven.client.tts.tts_prepare` call it — the sequence can't drift between them.

**High-level composers**:

- `prepare(pipeline, voice, text, speed, get_metadata) -> TTSResult` — the parallel of every other MaybeRemote's local-mode call. Does synthesize + finalize_metadata. One-line body. `MaybeRemote.TTS.synthesize`'s local path is pure delegation to this.
- `decode(encoded: EncodedTTSResult) -> TTSResult` — wire → in-process shape conversion. Decodes `audio_bytes` (MP3 etc.) to float32 numpy via `audio_codec.decode`. Empty-bytes fast path returns an empty `TTSResult` so the "cancelled" case (blank input, no phonemes) needs no special branch at the caller.

### `raven/server/modules/stt.py` (rewritten)

183 → 75 lines. Same external contract (`init_module`, `is_available`, `speech_to_text(stream, prompt, language) -> str`) so `app.py::api_stt_transcribe` needs no changes. Its only job is now: decode the audio container at the model's native rate via `audio_codec.decode`, wire a tqdm progress bar, forward to the common layer.

### `raven/server/modules/tts.py` (rewritten)

218 → 172 lines. Same external contract. Transport concerns (float→s16 cast, `urllib.parse.quote` on raw Unicode words/phonemes for ASCII HTTP headers, JSON header packing, segment-list handoff to `audio_codec.encode` for streaming preservation) live here. The `x-word-timestamps` wire format is byte-for-byte unchanged.

### `raven/server/app.py` (trimmed)

`api_stt_transcribe`'s docstring shrunk to transport concerns only (multipart format, parameter JSON shape, response shape). The Whisper-prompting guidance — pattern examples, 224-token limit, "not an instruction-following model" nuance, further-reading links — moved to the engine docstring in `common.audio.speech.stt.transcribe`. Source-of-truth at the layer that defines the `prompt` parameter's semantics.

### `raven/client/tts.py` (substantial refactor)

`tts_prepare`'s role in the refactor: it's the **wire receiver** — where `EncodedTTSResult` is constructed from the HTTP response. `WordTiming` objects are built at this boundary (URL-decoded words, URL-decoded phonemes, start/end times), then `speech_tts.finalize_metadata` is applied to give lipsync-ready output. Returns `EncodedTTSResult` (was: dict of `{"audio_bytes", "timestamps"}`).

Related changes:
- `tts_prepare` no longer returns `None` on blank input — returns an empty `EncodedTTSResult` (audio_bytes=b"", duration=0.0) as the "cancelled" signal. Callers check `not prep.audio_bytes`. This unblocks the one-line `speech_tts.decode(api.tts_prepare(...))` composition in MaybeRemote.
- `tts_speak` and `tts_speak_lipsynced` migrated to attribute access (`prep.audio_bytes`, `prep.word_metadata`). A pre-existing latent bug was flushed along the way: the old `None` check logged "Cancelled" but didn't actually `return`, so `final_prep.audio_bytes` would have crashed if `tts_prepare` ever returned None. Fixed: proper early-return on empty audio_bytes.
- Inline `isword` / `clean_timestamps` / `dipthong_vowel_to_ipa` table all deleted — `speech_tts.finalize_metadata` is the single source of truth.

### `raven/client/mayberemote.py` (TTS + STT additions)

All five MaybeRemoteServices now follow the same shape: `__init__` probes the server, local mode loads a common-layer engine if `allow_local=True`, methods dispatch on `self._local_model is None`, each method body is **one common-layer call per branch** (plus a shape conversion on TTS's remote side for the audio — see below).

**`STT.transcribe`**: pure delegator. Remote → `api.stt_transcribe_array(audio, ...)` (accepts float or s16, casts internally). Local → `speech_stt.transcribe(model, audio, ...)`. Both paths accept any sample rate — server resamples during decode; common layer auto-resamples.

**`TTS.synthesize`** (post all polish):

```python
if self._local_model is None:
    return speech_tts.decode(api.tts_prepare(voice=voice, text=text, speed=speed, get_metadata=get_metadata))
return speech_tts.prepare(self._local_model, voice=voice, text=text, speed=speed, get_metadata=get_metadata)
```

The `speech_tts.decode` wrap on the remote side is the *one genuine asymmetry with the rest of the fleet*: TTS's wire format is lossy-encoded audio bytes (`EncodedTTSResult`), not the same shape as the in-process representation (`TTSResult` with float audio). Every other MaybeRemote's wire rep matches its in-process rep, so no decode is needed there. The decode step is necessary and belongs where it is.

## Tests

All ml-tier (`@pytest.mark.ml`), skipped by default unless `-m ml` is passed.

| File | What | Count |
|---|---|---|
| `raven/common/audio/tests/test_resample.py` | Non-ml; type preservation, lengths, sine-wave integrity through round-trip, error paths | 16 |
| `raven/common/audio/speech/tests/test_stt.py` | Load / cache / silence → string / wrong-rate auto-resample / progress callback | 6 |
| `raven/common/audio/speech/tests/test_tts.py` | Load / voices / shape / metadata invariants / two-layer API equivalence / `clean_timestamps` two modes / `expand_phoneme_diphthongs` / `finalize_metadata` / `prepare` / `decode` empty-bytes fast path | 31 |
| `raven/common/audio/speech/tests/test_tts_stt_roundtrip.py` | Full in-process text → Kokoro → resample → Whisper → text, on pangram + numeric-token case | 2 |
| `raven/client/tests/test_api.py` (TTS cases) | prep returns audio_bytes / prep with metadata / blank-input empty-result | 3 |

Totals: 58 tests (16 non-ml resample, 39 ml-tier speech + prep, 3 client-side integration). ml-tier speech runs in ~35 s on CPU (model download on first run, then cached).

## Diagnostic tools

`raven/common/audio/speech/tests/bench_tts_nondeterminism.py` — runnable via `python -m raven.common.audio.speech.tests.bench_tts_nondeterminism`. Accepts `--text / --voice / --device / --runs / --repo`. Reports length spread, pairwise max / RMS diffs in both linear and dB-below-peak, regional breakdown (first / mid / last 1000 samples), and normalized cross-correlation at best lag.

**Baseline measurement captured** (CPU, af_alloy, pangram): identical length, zero phase offset, xcorr ≈ 0.993, RMS diff ≈ 3 % of peak (-37 dB). The diff concentrates in the voiced regions; leading / trailing silence stays near machine epsilon. Consistent with an unseeded stochastic vocoder sampler (standard for VITS-family TTS — naturalness feature, not a bug).

## Design-call summary

Ten choices worth re-reviewing against the actual code:

1. **torchaudio over scipy** for the resampler. Adds a dep; buys device-agnosticism.
2. **Constrained `TypeVar`** on `resample`, public in `__all__`. Type checker tracks "same in, same out".
3. **Float32 in common, s16 at the transport boundary**. Avoids a lossy round-trip for in-process consumers.
4. **All common-layer result types are `@dataclass`** (`TTSPipeline`, `STTModel`, `TTSSegment`, `TTSResult`, `EncodedTTSResult`, `WordTiming`). Local deviation from Raven's plain-class default, applied uniformly within the speech module.
5. **Absolute timestamps inside `synthesize_iter`**. `t0` accumulator isn't leaked to the consumer.
6. **Two-layer TTS API**. `synthesize_iter` + `synthesize`; reserves streaming without costing complexity.
7. **Auto-resample and finalize-metadata live in common, not MaybeRemote**. MaybeRemote stays a pure delegator — matches the existing `Dehyphenator` / `Embedder` / `NLP` pattern.
8. **`clean_timestamps` with `for_lipsync` mode**. Dedup always (engine-level noise); single-char drop is lipsync-specific and opt-outable.
9. **`EncodedTTSResult` as a first-class sibling of `TTSResult`**. Wire-encoded vs. raw-float; `speech_tts.decode` is the shape-conversion helper. Naming makes the family resemblance clear.
10. **`tts_prepare` returns empty `EncodedTTSResult` instead of `None` on blank input**. Uniform return type, one-liner composition at MaybeRemote. Pre-existing latent None-check bug in `tts_speak` / `tts_speak_lipsynced` fixed along the way.

## What's not in the refactor

- **Step 6** (manual smoke): Librarian + Avatar in remote and local modes, lipsync visually correct. JJ-side.
- `classify` and `translate` as MaybeRemote candidates — filed as deferred item.
- Server info endpoint for STT/TTS native sample rates — filed as deferred item.
- FLAC instead of MP3 for MaybeRemote TTS transport — filed as deferred item.
- `prepare_cached` variant of `speech_tts.prepare` for LRU caching at the common layer — filed as part of the `tts_prepare` generalization deferred item.
- Extract lipsync driver logic (`tts_speak_lipsynced`) into `raven.common.audio.speech.lipsync` — filed as deferred item, needed when the client-local avatar animator lands.
- The whole client-local avatar animator track — separate deferred item with its own authorship / licensing analysis.

## Known properties worth remembering

- **Kokoro is non-deterministic** (~3 % RMS diff in voiced regions between identical calls, xcorr 0.993). Intentional; common to VITS-family TTS. Not a bug. Test assertions that expected bit-equality had to be relaxed.
- **Whisper's CPU short-form path** kicks in when the feature stream length < 3000 — the `transformers` issue #30740 workaround from the pre-refactor code is preserved.
- **`maybe_install_models`** still auto-downloads on first `load_*` call. No local-cache skip path; first-run latency applies. Also true pre-refactor.
- **MP3 round-trip in MaybeRemote remote mode** — `TTS.synthesize` in remote mode goes `server-float → server-MP3 → wire → client-MP3 → client-float`, a lossy round-trip. Inaudible but not bit-identical. Switching the server's wire format to FLAC would fix this (deferred item).
