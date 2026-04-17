# Speech extraction — implementation overview

**Status**: steps 1–5 complete, step 6 (manual smoke) pending.
**Reading companion to**: `briefs/speech-extract-to-common.md` (the design brief).
**Purpose**: a guided tour of what actually landed in the code, so the diffs can be read with context.

This document is the reading guide, not the authoritative record — git log is. Use this to decide which diffs to read carefully and which to skim.

## TL;DR

Kokoro (TTS) and Whisper (STT) are no longer tied to the Flask server. Both can be called in-process from any Raven app, and the server modules are now thin transport wrappers. `raven.client.mayberemote` gained `TTS` and `STT` services so apps can transparently use the server when available or run locally otherwise.

Six new things landed along the way, plus the rewrites:

| Location | What | Size |
|---|---|---|
| `raven/common/audio/resample.py` | New — torchaudio-backed, device-agnostic resampler with quality presets | ~105 lines |
| `raven/common/audio/speech/stt.py` | New — in-process Whisper | ~195 lines |
| `raven/common/audio/speech/tts.py` | New — in-process Kokoro, two-layer synthesize API | ~310 lines |
| `raven/common/audio/speech/tests/*` | New — smoke tests + full in-process round-trip | 4 files |
| `raven/common/audio/speech/tests/bench_tts_nondeterminism.py` | New — diagnostic tool | ~140 lines |
| `raven/client/mayberemote.py` | Added `TTS` + `STT` classes | ~140 lines added |
| `raven/server/modules/stt.py` | Rewritten as thin wrapper | 183 → 75 lines |
| `raven/server/modules/tts.py` | Rewritten as thin wrapper | 218 → 172 lines |

External HTTP API unchanged. The existing `test_api.py::TestStt::test_tts_stt_roundtrip` integration test passes untouched. Lipsync wire format (`x-word-timestamps` header) byte-for-byte unchanged.

## Commit sequence

The work is a chain of focused commits, ordered so each one stands alone and bisect has clean targets:

```
5f0606e  briefs/speech-extract-to-common: design corrections after review
40e7a81  common.audio: add device-agnostic resample helper (torchaudio)
482134c  common.audio.speech.stt: extract Whisper engine out of server module
b0108b3  TODO_DEFERRED: four items raised during speech-extract discussion
c99a06f  TODO_DEFERRED: review reminder + licensing/cropping corrections
32c2656  TODO_DEFERRED: fix confused phrasing about AGPL reach
64eeea4  common.audio.speech.tts: extract Kokoro engine + in-process round-trip test
8d24f71  test_tts: correct the Kokoro nondeterminism comment
84912ee  bench_tts_nondeterminism: save the Kokoro measurement as a tool
e8d684f  mayberemote: add TTS and STT services (step 5 of speech extraction)
8a4eee9  TODO_DEFERRED: precise authorship provenance for server modules
4ef9a4d  TODO_DEFERRED: user-authored avatar code spans ST-extras era too
84a1196  TODO_DEFERRED: honest latency claim for client-local animator
6d687ef  speech refactor: review corrections (step 5 polish)
```

The `TODO_DEFERRED` commits are housekeeping — they can be skimmed last. The code commits are the substance.

## Module-by-module tour

### `raven/common/audio/resample.py` (new)

**Design call**: torchaudio over scipy. torchaudio adds a new dep (~20 MB wheel), and operates on torch tensors on any device — so a future GPU-flow-through path (keeping audio on the device between TTS and audio output) uses the same call with no code change. scipy would have worked fine today (audio is CPU-bound all the way through Kokoro→Whisper) but would have locked us out of GPU resampling forever.

**API shape**: `resample(audio, from_rate, to_rate, quality="default")` with a constrained `TypeVar` — `numpy in → numpy out`, `tensor in → tensor out on the same device`. `AudioT` and `Quality` are in `__all__` so callers can annotate their own code.

**Quality presets**: `"default"` (Hann window, width 6, speech-grade), `"kaiser_fast"` / `"kaiser_best"` (librosa naming). Baked in from the start so the helper can serve music/HQ audio later — defends its place in `common.audio`.

**Caveat captured in TODO_DEFERRED**: `torchaudio` on a fresh install can pull a CUDA-mismatched version from PyPI. Install via the pytorch index-url matching the installed torch CUDA build. Not fixable from `pyproject.toml` alone.

### `raven/common/audio/speech/stt.py` (new)

**`STTModel` @dataclass**: holds `(model, processor, device, dtype, sample_rate)`. The `sample_rate` mirror (of `processor.feature_extractor.sampling_rate`) is a convenience so callers don't have to reach through to HF internals.

**`load_stt_model(model_name, device_string, dtype)`**: memoized on `(model_name, device_string, str(dtype))`, same pattern as `nlptools._spacy_pipelines`.

**`transcribe(stt_model, audio, sample_rate, prompt=None, language=None, progress_callback=None)`**:

- Auto-resamples to the model's native rate. If the caller passes 24 kHz audio (Kokoro's output), the function resamples to 16 kHz internally. Moved here at review (was previously strict; JJ's call — auto-convenience belongs in common, not at MaybeRemote).
- `progress_callback` takes a plain `(current, total)` callback; common layer adapts that to HF's batch-tensor `monitor_progress` convention internally. Library callers never see the HF-specific tensor shape.
- Preserves the original Whisper-specific pre-processing path: the "feature length < 3000 means short-form" branch from the pre-refactor code.

### `raven/common/audio/speech/tts.py` (new)

**Dataclasses**: `TTSPipeline` holds `(kpipeline, modelsdir, lang_code, sample_rate)` — `modelsdir` is needed because Kokoro has no voice-enumeration API and `get_voices` scans `.pt` files on disk. `WordTiming`, `TTSSegment`, `TTSResult` are the result types; raw Unicode in `word` / `phonemes` (no URL-encoding — that's a transport concern).

**Two-layer synthesize API**:

- `synthesize_iter(pipeline, voice, text, ...)` yields per-segment `TTSSegment` with already-absolute word timestamps. The `t0` accumulator lives inside the iterator so consumers of either API entry point see absolute timings — a future streaming consumer doesn't have to re-implement accumulation.
- `synthesize(pipeline, voice, text, ...)` is the thin concatenating wrapper: collect all segments, flatten the audio, flatten the metadata, return one `TTSResult`. Current server path uses this.

**Audio dtype**: float32 in [-1, 1], Kokoro's native output. The `.cpu().numpy()` happens once per segment inside `synthesize_iter`, at the engine→common boundary. The server wrapper casts to s16 right before `audio_codec.encode` at the transport boundary.

**`clean_timestamps(timings, for_lipsync=True)`**: two-mode filter. Dedup of consecutive same-start_time entries (a Kokoro-FastAPI-era bug guard, kept as precaution) always applies; the single-char-token drop is lipsync-specific and opt-outable. MaybeRemote.TTS uses `for_lipsync=True` (default); a captioning/transcript caller can pass `for_lipsync=False`.

### `raven/server/modules/stt.py` (rewritten)

183 → 75 lines. Same external contract (`init_module`, `is_available`, `speech_to_text(stream, prompt, language) -> str`) so `app.py::api_stt_transcribe` needs no changes. Its only job is now: decode the audio container at the model's native rate via `audio_codec.decode`, wire a tqdm progress bar, forward to the common layer.

### `raven/server/modules/tts.py` (rewritten)

218 → 172 lines. Same external contract. Transport concerns (float→s16 cast, `urllib.parse.quote` on raw Unicode words/phonemes for ASCII HTTP headers, JSON header packing, segment-list handoff to `audio_codec.encode` for streaming preservation) live here. The `x-word-timestamps` wire format is byte-for-byte unchanged.

### `raven/client/mayberemote.py` (TTS + STT additions)

Both classes follow the existing `Dehyphenator` / `Embedder` / `NLP` pattern: `__init__` probes the server, local mode loads a common-layer engine if `allow_local=True`, methods dispatch on `self._local_model is None`.

**`STT.transcribe`**: pure delegator. Both paths accept any sample rate (remote server resamples during its decode pass; local common layer auto-resamples). No convenience functionality added by MaybeRemote itself.

**`TTS.synthesize`**: uniform return type (`speech_tts.TTSResult` with float32 audio + `WordTiming` list) regardless of mode. Remote path uses a local helper `_remote_tts_speak_raw` that calls the HTTP endpoint directly (bypassing `raven.client.tts.tts_prepare`'s lipsync-specific dict-level cleanup) so raw Kokoro metadata comes back. Then `speech_tts.clean_timestamps` (`for_lipsync=True` default) is applied uniformly to both local and remote results.

**Remote path round-trips through MP3**: the server's default transport format is lossy, so remote-mode sample values aren't bit-identical to the server's internal float. Inaudible; noted in the docstring. Deferred item suggests trying FLAC now that Raven is off Kokoro-FastAPI.

**`sample_rate` attribute**: set at construction. Local reads it off the loaded model; remote uses the canonical value (Whisper 16 kHz, Kokoro 24 kHz). Deferred item proposes a server info endpoint so this stops being hardcoded.

## Tests

All ml-tier (`@pytest.mark.ml`), skipped by default.

| File | What | Count |
|---|---|---|
| `raven/common/audio/tests/test_resample.py` | Non-ml; type preservation, lengths, sine-wave integrity through round-trip, error paths | 16 |
| `raven/common/audio/speech/tests/test_stt.py` | Load / cache / silence → string / wrong-rate auto-resample / progress callback | 6 |
| `raven/common/audio/speech/tests/test_tts.py` | Load / voices / shape / metadata invariants (absolute-timestamp monotonicity, raw-Unicode) / two-layer API equivalence / `clean_timestamps` two modes | 19 |
| `raven/common/audio/speech/tests/test_tts_stt_roundtrip.py` | Full in-process text → Kokoro → resample → Whisper → text, on pangram + numeric-token case | 2 |

Totals: 43 new tests (16 non-ml resample, 27 ml-tier speech). ml-tier speech runs in ~35 s on CPU (model download on first run, then cached).

## Diagnostic tools

`raven/common/audio/speech/tests/bench_tts_nondeterminism.py` — runnable via `python -m raven.common.audio.speech.tests.bench_tts_nondeterminism`. Accepts `--text / --voice / --device / --runs / --repo`. Reports length spread, pairwise max / RMS diffs in both linear and dB-below-peak, regional breakdown (first / mid / last 1000 samples), and normalized cross-correlation at best lag.

**Baseline measurement captured** (CPU, af_alloy, pangram): identical length, zero phase offset, xcorr ≈ 0.993, RMS diff ≈ 3 % of peak (-37 dB). The diff concentrates in the voiced regions; leading / trailing silence stays near machine epsilon. Consistent with an unseeded stochastic vocoder sampler (standard for VITS-family TTS — naturalness feature, not a bug).

## Design-call summary

Quick list of the choices worth re-reviewing against the actual code:

1. **torchaudio over scipy** for the resampler. Adds a dep; buys device-agnosticism.
2. **Constrained `TypeVar`** on `resample`, public in `__all__`. Better than `Union` — type checker tracks "same in, same out".
3. **Float32 in common, s16 at the transport boundary**. Avoids a lossy round-trip for in-process consumers.
4. **All common-layer result types are `@dataclass`** (`TTSPipeline`, `STTModel`, `TTSSegment`, `TTSResult`, `WordTiming`). Local deviation from Raven's plain-class default, applied uniformly within the speech module.
5. **Absolute timestamps inside `synthesize_iter`**. `t0` accumulator isn't leaked to the consumer.
6. **Two-layer TTS API**. `synthesize_iter` + `synthesize`; reserves streaming without costing complexity.
7. **Auto-resample inside `speech_stt.transcribe`**, not inside MaybeRemote. MaybeRemote stays a pure delegator.
8. **`clean_timestamps` with `for_lipsync` mode**. Dedup always (engine-level noise); single-char drop is lipsync-specific and opt-outable.
9. **Per-segment metadata propagation** in TTS. The server's streaming path keeps the per-segment encode boundary via `synthesize_iter`.
10. **Direct HTTP call** in MaybeRemote TTS (bypassing `tts_prepare`). So both paths return raw Kokoro metadata; cleanup is applied uniformly afterwards.

## What's not in the refactor

- **Step 6** (manual smoke): Librarian + Avatar in remote and local modes, lipsync visually correct. JJ-side.
- `classify` and `translate` as MaybeRemote candidates: mechanical, filed as deferred item.
- Server info endpoint for STT/TTS native sample rates: filed as deferred item.
- FLAC instead of MP3 for MaybeRemote TTS transport: filed as deferred item.
- Generalizing `raven.client.tts.tts_prepare` for local-mode offline precomputation: filed as deferred item.
- The whole client-local avatar animator track: separate deferred item with its own (thorny) authorship/licensing analysis.

## Known properties worth remembering

- **Kokoro is non-deterministic** (~3 % RMS diff in voiced regions between identical calls, xcorr 0.993). Intentional; common to VITS-family TTS. Not a bug. Test assertions that expected bit-equality had to be relaxed.
- **Whisper's CPU short-form path** kicks in when the feature stream length < 3000 — the `transformers` issue #30740 workaround from the pre-refactor code is preserved.
- **`maybe_install_models`** still auto-downloads on first `load_*` call. No local-cache skip path; first-run latency applies. Also true pre-refactor.
