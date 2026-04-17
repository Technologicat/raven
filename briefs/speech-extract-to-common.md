# Extract speech engines (TTS, STT) into `raven.common`

**Date:** 2026-04-17
**Status:** Proposed
**Trigger:** Following the `nlptools` extraction precedent, the TTS (Kokoro) and STT (Whisper) engines currently live inside `raven/server/modules/{tts,stt}.py` mixed with Flask response construction, HTTP header packing, and URL-quoting for transport.  This means the only way to exercise the engines in-process is via the full server HTTP stack.  The client-side integration test (`raven/client/tests/test_api.py::TestStt::test_tts_stt_roundtrip`) has to spin up a running Raven-server to run, and its value as a regression test is diluted — a failure could come from the engine, the Flask layer, the network, or the client.

## Context

- `raven.common.nlptools` is the precedent: pure in-process ML logic with model-load caching and device management, no Flask / HTTP.  The server's `classify`, `translate`, `sanitize`, `natlang`, `embeddings` modules all wrap `nlptools`.
- `raven/common/tests/test_nlptools.py` exercises that layer directly, in-process, under the `@pytest.mark.ml` tier — no HTTP, no Flask, no client/server handshake.  That's the model to replicate for speech.
- `raven.client.mayberemote` already provides `Dehyphenator`, `Embedder`, `NLP` as classes that transparently use either the server (if reachable) or a local in-process copy.  Adding `TTS` and `STT` to this family would let Librarian / Avatar apps degrade gracefully when the server isn't running, and would make the engines easier to test either way.

## Goal

1. Move the Kokoro and Whisper orchestration logic out of `raven.server.modules` and into `raven.common.audio.speech`, so the engines are callable in-process without Flask.
2. Reduce `raven/server/modules/{tts,stt}.py` to thin Flask wrappers: receive request, call common, wrap result in a `flask.Response` with the appropriate headers.
3. Add `TTS` and `STT` classes to `raven.client.mayberemote`, mirroring the existing `Dehyphenator` / `Embedder` / `NLP` pattern.
4. Land a local ML-tier round-trip test: text → TTS → STT → text, no HTTP involved.  The existing client-side roundtrip stays as an integration test of the server stack.

## Proposed module layout

```
raven/common/audio/
├── codec.py         # existing
├── player.py        # existing
├── recorder.py      # existing
├── utils.py         # existing
├── resample.py      # NEW — torchaudio-backed, device-agnostic resampler
└── speech/
    ├── __init__.py
    ├── tts.py       # Kokoro wrapper
    ├── stt.py       # Whisper wrapper
    └── tests/
        ├── __init__.py
        └── test_tts_stt_roundtrip.py   # ml-tier, in-process
```

### `raven.common.audio.resample`

New companion module for `audio.codec`, needed because TTS (Kokoro @ 24 kHz) and STT (Whisper @ 16 kHz) run at different sample rates and the round-trip test path feeds one into the other in-process.  Today the mismatch is papered over by `audio_codec.decode(stream, target_sample_rate=...)` on the server side (PyAV resamples as a side-effect of container decoding), but that helper takes a filelike, not a numpy array — useless for the in-process numpy → numpy path.

Backend: `torchaudio.functional.resample`.  Rationale over `scipy.signal.resample_poly`:

- **Device-agnostic.**  Operates on whatever device the input tensor lives on.  Aligns with the Raven convention of running heavy signal processing on the GPU when possible.
- **Future-proofs** a GPU-flow-through path (e.g. streaming lipsync keeping audio on device between TTS and audio output) — the resampler doesn't change, only the caller.
- `torchaudio` is a small addition (~20 MB wheel) next to `torch`, already a direct dep.

For today's TTS→STT round-trip the audio is already on CPU (Kokoro output is moved host-side inside `synthesize_iter` — see below), so GPU resampling would be a net slowdown.  The point of choosing torchaudio is the option, not the default.

API — polymorphic on input type, with a constrained `TypeVar` so the type checker tracks "same type in, same type out":

```python
from typing import Literal, TypeVar

Quality = Literal["default", "kaiser_fast", "kaiser_best"]
AudioT = TypeVar("AudioT", np.ndarray, torch.Tensor)

def resample(audio: AudioT,
             from_rate: int,
             to_rate: int,
             quality: Quality = "default") -> AudioT:
    """Resample a 1-D float audio signal.

    Returns the same type as input:
      - numpy in  → numpy out (torch stays on CPU under the hood)
      - tensor in → tensor out on the same device

    `quality`:
      "default"     — sinc_interp_hann, width=6.   Fast; speech-grade.
      "kaiser_fast" — sinc_interp_kaiser, beta≈8.56, width=16.   librosa "fast" preset.
      "kaiser_best" — sinc_interp_kaiser, beta≈14.77, width=64.  librosa "best" preset; music-grade.

    No-op when `from_rate == to_rate`.
    """
```

No caching or model loading — pure function.  The `quality` preset is there so the helper can carry its weight beyond speech (music, upscaled audio in cel animations, anything future).

### `raven.common.audio.speech.tts`

Proposed API — two layers, so streaming is a natural extension later:

```python
def load_tts_pipeline(repo_id: str, device_string: str, lang_code: str = "a") -> TTSPipeline: ...

def get_voices(pipeline: TTSPipeline) -> list[str]: ...

def synthesize_iter(pipeline: TTSPipeline,
                    voice: str,
                    text: str,
                    speed: float = 1.0,
                    get_metadata: bool = True) -> Iterator[TTSSegment]: ...

def synthesize(pipeline: TTSPipeline,
               voice: str,
               text: str,
               speed: float = 1.0,
               get_metadata: bool = True) -> TTSResult: ...
```

`synthesize_iter` yields one `TTSSegment` per Kokoro segment.  `synthesize` is a thin wrapper that concatenates the segments into a single `TTSResult`.  Current server usage (one big audio response per request) calls `synthesize`; a future streaming endpoint can consume `synthesize_iter` directly.

All four container types in this module are `@dataclass` (stdlib) — they are pure data records with no methods.  This is a local deviation from Raven's prevailing plain-class style, justified because dataclasses are a genuinely better fit here and uniform treatment inside the module beats matching the fleet default.

```python
@dataclass
class TTSPipeline:
    kpipeline: kokoro.KPipeline
    modelsdir: str            # needed by get_voices — Kokoro doesn't expose it
    lang_code: str
    sample_rate: int          # 24000 for Kokoro
```

Caching: follow the `nlptools._spacy_pipelines = {}` pattern — cache key `(repo_id, device_string, lang_code)`.

```python
@dataclass
class WordTiming:
    word: str                 # raw, NOT URL-encoded — encoding is a transport concern
    phonemes: str             # raw, NOT URL-encoded
    start_time: float | None  # seconds from start of whole audio (absolute, not segment-relative)
    end_time: float | None

@dataclass
class TTSSegment:
    audio: np.ndarray                          # rank-1 float32 in [-1, 1], shape [n_samples]
    sample_rate: int                           # Kokoro = 24000
    t0: float                                  # offset of this segment from start of whole audio, seconds
    word_metadata: list[WordTiming] | None     # None if get_metadata=False; timings already absolute

@dataclass
class TTSResult:
    audio: np.ndarray                          # rank-1 float32 in [-1, 1], concatenated
    sample_rate: int
    duration: float                            # seconds
    word_metadata: list[WordTiming] | None
```

Audio is float32 in the common layer — Kokoro's native output.  The server wrapper casts to s16 right before `audio_codec.encode` (transport concern).  This avoids a pointless `float→s16→float` round-trip when the common-layer API is consumed in-process (e.g. by the STT test fixture).

**Absolute timestamps throughout.**  Kokoro emits segment-relative timestamps.  `synthesize_iter` accumulates `t0` internally and rewrites every `WordTiming.start_time` / `end_time` to whole-audio-absolute before yielding.  Consumers of either API get the same convention — a streaming consumer does not have to re-implement accumulation.

### `raven.common.audio.speech.stt`

Proposed API:

```python
def load_stt_model(model_name: str, device_string: str,
                   dtype: str | torch.dtype) -> STTModel: ...

def transcribe(model: STTModel,
               audio: np.ndarray,          # rank-1 float, mono
               sample_rate: int,           # must match model.feature_extractor.sampling_rate
               prompt: str | None = None,
               language: str | None = None) -> str: ...
```

```python
@dataclass
class STTModel:
    model: transformers.AutoModelForSpeechSeq2Seq
    processor: transformers.AutoProcessor
    device: torch.device
    dtype: torch.dtype
```

An optional `progress_callback: Callable[[int, int], None] | None` parameter on `transcribe` lets the server wrapper plug in its `tqdm` adapter without the common layer growing a UX dependency.

The `transcribe` signature takes a decoded numpy array, not a filelike — audio container decoding is `audio.codec.decode`'s job, not STT's.  The server wrapper continues to call `audio_codec.decode` before handing off to `transcribe`.

### What stays in `raven.server.modules.tts`

Only the HTTP-facing parts:

```python
def text_to_speech(voice, text, speed, format, get_metadata, stream) -> flask.Response:
    result = speech_tts.synthesize(_pipeline, voice, text, speed, get_metadata=get_metadata)

    # float32 [-1, 1] -> s16 (transport format)
    audio_s16 = np.asarray(result.audio * 32767.0, dtype=np.int16)
    audio_bytes_or_streamer = audio_codec.encode(audio_s16, format=format, sample_rate=result.sample_rate, stream=stream)

    headers = {"Content-Type": f"audio/{format}",
               "x-audio-duration": result.duration}
    if get_metadata:
        headers["x-word-timestamps"] = json.dumps([
            {"word": urllib.parse.quote(w.word, safe=""),
             "phonemes": urllib.parse.quote(w.phonemes, safe=""),
             "start_time": w.start_time,
             "end_time": w.end_time}
            for w in result.word_metadata
        ])
    return flask.Response(audio_bytes_or_streamer, headers=headers)
```

The URL-encoding, the `x-*` header packing, the float→s16 cast, the audio-format encoding, the Flask `Response` construction — all belong in the server wrapper.  They are transport concerns, not engine concerns.

(The current in-process consumer of `word_metadata` is the **lipsync driver** in `raven/client/tts.py`, which today parses the URL-encoded `x-word-timestamps` header.  Once the common layer exists, a future local-mode lipsync path can consume `WordTiming` objects directly, with no URL-encoding on either end.)

### What stays in `raven.server.modules.stt`

Similar skeletal shape — decode audio, call `common.audio.speech.stt.transcribe`, return the string.

## MaybeRemoteService additions

Parallel to the existing pattern:

```python
class TTS(MaybeRemoteService):
    def __init__(self, allow_local, repo_id=None, device_string=None, lang_code="a"): ...
    def list_voices(self) -> list[str]: ...
    def synthesize(self, voice, text, speed=1.0, get_metadata=True) -> TTSResult: ...

class STT(MaybeRemoteService):
    def __init__(self, allow_local, model_name=None, device_string=None, dtype=None): ...
    def transcribe(self, audio, sample_rate, prompt=None, language=None) -> str: ...
```

Both keep the same "if remote: call HTTP API; if local: call common" shape as `Dehyphenator`.

**Design caveat**: TTS models are heavy (~360 MB for Kokoro-82M) and STT models are heavier (Whisper-base is 74M params, Whisper-large-v3-turbo is 800M).  For apps that only occasionally need these and would rather fail than auto-download-and-run locally, `allow_local=False` is the right default in the app-level glue.

## Tests

The ml-tier round-trip test (`test_tts_stt_roundtrip.py`):

```python
pytest.importorskip("kokoro")
pytest.importorskip("transformers")
pytest.importorskip("torch")
pytestmark = pytest.mark.ml

@pytest.fixture(scope="session")
def tts():
    return speech_tts.load_tts_pipeline(repo_id="hexgrad/Kokoro-82M", device_string="cpu")

@pytest.fixture(scope="session")
def stt():
    return speech_stt.load_stt_model("openai/whisper-base", device_string="cpu", dtype="float32")

def test_tts_stt_roundtrip_recognizes_word(tts, stt):
    original = "The quick brown fox jumps over the lazy dog."
    synth = speech_tts.synthesize(tts, voice=speech_tts.get_voices(tts)[0], text=original, get_metadata=False)

    # Common-layer TTS already emits float32 mono — feed straight into STT.
    # (If sample rates differ between Kokoro and the Whisper feature extractor, resample here.)
    transcribed = speech_stt.transcribe(stt, audio=synth.audio, sample_rate=synth.sample_rate, prompt=original, language="en")

    low = transcribed.lower()
    # At least one recognizable noun from the original should survive the round-trip.
    assert any(word in low for word in ("fox", "dog", "quick", "brown", "lazy"))

def test_synthesize_metadata_shape(tts):
    result = speech_tts.synthesize(tts, voice=..., text="Hello world.", get_metadata=True)
    assert result.word_metadata is not None
    assert all(hasattr(w, "word") and hasattr(w, "phonemes") for w in result.word_metadata)
    # Not URL-encoded in the common layer — percent signs would suggest transport leakage.
    assert not any("%" in w.word for w in result.word_metadata)
```

The existing `raven/client/tests/test_api.py::TestStt::test_tts_stt_roundtrip` stays as an integration test of the server HTTP layer — its failure mode is now "server/Flask is broken", not "engine is broken".

## Out of scope

- **Audio recording/playback hardware** (`raven/common/audio/{recorder,player}.py`) — orthogonal, stays as-is.  The round-trip test uses synthetic audio, not hardware.
- **Avatar lipsync driver** (the consumer of `word_metadata`) — no changes needed.  Consumers that parse the current URL-encoded `x-word-timestamps` header live in `raven/client/tts.py` and keep working; the new common-layer `WordTiming.word` / `.phonemes` are just not URL-encoded in-process.
- **Whisper-large-v3-turbo performance work** — the 800 MB model is heavy on CPU.  The test should use `whisper-base` (74 M) like the Linux default in `raven/server/config.py`.
- **`raven.server.config.speech_recognition_model` default switch** — leave the current default alone.
- **Format negotiation redesign** — `format` (mp3/wav/flac/opus/aac) stays a server-wrapper concern; in-process API works with numpy audio only.

## Success criteria

- `raven/server/modules/tts.py` and `raven/server/modules/stt.py` shrink to roughly <100 lines each (thin Flask wrappers).
- `raven/common/audio/speech/{tts,stt}.py` are independently usable in a REPL with no Flask context.
- `pytest -m ml` passes the new round-trip test on the dev box in under 60 s (after first model download).
- The existing client-side integration test in `test_api.py` continues to pass unchanged.
- `raven.client.mayberemote.TTS` and `.STT` exist and work in both remote and local modes.
- `raven/common/audio/speech/` has no `import flask`, no `urllib.parse` calls, no `x-*` header logic.

## Pitfalls

- **Phoneme Unicode in in-process API.**  Kokoro emits IPA phonemes with characters like `ˈ`, `ɪ`, `ɹ`.  The server currently URL-encodes these for header transport; the common-layer API should return them as raw `str`.  Any caller that needs URL-encoding (i.e. the server wrapper) applies it at the boundary.
- **Kokoro `init_module` side effects.**  The current `init_module` both installs models via `maybe_install_models` and instantiates the `KPipeline` on a module-global.  The common-layer `load_tts_pipeline` should return the pipeline as a value (no module globals) and let callers pass it around, following `nlptools.load_spacy_pipeline`.  The server's module globals move into the server wrapper.
- **Whisper `generate(monitor_progress=...)`.**  The current STT module wires a `tqdm` progress bar into `model.generate`.  That's a UX concern — fine for the server (logs to console), awkward for a library function.  `common.audio.speech.stt.transcribe` should accept an optional `progress_callback` and default to None; the server wrapper supplies its `tqdm` adapter.
- **Config module imports.**  The current `init_module` takes a `config_module_name: str` and does `importlib.import_module(config_module_name)` to look up model names and HF repo IDs.  That's a server-side concern; the common-layer loader takes the values directly as arguments.
- **Order-of-extraction matters.**  Do STT first — it's simpler (fewer moving parts, no token metadata).  Do TTS second — word metadata and the `get_metadata=True` path have the most subtle plumbing.  Commit each extraction separately so a bisect has a clean target.
- **Don't break Avatar lipsync.**  `raven/client/tts.py` parses the percent-encoded `x-word-timestamps` header today.  The server-wrapper URL-encoding must continue to match what the client decodes.  A regression here breaks lipsync silently — visual inspection during manual test is warranted.

## Sequencing

1. `briefs/speech-extract-to-common.md` lands (this document).
2. Add `raven/common/audio/resample.py` + tests + `torchaudio` dependency.  Commit.  (Needed before the round-trip test in step 3 can run.)
3. Extract STT first: `raven/common/audio/speech/stt.py` + server wrapper update + test.  Commit.
4. Extract TTS: `raven/common/audio/speech/tts.py` + server wrapper update + test.  Commit.
5. Add `MaybeRemoteService` subclasses (`TTS`, `STT`).  Commit.
6. Manual smoke test: Librarian + Avatar in both remote (server on) and local (server off) modes; confirm lipsync still works.
