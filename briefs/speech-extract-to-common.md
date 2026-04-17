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
raven/common/audio/speech/
├── __init__.py
├── tts.py       # Kokoro wrapper
├── stt.py       # Whisper wrapper
└── tests/
    ├── __init__.py
    └── test_tts_stt_roundtrip.py   # ml-tier, in-process
```

### `raven.common.audio.speech.tts`

Proposed API:

```python
def load_tts_pipeline(repo_id: str, device_string: str, lang_code: str = "a") -> TTSPipeline: ...

def get_voices(pipeline: TTSPipeline) -> list[str]: ...

def synthesize(pipeline: TTSPipeline,
               voice: str,
               text: str,
               speed: float = 1.0,
               get_metadata: bool = True) -> TTSResult: ...
```

Where `TTSResult` is a plain dataclass / `TypedDict`:

```python
@dataclass
class TTSResult:
    audio: np.ndarray                    # rank-1 s16, shape [n_samples]
    sample_rate: int                     # Kokoro = 24000
    duration: float                      # seconds
    word_metadata: list[WordTiming] | None   # None if get_metadata=False
```

```python
@dataclass
class WordTiming:
    word: str                 # raw, NOT URL-encoded — encoding is a transport concern
    phonemes: str             # raw, NOT URL-encoded
    start_time: float | None  # seconds from audio start
    end_time: float | None
```

Caching: follow the `nlptools._spacy_pipelines = {}` pattern — cache key `(repo_id, device_string, lang_code)`.

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

`STTModel` wraps the (`transformers.AutoModelForSpeechSeq2Seq`, `transformers.AutoProcessor`, `torch.device`, `dtype`) tuple — same small-dataclass style as `nlptools._Translator`.

The `transcribe` signature takes a decoded numpy array, not a filelike — audio container decoding is `audio.codec.decode`'s job, not STT's.  The server wrapper continues to call `audio_codec.decode` before handing off to `transcribe`.

### What stays in `raven.server.modules.tts`

Only the HTTP-facing parts:

```python
def text_to_speech(voice, text, speed, format, get_metadata, stream) -> flask.Response:
    result = speech_tts.synthesize(_pipeline, voice, text, speed, get_metadata=get_metadata)

    audio_bytes_or_streamer = audio_codec.encode(result.audio, format=format, sample_rate=result.sample_rate, stream=stream)

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

The URL-encoding, the `x-*` header packing, the audio-format encoding, the Flask `Response` construction — all belong in the server wrapper.  They are transport concerns, not engine concerns.

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

    # STT wants float mono at the model's native sample rate — resample if needed.
    audio_f = synth.audio.astype(np.float32) / 32767.0
    transcribed = speech_stt.transcribe(stt, audio=audio_f, sample_rate=synth.sample_rate, prompt=original, language="en")

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
2. Extract STT first: `raven/common/audio/speech/stt.py` + server wrapper update + test.  Commit.
3. Extract TTS: `raven/common/audio/speech/tts.py` + server wrapper update + test.  Commit.
4. Add `MaybeRemoteService` subclasses (`TTS`, `STT`).  Commit.
5. Manual smoke test: Librarian + Avatar in both remote (server on) and local (server off) modes; confirm lipsync still works.
