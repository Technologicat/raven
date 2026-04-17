"""Text-to-speech engine (Kokoro), as an in-process library.

Split out from `raven.server.modules.tts` so the engine can be exercised
without running Flask. The server wrapper now calls into this module
and applies transport-layer concerns (URL-encoding the raw phonemes,
s16 cast, audio-format encoding, Flask Response construction) at its
own boundary.

Two-layer API:

- `synthesize_iter` yields one `TTSSegment` per Kokoro segment, with
  timestamps already rewritten to absolute (from start of whole audio).
  A streaming consumer can forward segments as they come.

- `synthesize` is a thin concatenating wrapper: collect all segments,
  flatten the audio, concatenate the metadata list, return one
  `TTSResult`. Current single-response server path uses this.

Audio is float32 in [-1, 1] throughout — Kokoro's native output.
The server wrapper casts to s16 right before `audio_codec.encode`.

No Flask, no URL-encoding, no header packing.
"""

__all__ = ["WordTiming",
           "TTSSegment",
           "TTSResult",
           "TTSPipeline",
           "load_tts_pipeline",
           "get_voices",
           "synthesize_iter",
           "synthesize",
           "clean_timestamps"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

import kokoro

from ...hfutil import maybe_install_models


# Kokoro's native output sample rate. Documented in Kokoro's own README.
# Kept as a module constant so callers can reference it by name rather than
# reading it off `TTSPipeline.sample_rate` when they don't have a pipeline handy.
KOKORO_SAMPLE_RATE = 24000


@dataclass
class WordTiming:
    """One word (or word-like token) in the synthesized speech, with absolute timing.

    `word`, `phonemes`: raw Unicode strings — not URL-encoded. Any transport
                       that needs ASCII-only headers (e.g. HTTP) encodes at its own boundary.

    `start_time`, `end_time`: seconds from start of the whole audio (absolute, not segment-relative).
                              May be `None` if Kokoro didn't produce a timestamp for this token
                              (happens occasionally for short function words).
    """
    word: str
    phonemes: str
    start_time: Optional[float]
    end_time: Optional[float]


@dataclass
class TTSSegment:
    """One segment of synthesized audio, as produced by Kokoro per its internal chunking.

    `audio`: float32 in [-1, 1], rank-1 mono. Already moved host-side
             (Kokoro produces torch tensors; `synthesize_iter` does the one
             `.cpu().numpy()` per segment at this boundary).

    `sample_rate`: always `KOKORO_SAMPLE_RATE` today. Stored per-segment so
                   a future TTS engine could vary rate per segment without
                   breaking this interface.

    `t0`: offset of this segment from the start of the whole audio, in seconds.
          Useful for a streaming consumer that wants to know "where am I in
          the timeline?" without re-accumulating from segment durations.

    `word_metadata`: per-word timings for this segment. Timings are already
                     absolute (i.e. `t0` has been added to the raw per-segment
                     values Kokoro emits). `None` when `get_metadata=False`.
    """
    audio: np.ndarray
    sample_rate: int
    t0: float
    word_metadata: Optional[list[WordTiming]]


@dataclass
class TTSResult:
    """All segments concatenated into a single audio array + flat metadata list.

    `audio`: float32 in [-1, 1], rank-1 mono.

    `sample_rate`: Kokoro's 24 kHz.

    `duration`: seconds, equal to `len(audio) / sample_rate`.

    `word_metadata`: flat list of `WordTiming`, ordered by start time.
                     Timings are absolute. `None` when `get_metadata=False`.
    """
    audio: np.ndarray
    sample_rate: int
    duration: float
    word_metadata: Optional[list[WordTiming]]


@dataclass
class TTSPipeline:
    """Loaded Kokoro pipeline + the on-disk location of its voice files.

    `kpipeline`: the live `kokoro.KPipeline` instance.

    `modelsdir`: filesystem path where the Kokoro model repo is installed.
                 Needed by `get_voices` — Kokoro itself doesn't expose a voice
                 enumeration API, so we scan `.pt` files under this directory
                 (matching Kokoro-FastAPI's approach).

    `lang_code`: e.g. `"a"` for American English, `"b"` for British.
                 See `load_tts_pipeline` for the full list.

    `sample_rate`: mirror of `KOKORO_SAMPLE_RATE`, for API symmetry with `STTModel`.
    """
    kpipeline: kokoro.KPipeline
    modelsdir: str
    lang_code: str
    sample_rate: int


# Cache, keyed by (repo_id, device_string, lang_code). Same pattern as `nlptools._spacy_pipelines`.
_tts_pipelines: dict[tuple[str, str, str], TTSPipeline] = {}


def load_tts_pipeline(repo_id: str,
                      device_string: str,
                      lang_code: str = "a") -> TTSPipeline:
    """Load (and cache) a Kokoro pipeline.

    `repo_id`: HuggingFace repo, e.g. `"hexgrad/Kokoro-82M"`. Auto-downloaded
               via `maybe_install_models` if not present. Downloading the full
               repo is required even for in-process use, because `get_voices`
               reads the voice files from disk (Kokoro has no voice-listing API).

    `device_string`: e.g. `"cpu"`, `"cuda:0"`. Forwarded to Kokoro.

    `lang_code`: selects the phonemizer. From Kokoro's README:

        🇺🇸 'a' → American English, 🇬🇧 'b' → British English
        🇪🇸 'e' → Spanish,           🇫🇷 'f' → French
        🇮🇳 'h' → Hindi,             🇮🇹 'i' → Italian
        🇯🇵 'j' → Japanese           (pip install misaki[ja])
        🇧🇷 'p' → Brazilian Portuguese
        🇨🇳 'z' → Mandarin Chinese   (pip install misaki[zh])

        Word-level metadata (`get_metadata=True` in `synthesize`) currently
        supports English only (`"a"` or `"b"`).

    Repeat calls with the same `(repo_id, device_string, lang_code)` return
    the cached pipeline; the underlying model is loaded at most once per process.
    """
    cache_key = (repo_id, device_string, lang_code)
    if (cached := _tts_pipelines.get(cache_key)) is not None:
        logger.info(f"load_tts_pipeline: returning cached pipeline for {cache_key}")
        return cached

    logger.info(f"load_tts_pipeline: ensuring Kokoro models are installed at '{repo_id}'.")
    modelsdir = maybe_install_models(repo_id)

    logger.info(f"load_tts_pipeline: loading on '{device_string}', lang_code='{lang_code}'.")
    kpipeline = kokoro.KPipeline(lang_code=lang_code, device=device_string, repo_id=repo_id)

    pipeline = TTSPipeline(kpipeline=kpipeline,
                           modelsdir=modelsdir,
                           lang_code=lang_code,
                           sample_rate=KOKORO_SAMPLE_RATE)

    _tts_pipelines[cache_key] = pipeline
    return pipeline


def get_voices(pipeline: TTSPipeline) -> list[str]:
    """List the voices installed on disk for this pipeline, sorted alphabetically.

    Scans `pipeline.modelsdir` for `.pt` files — Kokoro ships one voice per `.pt`.
    The scan is a directory walk over ~30 small files; cheap enough to re-run,
    no internal caching.
    """
    voices = []
    for _root, _dirs, files in os.walk(pipeline.modelsdir, topdown=True):
        for filename in files:
            if filename.endswith(".pt"):
                voices.append(filename[:-3])  # drop the ".pt"
    return sorted(voices)


def _audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Seconds of audio in a rank-1 array."""
    return len(audio) / sample_rate


def _is_word(s: str) -> bool:
    """True if `s` is a lipsync-meaningful token (not an incidental single-char glyph)."""
    # Multi-char strings pass through unconditionally. Single-char: alphanumeric or a common
    # punctuation mark (used to close the mouth between phrases).
    return len(s) > 1 or s.isalnum() or s in ",.!?:;"


def clean_timestamps(timings: list[WordTiming], for_lipsync: bool = True) -> list[WordTiming]:
    """Filter raw Kokoro word metadata.

    Two filters compose this:

    - **Always**: drop consecutive entries with the same `start_time`. Originally
      seen with Kokoro-FastAPI; not yet confirmed with the in-process Kokoro path
      but kept as a precaution — cost is one comparison per entry. Not lipsync-
      specific; the duplicate timestamps are engine-level noise regardless of
      downstream use.

    - **Only when `for_lipsync=True`**: drop incidental single-char tokens that
      aren't letters, digits, or common end-of-phrase punctuation (``,.!?:;``).
      These tokens tend to be tokenization artifacts (stray apostrophes, dashes,
      whitespace markers) and confuse the lipsync driver if they slip through.
      For captioning / transcript display, these tokens may be relevant —
      disable the filter there.

    The filter preserves the `WordTiming` instances themselves (no copying) — the
    returned list is a subset of the input list.
    """
    out = []
    last_start_time = None
    for timing in timings:
        if timing.start_time == last_start_time:
            continue
        if for_lipsync and not _is_word(timing.word):
            continue
        out.append(timing)
        last_start_time = timing.start_time
    return out


def synthesize_iter(pipeline: TTSPipeline,
                    voice: str,
                    text: str,
                    speed: float = 1.0,
                    get_metadata: bool = True) -> Iterator[TTSSegment]:
    """Yield one `TTSSegment` per Kokoro output segment, in order.

    Timestamps in `segment.word_metadata` are **absolute** (from start of whole audio),
    not segment-relative as Kokoro natively emits them — `t0` is accumulated across
    segments inside this generator so consumers of either API layer see the same
    convention.

    `voice`: one of `get_voices(pipeline)`. Leading/trailing whitespace is stripped
             (SillyTavern compatibility: ST sends whitespace around comma-separated
             voice-list entries).

    `speed`: 1.0 = normal. Practical range ~0.8–1.2. Below that Kokoro starts sounding
             like a broken record player; above, it skips phonemes.

    `get_metadata`: if `True`, each segment's `word_metadata` is a populated list.
                    If `False`, each segment's `word_metadata` is `None`.
    """
    # SillyTavern compatibility
    voice = voice.strip()

    voices = get_voices(pipeline)
    if voice not in voices:
        raise ValueError(f"synthesize_iter: unknown voice '{voice}'; installed voices: {voices}")

    kpipeline = pipeline.kpipeline
    _unused, tokens = kpipeline.g2p(text)

    # Kokoro emits per-segment-relative timestamps; we rewrite to absolute here.
    t0 = 0.0
    for segment_num, result in enumerate(kpipeline.generate_from_tokens(tokens=tokens,
                                                                        voice=voice,
                                                                        speed=speed),
                                         start=1):
        logger.info(f"synthesize_iter: processing segment {segment_num}")

        # float32 [-1, 1], mono, on CPU. The one .cpu().numpy() per segment happens here —
        # this is the engine→common-layer boundary for device locality.
        audio_numpy = result.audio.cpu().numpy().astype(np.float32, copy=False)

        word_metadata: Optional[list[WordTiming]]
        if get_metadata:
            if not result.tokens:
                raise RuntimeError("synthesize_iter: no tokens in segment result, cannot build word metadata.")
            word_metadata = []
            for token in result.tokens:
                if not all(hasattr(token, field) for field in ("text", "start_ts", "end_ts", "phonemes")):
                    raise RuntimeError(f"synthesize_iter: token missing mandatory field (text, start_ts, end_ts, phonemes). Got: {token}")
                word_metadata.append(WordTiming(word=token.text,
                                                phonemes=token.phonemes,
                                                start_time=(t0 + token.start_ts) if token.start_ts is not None else None,
                                                end_time=(t0 + token.end_ts) if token.end_ts is not None else None))
        else:
            word_metadata = None

        yield TTSSegment(audio=audio_numpy,
                         sample_rate=pipeline.sample_rate,
                         t0=t0,
                         word_metadata=word_metadata)

        t0 += _audio_duration(audio_numpy, pipeline.sample_rate)


def synthesize(pipeline: TTSPipeline,
               voice: str,
               text: str,
               speed: float = 1.0,
               get_metadata: bool = True) -> TTSResult:
    """Synthesize `text` to a single concatenated `TTSResult`.

    Thin wrapper over `synthesize_iter`: collect segments, flatten audio,
    flatten metadata, compute duration. Use this when you want one blob
    of audio + one flat metadata list (the current HTTP server path).

    For future streaming consumers, prefer `synthesize_iter`.
    """
    segments = list(synthesize_iter(pipeline, voice, text, speed=speed, get_metadata=get_metadata))

    if not segments:
        # Kokoro yielded nothing — extremely short / empty input, or an internal bug.
        # Return a valid empty result rather than raising.
        return TTSResult(audio=np.zeros(0, dtype=np.float32),
                         sample_rate=pipeline.sample_rate,
                         duration=0.0,
                         word_metadata=[] if get_metadata else None)

    audio = np.concatenate([seg.audio for seg in segments])
    sample_rate = segments[0].sample_rate
    duration = _audio_duration(audio, sample_rate)

    if get_metadata:
        word_metadata: Optional[list[WordTiming]] = []
        for seg in segments:
            # `seg.word_metadata` is guaranteed non-None when get_metadata=True
            # (synthesize_iter raises otherwise), but mypy doesn't know that.
            if seg.word_metadata is not None:
                word_metadata.extend(seg.word_metadata)
    else:
        word_metadata = None

    return TTSResult(audio=audio,
                     sample_rate=sample_rate,
                     duration=duration,
                     word_metadata=word_metadata)
