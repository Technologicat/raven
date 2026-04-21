"""Engine-agnostic data types for the speech subsystem.

These dataclasses describe the shapes that flow through the whole speech stack —
common-layer synthesis output, HTTP wire representation, and client-side cache.
They carry no dependency on any specific TTS engine; the engine handle lives in
`raven.common.audio.speech.tts.TTSPipeline`.

Keeping these here (rather than next to the engine wrapper) lets consumers that
only deal with timing data — e.g. `raven.common.audio.speech.lipsync` — import
the shapes without dragging in the engine dependencies (Kokoro, huggingface_hub,
PyAV).
"""

__all__ = ["WordTiming",
           "TTSSegment",
           "TTSResult",
           "EncodedTTSResult"]

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WordTiming:
    """One word (or word-like token) in the synthesized speech, with absolute timing.

    `word`, `phonemes`: raw Unicode strings. Any transport that needs ASCII-only headers
                        (e.g. HTTP) URL-encodes this data at its own boundary.

    `start_time`, `end_time`: seconds from start of the whole audio (absolute, not segment-relative).
                              May be `None` if Kokoro didn't produce a timestamp for this token
                              (conditions under which this happens are not documented upstream;
                              handle defensively).
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

    `sample_rate`: always `SAMPLE_RATE` today. Stored per-segment so
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
class EncodedTTSResult:
    """A `TTSResult` where the audio is encoded (to a file format) instead of raw.

    Produced by the HTTP wire path (`raven.client.tts.tts_prepare`) and by any
    future local-mode precompute that wants to cache encoded bytes rather than
    numpy arrays. Consumed by audio players (pygame mixer, etc.) and by wire
    transports that forward the bytes as-is.

    `audio_bytes`: encoded audio file contents.
    `audio_format`: encoder format — `"mp3"`, `"wav"`, `"flac"`, `"opus"`, `"aac"`.
    `sample_rate`: the sample rate of the encoded audio, in Hz. Preserved across
                   all supported formats (encoders don't resample).
    `duration`: seconds of speech audio *before* encoding. Some codecs pad the
                tail to their frame size (on the time axis), so the decoded file
                may be slightly longer.
    `word_metadata`: flat list of `WordTiming`, ordered by start time; timings
                     are absolute (relative to start of whole audio). `None`
                     when `get_metadata=False`.
    """
    audio_bytes: bytes
    audio_format: str
    sample_rate: int
    duration: float
    word_metadata: Optional[list[WordTiming]]
