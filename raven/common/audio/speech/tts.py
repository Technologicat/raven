"""Text-to-speech engine (Kokoro), as an in-process library.

Two-layer API:

- `synthesize_iter` yields one `TTSSegment` per Kokoro segment, with
  timestamps already rewritten to absolute (from start of whole audio).
  A streaming consumer can forward segments as they come.

- `synthesize` is a thin concatenating wrapper: collect all segments,
  flatten the audio, concatenate the metadata list, return one
  `TTSResult`. The single-response server path uses this.

Audio is float32 in [-1, 1] throughout, at 24 kHz, mono — Kokoro's native output.

Transport concerns (s16 cast, URL-encoding raw phonemes for ASCII-only
HTTP headers, audio-format encoding, Flask Response construction) live
in the server wrapper at `raven.server.modules.tts`. Client-side
remote/local dispatch lives in `raven.client.mayberemote.TTS`.
"""

__all__ = ["SAMPLE_RATE",
           "TTSPipeline",
           "load_tts_pipeline",
           "get_voices",
           "synthesize_iter",
           "synthesize",
           "clean_timestamps",
           "dipthong_vowel_to_ipa",
           "expand_phoneme_diphthongs",
           "finalize_metadata",
           "prepare",
           "prepare_cached",
           "prepare_encoded_cached",
           "encode",
           "decode"]

import functools
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import io
import os
from dataclasses import dataclass, replace
from typing import Iterator, Optional

import numpy as np

import kokoro

from ...hfutil import maybe_install_models
from .. import codec as audio_codec
from .datatypes import WordTiming, TTSSegment, TTSResult, EncodedTTSResult


# Native output sample rate of the TTS engine (currently Kokoro, which runs at 24 kHz —
# documented in Kokoro's own README). Exposed as a module constant so callers can reference
# it by name rather than reading it off `TTSPipeline.sample_rate` when they don't have a
# pipeline handy. If the engine is ever swapped, update this value and any fixed-rate
# assumptions that follow.
SAMPLE_RATE = 24000


# Misaki's (Kokoro's phonemizer) shorthand for diphthongs → canonical IPA.
# Kokoro emits the single-letter form; most downstream consumers (lipsync, captioning)
# want the two-character IPA form. `expand_phoneme_diphthongs` does the substitution.
#
# Phoneme characters and per-phoneme comments come from Misaki docs:
#   https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md
dipthong_vowel_to_ipa = {
    "A": "eɪ",  # The "eh" vowel sound, like hey => hˈA.
    "I": "aɪ",  # The "eye" vowel sound, like high => hˈI.
    "W": "aʊ",  # The "ow" vowel sound, like how => hˌW.
    "Y": "ɔɪ",  # The "oy" vowel sound, like soy => sˈY.
    # 🇺🇸 American-only
    "O": "oʊ",  # American "oh" vowel sound.
    # 🇬🇧 British-only
    "Q": "əʊ",  # British "oh" vowel sound.
}


@dataclass(eq=False)
class TTSPipeline:
    """Loaded Kokoro pipeline + the on-disk location of its voice files.

    `model_name`: the HuggingFace repo identifier this pipeline was loaded
                  from (e.g. `"hexgrad/Kokoro-82M"`). Available as metadata
                  for diagnostics and info endpoints.

    `kpipeline`: the live `kokoro.KPipeline` instance.

    `modelsdir`: filesystem path where the Kokoro model repo is installed.
                 Needed by `get_voices` — Kokoro itself doesn't expose a voice
                 enumeration API, so we scan `.pt` files under this directory
                 (matching Kokoro-FastAPI's approach).

    `lang_code`: e.g. `"a"` for American English, `"b"` for British.
                 See `load_tts_pipeline` for the full list.

    `sample_rate`: mirror of `SAMPLE_RATE`, for API symmetry with `STTModel`.

    `eq=False` on the dataclass: identity-based hashing, so a `TTSPipeline`
    instance is usable as an `lru_cache` key (see `prepare_cached`). Safe
    because pipelines are singletons per `(model_name, device_string, lang_code)`
    via `_tts_pipelines`.
    """
    model_name: str
    kpipeline: kokoro.KPipeline
    modelsdir: str
    lang_code: str
    sample_rate: int


# Cache, keyed by (model_name, device_string, lang_code). Same pattern as `nlptools._spacy_pipelines`.
_tts_pipelines: dict[tuple[str, str, str], TTSPipeline] = {}


def load_tts_pipeline(model_name: str,
                      device_string: str,
                      lang_code: str = "a") -> TTSPipeline:
    """Load (and cache) a Kokoro pipeline.

    `model_name`: HuggingFace repo identifier, e.g. `"hexgrad/Kokoro-82M"`.
                  Auto-downloaded via `maybe_install_models` if not present.
                  Downloading the full repo is required even for in-process
                  use, because `get_voices` reads the voice files from disk
                  (Kokoro has no voice-listing API).

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

    **System-package prerequisite**: Kokoro's phonemizer (Misaki) falls back to
    `espeak-ng` for out-of-dictionary words. If Misaki fails to initialise or
    silently mis-pronounces uncommon text, the OS-level `espeak-ng` is probably
    missing.

    - Linux: `sudo apt install espeak-ng`
    - Windows: see installation notes at https://github.com/hexgrad/kokoro

    Repeat calls with the same `(model_name, device_string, lang_code)` return
    the cached pipeline; the underlying model is loaded at most once per process.
    """
    cache_key = (model_name, device_string, lang_code)
    if (cached := _tts_pipelines.get(cache_key)) is not None:
        logger.info(f"load_tts_pipeline: returning cached pipeline for {cache_key}")
        return cached

    logger.info(f"load_tts_pipeline: ensuring Kokoro models are installed at '{model_name}'.")
    modelsdir = maybe_install_models(model_name)

    logger.info(f"load_tts_pipeline: loading on '{device_string}', lang_code='{lang_code}'.")
    kpipeline = kokoro.KPipeline(lang_code=lang_code, device=device_string, repo_id=model_name)  # Kokoro's parameter is still `repo_id`

    pipeline = TTSPipeline(model_name=model_name,
                           kpipeline=kpipeline,
                           modelsdir=modelsdir,
                           lang_code=lang_code,
                           sample_rate=SAMPLE_RATE)

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
    """True if `s` is a lipsync-meaningful token (not an incidental single-char glyph).

    Multi-char string -> `True` unconditionally.

    Single-char string -> `True` if alphanumeric, or a common punctuation mark
                          (the latter are used in lipsync, to close the avatar's
                           mouth between phrases).
    """
    return len(s) > 1 or s.isalnum() or s in ",.!?:;"


def clean_timestamps(timings: list[WordTiming], for_lipsync: bool = True) -> list[WordTiming]:
    """Filter raw Kokoro word metadata.

    Consists of two filters:

    - **Always**: drop consecutive entries with the same `start_time`. Originally
      seen with Kokoro-FastAPI; not yet confirmed with the in-process Kokoro path
      but kept as a precaution — cost is one comparison per entry.

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


def expand_phoneme_diphthongs(timings: list[WordTiming]) -> list[WordTiming]:
    """Expand Misaki's single-letter diphthong shorthand to canonical IPA.

    Misaki (Kokoro's phonemizer) emits diphthongs as single capital letters:
    `A`, `I`, `W`, `Y`, `O`, `Q` (see `dipthong_vowel_to_ipa` for the mapping).
    Most downstream consumers want two-character IPA — lipsync phoneme lookup,
    captioning, display.

    Returns new `WordTiming` instances with expanded `.phonemes` (does not mutate
    the input). Idempotent: the IPA expansions contain none of the source letters,
    so running twice produces the same result as running once.
    """
    def _expand(phonemes: str) -> str:
        for shorthand, ipa in dipthong_vowel_to_ipa.items():
            phonemes = phonemes.replace(shorthand, ipa)
        return phonemes
    return [replace(t, phonemes=_expand(t.phonemes)) for t in timings]


def synthesize_iter(pipeline: TTSPipeline,
                    voice: str,
                    text: str,
                    speed: float = 1.0,
                    get_metadata: bool = True) -> Iterator[TTSSegment]:
    """Synthesize `text` as an iterable of `TTSSegment` objects.

    Yields one `TTSSegment` per Kokoro output segment, in order.

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

    For streaming consumers, prefer `synthesize_iter`.
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


def finalize_metadata(timings: list[WordTiming]) -> list[WordTiming]:
    """Apply the standard post-processing pipeline to raw Kokoro word metadata.

    Composes `clean_timestamps(for_lipsync=True)` and `expand_phoneme_diphthongs`
    into a single call. Single source of truth for "what does lipsync-ready TTS
    metadata look like?" — both the in-process `prepare` and the HTTP wire path
    in `raven.client.tts.tts_prepare` use this helper, so the sequence can't
    drift between paths.

    Returns a new list of new `WordTiming` instances (`clean_timestamps` returns
    a filtered subset, `expand_phoneme_diphthongs` then returns copies with
    expanded phonemes).
    """
    timings = clean_timestamps(timings, for_lipsync=True)
    timings = expand_phoneme_diphthongs(timings)
    return timings


def prepare(pipeline: TTSPipeline,
            voice: str,
            text: str,
            speed: float = 1.0,
            get_metadata: bool = True) -> TTSResult:
    """Synthesize `text`, with `word_metadata` post-processed for lipsync use.

    Thin composition over `synthesize` + `finalize_metadata`. Parallel to the
    single common-layer call that every other `MaybeRemoteService` delegates to
    in local mode.

    Use this when you want a ready-to-consume `TTSResult` and don't want to
    remember the post-processing sequence. Use bare `synthesize` when you want
    access to the raw Kokoro output (e.g. inspecting pre-cleanup timestamps).
    """
    result = synthesize(pipeline, voice, text, speed=speed, get_metadata=get_metadata)
    if result.word_metadata is not None:
        result = replace(result, word_metadata=finalize_metadata(result.word_metadata))
    return result


@functools.lru_cache(maxsize=128)
def prepare_cached(pipeline: TTSPipeline,
                   voice: str,
                   text: str,
                   speed: float = 1.0,
                   get_metadata: bool = True) -> TTSResult:
    """Memoized `prepare`. Same signature, cached across calls.

    Kokoro's vocoder is stochastic, so re-synthesizing the same input
    produces slightly different audio each time. The cached version returns
    the same audio for the same arguments as long as it is in the cache.

    Keyed by `(pipeline, voice, text, speed, get_metadata)`. Pipeline identity
    is the cache key (see `TTSPipeline` docstring). Cache size 128 matches
    `raven.client.tts.tts_prepare_cached`.
    """
    return prepare(pipeline, voice, text, speed=speed, get_metadata=get_metadata)


@functools.lru_cache(maxsize=128)
def prepare_encoded_cached(pipeline: TTSPipeline,
                           voice: str,
                           text: str,
                           speed: float,
                           get_metadata: bool,
                           format: str) -> EncodedTTSResult:
    """Cached `prepare` in encoded form. Other-shape companion to `prepare_cached`.

    Synthesizes once via `prepare_cached`, then encodes on top. Two-level cache:
    asking for the same text in two formats synthesizes once and encodes twice;
    asking for the same text in the same format hits both levels.

    `format`: any PyAV-supported encoding, e.g. `"mp3"`, `"flac"`. No default —
              callers who ask for encoded audio always care which encoding.
    """
    return encode(prepare_cached(pipeline, voice, text, speed=speed, get_metadata=get_metadata), format)


def encode(result: TTSResult, format: str) -> EncodedTTSResult:
    """Convert a `TTSResult` to an `EncodedTTSResult` by encoding the audio.

    Companion to `decode`: takes raw float audio and produces wire/playback-ready
    encoded bytes. Handles the float → s16 transport cast internally.

    `format`: any PyAV-supported audio format, e.g. `"mp3"`, `"flac"`, `"wav"`.
              Sample rate, duration, and `word_metadata` pass through unchanged.

    Empty `audio_bytes` → empty `EncodedTTSResult` (audio_bytes=b"").
    """
    if not len(result.audio):
        return EncodedTTSResult(audio_bytes=b"",
                                audio_format=format,
                                sample_rate=result.sample_rate,
                                duration=result.duration,
                                word_metadata=result.word_metadata)
    audio_s16 = np.array(result.audio * 32767.0, dtype=np.int16)
    audio_bytes = audio_codec.encode(audio_data=audio_s16,
                                     format=format,
                                     sample_rate=result.sample_rate)
    return EncodedTTSResult(audio_bytes=audio_bytes,
                            audio_format=format,
                            sample_rate=result.sample_rate,
                            duration=result.duration,
                            word_metadata=result.word_metadata)


def decode(encoded: EncodedTTSResult) -> TTSResult:
    """Convert an `EncodedTTSResult` to a `TTSResult` by decoding the audio.

    Companion to `EncodedTTSResult`: takes wire/playback-encoded audio (MP3 etc.)
    and decodes it to the raw float format expected for in-process consumption.
    Sample rate, duration, and word_metadata pass through unchanged.

    Empty `audio_bytes` → empty `TTSResult` (zero-length float array). That
    supports the "cancelled / no audio produced" case from `tts_prepare` without
    a special code path.
    """
    if not encoded.audio_bytes:
        return TTSResult(audio=np.zeros(0, dtype=np.float32),
                         sample_rate=encoded.sample_rate,
                         duration=encoded.duration,
                         word_metadata=encoded.word_metadata)

    _metadata, audio = audio_codec.decode(io.BytesIO(encoded.audio_bytes),
                                          target_sample_format="fltp",
                                          target_sample_rate=encoded.sample_rate,
                                          target_layout="mono")
    audio = audio.astype(np.float32, copy=False)
    return TTSResult(audio=audio,
                     sample_rate=encoded.sample_rate,
                     duration=encoded.duration,
                     word_metadata=encoded.word_metadata)
