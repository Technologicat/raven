"""TTS (text-to-speech, speech synthesizer) Python bindings of Raven's web API.

This module coordinates lipsync between the TTS and the avatar, and implements the actual client-side audio output.
"""

__all__ = ["tts_list_voices",
           "tts_warmup",
           "tts_prepare",
           "tts_prepare_cached",
           "tts_prepare_decoded_cached",
           "tts_speak",
           "tts_speak_lipsynced",
           "tts_stop",
           "tts_speaking"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import copy
import functools
import io
import json
import requests
import time
import traceback
from typing import Callable, List, Optional
import urllib.parse

from unpythonic import timer, sym
from unpythonic.env import env as envcls

from . import api  # for calling the avatar_* functions during lipsync
from . import util

from ..common.audio.speech import lipsync as speech_lipsync
from ..common.audio.speech import tts as speech_tts

# --------------------------------------------------------------------------------
# Data

# Phoneme to avatar animator morph table.
#
# Phoneme characters and individual-phoneme comments come from Misaki docs
# (Kokoro TTS uses the Misaki engine for phonemization):
#   https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md
#
# For the animator morph names, see `raven.server.modules.avatarutil`.
#
# The lipsync phoneme mapping is already acceptable. Possible future TODO:
#   - For more fine-grained control, we could use the morph value channel too, not just always set the morph to 1.0. OTOH the pose animation smooths this out.
#   - Could also combine morphs for a single phoneme if needed (use a list).
#
# The format is:
#   `phoneme: value`, where `value` is an animator morph name, or one of the special commands:
#     "!close_mouth" - as it says on the tin
#     "!keep" - keep previous mouth position
#     "!maybe_close_mouth" - close mouth only if the pause is at least half a second, else act like "!keep".
phoneme_to_morph = {
    # IPA Consonants
    "b": "!close_mouth",
    "d": "mouth_delta",
    "f": "mouth_eee_index",
    "h": "mouth_delta",
    "j": "mouth_aaa_index",  # As in yes => jˈɛs.
    "k": "mouth_aaa_index",
    "l": "mouth_eee_index",
    "m": "!close_mouth",
    "n": "mouth_eee_index",
    "p": "!close_mouth",
    "s": "mouth_eee_index",
    "t": "mouth_eee_index",
    "T": "mouth_eee_index",  # getting this too from Misaki
    "v": "mouth_iii_index",  # getting this too from Misaki
    "w": "mouth_ooo_index",
    "z": "mouth_eee_index",
    "ɡ": "mouth_aaa_index",  # Hard "g" sound, like get => ɡɛt. Visually looks like the lowercase letter g, but its actually U+0261.
    "ŋ": "mouth_aaa_index",  # The "ng" sound, like sung => sˈʌŋ.
    "ɹ": "mouth_aaa_index",  # Upside-down r is just an "r" sound, like red => ɹˈɛd.
    "ʃ": "mouth_ooo_index",  # The "sh" sound, like shin => ʃˈɪn.
    "ʒ": "mouth_eee_index",  # The "zh" sound, like Asia => ˈAʒə.
    "ð": "mouth_aaa_index",  # Soft "th" sound, like than => ðən.
    "θ": "mouth_aaa_index",  # Hard "th" sound, like thin => θˈɪn.
    # Consonant Clusters
    "ʤ": "mouth_ooo_index",  # A "j" or "dg" sound, merges dʒ, like jump => ʤˈʌmp or lunge => lˈʌnʤ.
    "ʧ": "mouth_ooo_index",  # The "ch" sound, merges tʃ, like chump => ʧˈʌmp or lunch => lˈʌnʧ.
    # IPA Vowels
    "ə": "mouth_delta",  # The schwa is a common, unstressed vowel sound, like a 🍌 => ə 🍌.
    "i": "mouth_iii_index",  # As in easy => ˈizi.
    "u": "mouth_uuu_index",  # As in flu => flˈu.
    "ɑ": "mouth_aaa_index",  # As in spa => spˈɑ.
    "ɔ": "mouth_ooo_index",  # As in all => ˈɔl.
    "ɛ": "mouth_eee_index",  # As in hair => hˈɛɹ or bed => bˈɛd. Possibly dubious, because those vowel sounds do not sound similar to my ear.
    "ɜ": "mouth_delta",  # As in her => hɜɹ. Easy to confuse with ɛ above.
    "ɪ": "mouth_iii_index",  # As in brick => bɹˈɪk.
    "ʊ": "mouth_uuu_index",  # As in wood => wˈʊd.
    "ʌ": "mouth_aaa_index",  # As in sun => sˈʌn.
    # Custom Vowel (Misaki)
    "ᵊ": "mouth_delta",  # Small schwa, muted version of ə, like pixel => pˈɪksᵊl. I made this one up, so I'm not entirely sure if it's correct.
    # 🇺🇸 American-only
    "æ": "mouth_delta",  # The vowel sound at the start of ash => ˈæʃ.
    "ᵻ": "mouth_delta",  # A sound somewhere in between ə and ɪ, often used in certain -s suffixes like boxes => bˈɑksᵻz.
    "ɾ": "mouth_eee_index",  # A sound somewhere in between t and d, like butter => bˈʌɾəɹ.
    # 🇬🇧 British-only
    "a": "mouth_aaa_index",  # The vowel sound at the start of ash => ˈaʃ.
    "ɒ": "mouth_ooo_index",  # The sound at the start of on => ˌɒn. Easy to confuse with ɑ, which is a shared phoneme.
    # Other
    "ː": "!keep",  # Vowel extender, visually looks similar to a colon. Possibly dubious, because Americans extend vowels too, but the gold US dictionary somehow lacks these. Often used by the Brits instead of ɹ: Americans say or => ɔɹ, but Brits say or => ɔː.
    # Stress Marks
    "ˈ": "!keep",  # Primary stress, visually looks similar to an apostrophe (but is U+02C8).
    "ˌ": "!keep",  # Secondary stress (not a comma, but U+02CC).
    # punctuation
    ",": "!keep",  # comma, U+2C
    ";": "!keep",  # semicolon, U+3B
    ":": "!keep",  # colon, U+3A
    ".": "!maybe_close_mouth",  # period, U+2E
    "!": "!maybe_close_mouth",  # exclamation mark, U+21
    "?": "!maybe_close_mouth",  # question mark, U+3F
    # from dipthong expansion
    "e": "mouth_eee_index",
    "o": "mouth_ooo_index",
}

# --------------------------------------------------------------------------------
# API

def tts_list_voices() -> List[str]:
    """Return a list of voice names supported by the TTS endpoint (if the endpoint is available).

    Return format is::

        [voice0, ...]
    """
    if not util.api_initialized:
        raise RuntimeError("tts_list_voices: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    response = requests.get(f"{util.api_config.raven_server_url}/api/tts/list_voices", headers=headers)
    util.yell_on_error(response)
    output_data = response.json()
    return output_data["voices"]

def tts_warmup(voice: str) -> None:
    """Warm up the TTS, before you need it for the first time.

    The first invocation of the TTS may take some extra time as it loads the voice;
    this function does that explicitly.
    """
    # Skip the LRU cache, as it would defeat the whole point of this function.
    logger.info(f"tts_warmup: Warming up TTS for voice '{voice}'.")
    tts_prepare(text="The quick brown fox jumps over the lazy dog.",
                voice=voice,
                speed=1.0,
                get_metadata=True)  # not sure if the phonemizer needs warmup, but let's do it anyway
    logger.info(f"tts_warmup: Warmup for voice '{voice}' done.")

def _empty_encoded_result(get_metadata: bool, format: str = "flac") -> speech_tts.EncodedTTSResult:
    """Zero-audio `EncodedTTSResult`, returned by `tts_prepare` on blank / no-phoneme input.

    Callers check `not prep.audio_bytes` to detect the cancelled case; the structure
    is otherwise uniform with a successful synthesis, so decode / consume paths don't
    need special-case handling.
    """
    return speech_tts.EncodedTTSResult(audio_bytes=b"",
                                       audio_format=format,
                                       sample_rate=speech_tts.SAMPLE_RATE,
                                       duration=0.0,
                                       word_metadata=[] if get_metadata else None)

def tts_prepare(text: str,
                voice: str,
                speed: float = 1.0,
                get_metadata: bool = True,
                format: str = "flac") -> speech_tts.EncodedTTSResult:
    """Using the speech synthesizer, precompute TTS speech audio for `text` using `voice`.

    Explicit remote mode. Not cached — each call round-trips to the server.
    For the cached variant (nearly always what callers want in app code, so
    that re-rendering the same sentence is free), use `tts_prepare_cached`.

    Optionally, compute also word-level timestamps and phoneme data.

    To get the list of available voices, call `tts_list_voices`.

    `speed`: For each voice, 1.0 is the default speed the voice is designed to speak at.
             Raising this too high may cause skipped words.

    `get_metadata`: If `True`, the returned `EncodedTTSResult` has populated
                    `word_metadata` (lipsync-ready — `clean_timestamps` + diphthong
                    expansion already applied). If `False`, `word_metadata` is `None`.

    `format`: audio-file encoding to request from the server. Any format supported
              by the server's encoder (see `raven.common.audio.codec.encode`) —
              e.g. `"mp3"`, `"flac"`. Default `"flac"` — lossless, so the
              samples round-trip bit-identical with the server's synthesis.
              The chosen format is also stored in the returned
              `EncodedTTSResult.audio_format`.

    Always returns an `EncodedTTSResult`. On blank input, or if `get_metadata=True`
    and the TTS produced no phoneme data, the returned value is a zero-audio result
    (`audio_bytes == b""`, `duration == 0.0`) — callers detect this with
    `not prep.audio_bytes`.
    """
    if not util.api_initialized:
        raise RuntimeError("tts_prepare: The `raven.client.api` module must be initialized before using the API.")
    if not text.strip():
        logger.info("tts_prepare: Ignoring blank `text`. Cancelled.")
        return _empty_encoded_result(get_metadata, format=format)
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"

    timings: Optional[List[speech_tts.WordTiming]] = None

    # Get audio, and if getting metadata, also the word-level timestamps
    with timer() as tim:
        logger.info(f"tts_prepare: getting TTS audio ({format}) with word-level timestamps and phonemes from Raven-server")
        data = {"voice": voice,
                "text": text,
                "format": format,
                "speed": speed,
                "stream": True,
                "get_metadata": get_metadata}
        stream_response = requests.post(f"{util.api_config.raven_server_url}/api/tts/speak", headers=headers, json=data, stream=True)
        util.yell_on_error(stream_response)

        # Parse word-level timestamps at the wire boundary: URL-decoded JSON dicts → WordTiming dataclasses.
        # From here on, the in-process representation is WordTiming; dicts only live on the wire.
        if get_metadata:
            raw_timestamps = json.loads(stream_response.headers["x-word-timestamps"])
            timings = [speech_tts.WordTiming(word=urllib.parse.unquote(ts["word"]),
                                             phonemes=urllib.parse.unquote(ts["phonemes"]),
                                             start_time=ts.get("start_time"),
                                             end_time=ts.get("end_time"))
                       for ts in raw_timestamps]

        # Stream the audio from the response body.
        # Save the audio in an in-memory buffer, thus obtaining a filelike that can be fed into the audio player.
        # This is not only a convenience. This facilitates LRU caching; as well as `stream_response.raw` can't be loaded into an audio player anyway, since it doesn't support seeking.
        it = stream_response.iter_content(chunk_size=4096)
        audio_buffer = io.BytesIO()
        try:
            while True:
                chunk = next(it)
                audio_buffer.write(chunk)
        except StopIteration:
            pass
    audio_buffer.seek(0)
    audio_bytes = audio_buffer.getvalue()

    total_audio_duration = json.loads(stream_response.headers["x-audio-duration"])
    logger.info(f"tts_prepare: got TTS audio (duration {total_audio_duration:0.6g}s) in {tim.dt:0.6g}s.")

    # Postprocess per-word phoneme data via the common-layer pipeline (dedup + lipsync filter + IPA expansion).
    if get_metadata:
        logger.info("tts_prepare: postprocessing per-word phoneme data")
        with timer() as tim:
            timings = speech_tts.finalize_metadata(timings)

            logger.info("tts_prepare: timestamped phonemes:")
            for timing in timings:
                logger.info(f"    {timing}")  # DEBUG once more, with feeling! (show timestamps, with phoneme data)

            if not timings:
                logger.info("tts_prepare: Metadata was requested, but the TTS did not generate any phonemes. Cancelled. The text was:")
                logger.info(text)
                return _empty_encoded_result(get_metadata, format=format)
        logger.info(f"tts_prepare: postprocessed per-word phoneme data in {tim.dt:0.6g}s.")

    return speech_tts.EncodedTTSResult(audio_bytes=audio_bytes,
                                       audio_format=format,
                                       sample_rate=speech_tts.SAMPLE_RATE,
                                       duration=total_audio_duration,
                                       word_metadata=timings)


@functools.lru_cache(maxsize=128)
def tts_prepare_cached(text: str,
                       voice: str,
                       speed: float = 1.0,
                       get_metadata: bool = True,
                       format: str = "flac") -> speech_tts.EncodedTTSResult:
    """Memoized `tts_prepare`. Same signature, cached across calls.

    Default choice for app code: precomputing the same sentence twice doesn't
    re-hit the server. Parallel to `raven.common.audio.speech.tts.prepare_cached`
    on the local side.

    Keyed by `(text, voice, speed, get_metadata, format)`.
    """
    return tts_prepare(text, voice, speed=speed, get_metadata=get_metadata, format=format)


@functools.lru_cache(maxsize=128)
def tts_prepare_decoded_cached(text: str,
                               voice: str,
                               speed: float = 1.0,
                               get_metadata: bool = True) -> speech_tts.TTSResult:
    """Cached remote TTS in decoded (float `TTSResult`) form. Other-shape companion to `tts_prepare_cached`.

    Fetches encoded audio via `tts_prepare_cached` (using FLAC on the wire for
    lossless round-trip), then decodes to float. Two-level cache: the FLAC
    bytes are cached by the inner call, and the decoded `TTSResult` is cached
    here on top.

    Parallel to `raven.common.audio.speech.tts.prepare_encoded_cached` on the
    local side, modulo the direction of the conversion — local synthesizes
    float natively and encodes on top; remote receives encoded natively and
    decodes on top.
    """
    return speech_tts.decode(tts_prepare_cached(text, voice, speed=speed, get_metadata=get_metadata, format="flac"))


def tts_speak(text: str,
              voice: str,
              speed: float = 1.0,
              on_audio_ready: Optional[Callable] = None,
              on_start: Optional[Callable] = None,
              on_stop: Optional[Callable] = None,
              prep: Optional[speech_tts.EncodedTTSResult] = None) -> None:
    """Using the speech synthesizer, speak `text` using `voice`.

    To get the list of available voices, call `tts_list_voices`.

    `speed`: For each voice, 1.0 is the default speed the voice is designed to speak at.
             Raising this too high may cause skipped words.

    If `on_audio_ready` is provided, call it after TTS synthesis is complete, before playback starts.
    It is expected to take one argument, `audio_data: bytes`, containing the speech audio encoded in whatever
    format `tts_prepare_cached` returned (default FLAC; read `final_prep.audio_format` if you need to know).
    Return value is ignored.

    If `on_start` is provided, call it when the TTS starts speaking. No arguments. Return value is ignored.
    If `on_stop` is provided, call it when the TTS has stopped speaking. No arguments. Return value is ignored.

    **Advanced mode**

    If `prep` is provided, ignore `voice`, `text`, and `speed`, and load preprocessed TTS audio from `prep`.
    To get a `prep`, use `tts_prepare_cached`. This allows precomputing TTS for more sentences while a previous one
    is still being spoken.

    The `on_audio_ready` event is called also in advanced mode, with precomputed audio.
    When we obtain the precomputed audio from `prep`, then `on_audio_ready` triggers.
    """
    if not util.api_initialized:
        raise RuntimeError("tts_speak: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"

    # We run this in the background
    def speak(task_env) -> None:
        if prep is None:
            logger.info(f"tts_speak.speak: instance {task_env.task_name}: getting audio")
            final_prep = tts_prepare_cached(text, voice, speed, get_metadata=False)
        else:
            logger.info(f"tts_speak.speak: instance {task_env.task_name}: using precomputed audio")
            final_prep = prep
        if not final_prep.audio_bytes:  # blank input or no-phoneme case — `tts_prepare` flags it via empty audio_bytes
            logger.info(f"tts_speak.speak: instance {task_env.task_name}: no audio produced. Cancelled.")
            return
        audio_bytes = final_prep.audio_bytes

        # Send TTS speech audio data (encoded per `final_prep.audio_format`, default FLAC) to caller if they want it
        if on_audio_ready is not None:
            on_audio_ready(audio_bytes)

        # play audio
        logger.info(f"tts_speak.speak: instance {task_env.task_name}: loading audio into mixer")
        audio_buffer = io.BytesIO()
        audio_buffer.write(audio_bytes)
        audio_buffer.seek(0)
        if util.api_config.audio_player.is_playing():
            util.api_config.audio_player.stop()
        try:
            util.api_config.audio_player.load(audio_buffer)
        except RuntimeError as exc:
            logger.error(f"tts_speak.speak: instance {task_env.task_name}: failed to load audio into mixer, reason {type(exc)}: {exc}")
            return

        logger.info(f"tts_speak.speak: instance {task_env.task_name}: starting playback")
        if on_start is not None:
            try:
                on_start()
            except Exception as exc:
                logger.error(f"tts_speak.speak: instance {task_env.task_name}: in start callback: {type(exc)}: {exc}")
                traceback.print_exc()
        util.api_config.audio_player.start()

        if on_stop is not None:
            while util.api_config.audio_player.is_playing():
                time.sleep(0.01)
                # NOTE: At this point, we don't take cancellations from `task_env.cancelled`; but the client can explicitly call `tts_stop`,
                # which stops the audio, thus exiting the loop. We are likely running on a different thread pool than the main app, anyway,
                # unless the main thread pool was passed to `raven.client.api.initialize` (implemented in `raven.client.util.initialize_api`).
            logger.info(f"tts_speak.speak: instance {task_env.task_name}: playback finished")
            try:
                on_stop()
            except Exception as exc:
                logger.error(f"tts_speak.speak: instance {task_env.task_name}: in stop callback: {type(exc)}: {exc}")
                traceback.print_exc()
        else:
            logger.info(f"tts_speak.speak: instance {task_env.task_name}: no stop callback, all done.")
    util.api_config.task_manager.submit(speak, envcls())

def tts_speak_lipsynced(instance_id: str,
                        text: str,
                        voice: str,
                        speed: float = 1.0,
                        video_offset: float = 0.0,
                        on_audio_ready: Optional[Callable] = None,
                        on_start: Optional[Callable] = None,
                        on_stop: Optional[Callable] = None,
                        prep: Optional[speech_tts.EncodedTTSResult] = None) -> None:
    """Like `tts_speak`, but with lipsync for the avatar.

    Using the speech synthesizer, speak `text` using `voice`.

    To get the list of available voices, call `tts_list_voices`.

    `instance_id`: The avatar instance ID you got from `raven.client.api.avatar_load`.
                   Which avatar instance to lipsync.

    `speed`: For each voice, 1.0 is the default speed the voice is designed to speak at.
             Raising this too high may cause skipped words.

    `video_offset`: seconds, for adjusting lipsync animation.
        - Positive values: Use if the video is early. Shifts video later with respect to the audio.
        - Negative values: Use if the video is late. Shifts video earlier with respect to the audio.

    If `on_audio_ready` is provided, call it after TTS synthesis is complete, before playback starts.
    It is expected to take one argument, `audio_data: bytes`, containing the speech audio encoded in whatever
    format `tts_prepare_cached` returned (default FLAC; read `final_prep.audio_format` if you need to know).
    Return value is ignored.

    If `on_start` is provided, call it when the TTS starts speaking. No arguments. Return value is ignored.
    If `on_stop` is provided, call it when the TTS has stopped speaking. No arguments. Return value is ignored.

    **Advanced mode**

    If `prep` is provided, ignore `voice`, `text`, and `speed`, and load preprocessed TTS audio and phonemes
    from `prep`. To get a `prep`, use `tts_prepare`. This allows precomputing TTS for more sentences while a
    previous one is still being spoken.

    The `on_audio_ready` event is called also in advanced mode, with precomputed audio.
    When we obtain the precomputed audio from `prep`, then `on_audio_ready` triggers.
    """
    if not util.api_initialized:
        raise RuntimeError("tts_speak_lipsynced: The `raven.client.api` module must be initialized before using the API.")

    def speak(task_env: envcls) -> None:
        if prep is None:
            logger.info("tts_speak_lipsynced.speak: getting audio and phonemes")
            final_prep = tts_prepare_cached(text, voice, speed, get_metadata=True)
        else:
            logger.info("tts_speak_lipsynced.speak: using precomputed audio and phonemes")
            final_prep = prep
        if not final_prep.audio_bytes:  # blank input or no-phoneme case — `tts_prepare` flags it via empty audio_bytes
            logger.info("tts_speak_lipsynced.speak: no audio produced. Cancelled.")
            return
        audio_bytes = final_prep.audio_bytes
        timestamps = final_prep.word_metadata

        # Send TTS speech audio data (encoded per `final_prep.audio_format`, default FLAC) to caller if they want it
        if on_audio_ready is not None:
            on_audio_ready(audio_bytes)

        # Transform word-level timestamps into a phoneme stream with interpolated
        # per-phoneme timings. Example input/output::
        #
        #   [WordTiming(word="Sharon", phonemes="ʃˈɛɹən", start_time=0.275, end_time=0.575),
        #    WordTiming(word="Apple",  phonemes="ˈæpᵊl",  start_time=0.575, end_time=0.875),
        #    ...]
        #
        #   → [PhonemeEvent(phoneme='ʃ', morph='mouth_ooo_index', start_time=0.275, end_time=0.325),
        #      PhonemeEvent(phoneme='ˈ', morph='!keep',           start_time=0.325, end_time=0.375),
        #      PhonemeEvent(phoneme='ɛ', morph='mouth_eee_index', start_time=0.375, end_time=0.425),
        #      PhonemeEvent(phoneme='ɹ', morph='mouth_aaa_index', start_time=0.425, end_time=0.475),
        #      PhonemeEvent(phoneme='ə', morph='mouth_delta',     start_time=0.475, end_time=0.525),
        #      PhonemeEvent(phoneme='n', morph='mouth_eee_index', start_time=0.525, end_time=0.575),
        #      PhonemeEvent(phoneme='ˈ', morph='!keep',           start_time=0.575, end_time=0.635),  # stress mark at a word boundary → "!keep"
        #      PhonemeEvent(phoneme='æ', morph='mouth_delta',     start_time=0.635, end_time=0.695),
        #      PhonemeEvent(phoneme='p', morph='!close_mouth',    start_time=0.695, end_time=0.755),  # plosive → "!close_mouth"
        #      PhonemeEvent(phoneme='ᵊ', morph='mouth_delta',     start_time=0.755, end_time=0.815),
        #      PhonemeEvent(phoneme='l', morph='mouth_eee_index', start_time=0.815, end_time=0.875),
        #      ...]
        phoneme_stream = speech_lipsync.label_phoneme_stream(speech_lipsync.build_phoneme_stream(timestamps),
                                                             phoneme_to_morph)

        try:
            # play audio
            logger.info("tts_speak_lipsynced.speak: loading audio into mixer")
            audio_buffer = io.BytesIO()
            audio_buffer.write(audio_bytes)
            audio_buffer.seek(0)
            if util.api_config.audio_player.is_playing():
                util.api_config.audio_player.stop()
            try:
                util.api_config.audio_player.load(audio_buffer)
            except RuntimeError as exc:
                logger.error(f"tts_speak_lipsynced.speak: failed to load audio into mixer, reason {type(exc)}: {exc}")
                return

            # All mouth morphs. These are what we will override while lipsyncing.
            # We use `avatar_modify_overrides` instead of `avatar_set_overrides` to control just the mouth morphs, so that other parts of the client can override other morphs or cel blends simultaneously if needed.
            mouth_morph_overrides = {
                "mouth_aaa_index": 0.0,
                "mouth_eee_index": 0.0,
                "mouth_iii_index": 0.0,
                "mouth_ooo_index": 0.0,
                "mouth_uuu_index": 0.0,
                "mouth_delta": 0.0,
            }

            def on_tick(t: float) -> sym:
                if not util.api_config.audio_player.is_playing():
                    return speech_lipsync.action_finish

                event = speech_lipsync.phoneme_at(phoneme_stream, t)

                # Pre-stream (before first phoneme): do nothing, let the avatar keep its current pose.
                # Post-stream (after last phoneme, audio still playing out trailing silence): close the mouth.
                if event is None:
                    if phoneme_stream and t > phoneme_stream[-1].end_time:
                        api.avatar_modify_overrides(instance_id, action="set", overrides=copy.copy(mouth_morph_overrides))
                    return speech_lipsync.action_continue

                overrides = copy.copy(mouth_morph_overrides)
                morph = event.morph

                if morph == "!close_mouth":
                    api.avatar_modify_overrides(instance_id, action="set", overrides=overrides)  # set all mouth morphs to zero -> close mouth
                elif morph == "!keep":
                    pass  # keep previous mouth position
                elif morph == "!maybe_close_mouth":  # close mouth only if the pause is at least half a second, else act like "!keep".
                    phoneme_length = event.end_time - event.start_time
                    if phoneme_length >= 0.5:
                        api.avatar_modify_overrides(instance_id, action="set", overrides=overrides)
                else:  # activate one mouth morph, set others to zero
                    overrides[morph] = 1.0
                    api.avatar_modify_overrides(instance_id, action="set", overrides=overrides)

                return speech_lipsync.action_continue

            logger.info("tts_speak_lipsynced.speak: starting playback")
            if on_start is not None:
                try:
                    on_start()
                except Exception as exc:
                    logger.error(f"tts_speak_lipsynced.speak: in start callback: {type(exc)}: {exc}")
                    traceback.print_exc()

            util.api_config.audio_player.start()
            # NOTE: `drive` doesn't take cancellations from `task_env.cancelled`; but the client can explicitly call `tts_stop`,
            # which stops the audio, making `is_playing()` return False — the `on_tick` closure then returns `action_finish`
            # and the loop exits. We are likely running on a different thread pool than the main app, anyway, unless the main
            # thread pool was passed to `raven.client.api.initialize` (implemented in `raven.client.util.initialize_api`).
            speech_lipsync.drive(on_tick=on_tick,
                                 clock=lambda: util.api_config.audio_player.get_position() - video_offset)
        finally:
            logger.info("tts_speak_lipsynced.speak: playback finished")
            if on_stop is not None:
                try:
                    on_stop()
                except Exception as exc:
                    logger.error(f"tts_speak_lipsynced.speak: in stop callback: {type(exc)}: {exc}")
                    traceback.print_exc()

            # TTS is exiting, so stop lipsyncing.
            #
            # NOTE: During app shutdown, we also get here if the avatar instance was deleted
            # (so an `api.avatar_modify_overrides` call raised, because the avatar instance was not found).
            # So at this point, we shouldn't trust that it's still there.
            try:
                api.avatar_modify_overrides(instance_id, action="unset", overrides=mouth_morph_overrides)  # Values are ignored by the "unset" action, which removes the overrides.
            except Exception:
                pass

    util.api_config.task_manager.submit(speak, envcls())

def tts_stop() -> None:
    """Stop the speech synthesizer."""
    if not util.api_initialized:
        raise RuntimeError("tts_stop: The `raven.client.api` module must be initialized before using the API.")
    logger.info("tts_stop: stopping audio")
    util.api_config.audio_player.stop()

def tts_speaking() -> bool:
    """Query whether the speech synthesizer is speaking."""
    if not util.api_initialized:
        raise RuntimeError("tts_stop: The `raven.client.api` module must be initialized before using the API.")
    is_speaking = util.api_config.audio_player.is_playing()
    logger.info(f"tts_speaking: is audio playing: {is_speaking}")
    return is_speaking
