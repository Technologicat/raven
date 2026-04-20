"""Text to speech — Flask wrapper around `raven.common.audio.speech.tts`.

The engine logic (Kokoro pipeline loading, per-segment synthesis,
absolute-timestamp accumulation, metadata extraction) lives in the
common layer. This module handles the transport concerns: float→s16
conversion, URL-encoding raw Unicode phonemes for ASCII-only HTTP
headers, audio-format encoding, Flask Response construction.

Lipsync for the AI avatar is coordinated by `raven.client.api`, using
the `get_metadata=True` mode of `text_to_speech`, which returns the
per-word phoneme timings as URL-encoded JSON in the `x-word-timestamps`
response header.
"""

__all__ = ["init_module", "is_available", "get_info", "get_voices", "text_to_speech"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import importlib
import json
import traceback
from typing import List, Optional
import urllib.parse

from colorama import Fore, Style

from unpythonic import timer

from flask import Response

import numpy as np

from ...common.audio import codec as audio_codec
from ...common.audio.speech import tts as speech_tts

_pipeline: Optional[speech_tts.TTSPipeline] = None

def init_module(config_module_name: str, device_string: str, lang_code: str = "a") -> None:
    """Initialize the speech synthesizer.

    Note that the `get_metadata` mode of `text_to_speech` currently
    supports English only (`lang_code="a"` or `lang_code="b"`).

    See `raven.common.audio.speech.tts.load_tts_pipeline` for the
    full list of supported `lang_code` values.
    """
    global _pipeline
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}tts{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}'...")
    try:
        server_config = importlib.import_module(config_module_name)
        _pipeline = speech_tts.load_tts_pipeline(model_name=server_config.kokoro_models,
                                                 device_string=device_string,
                                                 lang_code=lang_code)
    except Exception as exc:
        _pipeline = None
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'tts'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (_pipeline is not None)

def get_info() -> dict:
    """Return engine metadata as a JSON-serializable dict.

    Current fields:
        `sample_rate`: native output sample rate, Hz (Kokoro: 24000).
        `model`: HuggingFace repo identifier for the loaded TTS model.
    """
    if _pipeline is None:
        raise RuntimeError("get_info: pipeline not initialized (did `init_module` succeed?)")
    return {"sample_rate": _pipeline.sample_rate,
            "model": _pipeline.model_name}

def get_voices() -> List[str]:
    """Get a list of available voices.

    Scanned from the files of the installed Kokoro-82M model.
    """
    if _pipeline is None:
        raise RuntimeError("get_voices: pipeline not initialized (did `init_module` succeed?)")
    return speech_tts.get_voices(_pipeline)

def text_to_speech(voice: str,
                   text: str,
                   speed: float = 1.0,
                   format: str = "flac",
                   get_metadata: bool = True,
                   stream: bool = False) -> Response:
    """Convert `text` to speech with the speech synthesizer.

    The audio file is returned as the response content.

    `voice`: See `get_voices`, or the `raven-avatar-settings-editor` GUI app.

    `speed`: Speed, relative to the normal speed of the selected voice.

             Usually values in the range 0.8 ... 1.2 work. If less, the TTS may sound like
             a broken record player; if more, the TTS may start skipping phonemes.

    `format`: one of:
        "wav" (PCM signed 16-bit LE)
        "mp3"
        "opus"
        "flac"
        "aac"

    `get_metadata`: If `True`, get word-level timestamps and phonemes.

                    This is useful for e.g. captioning and lipsyncing.

                    These are returned as JSON in a header, "x-word-timestamps". The format is:

                        [{"word": "reasonably" (URL-encoded to ASCII with percent-escaped UTF-8),
                          "phonemes": "ɹˈizənəbli" (URL-encoded to ASCII with percent-escaped UTF-8),
                          "start_time": 2.15,
                          "end_time": 2.75},
                         ...]

                    The start and end times are measured in seconds from start of audio.

                    Note the phoneme string may have punctuation, e.g. "some.python.module"
                    may tokenize as one word.

                    The phoneme string may also have spaces when the word doesn't; the year "2025"
                    tokenizes as one word, with phonemes "twˈɛnti twˈɛnti fˈIv".

                    For how to interpret the phoneme data, see:
                        https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md

    `stream`: Stream the audio, or send the whole response content at once?

              All the audio is in any case generated in one go; this just streams the result
              over the network transport.

              If `stream=False`, the response includes a "Content-Length" header, in bytes.

    We always send one custom header, "x-audio-duration", which contains the duration
    of the speech audio data, in seconds, before encoding into `format`.

    Note that at least historically, some audio formats have had very long audio frames
    (over 1 second), so depending on `format`, the encoded audio file may be slightly longer
    than "x-audio-duration", end-padded with silence.
    """
    # Run synthesis; keep per-segment s16 arrays so streaming encode preserves segment boundaries.
    segment_audios_s16: List[np.ndarray] = []
    metadata: List[dict] = []
    with timer() as tim:
        for segment in speech_tts.synthesize_iter(_pipeline,
                                                  voice=voice,
                                                  text=text,
                                                  speed=speed,
                                                  get_metadata=get_metadata):
            # Transport-layer cast: float [-1, 1] → s16 for audio_codec.encode.
            segment_audios_s16.append(np.array(segment.audio * 32767.0, dtype=np.int16))

            if get_metadata and segment.word_metadata is not None:
                for w in segment.word_metadata:
                    # URL-encode the raw Unicode so the JSON fits in an ASCII-only HTTP header.
                    metadata.append({"word": urllib.parse.quote(w.word, safe=""),
                                     "phonemes": urllib.parse.quote(w.phonemes, safe=""),
                                     "start_time": w.start_time,
                                     "end_time": w.end_time})

    total_audio_duration = sum(len(chunk) / _pipeline.sample_rate for chunk in segment_audios_s16)
    plural_s = "s" if len(segment_audios_s16) != 1 else ""
    logger.info(f"text_to_speech: processing complete in {tim.dt:0.6g}s. "
                f"Got {len(segment_audios_s16)} TTS response segment{plural_s}, "
                f"total audio duration {total_audio_duration:0.6g}s.")

    # Our output format is inspired by Kokoro-FastAPI's "/dev/captioned_speech" endpoint (June 2025),
    # but we include the phonemes too (for lipsyncing) and the audio duration.
    output_headers = {"Content-Type": f"audio/{format}",
                      "x-audio-duration": total_audio_duration}  # seconds
    if get_metadata:
        output_headers["x-word-timestamps"] = json.dumps(metadata)

    if stream:
        streamer = audio_codec.encode(audio_data=segment_audios_s16,
                                      format=format,
                                      sample_rate=_pipeline.sample_rate,
                                      stream=True)
        return Response(streamer(), headers=output_headers)
    else:
        audio_bytes = audio_codec.encode(audio_data=segment_audios_s16,
                                         format=format,
                                         sample_rate=_pipeline.sample_rate)
        output_headers["Content-Length"] = len(audio_bytes)
        return Response(audio_bytes, headers=output_headers)
