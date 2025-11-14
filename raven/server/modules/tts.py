"""This module handles the TTS (text to speech, speech synthesizer) audio.

Lipsync for the AI avatar is coordinated by `raven.client.api`, using the `get_metadata` mode of `text_to_speech`.
"""

__all__ = ["init_module", "is_available", "get_voices", "text_to_speech"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import importlib
import json
import os
import traceback
from typing import List
import urllib.parse

from colorama import Fore, Style

from unpythonic import memoize

from kokoro import KPipeline

from flask import Response

import numpy as np

from unpythonic import timer

from ...common import audioutils
from ...common.hfutil import maybe_install_models

modelsdir = None
pipeline = None
lang = None

def init_module(config_module_name: str, device_string: str, lang_code="a") -> None:
    """Initialize the speech synthesizer.

    Note that the `get_metadata` mode of `text_to_speech` currently
    supports English only (`lang_code="a"` or `lang_code="b"`).

    Language codes:
      https://github.com/hexgrad/kokoro
      🇺🇸 'a' => American English, 🇬🇧 'b' => British English
      🇪🇸 'e' => Spanish es
      🇫🇷 'f' => French fr-fr
      🇮🇳 'h' => Hindi hi
      🇮🇹 'i' => Italian it
      🇯🇵 'j' => Japanese: pip install misaki[ja]
      🇧🇷 'p' => Brazilian Portuguese pt-br
      🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
    """
    global modelsdir
    global pipeline
    global lang
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}tts{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}'...")

    # Install Kokoro's AI models if not installed yet.
    #
    # We need to install the full repo to get a list of available voice names programmatically (like Kokoro-FastAPI does, see `Kokoro-FastAPI/api/src/core/paths.py`).
    # We can't download the model to "raven/vendor/", though, because Kokoro itself won't look for the files there - they must go into HF's default cache location.
    try:
        server_config = importlib.import_module(config_module_name)  # contains hf repo name for downloading Kokoro models if needed
        modelsdir = maybe_install_models(server_config.kokoro_models)
        pipeline = KPipeline(lang_code=lang_code, device=device_string, repo_id=server_config.kokoro_models)
        lang = lang_code
    except Exception as exc:
        modelsdir = None
        pipeline = None
        lang = None
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'tts'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (pipeline is not None)

@memoize
def get_voices() -> List[str]:
    """Get a list of available voices.

    These are automatically scanned from the files of the installed Kokoro-82M model.

    The scan occurs only once per server session.
    """
    if modelsdir is None:
        raise RuntimeError("get_voices: `modelsdir` not initialized, cannot get list of voices (did `init_module` succeed?)")
    voices = []
    for root, dirs, files in os.walk(os.path.join(modelsdir), topdown=True):
        for filename in files:
            if filename.endswith(".pt"):
                voices.append(filename[:-3])  # drop the ".pt"
    return list(sorted(voices))

def text_to_speech(voice: str,
                   text: str,
                   speed: float = 1.0,
                   format: str = "mp3",
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
    # Side effect: validate `format` argument
    sample_rate = 24000  # Kokoro uses 24kHz sample rate
    _, tokens = pipeline.g2p(text)
    metadata = []
    audios = []

    # SillyTavern compatibility: ST sends the whitespace too if the user configured the comma-separated voice list with whitespaces
    voice = voice.strip()

    if voice not in (voices := get_voices()):
        raise ValueError(f"Unknown voice '{voice}'; installed voices: {voices}")

    # In multi-segment output, Kokoro returns timestamps relative to the start of the current segment.
    # We want absolute timestamps for the whole audio, so keep track of the length of the completed segments.
    t0 = 0.0
    def get_audio_duration(audio_numpy: np.array) -> float:
        """Get time duration of NumPy audio data, in seconds."""
        return len(audio_numpy) / sample_rate
    with timer() as tim:
        for segment_num, result in enumerate(pipeline.generate_from_tokens(tokens=tokens,
                                                                           voice=voice,
                                                                           speed=speed),
                                             start=1):
            logger.info(f"text_to_speech: Processing TTS response segment {segment_num}")
            if get_metadata:
                if not result.tokens:
                    raise RuntimeError("text_to_speech: No tokens in result, don't know how to get metadata.")
                for token in result.tokens:
                    if not all(hasattr(token, field) for field in ("text", "start_ts", "end_ts", "phonemes")):
                        raise RuntimeError(f"text_to_speech: Token is missing at least one mandatory field ('text', 'start_ts', 'end_ts', 'phonemes'). Data: {token}")
                    metadata.append({"word": urllib.parse.quote(token.text, safe=""),
                                     "phonemes": urllib.parse.quote(token.phonemes, safe=""),
                                     "start_time": (t0 + token.start_ts) if token.start_ts is not None else None,
                                     "end_time": (t0 + token.end_ts) if token.end_ts is not None else None})
            audio_numpy = result.audio.cpu().numpy()
            audio_numpy = np.array(audio_numpy * 32767.0, dtype=np.int16)  # float [-1, 1] -> s16
            audios.append(audio_numpy)
            t0 += get_audio_duration(audio_numpy)  # add duration of this audio segment, in seconds
    total_audio_duration = sum(get_audio_duration(audio_numpy) for audio_numpy in audios)
    plural_s = "s" if len(audios) != 1 else ""
    logger.info(f"text_to_speech: Processing complete in {tim.dt:0.6g}s. Got {len(audios)} TTS response segment{plural_s}, with a total audio duration of {total_audio_duration:0.6g}s.")

    # Our output format is inspired by Kokoro-FastAPI's "/dev/captioned_speech" endpoint (June 2025), but we include the phonemes too, for lipsyncing, and the audio duration.
    output_headers = {"Content-Type": f"audio/{format}",
                      "x-audio-duration": total_audio_duration}  # seconds
    if get_metadata:
        output_headers["x-word-timestamps"] = json.dumps(metadata)

    if stream:
        streamer = audioutils.encode_audio(audio_data=audios,
                                           format=format,
                                           sample_rate=sample_rate,
                                           stream=True)
        return Response(streamer(), headers=output_headers)
    else:
        audio_bytes = audioutils.encode_audio(audio_data=audios,
                                              format=format,
                                              sample_rate=sample_rate)
        output_headers["Content-Length"] = len(audio_bytes)
        return Response(audio_bytes, headers=output_headers)
