__all__ = ["init_module", "is_available", "text_to_speech"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import io
import json
import os
import traceback
from typing import List
import urllib.parse

from colorama import Fore, Style

from kokoro import KPipeline

from flask import Response

from ..common import config
from ..common.hfutil import maybe_install_models
from ..vendor.streaming_audio_writer.streaming_audio_writer import StreamingAudioWriter

modelsdir = None
pipeline = None
lang = None

def init_module(device_string: str, lang_code="a") -> None:
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
    # We can't download the model to "raven/avatar/vendor/", though, because Kokoro itself won't look for the files there - they must go into HF's default cache location.
    try:
        modelsdir = maybe_install_models(config.KOKORO_MODELS)
        pipeline = KPipeline(lang_code=lang_code, device=device_string, repo_id=config.KOKORO_MODELS)
        lang = lang_code
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'tts'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        modelsdir = None
        pipeline = None
        lang = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (pipeline is not None)

def get_voices() -> List[str]:
    """Get a list of available voices.

    These are automatically scanned from the files of the installed Kokoro-82M model.
    """
    if modelsdir is None:
        raise RuntimeError("get_voices: `modelsdir` not initialized, cannot get list of voices (did `init_module` succeed?)")
    voices = []
    for root, dirs, files in os.walk(os.path.join(modelsdir), topdown=True):
        for filename in files:
            if filename.endswith(".pt"):
                voices.append(filename[:-3])  # drop the ".pt"
    return list(sorted(voices))

# def chunkify_arrays(arrays: Sequence[np.array], chunk_size: int = 4096) -> [np.array, None, None]:
#     """Yield fixed-size chunks from a sequence of arrays, papering over array boundaries, and without making a full copy."""
#
#     # To simplify the rest of the algorithm, combine small arrays until we have *at least* `chunk_size` elements each.
#     # This ensures that when filling up to `chunk_size`, we can always get enough data from the next array in the sequence.
#     newarrays = []
#     array_idx = 0
#     while array_idx < len(arrays):
#         arr = arrays[array_idx]
#         if len(arr) >= chunk_size:
#             newarrays.append(arr)
#             array_idx += 1
#         else:
#             tmps = []
#             total_len = 0
#             while total_len < chunk_size and array_idx < len(arrays):
#                 tmps.append(arrays[array_idx])
#                 total_len += len(arrays[array_idx])
#                 array_idx += 1
#             newarrays.append(np.concatenate(tmps))
#
#     # Then yield fixed-size chunks.
#     array_idx = 0
#     start_offs = 0
#     while True:
#         # start of an array
#         arr = newarrays[array_idx]
#         while (elems_remaining_in_arr := (len(arr) - start_offs)) >= chunk_size:
#             yield arr[start_offs:start_offs + chunk_size]
#             start_offs += chunk_size
#         assert elems_remaining_in_arr < chunk_size  # strictly less than a full chunk left in this array
#         assert elems_remaining_in_arr >= 0  # but still a nonnegative amount
#         last = array_idx == (len(newarrays) - 1)
#         if not last:  # fill the rest of the chunk from the next array, if available
#             if elems_remaining_in_arr > 0:  # did not end exactly at a chunk boundary -> need to combine
#                 next_arr = newarrays[array_idx + 1]
#                 elems_from_next = chunk_size - elems_remaining_in_arr
#                 yield np.concatenate((arr[start_offs:], next_arr[:elems_from_next]))
#                 array_idx += 1
#                 start_offs = elems_from_next
#             else:  # ended exactly at a chunk boundary
#                 array_idx += 1
#                 start_offs = 0
#         else:  # it was the last array
#             if elems_remaining_in_arr > 0:  # did not end exactly at a chunk boundary -> yield the final leftovers as one more chunk
#                 yield arr[start_offs:]
#             return  # and we're done

def text_to_speech(voice: str,
                   text: str,
                   speed: float = 1.0,
                   format: str = "mp3",
                   get_metadata: bool = True,
                   stream: bool = False) -> None:
    """Convert `text` to speech with the speech synthesizer.

    The audio file is sent as the response content.

    `voice`: See https://github.com/hexgrad/kokoro and https://github.com/remsky/Kokoro-FastAPI

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

                    This is useful for e.g. captioning and lip syncing.

                    These are returned as JSON in a header, "x-word-timestamps". The format is:

                        [{"word": "reasonably",
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
              over the transport.
    """
    # Side effect: validate `format` argument
    audio_encoder = StreamingAudioWriter(format=format,
                                         sample_rate=24000,  # Kokoro uses 24kHz sample rate
                                         channels=1)
    import numpy as np
    _, tokens = pipeline.g2p(text)
    metadata = []
    audios = []
    for result in pipeline.generate_from_tokens(tokens=tokens,
                                                voice=voice,
                                                speed=speed):
        if get_metadata:
            if not result.tokens:
                raise RuntimeError("get_speech: No tokens in result, don't know how to get metadata.")
            for token in result.tokens:
                if not all(hasattr(token, field) for field in ("text", "start_ts", "end_ts", "phonemes")):
                    raise RuntimeError(f"get_speech: Token is missing at least one mandatory field ('text', 'start_ts', 'end_ts', 'phonemes'). Data: {token}")
                metadata.append({"word": urllib.parse.quote(token.text, safe=""),
                                 "phonemes": urllib.parse.quote(token.phonemes, safe=""),
                                 "start_time": token.start_ts,
                                 "end_time": token.end_ts})
        audio_numpy = result.audio.cpu().numpy()
        audio_numpy = np.array(audio_numpy * 32767.0, dtype=np.int16)  # float [-1, 1] -> s16
        audios.append(audio_numpy)

    # Our output format is otherwise exactly like that of Kokoro-FastAPI's "/dev/captioned_speech" endpoint (June 2025),
    # but we include the phonemes too, for lip syncing.
    output_headers = {"Content-Type": f"audio/{format}"}
    if get_metadata:
        output_headers["x-word-timestamps"] = json.dumps(metadata)

    if stream:
        def generate():
            for audio_chunk in audios:
                yield audio_encoder.write_chunk(audio_chunk)
            yield audio_encoder.write_chunk(finalize=True)

        return Response(generate(), headers=output_headers)
    else:
        audio_buffer = io.BytesIO()
        for audio_chunk in audios:
            audio_buffer.write(audio_encoder.write_chunk(audio_chunk))
        audio_buffer.write(audio_encoder.write_chunk(finalize=True))
        audio_bytes = audio_buffer.getvalue()

        return Response(audio_bytes, headers=output_headers)
