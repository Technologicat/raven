"""Speech to text — Flask wrapper around `raven.common.audio.speech.stt`.

The engine logic (model loading, token generation, Whisper-specific
preprocessing) lives in the common layer. This module handles only the
transport concerns: audio container decoding, config-module lookup,
and the tqdm progress bar for server console output.
"""

__all__ = ["init_module", "is_available", "speech_to_text"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import importlib
import traceback
from typing import Optional, Union

from colorama import Fore, Style

import torch

from tqdm import tqdm

from ...common.audio import codec as audio_codec
from ...common.audio.speech import stt as speech_stt

_stt_model: Optional[speech_stt.STTModel] = None

def init_module(config_module_name: str, device_string: str, dtype: Union[str, torch.dtype]) -> None:
    """Initialize the speech recognizer (speech to text)."""
    global _stt_model
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}stt{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}'...")
    try:
        server_config = importlib.import_module(config_module_name)
        model_name = server_config.speech_recognition_model

        logger.info(f"init_module: loading model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}' on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}'.")
        _stt_model = speech_stt.load_stt_model(model_name, device_string, dtype)
    except Exception as exc:
        _stt_model = None
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'stt'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (_stt_model is not None)

# TODO: the input is a flask.request.file.stream; what's the type of that?
def speech_to_text(stream,
                   prompt: Optional[str],
                   language: Optional[str]) -> str:
    """Transcribe speech to text.

    `stream`: a `flask.request.file.stream` containing an audio file (any format supported by PyAV).

    `prompt`: optional Whisper conditioning text (list rare proper names, set context,
              or nudge transcription style). See `raven.server.app.api_stt_transcribe`.

    `language`: optional ISO-639-1 code (e.g. `"en"`). Default is to autodetect.

    Returns the transcribed text.
    """
    # Decode the audio container to mono float at Whisper's native sample rate,
    # then hand off to the engine.
    _unused_metadata, audio_numpy = audio_codec.decode(stream,
                                                       target_sample_format="fltp",
                                                       target_sample_rate=_stt_model.sample_rate,
                                                       target_layout="mono")

    # Log a tqdm progress bar to the server console during transcription.
    # `leave=True` keeps the completed bar visible in the log.
    with tqdm(desc="Transcribing", leave=True) as pbar:
        def on_progress(current: int, total: int) -> None:
            pbar.total = total
            pbar.n = current
            pbar.refresh()  # note `refresh`, not `update` — `update` also increments `n`
        return speech_stt.transcribe(_stt_model,
                                     audio=audio_numpy,
                                     sample_rate=_stt_model.sample_rate,
                                     prompt=prompt,
                                     language=language,
                                     progress_callback=on_progress)
