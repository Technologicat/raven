"""Utilities for the Python bindings of Raven's web API."""

# Technically, the `api_initialized` flag is part of this public API of this module,
# but since it's a bare boolean (not boxed), from-importing it doesn't make sense,
# so we don't include it in `__all__`.
#
# The correct way is to look it up on this module (`raven.client.util.api_initialized`)
# when its current value is needed.
__all__ = ["api_config",  # configuration namespace
           "initialize_api",
           "yell_on_error"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import atexit
import concurrent.futures
import os
import pathlib
import requests
import traceback
from typing import Optional, Union

from bs4 import BeautifulSoup  # for error message prettification (strip HTML)

import pygame  # for audio (text to speech) support

from unpythonic import equip_with_traceback
from unpythonic.env import env as envcls

from ..common import bgtask

api_initialized = False
api_config = envcls(raven_default_headers={},
                    audio_frequency=44100,
                    audio_buffer_size=2048)  # slightly larger than the default 512 to prevent xruns while the AI translator/subtitler is running simultaneously
def initialize_api(raven_server_url: str,
                   raven_api_key_file: Optional[Union[pathlib.Path, str]],
                   tts_playback_audio_device: Optional[str],
                   executor: Optional = None):
    """Set up URLs and API keys, and initialize the audio mixer.

    Call this before calling any of the actual API functions in `raven.client.api`.

    Suggested values for the `raven_*` arguments are provided in `raven.client.config`.

    `tts_playback_audio_device`: One of the names listed by `raven-check-audio-devices`,
                                 or `None` to use the system default.

    `executor`: `concurrent.futures.ThreadPoolExecutor` or something duck-compatible with it.
                Used for playing TTS audio in the background.

                If not provided, an executor is instantiated automatically.
    """
    global api_initialized

    # HACK: Here it is very useful to know where the call came from, to debug mysterious extra initializations (since only the settings sent the first time will take).
    dummy_exc = Exception()
    dummy_exc = equip_with_traceback(dummy_exc, stacklevel=2)  # 2 = ignore `equip_with_traceback` itself, and its caller, i.e. us
    tb = traceback.extract_tb(dummy_exc.__traceback__)
    top_frame = tb[-1]
    called_from = f"{top_frame[0]}:{top_frame[1]}"  # e.g. "/home/xxx/foo.py:52"
    logger.info(f"initialize_api: called from: {called_from}")

    if api_initialized:  # initialize only once
        logger.info("initialize_api: `raven.client.api` is already initialized. Using existing initialization.")
        return

    logger.info(f"initialize_api: Initializing `raven.client.api` with raven_server_url = '{raven_server_url}', raven_api_key_file = '{str(raven_api_key_file)}', tts_playback_audio_device = '{tts_playback_audio_device}', executor = {executor}.")

    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor()
    api_config.task_manager = bgtask.TaskManager(name="raven_client_api",
                                                 mode="concurrent",
                                                 executor=executor)
    def clear_background_tasks():
        api_config.task_manager.clear(wait=False)  # signal background tasks to exit
    atexit.register(clear_background_tasks)

    api_config.raven_server_url = raven_server_url

    if raven_api_key_file is not None and os.path.exists(raven_api_key_file):  # TODO: test this (I have no idea what I'm doing)
        with open(raven_api_key_file, "r", encoding="utf-8") as f:
            raven_api_key = f.read().replace('\n', '')
        # See `raven.server.app`.
        api_config.raven_default_headers["Authorization"] = raven_api_key.strip()

    # Initialize audio mixer for playing back TTS audio
    # https://www.pygame.org/docs/ref/mixer.html
    if tts_playback_audio_device is not None:
        logger.info(f"initialize_api: Initializing TTS audio playback on non-default audio device '{tts_playback_audio_device}' (from `raven.client.config`).")
    else:
        logger.info("initialize_api: Initializing TTS audio playback on default audio device. If you want to use a non-default device, see `raven.client.config`, and run `raven-check-audio-devices` to get available choices.")
    pygame.mixer.init(frequency=api_config.audio_frequency,
                      size=-16,  # minus: signed values will be used
                      channels=2,
                      buffer=api_config.audio_buffer_size,  # There seems to be no way to *get* the buffer size from `pygame.mixer`, so we must *set* it to know it.
                      devicename=tts_playback_audio_device)  # `None` is the default, and means to use the system's default playback device.

    api_initialized = True

def _strip_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, features='html.parser')
        return soup.get_text()
    except Exception:
        return html  # used for cleaning error messages; important to see the original text if HTML stripping fails

def yell_on_error(response: requests.Response) -> None:
    if response.status_code != 200:
        logger.error(f"Raven-server returned error: {response.status_code} {response.reason}. Content of error response follows.")
        logger.error(_strip_html(response.text))
        raise RuntimeError(f"While calling Raven-server: HTTP {response.status_code} {response.reason}")
