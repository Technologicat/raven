"""Python client-side API for Talkinghead. This talks with the server through the web API so you don't have to.

We support:

  - classify
  - talkinghead
  - TTS, via https://github.com/remsky/Kokoro-FastAPI

This module is licensed under the 2-clause BSD license, to facilitate Talkinghead integration anywhere.
"""

__all__ = ["init_module"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import concurrent.futures
import copy
import io
import json
import os
import pathlib
import re
import requests
import time
from typing import Callable, Dict, Generator, List, Optional, Union

from unpythonic.env import env as envcls

import pygame  # for audio (text to speech) support

from .. import bgtask

from . import netutil

# ----------------------------------------
# Module bootup

config = envcls(default_headers={},
                tts_default_headers={})
def init_module(avatar_url: str,
                avatar_api_key_file: Optional[str],
                tts_url: Optional[str],
                tts_api_key_file: Optional[str],
                executor: Optional = None):
    """Set up URLs and API keys, and initialize the audio mixer.

    `executor`: `concurrent.futures.ThreadPoolExecutor` or something duck-compatible with it.
                Used for playing TTS audio in the background.

                If not provided, an executor is instantiated automatically.
    """
    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor()
    config.task_manager = bgtask.TaskManager(name="talkinghead_client_api",
                                             mode="concurrent",
                                             executor=executor)

    config.avatar_url = avatar_url
    config.tts_url = tts_url

    if avatar_api_key_file is not None and os.path.exists(avatar_api_key_file):  # TODO: test this (I have no idea what I'm doing)
        with open(avatar_api_key_file, "r", encoding="utf-8") as f:
            avatar_api_key = f.read()
        # See `server.py`.
        config.default_headers["Authorization"] = avatar_api_key.strip()

    if tts_api_key_file is not None and os.path.exists(tts_api_key_file):  # TODO: test this
        with open(tts_api_key_file, "r", encoding="utf-8") as f:
            tts_api_key = f.read()
        # Format for OpenAI compatible endpoints is "Authorization: Bearer xxxx"; the API key file should contain the "Bearer xxxx" part.
        config.default_headers["Authorization"] = tts_api_key.strip()

    pygame.mixer.init()

def yell_on_error(response: requests.Response) -> None:
    if response.status_code != 200:
        logger.error(f"Avatar server returned error: {response.status_code} {response.reason}. Content of error response follows.")
        logger.error(response.text)
        raise RuntimeError(f"While calling avatar server: HTTP {response.status_code} {response.reason}")

# --------------------------------------------------------------------------------
# Classify

def classify_labels() -> List[str]:
    """Get list of emotion names from server."""
    headers = copy.copy(config.default_headers)
    response = requests.get(f"{config.avatar_url}/api/classify/labels", headers=headers)
    yell_on_error(response)
    output_data = response.json()  # -> {"labels": [emotion0, ...]}
    return list(sorted(output_data["labels"]))

def classify(text: str) -> Dict[str, float]:  # TODO: feature orthogonality
    """Classify the emotion of `text` and auto-update the avatar's emotion from that."""
    headers = copy.copy(config.default_headers)
    headers["Content-Type"] = "application/json"
    input_data = {"text": text}
    response = requests.post(f"{config.avatar_url}/api/classify", headers=headers, json=input_data)
    yell_on_error(response)
    output_data = response.json()  # -> ["classification": [{"label": "curiosity", "score": 0.5329479575157166}, ...]]

    sorted_records = output_data["classification"]  # sorted already
    return {record["label"]: record["score"] for record in sorted_records}

# --------------------------------------------------------------------------------
# Talkinghead

def talkinghead_load(filename: Union[pathlib.Path, str]) -> None:
    """Send a character (512x512 RGBA PNG image) to the animator.

    Then, if the animator is not running, start it automatically.
    """
    headers = copy.copy(config.default_headers)
    # Flask expects the file as multipart/form-data. `requests` sets this automatically when we send files, if we don't set a 'Content-Type' header.
    with open(filename, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(f"{config.avatar_url}/api/talkinghead/load", headers=headers, files=files)
    yell_on_error(response)

def talkinghead_unload() -> None:
    """Actually just pause the animator, don't unload anything."""
    headers = copy.copy(config.default_headers)
    response = requests.get(f"{config.avatar_url}/api/talkinghead/unload", headers=headers)
    yell_on_error(response)

def talkinghead_reload() -> None:
    """Resume the animator after it was paused via `talkinghead_unload`, without sending a new character."""
    headers = copy.copy(config.default_headers)
    response = requests.get(f"{config.avatar_url}/api/talkinghead/reload", headers=headers)
    yell_on_error(response)

def talkinghead_load_emotion_templates(emotions: Dict) -> None:
    headers = copy.copy(config.default_headers)
    headers["Content-Type"] = "application/json"
    response = requests.post(f"{config.avatar_url}/api/talkinghead/load_emotion_templates", json=emotions, headers=headers)
    yell_on_error(response)

def talkinghead_load_emotion_templates_from_file(filename: Union[pathlib.Path, str]) -> None:
    with open(filename, "r", encoding="utf-8") as json_file:
        emotions = json.load(json_file)
    talkinghead_load_emotion_templates(emotions)

def talkinghead_load_animator_settings(animator_settings: Dict) -> None:
    headers = copy.copy(config.default_headers)
    headers["Content-Type"] = "application/json"
    response = requests.post(f"{config.avatar_url}/api/talkinghead/load_animator_settings", json=animator_settings, headers=headers)
    yell_on_error(response)

def talkinghead_load_animator_settings_from_file(filename: Union[pathlib.Path, str]) -> None:
    with open(filename, "r", encoding="utf-8") as json_file:
        animator_settings = json.load(json_file)
    talkinghead_load_animator_settings(animator_settings)

def talkinghead_start_talking() -> None:
    headers = copy.copy(config.default_headers)
    response = requests.get(f"{config.avatar_url}/api/talkinghead/start_talking", headers=headers)
    yell_on_error(response)

def talkinghead_stop_talking() -> None:
    headers = copy.copy(config.default_headers)
    response = requests.get(f"{config.avatar_url}/api/talkinghead/stop_talking", headers=headers)
    yell_on_error(response)

def talkinghead_set_emotion(emotion_name: str) -> None:
    headers = copy.copy(config.default_headers)
    headers["Content-Type"] = "application/json"
    data = {"emotion_name": emotion_name}
    response = requests.post(f"{config.avatar_url}/api/talkinghead/set_emotion", headers=headers, json=data)
    yell_on_error(response)

def talkinghead_result_feed(chunk_size: int = 4096, expected_format: Optional[str] = None) -> Generator[bytes, None, None]:
    """Return a generator that yields `bytes` objects, one per video frame, in the image file format received from the server.

    `expected_format`: If provided, string identifying the file format for video frames expected by your client, e.g. "PNG".
    If the server sends some other format, `ValueError` is raised. If not provided, no format checking is done.

    Due to the server's framerate control, the result feed attempts to feed data to the client at TARGET_FPS (default 25).
    New frames are not generated until the previous one has been consumed. Thus, while the animator is in the running state,
    it is recommended to continuously read the stream in a background thread.

    To close the connection (so that the server stops sending), call the `.close()` method of the generator.
    The connection also auto-closes when the generator is garbage-collected.
    """
    headers = copy.copy(config.default_headers)
    headers["Accept"] = "multipart/x-mixed-replace"
    stream_response = requests.get(f"{config.avatar_url}/api/talkinghead/result_feed", headers=headers, stream=True)
    yell_on_error(stream_response)

    stream_iterator = stream_response.iter_content(chunk_size=chunk_size)
    boundary = re.search(r"boundary=(\S+)", stream_response.headers["Content-Type"]).group(1)
    boundary_prefix = f"--{boundary}"  # e.g., '--frame'
    mimetype = f"image/{expected_format.lower()}" if (expected_format is not None) else None
    gen = netutil.multipart_x_mixed_replace_payload_extractor(source=stream_iterator,
                                                              boundary_prefix=boundary_prefix,
                                                              expected_mimetype=mimetype)
    return gen

# # DEBUG/TEST - exercise each of the API endpoints
# print(classify_labels())  # get available emotion names from server
# talkinghead_load("example.png")  # send the avatar - mandatory
# talkinghead_load_animator_settings(os.path.join(os.path.dirname(__file__), "animator.json"))  # send animator config - optional, server defaults used if not sent
# talkinghead_load_emotion_templates(os.path.join(os.path.dirname(__file__), "emotions", "_defaults.json"))  # send the morph parameters for emotions - optional, server defaults used if not sent
# gen = talkinghead_result_feed()
# talkinghead_start_talking()  # start "talking right now" animation
# print(classify("What is the airspeed velocity of an unladen swallow?"))  # classify some text, auto-update emotion from result
# talkinghead_set_emotion("surprise")  # manually update emotion
# for _ in range(5):
#     image_data = next(gen)  # next-gen lol
#     image_file = io.BytesIO(image_data)
#     image = PIL.Image.open(image_file)
# talkinghead_stop_talking()  # stop "talking right now" animation
# talkinghead_unload()  # pause animating the talkinghead
# talkinghead_reload()  # resume animating the talkinghead
# import sys
# sys.exit(0)

# --------------------------------------------------------------------------------
# AI speech synthesizer client (for an OpenAI-compatible endpoint)

def tts_available() -> bool:
    """Return whether the speech synthesizer is available."""
    if config.tts_url is None:
        return False
    headers = copy.copy(config.tts_default_headers)
    try:
        response = requests.get(f"{config.tts_url}/health", headers=headers)
    except Exception as exc:
        logger.error(f"TalkingheadExampleGUI.tts_available: {type(exc)}: {exc}")
        return False
    if response.status_code != 200:
        return False
    return True

def tts_voices() -> None:
    """Return a list of voice names supported by the TTS endpoint (if the endpoint is available)."""
    if config.tts_url is None:
        return []
    headers = copy.copy(config.tts_default_headers)
    response = requests.get(f"{config.tts_url}/v1/audio/voices", headers=headers)
    yell_on_error(response)
    output_data = response.json()
    return output_data["voices"]

def tts_speak(voice: str,
              text: str,
              start_callback: Optional[Callable],
              stop_callback: Optional[Callable]) -> None:
    """Using the speech synthesizer, speak `text` using `voice`.

    If `start_callback` is provided, call it when the TTS starts speaking.
    If `stop_callback` is provided, call it when the TTS has stopped speaking.
    """
    if config.tts_url is None:
        return
    headers = copy.copy(config.tts_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"model": "kokoro",  # https://github.com/remsky/Kokoro-FastAPI
            "voice": voice,
            "input": text,
            "response_format": "mp3",  # flac would be nice (faster to encode), but seems currently broken in kokoro (the audio may repeat twice)
            "speed": 1,
            "stream": True,
            "return_download_link": False}
    stream_response = requests.post(f"{config.tts_url}/v1/audio/speech", headers=headers, json=data, stream=True)
    yell_on_error(stream_response)

    # We run this in the background to
    def speak(task_env) -> None:
        it = stream_response.iter_content(chunk_size=4096)
        audio_buffer = io.BytesIO()
        try:
            while True:
                if task_env.cancelled:
                    return
                chunk = next(it)
                audio_buffer.write(chunk)
        except StopIteration:
            pass

        # # DEBUG - dump response to audio file
        # audio_buffer.seek(0)
        # with open("temp.mp3", "wb") as audio_file:
        #     audio_file.write(audio_buffer.getvalue())

        # play audio
        audio_buffer.seek(0)
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.load(audio_buffer)
        # pygame.mixer.music.load(stream_response.raw)  # can't do this at least with mp3 since the raw stream doesn't support seeking.

        if start_callback is not None:
            start_callback()
        pygame.mixer.music.play()

        if stop_callback is not None:
            while pygame.mixer.music.get_busy():
                time.sleep(0.01)
            stop_callback()
    config.task_manager.submit(speak, envcls())
