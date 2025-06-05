"""Python bindings for raven-avatar web API.

This talks with the server so you can just call regular Python functions.

For documentation, see the server side in `raven.avatar.server.app`.

We support all modules served by `raven.avatar.server.app`:

  - classify
  - embeddings (to enable this module, start the server with the "--embeddings" command-line option)
  - talkinghead
  - TTS with and without lip-syncing the talkinghead, via https://github.com/remsky/Kokoro-FastAPI
  - websearch

This must be initialized before the API is used; see `init_module`. Suggested default settings
for the parameters are provided in `raven.avatar.client.config`.

This module is licensed under the 2-clause BSD license.
"""

__all__ = ["init_module",
           "avatar_available",
           "tts_available",
           "classify_labels", "classify",
           "talkinghead_load", "talkinghead_stop", "talkinghead_start",
           "talkinghead_load_emotion_templates", "talkinghead_load_emotion_templates_from_file",
           "talkinghead_load_animator_settings", "talkinghead_load_animator_settings_from_file",
           "talkinghead_start_talking", "talkinghead_stop_talking",
           "talkinghead_set_emotion",
           "talkinghead_result_feed",
           "tts_voices",
           "tts_speak", "tts_speak_lipsynced",
           "tts_stop"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import bisect
import concurrent.futures
import copy
import io
import json
import os
import pathlib
import re
import requests
import time
import traceback
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

from unpythonic.env import env as envcls

import numpy as np

import pygame  # for audio (text to speech) support

from ...common import bgtask

from ..common import netutil

# ----------------------------------------
# Module bootup

module_initialized = False
api_config = envcls(avatar_default_headers={},
                    tts_default_headers={},
                    audio_frequency=44100,
                    audio_buffer_size=512)
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
    global module_initialized

    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor()
    api_config.task_manager = bgtask.TaskManager(name="talkinghead_client_api",
                                                 mode="concurrent",
                                                 executor=executor)

    api_config.avatar_url = avatar_url
    api_config.tts_url = tts_url

    if avatar_api_key_file is not None and os.path.exists(avatar_api_key_file):  # TODO: test this (I have no idea what I'm doing)
        with open(avatar_api_key_file, "r", encoding="utf-8") as f:
            avatar_api_key = f.read()
        # See `raven.avatar.server.app`.
        api_config.avatar_default_headers["Authorization"] = avatar_api_key.strip()

    if tts_api_key_file is not None and os.path.exists(tts_api_key_file):  # TODO: test this
        with open(tts_api_key_file, "r", encoding="utf-8") as f:
            tts_api_key = f.read()
        # Format for OpenAI compatible endpoints is "Authorization: Bearer xxxx"; the API key file should contain the "Bearer xxxx" part.
        api_config.tts_default_headers["Authorization"] = tts_api_key.strip()

    # https://www.pygame.org/docs/ref/mixer.html
    pygame.mixer.init(frequency=api_config.audio_frequency,
                      size=-16,
                      channels=2,
                      buffer=api_config.audio_buffer_size)  # There seems to be no way to *get* the buffer size from `pygame.mixer`, so we must *set* it to know it.

    module_initialized = True

def yell_on_error(response: requests.Response) -> None:
    if response.status_code != 200:
        logger.error(f"Avatar server returned error: {response.status_code} {response.reason}. Content of error response follows.")
        logger.error(response.text)
        raise RuntimeError(f"While calling avatar server: HTTP {response.status_code} {response.reason}")

# --------------------------------------------------------------------------------
# General utilities

def avatar_available() -> bool:
    """Return whether the avatar server (everything except TTS) is available."""
    if not module_initialized:
        raise RuntimeError("avatar_available: The `raven.avatar.client.api` module must be initialized before using the API.")
    if api_config.avatar_url is None:
        return False
    headers = copy.copy(api_config.avatar_default_headers)
    try:
        response = requests.get(f"{api_config.avatar_url}/health", headers=headers)
    except requests.exceptions.ConnectionError as exc:
        logger.error(f"avatar_available: {type(exc)}: {exc}")
        return False
    if response.status_code != 200:
        return False
    return True

def tts_available() -> bool:
    """Return whether the speech synthesizer is available."""
    if not module_initialized:
        raise RuntimeError("tts_available: The `raven.avatar.client.api` module must be initialized before using the API.")
    if api_config.tts_url is None:
        return False
    headers = copy.copy(api_config.tts_default_headers)
    try:
        response = requests.get(f"{api_config.tts_url}/health", headers=headers)
    except requests.exceptions.ConnectionError as exc:
        logger.error(f"tts_available: {type(exc)}: {exc}")
        return False
    if response.status_code != 200:
        return False
    return True

# --------------------------------------------------------------------------------
# Classify

def classify_labels() -> List[str]:
    """Get list of emotion names from server.

    Return format is::

        [emotion0, ...]
    """
    if not module_initialized:
        raise RuntimeError("classify_labels: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    response = requests.get(f"{api_config.avatar_url}/api/classify/labels", headers=headers)
    yell_on_error(response)
    output_data = response.json()  # -> {"labels": [emotion0, ...]}
    return list(sorted(output_data["labels"]))

def classify(text: str) -> Dict[str, float]:
    """Classify the emotion of `text`.

    Return format is::

        {emotion0: score0,
         ...}
    """
    if not module_initialized:
        raise RuntimeError("classify: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    headers["Content-Type"] = "application/json"
    input_data = {"text": text}
    response = requests.post(f"{api_config.avatar_url}/api/classify", headers=headers, json=input_data)
    yell_on_error(response)
    output_data = response.json()  # -> ["classification": [{"label": "curiosity", "score": 0.5329479575157166}, ...]]

    sorted_records = output_data["classification"]  # sorted already
    return {record["label"]: record["score"] for record in sorted_records}

# --------------------------------------------------------------------------------
# Embeddings

def embeddings_compute(text: Union[str, List[str]]) -> np.array:
    """Compute vector embeddings (semantic embeddings).

    Useful e.g. for semantic similarity comparison and RAG search.

    Return format is `np.array`, with shape:

        - `(ndim,)` if `text` is a single string
        - `(nbatch, ndim)` if `text` is a list of strings.

    Here `ndim` is the dimensionality of the vector embedding model that the avatar server is using.
    """
    if not module_initialized:
        raise RuntimeError("embeddings_compute: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    headers["Content-Type"] = "application/json"
    input_data = {"text": text}
    response = requests.post(f"{api_config.avatar_url}/api/embeddings/compute", json=input_data, headers=headers)
    yell_on_error(response)
    output_data = response.json()

    vectors = output_data["embedding"]
    return np.array(vectors)

# --------------------------------------------------------------------------------
# Talkinghead

def talkinghead_load(filename: Union[pathlib.Path, str]) -> None:
    """Send a character (512x512 RGBA PNG image) to the animator.

    Then, to start the animator, call `talkinghead_start`.
    """
    if not module_initialized:
        raise RuntimeError("talkinghead_load: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    # Flask expects the file as multipart/form-data. `requests` sets this automatically when we send files, if we don't set a 'Content-Type' header.
    with open(filename, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(f"{api_config.avatar_url}/api/talkinghead/load", headers=headers, files=files)
    yell_on_error(response)

def talkinghead_load_emotion_templates(emotions: Dict) -> None:
    if not module_initialized:
        raise RuntimeError("talkinghead_load_emotion_templates: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    headers["Content-Type"] = "application/json"
    response = requests.post(f"{api_config.avatar_url}/api/talkinghead/load_emotion_templates", json=emotions, headers=headers)
    yell_on_error(response)

def talkinghead_load_emotion_templates_from_file(filename: Union[pathlib.Path, str]) -> None:
    if not module_initialized:
        raise RuntimeError("talkinghead_load_emotion_templates_from_file: The `raven.avatar.client.api` module must be initialized before using the API.")
    with open(filename, "r", encoding="utf-8") as json_file:
        emotions = json.load(json_file)
    talkinghead_load_emotion_templates(emotions)

def talkinghead_load_animator_settings(animator_settings: Dict) -> None:
    if not module_initialized:
        raise RuntimeError("talkinghead_load_animator_settings: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    headers["Content-Type"] = "application/json"
    response = requests.post(f"{api_config.avatar_url}/api/talkinghead/load_animator_settings", json=animator_settings, headers=headers)
    yell_on_error(response)

def talkinghead_load_animator_settings_from_file(filename: Union[pathlib.Path, str]) -> None:
    if not module_initialized:
        raise RuntimeError("talkinghead_load_animator_settings_from_file: The `raven.avatar.client.api` module must be initialized before using the API.")
    with open(filename, "r", encoding="utf-8") as json_file:
        animator_settings = json.load(json_file)
    talkinghead_load_animator_settings(animator_settings)

def talkinghead_start() -> None:
    """Start or resume the animator."""
    if not module_initialized:
        raise RuntimeError("talkinghead_start: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    response = requests.get(f"{api_config.avatar_url}/api/talkinghead/start", headers=headers)
    yell_on_error(response)

def talkinghead_stop() -> None:
    """Pause the animator."""
    if not module_initialized:
        raise RuntimeError("talkinghead_stop: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    response = requests.get(f"{api_config.avatar_url}/api/talkinghead/stop", headers=headers)
    yell_on_error(response)

def talkinghead_start_talking() -> None:
    if not module_initialized:
        raise RuntimeError("talkinghead_start_talking: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    response = requests.get(f"{api_config.avatar_url}/api/talkinghead/start_talking", headers=headers)
    yell_on_error(response)

def talkinghead_stop_talking() -> None:
    if not module_initialized:
        raise RuntimeError("talkinghead_stop_talking: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    response = requests.get(f"{api_config.avatar_url}/api/talkinghead/stop_talking", headers=headers)
    yell_on_error(response)

def talkinghead_set_emotion(emotion_name: str) -> None:
    if not module_initialized:
        raise RuntimeError("talkinghead_set_emotion: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"emotion_name": emotion_name}
    response = requests.post(f"{api_config.avatar_url}/api/talkinghead/set_emotion", headers=headers, json=data)
    yell_on_error(response)

def talkinghead_set_overrides(data: Dict[str, float]) -> None:
    if not module_initialized:
        raise RuntimeError("talkinghead_set_overrides: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    headers["Content-Type"] = "application/json"
    response = requests.post(f"{api_config.avatar_url}/api/talkinghead/set_overrides", json=data, headers=headers)
    yell_on_error(response)

def talkinghead_result_feed(chunk_size: int = 4096, expected_mimetype: Optional[str] = None) -> Generator[Tuple[Optional[str], bytes], None, None]:
    """Return a generator that yields video frames, in the image file format received from the server.

    The yielded value is the tuple `(received_mimetype, payload)`, where `received_mimetype` is set to whatever the server
    sent in the Content-Type header. Talkinghead always sends a mimetype, which specifies the file format of `payload`.

    `expected_mimetype`: If provided, string identifying the mimetype for video frames expected by your client, e.g. "image/png".
    If the server sends some other format, `ValueError` is raised. If not provided, no format checking is done.

    Due to the server's framerate control, the result feed attempts to feed data to the client at TARGET_FPS (default 25).
    New frames are not generated until the previous one has been consumed. Thus, while the animator is in the running state,
    it is recommended to continuously read the stream in a background thread.

    To close the connection (so that the server stops sending), call the `.close()` method of the generator.
    The connection also auto-closes when the generator is garbage-collected.
    """
    if not module_initialized:
        raise RuntimeError("talkinghead_result_feed: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    headers["Accept"] = "multipart/x-mixed-replace"
    stream_response = requests.get(f"{api_config.avatar_url}/api/talkinghead/result_feed", headers=headers, stream=True)
    yell_on_error(stream_response)

    stream_iterator = stream_response.iter_content(chunk_size=chunk_size)
    boundary = re.search(r"boundary=(\S+)", stream_response.headers["Content-Type"]).group(1)
    boundary_prefix = f"--{boundary}"  # e.g., '--frame'
    gen = netutil.multipart_x_mixed_replace_payload_extractor(source=stream_iterator,
                                                              boundary_prefix=boundary_prefix,
                                                              expected_mimetype=expected_mimetype)
    return gen

# --------------------------------------------------------------------------------
# TTS - AI speech synthesizer client

def tts_voices() -> List[str]:
    """Return a list of voice names supported by the TTS endpoint (if the endpoint is available).

    Return format is::

        [voice0, ...]
    """
    if not module_initialized:
        raise RuntimeError("tts_voices: The `raven.avatar.client.api` module must be initialized before using the API.")
    if api_config.tts_url is None:
        return []
    headers = copy.copy(api_config.tts_default_headers)
    response = requests.get(f"{api_config.tts_url}/v1/audio/voices", headers=headers)
    yell_on_error(response)
    output_data = response.json()
    return output_data["voices"]

def tts_speak(voice: str,
              text: str,
              speed: float = 1.0,
              start_callback: Optional[Callable] = None,
              stop_callback: Optional[Callable] = None) -> None:
    """Using the speech synthesizer, speak `text` using `voice`.

    If `start_callback` is provided, call it when the TTS starts speaking.
    If `stop_callback` is provided, call it when the TTS has stopped speaking.
    """
    if not module_initialized:
        raise RuntimeError("tts_speak: The `raven.avatar.client.api` module must be initialized before using the API.")
    if api_config.tts_url is None:
        return
    headers = copy.copy(api_config.tts_default_headers)
    headers["Content-Type"] = "application/json"

    # We run this in the background
    def speak(task_env) -> None:
        logger.info("tts_speak.speak: getting audio")

        # Audio format
        data = {"model": "kokoro",  # https://github.com/remsky/Kokoro-FastAPI
                "voice": voice,
                "input": text,
                "response_format": "mp3",  # flac would be nice (small, fast to encode), but seems currently broken in kokoro (the audio may repeat twice).
                "speed": speed,
                "stream": True,
                "return_download_link": False}
        stream_response = requests.post(f"{api_config.tts_url}/v1/audio/speech", headers=headers, json=data, stream=True)
        yell_on_error(stream_response)

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
        logger.info("tts_speak.speak: loading audio into mixer")
        audio_buffer.seek(0)
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.load(audio_buffer)
        # pygame.mixer.music.load(stream_response.raw)  # can't do this at least with mp3 since the raw stream doesn't support seeking.

        logger.info("tts_speak.speak: starting playback")
        if start_callback is not None:
            try:
                start_callback()
            except Exception as exc:
                logger.error(f"tts_speak.speak: in start callback: {type(exc)}: {exc}")
                traceback.print_exc()
        pygame.mixer.music.play()

        if stop_callback is not None:
            while pygame.mixer.music.get_busy():
                time.sleep(0.01)
            logger.info("tts_speak.speak: playback finished")
            try:
                stop_callback()
            except Exception as exc:
                logger.error(f"tts_speak.speak: in stop callback: {type(exc)}: {exc}")
                traceback.print_exc()
        else:
            logger.info("tts_speak.speak: no stop callback, all done.")
    api_config.task_manager.submit(speak, envcls())

def tts_speak_lipsynced(voice: str,
                        text: str,
                        speed: float = 1.0,
                        video_offset: float = 0.0,
                        start_callback: Optional[Callable] = None,
                        stop_callback: Optional[Callable] = None) -> None:
    """Like `tts_speak`, but with lip sync for the talkinghead.

    Requires the Kokoro-FastAPI TTS backend so that we can get the phoneme data
    and timestamps.

    `video_offset`: seconds, for adjusting lipsync animation.
        - Positive values: Use if the video is early. Shifts video later with respect to the audio.
        - Negative values: Use if the video is late. Shifts video earlier with respect to the audio.

    See:
        https://github.com/remsky/Kokoro-FastAPI
    """
    if not module_initialized:
        raise RuntimeError("tts_speak_lipsynced: The `raven.avatar.client.api` module must be initialized before using the API.")
    if api_config.tts_url is None:
        return
    headers = copy.copy(api_config.tts_default_headers)
    headers["Content-Type"] = "application/json"

    # Phonemize and word-level timestamping treat underscores differently: phonemize treats them as spaces,
    # whereas word-level timestamping doesn't (no word split at underscore). Better to remove them.
    #
    # TODO: See if we could use Misaki directly to phonemize the individual timestamped words returned by Kokoro, bypassing the thorny subtly-different-word-splittings issue.
    def prefilter(text):
        return text.replace("_", " ")
    text = prefilter(text)

    # Phoneme characters and individual phoneme comments from Misaki docs (Kokoro uses the Misaki engine for phonemization):
    #   https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md
    #
    # For the animator morph names, see `util.py`
    dipthong_vowel_to_ipa = {
        "A": "eɪ",  # The "eh" vowel sound, like hey => hˈA. Expands to eɪ in IPA.
        "I": "aɪ",  # The "eye" vowel sound, like high => hˈI. Expands to aɪ in IPA.
        "W": "aʊ",  # The "ow" vowel sound, like how => hˌW. Expands to aʊ in IPA.
        "Y": "ɔɪ",  # The "oy" vowel sound, like soy => sˈY. Expands to ɔɪ in IPA.
        # 🇺🇸 American-only
        "O": "oʊ",  # Capital letter representing the American "oh" vowel sound. Expands to oʊ in IPA.
        # 🇬🇧 British-only
        "Q": "əʊ",  # Capital letter representing the British "oh" vowel sound. Expands to əʊ in IPA.
    }
    # TODO: Lipsync: improve the phoneme to morph conversion table.
    #   - For more fine-grained control, we could use the value channel too, not just always set the morph to 1.0. Could also combine morphs (use a list).
    #
    # value: animator morph name, or one of the special commands:
    #   "!close_mouth" - as it says on the tin
    #   "!keep" - keep previous mouth position
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
        # for dipthong expansion
        "e": "mouth_eee_index",
        "o": "mouth_ooo_index",
    }

    def get_phonemes(task_env) -> None:
        logger.info("tts_speak_lipsynced.get_phonemes: starting")
        # Language codes:
        #   https://github.com/hexgrad/kokoro
        #   🇺🇸 'a' => American English, 🇬🇧 'b' => British English
        #   🇪🇸 'e' => Spanish es
        #   🇫🇷 'f' => French fr-fr
        #   🇮🇳 'h' => Hindi hi
        #   🇮🇹 'i' => Italian it
        #   🇯🇵 'j' => Japanese: pip install misaki[ja]
        #   🇧🇷 'p' => Brazilian Portuguese pt-br
        #   🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
        data = {"text": text,
                "language": "a"}
        response = requests.post(f"{api_config.tts_url}/dev/phonemize", headers=headers, json=data, stream=True)
        yell_on_error(response)
        response_json = response.json()
        phonemes = response_json["phonemes"]
        for dipthong, ipa_expansion in dipthong_vowel_to_ipa.items():
            phonemes = phonemes.replace(dipthong, ipa_expansion)
        # phonemess = phonemes.split()  # -> [word0, word1, ...]
        # phonemess = re.split(r"\s|,|;|:|\.|!|\?|“|”", phonemes)  # -> [word0, word1, ...], dropping punctuation
        # Word-level timestamping splits at ":" (even if inside a word), so we should too. But it doesn't split at periods in dotted names, so we shouldn't either.
        phonemess = re.split(r"\s|:", phonemes)  # -> [word0, word1, ...]
        phonemess = [p for p in phonemess if p]  # drop empty strings
        task_env.phonemess = phonemess
        task_env.done = True
        logger.info("tts_speak_lipsynced.get_phonemes: done")
    phonemes_task_env = envcls(done=False)
    api_config.task_manager.submit(get_phonemes, phonemes_task_env)

    def speak(task_env) -> None:
        logger.info("tts_speak_lipsynced.speak: starting")
        def isword(s):
            return len(s) > 1 or s.isalnum()

        def clean_timestamps(timestamps):
            """Remove consecutive duplicate timestamps (some versions of Kokoro produce those) and any timestamps for punctuation."""
            out = []
            last_start_time = None
            for record in timestamps:  # format: [{"word": "blah", "start_time": 1.23, "end_time": 1.45}, ...]
                if record["start_time"] != last_start_time and isword(record["word"]):
                    out.append(record)
                    last_start_time = record["start_time"]
            return out

        # Get audio and word timestamps
        logger.info("tts_speak_lipsynced.speak: getting audio with word timestamps")
        data = {"model": "kokoro",
                "voice": voice,
                "input": text,
                "response_format": "mp3",
                "speed": speed,
                "stream": True,
                "return_timestamps": True}
        stream_response = requests.post(f"{api_config.tts_url}/dev/captioned_speech", headers=headers, json=data, stream=True)
        yell_on_error(stream_response)

        # The API docs are wrong; using a running Kokoro-FastAPI and sending an example
        # request at http://localhost:8880/docs helped to figure out what to do here.
        timestamps = json.loads(stream_response.headers["x-word-timestamps"])
        timestamps = clean_timestamps(timestamps)
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
        logger.info("tts_speak_lipsynced.speak: getting audio: done")

        # # DEBUG - dump response to audio file
        # audio_buffer.seek(0)
        # with open("temp.mp3", "wb") as audio_file:
        #     audio_file.write(audio_buffer.getvalue())

        # Wait until phonemization background task completes (usually it completes faster than audio, so likely completed already)
        while not phonemes_task_env.done:
            time.sleep(0.01)

        # for record in timestamps:
        #     print(record)  # DEBUG, show word-level timestamps
        # print(phonemes_task_env.phonemess)  # DEBUG, show phoneme sequences

        # Now we have:
        #   - `timestamps`: words, with word-level start and end times
        #   - `phonemes`: phonemes for each word
        #
        # Consolidate data for convenience
        if len(phonemes_task_env.phonemess) != len(timestamps):  # should have exactly one phoneme sequence for each word
            logger.error(f"Number of phoneme sequences ({len(phonemes_task_env.phonemess)}) does not match number of words ({len(timestamps)}), can't lipsync. Use `tts_speak` instead.")
            for record in timestamps:
                print(record)  # DEBUG, show timestamped words
            print(phonemes_task_env.phonemess)  # DEBUG, show phoneme sequences
            assert False

        for timestamp, phonemes in zip(timestamps, phonemes_task_env.phonemess):
            timestamp["phonemes"] = phonemes

        # for record in timestamps:
        #     print(record)  # DEBUG once more, with feeling! (show where each phoneme went)

        # Transform data into phoneme stream with interpolated timestamps
        def get_timestamp_for_phoneme(t0, t1, phonemes, idx):
            """Given word start/end times `t0` and `t1`, linearly interpolate the start/end times for a phoneme in the word."""
            L = len(phonemes)
            rel_start = idx / L
            rel_end = (idx + 1) / L
            dt = t1 - t0
            t_start = t0 + dt * rel_start
            t_end = t0 + dt * rel_end
            return t_start, t_end
        phoneme_stream = []  # [(phoneme0, morph0, t_start, t_end), ...]
        for record in timestamps:
            phonemes = record["phonemes"]
            for idx, phoneme in enumerate(phonemes):  # mˈaɪnd -> m, ˈ, a, ɪ, n, d
                t_start, t_end = get_timestamp_for_phoneme(record["start_time"], record["end_time"], phonemes, idx)
                if phoneme in phoneme_to_morph:  # accept only phonemes we know about
                    phoneme_stream.append((phoneme, phoneme_to_morph[phoneme], t_start, t_end))
        phoneme_start_times = [item[2] for item in phoneme_stream]  # for mapping playback time -> position in phoneme stream
        phoneme_end_times = [item[3] for item in phoneme_stream]  # for mapping playback time -> position in phoneme stream

        # for record in phoneme_stream:
        #     logger.info(f"tts_speak_lipsynced.speak: phoneme data: {record}")  # DEBUG, show final phoneme stream

        # Example of phoneme stream data:
        # [
        #   ('ʃ', 'mouth_ooo_index', 0.275, 0.325),
        #   ('ˈ', '!keep', 0.325, 0.375),
        #   ('ɛ', 'mouth_eee_index', 0.375, 0.425),
        #   ('ɹ', 'mouth_aaa_index', 0.425, 0.475),
        #   ('ə', 'mouth_delta', 0.475, 0.5249999999999999),
        #   ('n', 'mouth_eee_index', 0.5249999999999999, 0.575),
        #   ('ˈ', '!keep', 0.575, 0.635),
        #   ('æ', 'mouth_delta', 0.635, 0.695),
        #   ('p', '!close_mouth', 0.695, 0.755),
        #   ('ᵊ', 'mouth_delta', 0.755, 0.815),
        #   ('l', 'mouth_eee_index', 0.815, 0.875),
        #   ...,
        # }

        # play audio
        logger.info("tts_speak_lipsynced.speak: loading audio into mixer")
        audio_buffer.seek(0)
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.load(audio_buffer)

        def apply_lipsync_at_audio_time(t):
            # Sanity check: don't do anything before the first phoneme.
            if t < phoneme_start_times[0]:
                return

            # Mouth morphs
            overrides = {
                "mouth_aaa_index": 0.0,
                "mouth_eee_index": 0.0,
                "mouth_iii_index": 0.0,
                "mouth_ooo_index": 0.0,
                "mouth_uuu_index": 0.0,
                "mouth_delta": 0.0,
            }

            # Close the mouth if the last phoneme has ended (but the audio stream is still running, likely with silence at the end).
            if t > phoneme_end_times[-1]:
                talkinghead_set_overrides(overrides)
                return

            # Find position in phoneme stream
            idx = bisect.bisect_right(phoneme_start_times, t) - 1
            assert 0 <= idx <= len(phoneme_start_times)

            morph = phoneme_stream[idx][1]
            # print(t, phoneme_stream[idx][0], morph)  # DEBUG (very spammy, 100 messages per second)

            # Set mouth position
            if morph == "!close_mouth":
                talkinghead_set_overrides(overrides)  # set all mouth morphs to zero -> close mouth
            elif morph == "!keep":
                pass  # keep previous mouth position
            elif morph == "!maybe_close_mouth":  # close mouth only if the pause is at least half a second, else act like "!keep".
                phoneme_length = phoneme_end_times[idx] - phoneme_start_times[idx]
                if phoneme_length >= 0.5:
                    talkinghead_set_overrides(overrides)
            else:  # activate one mouth morph, set others to zero
                overrides[morph] = 1.0
                talkinghead_set_overrides(overrides)

        logger.info("tts_speak_lipsynced.speak: starting playback")
        if start_callback is not None:
            try:
                start_callback()
            except Exception as exc:
                logger.error(f"tts_speak_lipsynced.speak: in start callback: {type(exc)}: {exc}")
                traceback.print_exc()
        try:
            playback_start_time = time.time_ns()
            pygame.mixer.music.play()

            latency = api_config.audio_buffer_size / api_config.audio_frequency  # seconds
            while pygame.mixer.music.get_busy():
                # TODO: Lipsync: account for audio playback latency, how?
                t = (time.time_ns() - playback_start_time) / 10**9 - latency - video_offset  # seconds from start of audio
                apply_lipsync_at_audio_time(t)  # lipsync
                time.sleep(0.01)
        finally:
            logger.info("tts_speak_lipsynced.speak: playback finished")

            if stop_callback is not None:
                try:
                    stop_callback()
                except Exception as exc:
                    logger.error(f"tts_speak_lipsynced.speak: in stop callback: {type(exc)}: {exc}")
                    traceback.print_exc()

            # TTS is exiting, so stop lipsyncing.
            talkinghead_set_overrides({})

    api_config.task_manager.submit(speak, envcls())

def tts_stop():
    """Stop the speech synthesizer."""
    if not module_initialized:
        raise RuntimeError("tts_stop: The `raven.avatar.client.api` module must be initialized before using the API.")
    logger.info("tts_stop: stopping audio")
    pygame.mixer.music.stop()

# --------------------------------------------------------------------------------
# Websearch

def websearch_search(query: str, engine: str = "duckduckgo", max_links: int = 10) -> Tuple[str, Dict]:
    """Perform a websearch, using the Avatar server to handle the interaction with the search engine and the parsing of the results page.

    Uses the "/api/websearch2" endpoint on the server, which see.
    """
    if not module_initialized:
        raise RuntimeError("websearch_search: The `raven.avatar.client.api` module must be initialized before using the API.")
    headers = copy.copy(api_config.avatar_default_headers)
    headers["Content-Type"] = "application/json"
    input_data = {"query": query,
                  "engine": engine,
                  "max_links": max_links}
    response = requests.post(f"{api_config.avatar_url}/api/websearch2", headers=headers, json=input_data)
    yell_on_error(response)

    output_data = response.json()
    return output_data

# --------------------------------------------------------------------------------

def selftest():
    """DEBUG/TEST - exercise each of the API endpoints."""
    from colorama import Fore, Style, init as colorama_init
    import PIL.Image
    from . import config as client_config

    colorama_init()

    logger.info("selftest: initialize module")
    init_module(avatar_url=client_config.avatar_url,
                avatar_api_key_file=client_config.avatar_api_key_file,
                tts_url=client_config.tts_url,
                tts_api_key_file=client_config.tts_api_key_file)  # let it create a default executor

    logger.info(f"selftest: check server availability at {client_config.avatar_url}")
    if avatar_available():
        print(f"{Fore.GREEN}{Style.BRIGHT}Connected to avatar server at {client_config.avatar_url}.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{Style.BRIGHT}Proceeding with self-test.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}{Style.BRIGHT}ERROR: Cannot connect to avatar server at {client_config.avatar_url}.{Style.RESET_ALL} Is the avatar server running?")
        print(f"{Fore.RED}{Style.BRIGHT}Canceling self-test.{Style.RESET_ALL}")
        return

    logger.info("selftest: classify_labels")
    print(classify_labels())  # get available emotion names from server

    logger.info("selftest: initialize talkinghead")
    talkinghead_load(os.path.join(os.path.dirname(__file__), "..", "images", "example.png"))  # send an avatar - mandatory
    talkinghead_load_animator_settings_from_file(os.path.join(os.path.dirname(__file__), "..", "animator.json"))  # send animator config - optional, server defaults used if not sent
    talkinghead_load_emotion_templates_from_file(os.path.join(os.path.dirname(__file__), "..", "emotions", "_defaults.json"))  # send the morph parameters for emotions - optional, server defaults used if not sent
    gen = talkinghead_result_feed()  # start receiving animation frames
    talkinghead_start_talking()  # start "talking right now" animation (generic, non-lipsync, random mouth)

    logger.info("selftest: classify")
    text = "What is the airspeed velocity of an unladen swallow?"
    print(classify(text))  # classify some text, auto-update avatar's emotion from result

    # logger.info("selftest: websearch")
    # print(f"{text}\n")
    # out = websearch_search(text, max_links=3)
    # for item in out["data"]:
    #     if "title" in item and "link" in item:
    #         print(f"{item['title']}\n{item['link']}\n")
    #     elif "title" in item:
    #         print(f"{item['title']}\n")
    #     elif "link" in item:
    #         print(f"{item['link']}\n")
    #     print(f"{item['text']}\n")
    # # There's also out["results"] with preformatted text only.

    logger.info("selftest: embeddings")
    try:
        print(embeddings_compute(text).shape)  # needs `raven.avatar.server.app` to be running with the "--embeddings" command-line option
        print(embeddings_compute([text, "Testing, 1, 2, 3."]).shape)
    except RuntimeError as exc:
        logger.error(f"selftest: Failed to call `raven.avatar.server`'s `embeddings` module. If the error is a 403, the module likely isn't running. {type(exc)}: {exc}")

    logger.info("selftest: more talkinghead tests")
    talkinghead_set_emotion("surprise")  # manually update emotion
    for _ in range(5):  # get a few frames
        image_format, image_data = next(gen)  # next-gen lol
        print(image_format, len(image_data))
        image_file = io.BytesIO(image_data)
        image = PIL.Image.open(image_file)  # noqa: F841, we're only interested in testing whether the transport works.
    talkinghead_stop_talking()  # stop "talking right now" animation
    talkinghead_stop()  # pause animating the talkinghead
    talkinghead_start()  # resume animating the talkinghead
    gen.close()  # close the connection

    logger.info("selftest: all done")

if __name__ == "__main__":
    selftest()
