"""Python client-side API for Talkinghead. This talks with the server through the web API so you don't have to.

We support:

  - classify
  - talkinghead
  - TTS, via https://github.com/remsky/Kokoro-FastAPI

This module is licensed under the 2-clause BSD license, to facilitate Talkinghead integration anywhere.
"""

__all__ = ["init_module",
           "classify_labels", "classify",
           "talkinghead_load", "talkinghead_unload", "talkinghead_reload",
           "talkinghead_load_emotion_templates", "talkinghead_load_emotion_templates_from_file",
           "talkinghead_load_animator_settings", "talkinghead_load_animator_settings_from_file",
           "talkinghead_start_talking", "talkinghead_stop_talking",
           "talkinghead_set_emotion",
           "talkinghead_result_feed",
           "tts_available", "tts_voices",
           "tts_speak", "tts_speak_lipsynced", "tts_stop"]

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

def talkinghead_set_overrides(data: Dict[str, float]) -> None:
    headers = copy.copy(config.default_headers)
    headers["Content-Type"] = "application/json"
    response = requests.post(f"{config.avatar_url}/api/talkinghead/set_overrides", json=data, headers=headers)
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
        logger.error(f"tts_available: {type(exc)}: {exc}")
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
              speed: float = 1.0,
              start_callback: Optional[Callable] = None,
              stop_callback: Optional[Callable] = None) -> None:
    """Using the speech synthesizer, speak `text` using `voice`.

    If `start_callback` is provided, call it when the TTS starts speaking.
    If `stop_callback` is provided, call it when the TTS has stopped speaking.
    """
    if config.tts_url is None:
        return
    headers = copy.copy(config.tts_default_headers)
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
        stream_response = requests.post(f"{config.tts_url}/v1/audio/speech", headers=headers, json=data, stream=True)
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
    config.task_manager.submit(speak, envcls())

def tts_speak_lipsynced(voice: str,
                        text: str,
                        speed: float = 1.0,
                        lipsync_offset: float = -0.4,
                        start_callback: Optional[Callable] = None,
                        stop_callback: Optional[Callable] = None) -> None:
    """Like `tts_speak`, but with lip sync.

    Requires the Kokoro-FastAPI TTS backend so that we can get the phoneme data
    and timestamps.

    `lipsync_offset`: seconds. Positive values delay the animation (with respect to the audio).

    See:
        https://github.com/remsky/Kokoro-FastAPI
    """
    if config.tts_url is None:
        return
    headers = copy.copy(config.tts_default_headers)
    headers["Content-Type"] = "application/json"

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
        "f": "mouth_iii_index",
        "h": "mouth_delta",
        "j": "mouth_aaa_index",  # As in yes => jˈɛs.
        "k": "mouth_aaa_index",
        "l": "mouth_iii_index",
        "m": "!close_mouth",
        "n": "mouth_iii_index",
        "p": "!close_mouth",
        "s": "mouth_iii_index",
        "t": "mouth_iii_index",
        "w": "mouth_ooo_index",
        "z": "mouth_iii_index",
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
        "ɾ": "mouth_iii_index",  # A sound somewhere in between t and d, like butter => bˈʌɾəɹ.
        # 🇬🇧 British-only
        "a": "mouth_aaa_index",  # The vowel sound at the start of ash => ˈaʃ.
        "ɒ": "mouth_ooo_index",  # The sound at the start of on => ˌɒn. Easy to confuse with ɑ, which is a shared phoneme.
        # Other
        "ː": "!keep",  # Vowel extender, visually looks similar to a colon. Possibly dubious, because Americans extend vowels too, but the gold US dictionary somehow lacks these. Often used by the Brits instead of ɹ: Americans say or => ɔɹ, but Brits say or => ɔː.
        # Stress Marks
        "ˈ": "!keep",  # Primary stress, visually looks similar to an apostrophe (but is U+02C8).
        "ˌ": "!keep",  # Secondary stress (not a comma, but U+02CC).
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
        response = requests.post(f"{config.tts_url}/dev/phonemize", headers=headers, json=data, stream=True)
        yell_on_error(response)
        response_json = response.json()
        phonemes = response_json["phonemes"]
        for dipthong, ipa_expansion in dipthong_vowel_to_ipa.items():
            phonemes = phonemes.replace(dipthong, ipa_expansion)
        phonemess = phonemes.split()  # -> [word0, word1, ...]
        task_env.phonemess = phonemess
        task_env.done = True
        logger.info("tts_speak_lipsynced.get_phonemes: done")
    phonemes_task_env = envcls(done=False)
    config.task_manager.submit(get_phonemes, phonemes_task_env)

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
        stream_response = requests.post(f"{config.tts_url}/dev/captioned_speech", headers=headers, json=data, stream=True)
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

        # Now we have:
        #   - `timestamps`: words, with word-level start and end times
        #   - `phonemes`: phonemes for each word
        #
        # Consolidate data for convenience
        for timestamp, phonemes in zip(timestamps, phonemes_task_env.phonemess):
            timestamp["phonemes"] = phonemes

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
                if phoneme in phoneme_to_morph:
                    phoneme_stream.append((phoneme, phoneme_to_morph[phoneme], t_start, t_end))
                else:  # e.g. punctuation
                    phoneme_stream.append((phoneme, "!close_mouth", t_start, t_end))
        phoneme_start_times = [item[2] for item in phoneme_stream]  # for mapping playback time -> position in phoneme stream
        phoneme_end_times = [item[3] for item in phoneme_stream]  # for mapping playback time -> position in phoneme stream

        # Example of phoneme stream data:
        # [
        #   ('ʃ', 'mouth_ooo_index', 0.4, 0.45416666666666666),
        #   ('ˈ', '!keep', 0.45416666666666666, 0.5083333333333333),
        #   ('ɛ', 'mouth_eee_index', 0.5083333333333333, 0.5625),
        #   ('ɹ', 'mouth_aaa_index', 0.5625, 0.6166666666666667),
        #   ('ə', 'mouth_delta', 0.6166666666666667, 0.6708333333333334),
        #   ('n', 'mouth_iii_index', 0.6708333333333334, 0.725),
        #   ('ˈ', '!keep', 0.725, 0.8125),
        #   ('æ', 'mouth_delta', 0.8125, 0.8999999999999999),
        #   ('p', '!close_mouth', 0.8999999999999999, 0.9875),
        #   ('ᵊ', 'mouth_delta', 0.9875, 1.075),
        #   ('l', 'mouth_iii_index', 1.075, 1.1625),
        #   ('.', '!close_mouth', 1.1625, 1.25),
        #   ...,
        # }
        #
        # for record in phoneme_stream:
        #     print(record)  # DEBUG

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

            # Set mouth position
            if morph == "!close_mouth":
                talkinghead_set_overrides(overrides)  # set all mouth morphs to zero
            elif morph == "!keep":
                pass  # keep previous mouth position
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

            while pygame.mixer.music.get_busy():
                # TODO: Lipsync: account for audio playback latency, how?
                t = (time.time_ns() - playback_start_time) / 10**9 - lipsync_offset  # seconds from start of audio
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

    config.task_manager.submit(speak, envcls())

def tts_stop():
    """Stop the speech synthesizer."""
    logger.info("tts_stop: stopping audio")
    pygame.mixer.music.stop()
