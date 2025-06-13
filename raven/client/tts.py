"""TTS (text-to-speech, speech synthesizer) Python bindings of Raven's web API.

This module coordinates lipsync between the TTS and the avatar, and implements the actual client-side audio output.
"""

__all__ = ["tts_list_voices",
           "tts_speak",
           "tts_speak_lipsynced",
           "tts_stop"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import bisect
import copy
import io
import json
import re
import requests
import time
import traceback
from typing import Callable, List, Optional
import urllib.parse

import pygame  # for audio (text to speech) support

from unpythonic import timer
from unpythonic.env import env as envcls

from . import api  # for calling the avatar_* functions during lipsync
from . import util

def tts_list_voices() -> List[str]:
    """Return a list of voice names supported by the TTS endpoint (if the endpoint is available).

    Return format is::

        [voice0, ...]
    """
    if not util.api_initialized:
        raise RuntimeError("tts_list_voices: The `raven.client.api` module must be initialized before using the API.")
    if util.api_config.tts_url is None:
        return []
    headers = copy.copy(util.api_config.tts_default_headers)
    if util.api_config.tts_server_type == "kokoro":
        response = requests.get(f"{util.api_config.tts_url}/v1/audio/voices", headers=headers)
    else:  # util.api_config.tts_server_type == "raven"
        response = requests.get(f"{util.api_config.tts_url}/api/tts/list_voices", headers=headers)
    util.yell_on_error(response)
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
    if not util.api_initialized:
        raise RuntimeError("tts_speak: The `raven.client.api` module must be initialized before using the API.")
    if util.api_config.tts_url is None:
        return
    headers = copy.copy(util.api_config.tts_default_headers)
    headers["Content-Type"] = "application/json"

    # We run this in the background
    def speak(task_env) -> None:
        logger.info("tts_speak.speak: getting audio")

        if util.api_config.tts_server_type == "kokoro":  # https://github.com/remsky/Kokoro-FastAPI
            data = {"model": "kokoro",
                    "voice": voice,
                    "input": text,
                    "response_format": "mp3",  # flac would be nice (small, fast to encode), but seems currently broken in kokoro (the audio may repeat twice).
                    "speed": speed,
                    "stream": True,
                    "return_download_link": False}
            stream_response = requests.post(f"{util.api_config.tts_url}/v1/audio/speech", headers=headers, json=data, stream=True)
        else:  # util.api_config.tts_server_type == "raven"
            data = {"voice": voice,
                    "text": text,
                    "format": "mp3",
                    "speed": speed,
                    "stream": True,
                    "get_metadata": False}
            stream_response = requests.post(f"{util.api_config.tts_url}/api/tts/speak", headers=headers, json=data, stream=True)
        util.yell_on_error(stream_response)

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
    util.api_config.task_manager.submit(speak, envcls())

def tts_speak_lipsynced(instance_id: str,
                        voice: str,
                        text: str,
                        speed: float = 1.0,
                        video_offset: float = 0.0,
                        start_callback: Optional[Callable] = None,
                        stop_callback: Optional[Callable] = None) -> None:
    """Like `tts_speak`, but with lipsync for the avatar.

    `instance_id`: The avatar instance ID you got from `raven.client.api.avatar_load`.
                   Which avatar instance to lipsync.

    `video_offset`: seconds, for adjusting lipsync animation.
        - Positive values: Use if the video is early. Shifts video later with respect to the audio.
        - Negative values: Use if the video is late. Shifts video earlier with respect to the audio.

    See:
        https://github.com/remsky/Kokoro-FastAPI
    """
    if not util.api_initialized:
        raise RuntimeError("tts_speak_lipsynced: The `raven.client.api` module must be initialized before using the API.")
    if util.api_config.tts_url is None:
        return
    headers = copy.copy(util.api_config.tts_default_headers)
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
    #   - For more fine-grained control, we could use the morph value channel too, not just always set the morph to 1.0. OTOH the pose animation smooths this out.
    #   - Could also combine morphs for a single phoneme if needed (use a list).
    #
    # value: animator morph name, or one of the special commands:
    #   "!close_mouth" - as it says on the tin
    #   "!keep" - keep previous mouth position
    #   "!maybe_close_mouth" - close mouth only if the pause is at least half a second, else act like "!keep".
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

    # TODO: Kokoro-FastAPI doesn't support returning phonemes with word-level timestamps, so we jury-rig this. It often works, but not always (e.g. a year number produces one word, but multiple phoneme sequences).
    #
    # The robust solution is to use a local Kokoro installation (`tts_server_type="raven"`), which allows us to get word-level phonemes with the same word boundaries as in the timestamps.
    # But that makes Raven harder to install due to Kokoro's `espeak-ng` dependency, so we retain the option to use Kokoro-FastAPI.
    if util.api_config.tts_server_type == "kokoro":
        # Phonemize and word-level timestamping treat underscores differently: phonemize treats them as spaces,
        # whereas word-level timestamping doesn't (no word split at underscore). Better to remove them.
        def prefilter(text):
            return text.replace("_", " ")
        text = prefilter(text)

        def get_phonemes(task_env) -> None:
            with timer() as tim:
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
                response = requests.post(f"{util.api_config.tts_url}/dev/phonemize", headers=headers, json=data, stream=True)
                util.yell_on_error(response)
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
            logger.info(f"tts_speak_lipsynced.get_phonemes: done in {tim.dt:0.6g}s.")
        phonemes_task_env = envcls(done=False)
        util.api_config.task_manager.submit(get_phonemes, phonemes_task_env)
    # When `util.api_config.tts_server_type == "raven"`, we'll get the phonemes in the metadata, no need for a separate fetch.

    def speak(task_env) -> None:
        logger.info("tts_speak_lipsynced.speak: starting")
        def isword(s):
            return len(s) > 1 or s.isalnum()

        def clean_timestamps(timestamps):
            """Remove consecutive duplicate timestamps (some versions of Kokoro-FastAPI produce those) and any timestamps for punctuation."""
            out = []
            last_start_time = None
            for record in timestamps:  # format: [{"word": "blah", "start_time": 1.23, "end_time": 1.45}, ...]
                if record["start_time"] != last_start_time and isword(record["word"]):
                    out.append(record)
                    last_start_time = record["start_time"]
            return out

        # Get audio and word timestamps
        with timer() as tim:
            if util.api_config.tts_server_type == "kokoro":
                logger.info("tts_speak_lipsynced.speak: getting audio with word-level timestamps")
                data = {"model": "kokoro",
                        "voice": voice,
                        "input": text,
                        "response_format": "mp3",
                        "speed": speed,
                        "stream": True,
                        "return_timestamps": True}
                stream_response = requests.post(f"{util.api_config.tts_url}/dev/captioned_speech", headers=headers, json=data, stream=True)
            else:  # util.api_config.tts_server_type == "raven"
                logger.info("tts_speak_lipsynced.speak: getting audio with word-level timestamps and phonemes")
                data = {"voice": voice,
                        "text": text,
                        "format": "mp3",
                        "speed": speed,
                        "stream": True,
                        "get_metadata": True}
                stream_response = requests.post(f"{util.api_config.tts_url}/api/tts/speak", headers=headers, json=data, stream=True)
            util.yell_on_error(stream_response)

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
        logger.info(f"tts_speak_lipsynced.speak: getting audio: done in {tim.dt:0.6g}s.")

        # # DEBUG - dump response to audio file
        # audio_buffer.seek(0)
        # with open("temp.mp3", "wb") as audio_file:
        #     audio_file.write(audio_buffer.getvalue())

        if util.api_config.tts_server_type == "kokoro":
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
                logger.error(f"tts_speak_lipsynced.speak: Number of phoneme sequences ({len(phonemes_task_env.phonemess)}) does not match number of words ({len(timestamps)}), can't lipsync. Use `tts_speak` instead.")
                for record in timestamps:
                    print(record)  # DEBUG, show timestamped words
                print(phonemes_task_env.phonemess)  # DEBUG, show phoneme sequences
                assert False

            for timestamp, phonemes in zip(timestamps, phonemes_task_env.phonemess):
                timestamp["phonemes"] = phonemes
        else:  # util.api_config.tts_server_type == "raven"
            # The timestamp metadata contains the phonemes too.
            # But they're URL-encoded to ASCII with percent-escaped UTF-8, due to HTTP limitations (can't send Unicode in HTTP headers).
            # Also the words, just to be safe.
            for timestamp in timestamps:
                timestamp["word"] = urllib.parse.unquote(timestamp["word"])
                timestamp["phonemes"] = urllib.parse.unquote(timestamp["phonemes"])
        # The rest of the code is identical regardless of TTS server type.

        logger.info("tts_speak_lipsynced.speak: timestamped phonemes:")
        for record in timestamps:
            logger.info(f"    {record}")  # DEBUG once more, with feeling! (show timestamps, with phoneme data)

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
                api.avatar_set_overrides(instance_id, overrides)
                return

            # Find position in phoneme stream
            idx = bisect.bisect_right(phoneme_start_times, t) - 1
            assert 0 <= idx <= len(phoneme_start_times)

            morph = phoneme_stream[idx][1]
            # print(t, phoneme_stream[idx][0], morph)  # DEBUG (very spammy, 100 messages per second)

            # Set mouth position
            if morph == "!close_mouth":
                api.avatar_set_overrides(instance_id, overrides)  # set all mouth morphs to zero -> close mouth
            elif morph == "!keep":
                pass  # keep previous mouth position
            elif morph == "!maybe_close_mouth":  # close mouth only if the pause is at least half a second, else act like "!keep".
                phoneme_length = phoneme_end_times[idx] - phoneme_start_times[idx]
                if phoneme_length >= 0.5:
                    api.avatar_set_overrides(instance_id, overrides)
            else:  # activate one mouth morph, set others to zero
                overrides[morph] = 1.0
                api.avatar_set_overrides(instance_id, overrides)

        logger.info("tts_speak_lipsynced.speak: starting playback")
        if start_callback is not None:
            try:
                start_callback()
            except Exception as exc:
                logger.error(f"tts_speak_lipsynced.speak: in start callback: {type(exc)}: {exc}")
                traceback.print_exc()
        try:
            latency = util.api_config.audio_buffer_size / util.api_config.audio_frequency  # seconds
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                t = pygame.mixer.music.get_pos() / 1000 - latency - video_offset
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
            api.avatar_set_overrides(instance_id, {})

    util.api_config.task_manager.submit(speak, envcls())

def tts_stop():
    """Stop the speech synthesizer."""
    if not util.api_initialized:
        raise RuntimeError("tts_stop: The `raven.client.api` module must be initialized before using the API.")
    if util.api_config.tts_url is None:
        return
    logger.info("tts_stop: stopping audio")
    pygame.mixer.music.stop()
