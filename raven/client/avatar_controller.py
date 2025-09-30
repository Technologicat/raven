"""Avatar TTS (text to speech) and subtitling system controller.

Also starts/stops the avatar's "data eyes" effect (LLM tool access indicator).

Contrast `avatar_renderer`, which is concerned with blitting the avatar video into the GUI.

This takes in text to be spoken by the TTS and optionally subtitled.

The text is sent into an input queue, which is processed by a background task. First, the text is stripped of Markdown and emoji.
Then the text is split into sentences (using spaCy), which are then translated via Raven-server's translator (if subtitles are enabled).
It is also possible to produce closed captions (CC), i.e. subtitles with no translation.

Finally, TTS audio and phonemes are precomputed, and the result goes into an output queue, one item per sentence.

Another background task reads this output queue and controls the TTS playback and showing/hiding the subtitles.

That second background task also takes care of auto-resetting the avatar's emotion back to neutral, if the avatar is idle
for at least a few seconds after the TTS has finished speaking.
"""

__all__ = ["initialize",
           "shutdown",
           "update_emotion_from_text",
           "send_text_to_tts",
           "stop_tts",
           "start_data_eyes", "stop_data_eyes",
           "avatar_controller_config"]  # global configuration object; write into this to change settings on the fly (e.g. subtitles enabled/disabled)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import concurrent.futures
import functools
import queue
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

import emoji
import strip_markdown

import dearpygui.dearpygui as dpg

from unpythonic import equip_with_traceback, gensym, slurp
from unpythonic.env import env

from ..common import bgtask
from ..common import numutils

from ..common.gui import animation as gui_animation
from ..common.gui import utils as guiutils

from . import api  # Raven-server support

# --------------------------------------------------------------------------------

tts_input_queue = queue.Queue()  # for TTS input preprocessing and subtitle generation; see `send_text_to_tts`
tts_output_queue = queue.Queue()  # for TTS and subtitle playback; see `preprocess_task`

emotion_autoreset_t0 = time.time_ns()  # emotion autoreset is handled by `speak_task`
tts_idle_check_t0 = time.time_ns()

# --------------------------------------------------------------------------------
# For CPU-friendliness, LRU-cache all AI-heavy parts (for the "speak again" feature).
#
# NOTE: `raven.client.tts.tts_prepare` also LRU caches its results internally.

@functools.lru_cache(maxsize=128)
def _avatar_get_emotion_from_text(text: str) -> str:
    """Internal helper for computing avatar's emotion from text."""
    try:
        if not text:
            return "neutral"
        detected_emotions = api.classify(text)  # -> `{emotion0: score0, ...}`, sorted by score, descending
        filtered_emotions = [emotion_name for emotion_name in detected_emotions.keys() if emotion_name not in avatar_controller_config.emotion_blacklist]
        winning_emotion = filtered_emotions[0]
        return winning_emotion
    except Exception:
        return "neutral"

@functools.lru_cache(maxsize=128)
def _translate_sentence(sentence: str) -> str:
    """Internal helper for subtitle translation with LRU caching."""
    subtitle = api.translate_translate(sentence,
                                       source_lang=avatar_controller_config.translator_source_lang,
                                       target_lang=avatar_controller_config.translator_target_lang)
    return subtitle

@functools.lru_cache(maxsize=128)
def _natlang_analyze(line: str) -> List[List["spacy.tokens.token.Token"]]:  # noqa: F821: we don't want to import (torch and) spaCy just for one type annotation.
    """Internal helper for natural-language translation with LRU caching."""
    docs = api.natlang_analyze(line,
                               pipes=["tok2vec", "parser", "senter"])
    return docs

# --------------------------------------------------------------------------------
# API

avatar_controller_initialized = False
avatar_controller_config = env()
def initialize(avatar_instance_id: str,
               emotion_autoreset_interval: Optional[float],
               emotion_blacklist: List[str],
               data_eyes_fadeout_duration: float,
               stop_tts_button_gui_widget: Optional[Union[str, int]],
               on_tts_idle: Optional[Callable],
               tts_idle_check_interval: Optional[float],
               subtitles_enabled: bool,
               subtitle_text_gui_widget: Optional[Union[str, int]],
               subtitle_left_x0: int,
               subtitle_bottom_y0: int,
               translator_source_lang: str,
               translator_target_lang: Optional[str],
               main_window_w: int,
               main_window_h: int,
               executor: Optional = None) -> None:
    """Avatar TTS (text to speech) and subtitling system controller.

    Call **after** your app's GUI is alive.

    `avatar_instance_id`: Avatar instance to control. You get this from `raven.client.api.avatar_load`.

    `emotion_autoreset_interval`: seconds. This controller resets the avatar's emotion back to "neutral"
                                  after a few seconds of idle time (when the TTS is not speaking).

                                  Set to `None` to disable.

    `emotion_blacklist`: Prevent the avatar from entering any of the listed emotions.
                         In `update_emotion_from_text`, the most matching non-blacklisted emotion wins.

                         Can be useful if the emotion detector is misbehaving.

    `data_eyes_fadeout_duration`: seconds; how long it takes for the "data eyes" effect
                                  (LLM tool access indicator) to fade out when `stop_data_eyes`
                                  is called.

                                  Calling `start_data_eyes` always sets the effect to full strength
                                  instantly.

    `stop_tts_button_gui_widget`: DPG tag or ID of the DPG button widget that will call `stop_tts`
                                  if clicked. Used for automatically enabling/disabling the button
                                  depending on the TTS state (speaking / not speaking).

                                  Set this to `None` to disable the feature.

    `on_tts_idle`: 0-argument callable. Called periodically when the TTS is not speaking.
                   The return value is ignored.

                   This can be used to trigger additional GUI actions when the avatar stops speaking.

                   Note, however, that this will be called again every `tts_idle_check_interval`
                   as long as the avatar is not speaking, so the actions should be idempotent
                   (i.e. have no further effect if called more than once).

    `tts_idle_check_interval`: seconds. How often to check whether TTS has become idle,
                               and trigger `on_tts_idle` if it has.

                               Set to `None` to disable.

    `subtitles_enabled`: Whether subtitles are initially enabled when the module starts.
                         To change the status later, just write to
                         `avatar_controller_config.subtitles_enabled: bool`.

    `subtitle_text_gui_widget`: DPG tag or ID of the DPG text widget to send the subtitle text to.
                                The widget can start hidden - we will show/hide it automatically.

                                If `subtitles_enabled=False` and you intend to keep it that way
                                (i.e. don't intend to use subtitles), you can set this to `None`.

    `subtitle_left_x0`: Left edge of subtitle text, pixels. Used for re-positioning the text widget.

    `subtitle_bottom_y0`: Bottom edge of subtitle text, pixels. Used for re-positioning the text widget.
                          Whenever a subtitle appears, the text widget is re-positioned dynamically,
                          accounting for the rendered size of the text.

    `translator_source_lang`: For subtitling. Language code for the source language of the input text,
                              assumed monolingual. Usually "en", for English.
    `translator_target_lang`: For subtitling. Language code for the subtitle language.

                              What is available depends on what language pairs you have configured
                              Raven-server's `translate` module for.

                              Use the special value `None` for no translation, i.e. to replace subtitles
                              with closed captions (CC) in the source language.

    `main_window_w`: Width of main window, in pixels. Used for temporarily positioning the subtitle
                     offscreen for rendered size measurement.

    `main_window_h`: Height of main window, in pixels. Used for temporarily positioning the subtitle
                     offscreen for rendered size measurement.

    `executor`: A `ThreadPoolExecutor` or something duck-compatible with it.
                Used for running the background tasks.
    """
    global avatar_controller_initialized

    # HACK: Here it is very useful to know where the call came from, to debug mysterious extra initializations (since only the settings sent the first time will take).
    dummy_exc = Exception()
    dummy_exc = equip_with_traceback(dummy_exc, stacklevel=2)  # 2 = ignore `equip_with_traceback` itself, and its caller, i.e. us
    tb = traceback.extract_tb(dummy_exc.__traceback__)
    top_frame = tb[-1]
    called_from = f"{top_frame[0]}:{top_frame[1]}"  # e.g. "/home/xxx/foo.py:52"
    logger.info(f"initialize: called from: {called_from}")

    if avatar_controller_initialized:  # initialize only once
        logger.info("initialize: `raven.client.avatar_controller` is already initialized. Using existing initialization.")
        return

    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor()
    # Use separate task managers (but the same thread pool) so that we can easily power-cycle a component when needed.
    # TODO: This takes two slots in the thread pool for the whole duration of the app. Consider the implications.
    avatar_controller_config.input_queue_task_manager = bgtask.TaskManager(name="avatar_controller_input_queue",
                                                                           mode="concurrent",
                                                                           executor=executor)
    avatar_controller_config.output_queue_task_manager = bgtask.TaskManager(name="avatar_controller_output_queue",
                                                                            mode="concurrent",
                                                                            executor=executor)

    avatar_controller_config.avatar_instance_id = avatar_instance_id
    avatar_controller_config.emotion_autoreset_interval = emotion_autoreset_interval
    avatar_controller_config.emotion_blacklist = emotion_blacklist
    avatar_controller_config.data_eyes_fadeout_duration = data_eyes_fadeout_duration
    avatar_controller_config.stop_tts_button_gui_widget = stop_tts_button_gui_widget
    avatar_controller_config.on_tts_idle = on_tts_idle
    avatar_controller_config.tts_idle_check_interval = tts_idle_check_interval
    avatar_controller_config.subtitles_enabled = subtitles_enabled
    avatar_controller_config.subtitle_text_gui_widget = subtitle_text_gui_widget
    avatar_controller_config.subtitle_left_x0 = subtitle_left_x0
    avatar_controller_config.subtitle_bottom_y0 = subtitle_bottom_y0
    avatar_controller_config.translator_source_lang = translator_source_lang
    avatar_controller_config.translator_target_lang = translator_target_lang
    avatar_controller_config.main_window_w = main_window_w
    avatar_controller_config.main_window_h = main_window_h

    avatar_controller_config.gui_alive = True  # for app shutdown

    # Start background tasks
    avatar_controller_config.input_queue_task_manager.submit(preprocess_task, env())
    avatar_controller_config.output_queue_task_manager.submit(speak_task, env())

    avatar_controller_initialized = True

def shutdown() -> None:
    """Prepare module for app shutdown.

    This signals the background tasks to exit.
    """
    avatar_controller_config.gui_alive = False  # shutting down, GUI updates no longer safe
    avatar_controller_config.input_queue_task_manager.clear(wait=True)
    avatar_controller_config.output_queue_task_manager.clear(wait=True)

def update_emotion_from_text(text: str) -> str:
    """Update the emotion for the AI avatar from `text`, and reset the emotion autoreset (return-to-neutral) timer.

    The analysis results are LRU cached (cache size 128) to facilitate running also on CPU setups, where the analysis can be slow,
    so that switching back and forth between the same AI messages won't cause slowdowns.

    For convenience, return the name of the emotion.
    """
    global emotion_autoreset_t0
    try:
        emotion = _avatar_get_emotion_from_text(text)
        logger.info(f"update_emotion_from_text: updating emotion to '{emotion}'")
        api.avatar_set_emotion(instance_id=avatar_controller_config.avatar_instance_id,
                               emotion_name=emotion)
        logger.info("update_emotion_from_text: emotion updated")
        return emotion
    finally:
        # Reset the timer last. If running on CPU, the emotion analysis may be slow.
        emotion_autoreset_t0 = time.time_ns()

def send_text_to_tts(text: str,
                     voice: str,
                     voice_speed: float,
                     video_offset: float,
                     on_audio_ready: Optional[Callable] = None,
                     on_start_speaking: Optional[Callable] = None,
                     on_stop_speaking: Optional[Callable] = None,
                     on_start_sentence: Optional[Callable] = None,
                     on_stop_sentence: Optional[Callable] = None) -> str:
    """Send a complete piece of text into the TTS queue.

    Returns a batch UUID. This is used in the events to identify which call to `send_text_to_tts`
    the triggered event refers to.

    `voice`: TTS voice name. To get the list of available voices, call `raven.client.api.tts_list_voices`,
             or use the `raven-avatar-settings-editor` GUI app.

    `voice_speed`: For each voice, 1.0 is the default speed the voice is designed to speak at.
                   Raising this too high may cause skipped words.

    `video_offset`: seconds, for adjusting lipsync animation.
        - Positive values: Use if the video is early. Shifts video later with respect to the audio.
        - Negative values: Use if the video is late. Shifts video earlier with respect to the audio.

    `on_audio_ready`: The TTS audio for a sentence is ready.
                      Expected to take arguments:
                          `(output_record: Dict[str, Any], audio_data: bytes)`.
                      The return value is ignored.

                      Useful for saving the audio to disk, pre-split into sentences.

                      Because we precompute the TTS audio as soon as possible, `on_audio_ready`
                      may trigger long before the sentence is actually spoken out loud.

    `on_start_speaking`: The TTS is about to start speaking this batch.
                         Expected to take one argument: `output_record: Dict[str, Any]`.
                         The return value is ignored.

    `on_stop_speaking`: The TTS is done speaking this batch.
                        Expected to take one argument: `output_record: Dict[str, Any]`.
                        The return value is ignored.

    `on_start_sentence`: The TTS is about to start speaking a sentence.
                         Expected to take one argument: `output_record: Dict[str, Any]`.
                         The return value is ignored.

                         Useful mainly if you are recording avatar video, so that your
                         event handler can note down the video frame number and/or timestamp.

    `on_stop_sentence`: The TTS is done speaking a sentence.
                        Expected to take one argument: `output_record: Dict[str, Any]`.
                        The return value is ignored.

                        Useful mainly if you are recording avatar video, so that your
                        event handler can note down the video frame number and/or timestamp.

    For the content of `output_record`, the authoritative source is the source code of
    `preprocess_task.process_item`, which generates them. Generally, you can identify
    the batch and the sentence from there, and it also has a copy of most of the arguments
    that you passed to `send_text_to_tts`. (`text` is named `batch_text`; and `on_audio_ready`
    is gone, because it has already been handled by that point.)
    """
    batch_uuid = str(gensym("tts_job"))
    logger.info("send_text_to_tts: adding text to TTS queue, batch {batch_uuid}.")
    tts_input_queue.put({"batch_uuid": batch_uuid,
                         "batch_text": text,
                         "voice": voice,
                         "voice_speed": voice_speed,
                         "video_offset": video_offset,
                         "on_audio_ready": on_audio_ready,
                         "on_start_speaking": on_start_speaking,
                         "on_stop_speaking": on_stop_speaking,
                         "on_start_sentence": on_start_sentence,
                         "on_stop_sentence": on_stop_sentence})
    return batch_uuid

def stop_tts() -> None:
    """Stop the TTS, clearing also all speech pending in the queues.

    This triggers the `on_stop_speaking` event of the current batch.
    """
    logger.info("stop_tts: entered.")
    # Clear TTS input preprocess queue, so that no new preprocess jobs start.
    logger.info("stop_tts: clearing TTS input preprocess queue.")
    slurp(tts_input_queue)
    # Power-cycle the TTS input preprocessor, to cancel the current job.
    logger.info("stop_tts: power-cycling TTS input preprocessor.")
    avatar_controller_config.input_queue_task_manager.clear()
    avatar_controller_config.input_queue_task_manager.submit(preprocess_task, env())
    # Then clear the output queue, so that no new speak jobs start.
    logger.info("stop_tts: clearing TTS playback queue.")
    slurp(tts_output_queue)
    # Power-cycle the TTS playback controller, to cancel the current job.
    logger.info("stop_tts: power-cycling TTS playback controller.")
    avatar_controller_config.output_queue_task_manager.clear()
    avatar_controller_config.output_queue_task_manager.submit(speak_task, env())
    # We must still stop the TTS, to actually make the TTS playback task exit.
    # The current TTS task will end, and the old controller will then exit.
    logger.info("stop_tts: stopping TTS.")
    api.tts_stop()
    logger.info("stop_tts: all done.")

# --------------------------------------------------------------------------------
# Integration with avatar's "data eyes" effect (LLM tool access indicator)

# We use `avatar_modify_overrides` instead of `avatar_set_overrides` to control just the "data1" cel blend; hence the TTS can still override the mouth morphs simultaneously.

class DataEyesFadeOut(gui_animation.Animation):
    def __init__(self, duration: float):
        """Animation to fade out the avatar data eyes effect.

        `duration`: Fade duration, seconds.
        """
        super().__init__()
        self.duration = duration

    def render_frame(self, t):
        dt = (t - self.t0) / 10**9  # seconds since t0
        animation_pos = dt / self.duration
        if animation_pos >= 1.0:
            api.avatar_modify_overrides(avatar_controller_config.avatar_instance_id, action="unset", overrides={"data1": 0.0})  # Values are ignored by the "unset" action, which removes the overrides.
            return gui_animation.action_finish

        r = numutils.clamp(animation_pos)
        r = numutils.nonanalytic_smooth_transition(r)
        api.avatar_modify_overrides(avatar_controller_config.avatar_instance_id, action="set", overrides={"data1": 1.0 - r})

        return gui_animation.action_continue

_data_eyes_state = False
_data_eyes_state_lock = threading.RLock()
_data_eyes_fadeout_animation = None
def start_data_eyes() -> None:
    """Start the scifi "data eyes" effect (LLM tool access indicator), if the current character supports it."""
    global _data_eyes_state
    global _data_eyes_fadeout_animation
    with _data_eyes_state_lock:
        if _data_eyes_state:  # no-op if already active
            return
        if _data_eyes_fadeout_animation is not None:  # cancel latest fadeout animation if any (no-op if it's no longer running)
            gui_animation.animator.cancel(_data_eyes_fadeout_animation)
        api.avatar_modify_overrides(avatar_controller_config.avatar_instance_id, action="set", overrides={"data1": 1.0})
        _data_eyes_state = True

def stop_data_eyes() -> None:
    """Stop the scifi "data eyes" effect (LLM tool access indicator), if the current character supports it.

    The effect fades out as configured in `initialize`.
    """
    global _data_eyes_state
    global _data_eyes_fadeout_animation
    with _data_eyes_state_lock:
        if not _data_eyes_state:  # no-op (no fadeout animation!) if not active
            return
        if _data_eyes_fadeout_animation is not None:  # cancel latest previous instance if any (no-op if it's no longer running)
            gui_animation.animator.cancel(_data_eyes_fadeout_animation)
        _data_eyes_fadeout_animation = gui_animation.animator.add(DataEyesFadeOut(duration=avatar_controller_config.data_eyes_fadeout_duration))
        _data_eyes_state = False

# --------------------------------------------------------------------------------
# Background task: TTS input preprocessor

def preprocess_task(task_env: env) -> None:
    """Preprocess text for TTS (AI speech synthesizer) from the TTS input preprocess queue.

    The text should be a whole chat message (without the role name), or at least a complete paragraph.

    The text is cleaned, and then split into sentences. Depending on settings passed to `initialize`,
    each sentence may be translated for subtitling, closed-captioned (CC) as-is, or the subtitler may be skipped.

    Finally, the TTS audio and phonemes are precomputed.

    The resulting clean sentence, its possible subtitle, and the precomputed data, are added to the TTS playback queue.
    """
    logger.info(f"preprocess_task: instance {task_env.task_name}: TTS input preprocessor starting")

    def strip_emoji(text: str) -> str:
        return emoji.get_emoji_regexp().sub(r"", text)

    def process_item(input_record: Dict[str, Any]) -> None:
        batch_uuid = input_record["batch_uuid"]
        batch_text = input_record["batch_text"]
        logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}: analyzing '{batch_text}'")

        batch_text = batch_text.strip()
        if not batch_text:
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}: Text is empty after pre-stripping leading/trailing whitespace. Skipping.")
            return
        if batch_text.startswith("<tool_call>"):  # don't speak and subtitle tool call invocations generated by the LLM
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}: Text is a tool call invocation. Skipping.")
            return
        batch_text = strip_markdown.strip_markdown(batch_text)  # remove formatting for TTS and subtitling
        if batch_text is None:
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}: Text is `None` after stripping markdown. Skipping.")
            return
        batch_text = strip_emoji(batch_text)
        batch_text = batch_text.strip()  # once more, with feeling!
        if not batch_text:
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}: Text is empty after post-stripping emoji and leading/trailing whitespace. Skipping.")
            return
        # Now we actually have some text that is worth sending to the TTS and to the translation/subtitling system.

        # Break into lines, and break each line into sentences.
        # TODO: This relies on the fact that LLMs don't insert newlines except as paragraph breaks.
        lines = batch_text.split("\n")
        plural_s = "s" if len(lines) != 1 else ""
        logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}: detected {len(lines)} line{plural_s}.")
        lines = [line.strip() for line in lines if line.strip() != ""]  # This (vs. doing it on the fly) buys us that we know when we are processing the last item.
        for lineno, line in enumerate(lines, start=1):
            if task_env.cancelled:
                return

            is_first_line = (lineno == 1)
            is_last_line = (lineno == len(lines))

            docs = _natlang_analyze(line)
            assert len(docs) == 1
            doc = docs[0]
            sentences = list(doc.sents)
            plural_s = "s" if len(sentences) != 1 else ""
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}: detected {len(sentences)} sentence{plural_s} on this line.")

            sentences = [str(sentence) for sentence in sentences]  # from spaCy rich internal format
            sentences = [sentence.strip() for sentence in sentences if sentence.strip() != ""]  # Same here - now we know when we're processing the last item.
            for sentenceno, sentence in enumerate(sentences, start=1):
                if task_env.cancelled:
                    return

                sentence_uuid = str(gensym("tts_sentence"))
                is_first_sentence = (sentenceno == 1)
                is_last_sentence = (sentenceno == len(sentences))

                logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): starting processing")

                if avatar_controller_config.subtitles_enabled and avatar_controller_config.subtitle_text_gui_widget is not None:
                    if avatar_controller_config.translator_source_lang is not None and avatar_controller_config.translator_target_lang is not None:  # Call the AI translator on Raven-server
                        subtitle = _translate_sentence(sentence)
                    else:  # Subtitles but no translation -> English closed captions (CC)
                        subtitle = sentence
                    logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): original: {sentence}")
                    logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): subtitle: {subtitle}")
                else:
                    logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): original: {sentence}")
                    logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): subtitler is off.")
                    subtitle = None

                if task_env.cancelled:
                    return

                # Precompute TTS audio and phoneme data.
                # We have plenty of wall time to precompute more, even when running the TTS on CPU, while the first sentence is being spoken.
                logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): precomputing TTS audio and phoneme data")
                prep = api.tts_prepare(text=sentence,
                                       voice=input_record["voice"],
                                       speed=input_record["voice_speed"],
                                       get_metadata=True)
                if prep is None:
                    logger.warning(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): error during precomputing, skipping sentence")
                    continue

                logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): processing done")
                if task_env.cancelled:  # IMPORTANT: don't queue the result (and trigger event) if cancelled
                    return

                # The first part is for introspection/logging/debug; the second part is needed when speaking.
                output_record = {"batch_uuid": input_record["batch_uuid"],
                                 "batch_text": input_record["batch_text"],
                                 "voice": input_record["voice"],
                                 "voice_speed": input_record["voice_speed"],
                                 "line_number": lineno,
                                 "lines_count": len(lines),
                                 "sentence_number_on_line": sentenceno,
                                 "sentences_count_on_line": len(sentences),
                                 "sentence_uuid": sentence_uuid,
                                 "sentence": sentence,
                                 # ----------------------------------------
                                 "subtitle": subtitle,
                                 "prep": prep,
                                 "is_first_sentence_in_batch": (is_first_line and is_first_sentence),
                                 "is_last_sentence_in_batch": (is_last_line and is_last_sentence),
                                 "video_offset": input_record["video_offset"],  # needed when actually speaking
                                 "on_start_speaking": input_record["on_start_speaking"],
                                 "on_stop_speaking": input_record["on_stop_speaking"],
                                 "on_start_sentence": input_record["on_start_sentence"],
                                 "on_stop_sentence": input_record["on_stop_sentence"]}

                if (on_audio_ready := input_record["on_audio_ready"]) is not None:
                    audio_bytes = prep["audio_bytes"]
                    on_audio_ready(output_record, audio_bytes)

                tts_output_queue.put(output_record)

    # background task main loop
    try:
        while True:
            if task_env.cancelled:  # co-operative shutdown
                return

            try:
                input_record = tts_input_queue.get(block=False)
            except queue.Empty:
                time.sleep(0.2)
                continue

            try:
                process_item(input_record)
            except Exception as exc:
                logger.error(f"preprocess_task: during `process_item`: {type(exc)}: {exc}")
                traceback.print_exc()
    finally:
        logger.info(f"preprocess_task: instance {task_env.task_name}: TTS input preprocessor exiting")

# --------------------------------------------------------------------------------
# Background task: TTS playback controller

def speak_task(task_env: env) -> None:
    """TTS, with avatar lipsync and subtitles (from AI translator)."""
    global emotion_autoreset_t0
    global tts_idle_check_t0

    logger.info(f"speak_task: instance {task_env.task_name}: TTS playback controller starting")

    def process_item(output_record: Dict[str, Any]) -> None:
        batch_uuid = output_record["batch_uuid"]
        sentence_uuid = output_record["sentence_uuid"]
        # sentence = output_record["sentence"]  # not actually used during speaking
        subtitle = output_record["subtitle"]
        logger.info(f"speak_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, sentence {sentence_uuid}: starting processing")
        with task_env.lock:
            task_env.speaking = True

        def speak_task_on_start_speaking():
            logger.info(f"speak_task.process_item.speak_task_on_start_speaking: instance {task_env.task_name}: sentence {sentence_uuid}: TTS starting to speak.")
            if output_record["is_first_sentence_in_batch"] and (custom_on_start_speaking := output_record["on_start_speaking"]) is not None:
                custom_on_start_speaking(output_record)
            if (custom_on_start_sentence := output_record["on_start_sentence"]) is not None:
                custom_on_start_sentence(output_record)
            if avatar_controller_config.gui_alive:
                # Show subtitle if any
                if avatar_controller_config.subtitle_text_gui_widget is not None and subtitle is not None:
                    dpg.set_value(avatar_controller_config.subtitle_text_gui_widget, subtitle)
                    dpg.show_item(avatar_controller_config.subtitle_text_gui_widget)

                    # position subtitle offscreen to measure size
                    dpg.set_item_pos(avatar_controller_config.subtitle_text_gui_widget, (avatar_controller_config.main_window_w,
                                                                                         avatar_controller_config.main_window_h))
                    dpg.split_frame()
                    w, h = guiutils.get_widget_size(avatar_controller_config.subtitle_text_gui_widget)

                    # position subtitle at bottom
                    dpg.set_item_pos(avatar_controller_config.subtitle_text_gui_widget, (avatar_controller_config.subtitle_left_x0,
                                                                                         avatar_controller_config.subtitle_bottom_y0 - h))
                    dpg.split_frame()

                # Allow the user to cancel the TTS
                if avatar_controller_config.stop_tts_button_gui_widget is not None:
                    dpg.enable_item(avatar_controller_config.stop_tts_button_gui_widget)

        def speak_task_on_stop_speaking():
            global emotion_autoreset_t0
            logger.info(f"speak_task.process_item.speak_task_on_stop_speaking: instance {task_env.task_name}: sentence {sentence_uuid}: TTS finished.")
            if (custom_on_stop_sentence := output_record["on_stop_sentence"]) is not None:
                custom_on_stop_sentence(output_record)
            # The `task_env.cancelled` check catches the case where `speak_task` is being power-cycled. In that case, we must emit the `on_stop_speaking` event (if configured).
            if (output_record["is_last_sentence_in_batch"] or task_env.cancelled) and (custom_on_stop_speaking := output_record["on_stop_speaking"]) is not None:
                custom_on_stop_speaking(output_record)
            if avatar_controller_config.gui_alive:  # Be careful - the user might have closed the app while the TTS was speaking.
                if avatar_controller_config.subtitle_text_gui_widget is not None:
                    dpg.hide_item(avatar_controller_config.subtitle_text_gui_widget)
                if avatar_controller_config.stop_tts_button_gui_widget is not None:
                    dpg.disable_item(avatar_controller_config.stop_tts_button_gui_widget)
            with task_env.lock:
                emotion_autoreset_t0 = time.time_ns()  # reset the emotion autoreset timer, so that the last emotion (from `on_done` or sibling switching at an AI message) stays for a couple more seconds once speaking ends.
                # Set the speaking state flag very last, because these events are called from a different thread (the TTS client's background task),
                # and our `speak_task` thread starts processing the next item in the queue immediately when it detects this flag.
                task_env.speaking = False

        logger.info(f"speak_task.process_item: instance {task_env.task_name}: sentence {sentence_uuid}: submitting TTS task.")
        api.tts_speak_lipsynced(instance_id=avatar_controller_config.avatar_instance_id,
                                voice="ignored_due_to_prep",
                                text="ignored_due_to_prep",
                                speed=1.0,  # ignored due to prep
                                video_offset=output_record["video_offset"],
                                on_audio_ready=None,
                                on_start=speak_task_on_start_speaking,
                                on_stop=speak_task_on_stop_speaking,
                                prep=output_record["prep"])
        logger.info(f"speak_task.process_item: instance {task_env.task_name}: sentence {sentence_uuid}: processing done")

    task_env.lock = threading.RLock()
    task_env.speaking = False
    try:
        while True:
            if task_env.cancelled:  # co-operative shutdown
                return
            with task_env.lock:
                speaking = task_env.speaking  # we must release the lock as soon as possible (so that `speak_task_on_stop_speaking` can lock it, if it happens to be run), so get the state into a temporary variable.
            if speaking:  # wait until TTS is free (previous speech ended)
                time.sleep(0.1)
                continue
            try:
                output_record = tts_output_queue.get(block=False)
            except queue.Empty:  # wait until we have a sentence to speak
                time_now = time.time_ns()
                if avatar_controller_config.emotion_autoreset_interval is not None:  # if the feature is enabled, reset emotion after a few seconds of idle time (when the TTS is not speaking)
                    dt = (time_now - emotion_autoreset_t0) / 10**9
                    if not task_env.speaking and dt > avatar_controller_config.emotion_autoreset_interval:
                        emotion_autoreset_t0 = time_now
                        logger.info(f"speak_task: instance {task_env.task_name}: avatar idle for at least {avatar_controller_config.emotion_autoreset_interval} seconds; updating emotion to 'neutral' (default idle state)")
                        api.avatar_set_emotion(instance_id=avatar_controller_config.avatar_instance_id,
                                               emotion_name="neutral")
                if avatar_controller_config.tts_idle_check_interval is not None:  # trigger the TTS idle event if relevant now (if configured)
                    dt = (time_now - tts_idle_check_t0) / 10**9
                    if not task_env.speaking and dt > avatar_controller_config.tts_idle_check_interval:
                        tts_idle_check_t0 = time_now
                        if avatar_controller_config.on_tts_idle is not None:
                            avatar_controller_config.on_tts_idle()
                time.sleep(0.2)
                continue

            try:
                process_item(output_record)
            except Exception as exc:
                logger.error(f"speak_task: during `process_item`: {type(exc)}: {exc}")
                traceback.print_exc()
    finally:
        logger.info(f"speak_task: instance {task_env.task_name}: TTS playback controller exiting")
