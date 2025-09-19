"""Avatar TTS (text to speech) and subtitling system controller.

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

# TODO: This still reads some settings from `raven.librarian.config`; these might be better handled as parameters to `initialize` to facilitate reusability in other apps.

__all__ = ["initialize",
           "shutdown",
           "configure_subtitles",
           "update_emotion_from_text",
           "send_text_to_tts",
           "stop_tts"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import concurrent.futures
import functools
import queue
import threading
import time
import traceback
from typing import List, Optional, Union

import emoji
import strip_markdown

import dearpygui.dearpygui as dpg

from unpythonic import equip_with_traceback, slurp
from unpythonic.env import env

from ..client import api  # Raven-server support

from ..common import bgtask

from ..common.gui import utils as guiutils

from . import config as librarian_config

avatar_input_queue = queue.Queue()  # for TTS preprocessing and subtitle generation: [text0, ...]
avatar_output_queue = queue.Queue()  # for TTS and subtitle output: [(sentence0, translation0, prep0), ...]

avatar_emotion_autoreset_t0 = time.time_ns()  # emotion autoreset is handled by `speak_task`

avatar_controller_initialized = False
avatar_controller_config = env()
def initialize(avatar_instance_id: str,
                stop_tts_button_gui_widget: Optional[Union[str, int]],
                subtitles_enabled: bool,
                subtitle_text_gui_widget: Union[str, int],
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

    `stop_tts_button_gui_widget`: DPG tag or ID of the DPG button widget that will call `stop_tts`
                                  if clicked. Used for automatically enabling/disabling the button
                                  depending on the TTS state (speaking / not speaking).

                                  Set this to `None` to disable the feature.

    `subtitles_enabled`: Whether subtitles are initially enabled when the module starts.
                         Call `configure_subtitles` to enable/disable later.

    `subtitle_text_gui_widget`: DPG tag or ID of the DPG text widget to send the subtitle text to.
                                The widget can start hidden - we will show/hide it automatically.

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
        logger.info("initialize: `raven.librarian.avatar_controller` is already initialized. Using existing initialization.")
        return

    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor()
    # Use separate task managers (but the same thread pool) so that we can easily power-cycle a component when needed.
    # TODO: This takes two slots in the thread pool for the whole duration of the app. Consider the implications.
    avatar_controller_config.input_queue_task_manager = bgtask.TaskManager(name="librarian_avatar_input_queue",
                                                                           mode="concurrent",
                                                                           executor=executor)
    avatar_controller_config.output_queue_task_manager = bgtask.TaskManager(name="librarian_avatar_output_queue",
                                                                            mode="concurrent",
                                                                            executor=executor)

    avatar_controller_config.avatar_instance_id = avatar_instance_id
    avatar_controller_config.stop_tts_button_gui_widget = stop_tts_button_gui_widget
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

def configure_subtitles(enable: bool) -> None:
    """Enable or disable subtitles.

    Should be called when the relevant part of the app state changes.
    """
    avatar_controller_config.subtitles_enabled = enable

def update_emotion_from_text(text: str) -> str:
    """Update the emotion for the AI avatar from `text`, and reset the emotion autoreset (return-to-neutral) timer.

    The analysis results are LRU cached (cache size 128) to facilitate running also on CPU setups, where the analysis can be slow,
    so that switching back and forth between the same AI messages won't cause slowdowns.

    For convenience, return the name of the emotion.
    """
    global avatar_emotion_autoreset_t0
    try:
        emotion = _avatar_get_emotion_from_text(text)
        logger.info(f"update_emotion_from_text: updating emotion to '{emotion}'")
        api.avatar_set_emotion(instance_id=avatar_controller_config.avatar_instance_id,
                               emotion_name=emotion)
        logger.info("update_emotion_from_text: emotion updated")
        return emotion
    finally:
        # Reset the timer last. If running on CPU, the emotion analysis may be slow.
        avatar_emotion_autoreset_t0 = time.time_ns()

@functools.lru_cache(maxsize=128)
def _avatar_get_emotion_from_text(text: str) -> str:
    try:
        if not text:
            return "neutral"
        detected_emotions = api.classify(text)  # -> `{emotion0: score0, ...}`, sorted by score, descending
        filtered_emotions = [emotion_name for emotion_name in detected_emotions.keys() if emotion_name not in librarian_config.avatar_config.emotion_blacklist]
        winning_emotion = filtered_emotions[0]
        return winning_emotion
    except Exception:
        return "neutral"

def stop_tts() -> None:
    """Stop the TTS, clearing also all speech pending in the queues."""
    logger.info("stop_tts: entered.")
    # Clear preprocess queue, so that no new preprocess jobs start.
    logger.info("stop_tts: clearing TTS preprocess queue.")
    slurp(avatar_input_queue)
    # Power-cycle the TTS preprocessor, to cancel the current job.
    logger.info("stop_tts: power-cycling TTS preprocessor.")
    avatar_controller_config.input_queue_task_manager.clear()
    avatar_controller_config.input_queue_task_manager.submit(preprocess_task, env())
    # Then clear the output queue, so that no new speak jobs start.
    logger.info("stop_tts: clearing TTS output queue.")
    slurp(avatar_output_queue)
    # Then stop the TTS - the current TTS task will end, and `speak_task` will then notice that the output queue is empty.
    logger.info("stop_tts: stopping TTS.")
    api.tts_stop()
    logger.info("stop_tts: all done.")

def send_text_to_tts(text: str) -> None:
    """Add a complete piece of text to the preprocess queue for TTS."""
    logger.info("send_text_to_tts: adding text to TTS queue.")
    avatar_input_queue.put(text)

# For CPU-friendliness, LRU-cache the AI-heavy parts (for the "speak again" feature).
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

def preprocess_task(task_env: env) -> None:
    """Preprocess text for TTS output (AI speech synthesizer) from the TTS preprocess queue.

    The text should be a whole chat message (without the role name), or at least a complete paragraph.

    The text is cleaned, and then split into sentences. Depending on Librarian settings (see `raven.librarian.config`),
    each sentence may be translated for subtitling, closed-captioned (CC) as-is, or the subtitler may be skipped.

    The resulting clean sentence and its possible subtitle are added to the TTS output queue.
    """
    logger.info(f"preprocess_task: instance {task_env.task_name}: TTS preprocessor starting")

    def strip_emoji(text: str) -> str:
        return emoji.get_emoji_regexp().sub(r"", text)

    def process_item(text: str) -> None:
        logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}: analyzing '{text}'")
        text = text.strip()
        if not text:
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}: Text is empty after pre-stripping leading/trailing whitespace. Skipping.")
            return
        if text.startswith("<tool_call>"):  # don't speak and subtitle tool call invocations generated by the LLM
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}: Text is a tool call invocation. Skipping.")
            return
        text = strip_markdown.strip_markdown(text)  # remove formatting for TTS and subtitling
        text = strip_emoji(text)
        if text is None:
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}: Text is `None` after stripping markdown and emoji. Skipping .")
            return
        text = text.strip()  # once more, with feeling!
        if not text:
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}: Text is empty after post-stripping leading/trailing whitespace. Skipping.")
            return
        # Now we actually have some text that is worth sending to the TTS and to the translation/subtitling system.

        def check_exit_and_print_reason() -> bool:  # True: exit, False: keep running
            if task_env.cancelled or not avatar_controller_config.gui_alive:
                reason = "Cancelled" if task_env.cancelled else "App is shutting down"
                logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: {reason}, exiting.")
                return True
            return False

        # Break into lines, and break each line into sentences.
        # TODO: This relies on the fact that LLMs don't insert newlines except as paragraph breaks.
        lines = text.split("\n")
        plural_s = "s" if len(lines) != 1 else ""
        logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}: detected {len(lines)} line{plural_s}.")
        for lineno, line in enumerate(lines, start=1):
            if check_exit_and_print_reason():
                return

            line = line.strip()
            if not line:
                continue

            docs = _natlang_analyze(line)
            assert len(docs) == 1
            doc = docs[0]
            sentences = list(doc.sents)
            plural_s = "s" if len(sentences) != 1 else ""
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}, line {lineno}: detected {len(sentences)} sentence{plural_s} on this line.")

            for sentenceno, sentence in enumerate(sentences, start=1):
                if check_exit_and_print_reason():
                    return

                sentence = str(sentence)  # from spaCy rich internal format

                logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}, line {lineno}, sentence {sentenceno} (0x{id(sentence):x}): starting processing")

                if avatar_controller_config.subtitles_enabled:
                    if avatar_controller_config.translator_target_lang is not None:  # Call the AI translator on Raven-server
                        subtitle = _translate_sentence(sentence)
                    else:  # Subtitles but no translation -> English closed captions (CC)
                        subtitle = sentence
                    logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}, line {lineno}, sentence {sentenceno} (0x{id(sentence):x}): original: {sentence}")
                    logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}, line {lineno}, sentence {sentenceno} (0x{id(sentence):x}): subtitle: {subtitle}")
                else:
                    logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}, line {lineno}, sentence {sentenceno} (0x{id(sentence):x}): original: {sentence}")
                    logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}, line {lineno}, sentence {sentenceno} (0x{id(sentence):x}): subtitler is off.")
                    subtitle = None

                if check_exit_and_print_reason():
                    return

                # Precompute TTS audio and phoneme data.
                # We have plenty of wall time to precompute more, even when running the TTS on CPU, while the first sentence is being spoken.
                logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}, line {lineno}, sentence {sentenceno} (0x{id(sentence):x}): precomputing TTS audio and phoneme data")
                prep = api.tts_prepare(text=sentence,
                                       voice=librarian_config.avatar_config.voice,
                                       speed=librarian_config.avatar_config.voice_speed)
                if prep is None:
                    logger.warning(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}, line {lineno}, sentence {sentenceno} (0x{id(sentence):x}): error during precomputing, skipping sentence")
                    continue

                logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: text 0x{id(text)}, line {lineno}, sentence {sentenceno} (0x{id(sentence):x}): processing done")
                if check_exit_and_print_reason():  # IMPORTANT: don't queue the result if cancelled
                    return

                avatar_output_queue.put((sentence, subtitle, prep))

    # background task main loop
    try:
        while True:
            if task_env.cancelled or not avatar_controller_config.gui_alive:  # co-operative shutdown (e.g. app exit)
                return

            try:
                text = avatar_input_queue.get(block=False)
            except queue.Empty:
                time.sleep(0.2)
                continue

            try:
                process_item(text)
            except Exception as exc:
                logger.error(f"preprocess_task: {type(exc)}: {exc}")
                traceback.print_exc()
    finally:
        reason = "Cancelled" if task_env.cancelled else "App is shutting down"
        logger.info(f"preprocess_task: instance {task_env.task_name}: {reason}, TTS preprocessor exiting.")

def speak_task(task_env: env) -> None:
    """TTS, with avatar lipsync and subtitles (from AI translator)."""
    global avatar_emotion_autoreset_t0

    logger.info(f"speak_task: instance {task_env.task_name}: TTS playback controller starting")

    def process_item(sentence, subtitle, prep):
        logger.info(f"speak_task.process_item: instance {task_env.task_name}: sentence 0x{id(sentence):x}: starting processing")
        with task_env.lock:
            task_env.speaking = True

        def on_start_lipsync_speaking():
            logger.info(f"speak_task.process_item.on_start_lipsync_speaking: instance {task_env.task_name}: sentence 0x{id(sentence):x}: TTS starting to speak.")
            if avatar_controller_config.gui_alive:
                # Show subtitle if any
                if subtitle is not None:
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

        def on_stop_lipsync_speaking():
            global avatar_emotion_autoreset_t0
            logger.info(f"speak_task.process_item.on_stop_lipsync_speaking: instance {task_env.task_name}: sentence 0x{id(sentence):x}: TTS finished.")
            if avatar_controller_config.gui_alive:  # Be careful - the user might have closed the app while the TTS was speaking.
                dpg.hide_item(avatar_controller_config.subtitle_text_gui_widget)
                if avatar_controller_config.stop_tts_button_gui_widget is not None:
                    dpg.disable_item(avatar_controller_config.stop_tts_button_gui_widget)
            with task_env.lock:
                avatar_emotion_autoreset_t0 = time.time_ns()  # reset the emotion autoreset timer, so that the last emotion (from `on_done` or sibling switching at an AI message) stays for a couple more seconds once speaking ends.
                # Set the speaking state flag very last, because these events are called from a different thread (the TTS client's background task),
                # and our `speak_task` thread starts processing the next item in the queue immediately when it detects this flag.
                task_env.speaking = False

        logger.info(f"speak_task.process_item: instance {task_env.task_name}: sentence 0x{id(sentence):x}: submitting TTS task.")
        api.tts_speak_lipsynced(instance_id=avatar_controller_config.avatar_instance_id,
                                voice="ignored_due_to_prep",
                                text="ignored_due_to_prep",
                                speed=1.0,  # ignored due to prep
                                video_offset=librarian_config.avatar_config.video_offset,
                                on_audio_ready=None,
                                on_start=on_start_lipsync_speaking,
                                on_stop=on_stop_lipsync_speaking,
                                prep=prep)
        logger.info(f"speak_task.process_item: instance {task_env.task_name}: sentence 0x{id(sentence):x}: processing done")

    task_env.lock = threading.RLock()
    task_env.speaking = False
    try:
        while True:
            if task_env.cancelled:  # co-operative shutdown (e.g. app exit)
                break
            with task_env.lock:
                speaking = task_env.speaking  # we must release the lock as soon as possible (so that `on_stop_lipsync_speaking` can lock it, if it happens to be run), so get the state into a temporary variable.
            if speaking:  # wait until TTS is free (previous speech ended)
                time.sleep(0.1)
                continue
            try:
                sentence, subtitle, prep = avatar_output_queue.get(block=False)
            except queue.Empty:  # wait until we have a sentence to speak
                time_now = time.time_ns()
                dt = (time_now - avatar_emotion_autoreset_t0) / 10**9
                if not task_env.speaking and dt > librarian_config.avatar_config.emotion_autoreset_interval:  # reset emotion after a few seconds of idle time (when the TTS is not speaking)
                    avatar_emotion_autoreset_t0 = time_now
                    logger.info(f"speak_task: instance {task_env.task_name}: avatar idle for at least {librarian_config.avatar_config.emotion_autoreset_interval} seconds; updating emotion to 'neutral' (default idle state)")
                    api.avatar_set_emotion(instance_id=avatar_controller_config.avatar_instance_id,
                                           emotion_name="neutral")
                time.sleep(0.2)
                continue
            else:
                process_item(sentence, subtitle, prep)
    finally:
        logger.info(f"speak_task: instance {task_env.task_name}: TTS playback controller exiting")
