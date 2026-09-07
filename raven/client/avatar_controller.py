"""Avatar TTS (text to speech) and subtitling system controller.

Handles also two other features:

  - Autoresets the avatar emotion when not speaking and a timeout has elapsed.
  - Starts/stops the avatar's "data eyes" effect (LLM tool access indicator).

Contrast `avatar_renderer`, which is concerned with blitting the avatar video into the GUI.

This takes in text to be spoken by the TTS and optionally subtitled.

The text is sent into an input queue, which is processed by a background task. First, the text is stripped of Markdown and emoji.
Then the text is split into sentences (using spaCy), which are then translated via Raven-server's translator (if subtitles are enabled).
It is also possible to produce closed captions (CC), i.e. subtitles with no translation.

Finally, TTS audio and phonemes are precomputed, and the result goes into an output queue, one item per sentence.
The sentences are guaranteed to be spoken in the same order that the queued texts were sent in.

Another background task reads this output queue and controls the TTS playback and showing/hiding the subtitles.

That second background task also takes care of triggering a global `on_tts_idle` event once the queue has been
empty and nothing spoken for `tts_idle_check_interval` — long enough that the gap between two sentences of one
reply does not read as the end of it. If you want a per-avatar-instance trigger for end-of-speaking, it is
better to use the `on_stop_speaking` event of `dpg_avatar_controller.send_text_to_tts`.
"""

__all__ = ["DEFAULT_DISCONTINUITY_EFFECT",
           "DPGAvatarController"]

import logging
logger = logging.getLogger(__name__)

import concurrent.futures
import contextlib
import copy
import functools
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import emoji
import strip_markdown

import dearpygui.dearpygui as dpg

from unpythonic import gensym, slurp
from unpythonic.env import env

from ..common import bgtask
from ..common import text as common_text
from ..common.audio import player as audio_player

from ..common.gui import utils as guiutils

from . import api  # Raven-server support
from . import config as client_config
from . import mayberemote
from .avatar_renderer import DPGAvatarRenderer

# --------------------------------------------------------------------------------
# For CPU-friendliness, LRU-cache all AI-heavy parts (for the "speak again" feature).
#
# NOTE: `raven.client.tts.tts_prepare` also LRU caches its results internally.

@functools.lru_cache(maxsize=128)
def _avatar_get_emotion_from_text(emotion_blacklist: Tuple[str],
                                  text: str) -> str:
    """Internal helper for computing avatar's emotion from text."""
    try:
        if not text:
            return "neutral"
        detected_emotions = api.classify(text)  # -> `{emotion0: score0, ...}`, sorted by score, descending
        filtered_emotions = [emotion_name for emotion_name in detected_emotions.keys() if emotion_name not in emotion_blacklist]
        winning_emotion = filtered_emotions[0]
        return winning_emotion
    except Exception:
        return "neutral"

@functools.lru_cache(maxsize=128)
def _translate_sentence(sentence: str,
                        source_lang: str,
                        target_lang: str) -> str:
    """Internal helper for subtitle translation with LRU caching."""
    subtitle = api.translate_translate(sentence,
                                       source_lang=source_lang,
                                       target_lang=target_lang)
    return subtitle

@functools.lru_cache(maxsize=128)
def _natlang_analyze(text: str) -> List[List["spacy.tokens.token.Token"]]:  # noqa: F821 -- type annotation only, avoid importing spaCy here
    """Internal helper for natural-language translation with LRU caching."""
    docs = api.natlang_analyze(text,
                               pipes=["tok2vec", "parser", "senter"])
    return docs

# --------------------------------------------------------------------------------
# API

# The effect `mark_discontinuity` overlays when its caller names none. A postprocessor chain fragment, in the
# same format as the animator settings' own `postprocessor_chain`.
#
# This is a fallback rather than the policy: the app that calls `mark_discontinuity` is where the choice
# belongs, since taste in this varies and only the app knows its own users. Raven-librarian configures it in
# `raven.librarian.config`, which is also where the parameters are documented for someone tuning them.
#
# Tuned from `raven/avatar/assets/settings/glitchyholo.json`, which runs this same filter as a continuous
# ambient effect; a one-off flourish wants to be more prominent than an ambient one. `unboost` runs
# *backwards*: higher makes glitches rarer and fewer (`postprocessor.digital_glitches` computes
# `rand()**unboost`). `glitchyholo` sits at 10.0 and the filter's own default is 4.0.
DEFAULT_DISCONTINUITY_EFFECT = [["digital_glitches", {"strength": 0.02,
                                                      "unboost": 1.5,
                                                      "max_glitches": 8,
                                                      "min_glitch_height": 12,
                                                      "max_glitch_height": 40,
                                                      "hold_min": 1,
                                                      "hold_max": 2}]]


class DPGAvatarController:
    def __init__(self,
                 stop_tts_button_gui_widget: Optional[Union[str, int]],
                 on_tts_idle: Optional[Callable],
                 tts_idle_check_interval: Optional[float],
                 subtitles_enabled: bool,
                 subtitle_text_gui_widget: Optional[Union[str, int]],
                 subtitle_left_x0: int,
                 subtitle_bottom_y0: int,
                 translator_source_lang: str,
                 translator_target_lang: Optional[str],
                 executor: Optional[concurrent.futures.Executor] = None):
        """Avatar TTS (text to speech) and subtitling system controller.

        Instantiate this **after** your app's GUI is alive.

        NOTE: There is just one preprocessor and one TTS per client process,
              so your app should instantiate *at most one* of these.

        `stop_tts_button_gui_widget`: DPG tag or ID of the DPG button widget that will call `stop_tts`
                                      if clicked. Used for automatically enabling/disabling the button
                                      depending on the TTS state (speaking / not speaking).

                                      Set this to `None` to disable the feature.

        `on_tts_idle`: 0-argument callable. Called periodically when the TTS is not speaking.
                       The return value is ignored.

                       This can be used to trigger additional GUI actions when the avatar stops speaking.

                       Called once each time the TTS falls silent, so a handler may do something that only
                       makes sense once.

                       Note that in the case of multiple avatars, this event does not distinguish
                       between them; this is global for the TTS system.

                       Distinct from `on_idle` in `register_avatar_instance`, which the similar name
                       invites confusing it with. That one is per avatar instance and is about *activity*:
                       it fires when an instance has been idle long enough to have its video switched off.
                       This one is about *speech*.

        `tts_idle_check_interval`: seconds. How much quiet counts as the TTS having become idle, before
                                   `on_tts_idle` triggers. Long enough to sit through the gap between two
                                   sentences of one reply, which is a real gap: the queue is empty there
                                   whenever preparing the next sentence outlasts speaking the previous one.

                                   Set to `None` to disable.

        `subtitles_enabled`: Whether subtitles are initially enabled when the module starts.
                             To change the status later, just write to
                             `dpg_avatar_controller.subtitles_enabled: bool`.

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

        `executor`: A `ThreadPoolExecutor` or something duck-compatible with it.
                    Used for running the background tasks.
        """
        self.stop_tts_button_gui_widget = stop_tts_button_gui_widget
        self.on_tts_idle = on_tts_idle
        self.tts_idle_check_interval = tts_idle_check_interval
        self.tts_idle_check_t0 = time.monotonic_ns()
        self.subtitles_enabled = subtitles_enabled
        self.subtitle_text_gui_widget = subtitle_text_gui_widget
        self.subtitle_left_x0 = subtitle_left_x0
        self.subtitle_bottom_y0 = subtitle_bottom_y0
        self.translator_source_lang = translator_source_lang
        self.translator_target_lang = translator_target_lang

        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor()
        # Use separate task managers (but the same thread pool) so that we can easily power-cycle a component when needed.
        # TODO: This takes two slots in the thread pool for the whole duration of the app. Consider the implications.
        self.input_queue_task_manager = bgtask.TaskManager(name="avatar_controller_input_queue",
                                                           mode="concurrent",
                                                           executor=executor)
        self.output_queue_task_manager = bgtask.TaskManager(name="avatar_controller_output_queue",
                                                            mode="concurrent",
                                                            executor=executor)
        self.emotion_autoreset_task_manager = bgtask.TaskManager(name="avatar_controller_emotion_autoreset",
                                                                 mode="concurrent",
                                                                 executor=executor)

        self.tts_input_queue = queue.Queue()  # for TTS input preprocessing and subtitle generation; see `send_text_to_tts`
        self.tts_output_queue = queue.Queue()  # for TTS and subtitle playback; see `preprocess_task`
        self.gui_updates_safe = True  # At app shutdown, they aren't. Used by the subtitle system.

        # TTS dispatcher — settings come from `raven.client.config`. Device record
        # validated by `deviceinfo.validate` during `api.initialize`. Apps that want
        # standalone TTS flip `tts_allow_local` to True in `raven/client/config.py`
        # (default False matches today's Librarian — the avatar requires the server anyway).
        self.tts = mayberemote.TTS(allow_local=client_config.tts_allow_local,
                                   model_name=client_config.tts_model_name,
                                   device_string=client_config.devices["tts"]["device_string"],
                                   lang_code=client_config.tts_lang_code)

        # Start background tasks
        self.input_queue_task_manager.submit(self.preprocess_task, env())
        self.output_queue_task_manager.submit(self.speak_task, env())

    def shutdown(self) -> None:
        """Prepare module for app shutdown.

        This signals the background tasks to exit.
        """
        self.gui_updates_safe = False  # GUI may go bye-bye shortly
        self.input_queue_task_manager.clear(wait=True)
        self.output_queue_task_manager.clear(wait=True)
        self.emotion_autoreset_task_manager.clear(wait=True)

    def register_avatar_instance(self,
                                 avatar_instance_id: str,
                                 avatar_renderer: Optional[DPGAvatarRenderer],
                                 voice: Optional[str],
                                 voice_speed: Optional[float],
                                 emotion_blacklist: Tuple[str],
                                 emotion_autoreset_interval: Optional[float],
                                 idle_timeout: Optional[float],
                                 on_idle: Optional[Callable] = None) -> env:
        """Register an avatar instance, for methods that take a `config` parameter.

        Returns `config: unpythonic.env.env`, the avatar-instance-specific configuration record.

        The fields of `config` which have the same names as the parameters of this function are public:

        `avatar_instance_id`: Avatar instance to control. You get this from `raven.client.api.avatar_load`.

        `avatar_renderer`: The renderer instance that is rendering this avatar instance in the GUI.

        `voice`: TTS voice name. To get the list of available voices, call `raven.client.api.tts_list_voices`,
                 or use the `raven-avatar-settings-editor` GUI app.

        `voice_speed`: For each voice, 1.0 is the default speed the voice is designed to speak at.
                       Raising this too high may cause skipped words.

                 Use `None` ONLY IF you intend to populate `voice` and `voice_speed` later; trying to send
                 text to the TTS while the voice or the voice speed are set to `None` will raise `ValueError`.

        `emotion_blacklist`: Prevent this avatar instance from automatically entering any of the listed emotions
                             when the emotion is updated with `dpg_avatar_controller.update_emotion_from_text`.
                             The most matching non-blacklisted emotion wins.

                             Can be useful if the emotion detector is misbehaving.

        `emotion_autoreset_interval`: seconds, or `None` to disable.

                                      When enabled, this registers a handler to automatically reset the avatar
                                      instance's emotion to "neutral" whenever that avatar instance is not TTS-speaking,
                                      and at least `emotion_autoreset_interval` seconds have passed since:

                                        - The end of speaking, and
                                        - The last update to the avatar's emotion using `update_emotion_from_text`.

        `idle_timeout`: seconds, or `None` to disable. How long of no activity (for this avatar instance)
                        until its video is switched off, and `on_idle` triggers.

                        To reset the timeout, call the `ping` method.

                        To temporarily override the timeout, see the `idle_override` context manager.

        `on_idle`: Called when this avatar instance has been idle for `idle_timeout` and its video is about
                   to be switched off. Takes one argument, this instance's `config`. The return value is
                   ignored, and an exception it raises is logged and swallowed — a handler cannot call off
                   the pause it is being told about.

                   Called *before* the renderer pauses, which is the point of it: an app that shows
                   something else in the avatar's place can do so first, instead of after the renderer has
                   drawn its "video is off" indicator into a panel that is about to be replaced.

                   `video_available` already answers `False` by the time this is called, so a handler that
                   asks gets the answer it was woken for. It is called with no lock of this controller's
                   held, so a handler is free to take its own and to call back in here.

                   Fires once per switch-off, the video then staying off until something pings the instance
                   awake again.

                   Not to be confused with `on_tts_idle` in the constructor, which the name invites. That
                   one is about *speech* — the TTS queue has drained and nothing is being spoken, which is
                   how a caller learns that a reply has finished being read out across all its sentences.
                   This one is about *activity*.

        The fadeout duration of the "data eyes" effect (LLM tool access indicator) is the animator setting
        `data_eyes_fadeout_duration`.
        """
        config = env()

        config.avatar_instance_id = avatar_instance_id
        config.avatar_renderer = avatar_renderer
        config.voice = voice
        config.voice_speed = voice_speed
        config.emotion_blacklist = tuple(emotion_blacklist)  # Ensure it's hashable, for LRU cache
        config.emotion_autoreset_interval = emotion_autoreset_interval
        config.idle_timeout = idle_timeout
        config.on_idle = on_idle

        config._emotion_autoreset_t0 = time.monotonic_ns()
        config._current_emotion = "neutral"  # last emotion we sent; a fresh avatar instance starts neutral
        config._idle_detector_lock = threading.RLock()
        config._idle_detector_overrides = 0
        config._idle_detector_t0 = time.monotonic_ns()
        # The two ways the video can be off, kept apart because a caller deciding *whether to show the
        # avatar at all* must not read its own effect. Suppression is that caller's own doing — it turns
        # the video off because nothing is displaying it — so a predicate that counted suppression as
        # unavailability would answer "unavailable" for as long as it kept the avatar hidden, and the
        # avatar would never come back. See `set_video_suppressed` and `video_available`.
        config._video_suppressed = False  # the app says nothing is showing this avatar right now
        config._idle_paused = False  # the idle detector switched the video off
        # Counted rather than a flag, because more than one thing can be consulting an external source at
        # once: a turn's tool call runs on the turn's thread while an attachment is being read on a
        # background one, and whichever finished first would otherwise switch the effect off under the
        # other. See `start_data_eyes`.
        config._data_eyes_lock = threading.RLock()
        config._data_eyes_users = 0
        # The animator settings currently in force, and the transient effect overlaid on them. The settings
        # are remembered here because the server offers no getter for them, and restoring the chain after a
        # temporary filter means knowing what it was. See `mark_discontinuity`.
        config._animator_settings_lock = threading.RLock()
        config._animator_settings = None
        config._effect_timer = None
        config._effect_started_at = None
        config._avatar_speaking = False  # per-avatar-instance flag, set/reset by start/stop events in `speak_task`

        # Reset emotion after a few seconds of idle time (when the TTS is not speaking).
        def emotion_autoreset_task(task_env: env) -> None:
            while True:
                if task_env.cancelled:
                    return
                if config.emotion_autoreset_interval is not None:
                    time_now = time.monotonic_ns()
                    dt = (time_now - config._emotion_autoreset_t0) / 10**9
                    if not config._avatar_speaking and dt > config.emotion_autoreset_interval:
                        config._emotion_autoreset_t0 = time_now
                        # The reset is idempotent, and runs once per interval for as long as the avatar sits idle —
                        # so announce only when it actually returns the avatar from an expression. Reporting every
                        # tick would bury a quiet session's real log lines under twenty a minute that mean nothing.
                        if config._current_emotion != "neutral":
                            logger.info(f"emotion_autoreset_task: instance {task_env.task_name}: avatar idle for at least {config.emotion_autoreset_interval} seconds; updating emotion from '{config._current_emotion}' to 'neutral' (default idle state)")
                        config._current_emotion = "neutral"
                        try:
                            api.avatar_set_emotion(instance_id=config.avatar_instance_id,
                                                   emotion_name="neutral")
                        except Exception:  # exit task if the avatar instance is gone
                            logger.info(f"emotion_autoreset_task: instance {task_env.task_name}: avatar instance is gone, exiting.")
                            return
                # Decide under the lock, act outside it. The second half is the one that matters: acting
                # means calling `on_idle`, which belongs to the app, and a handler that puts something else
                # in the avatar's place calls back into this controller — Raven's does, through
                # `set_video_suppressed`, which takes this very lock. Every other path takes the app's lock
                # first and this one second, so holding it across the call inverts that order and the two
                # threads meet in the middle.
                switching_off = False
                with config._idle_detector_lock:
                    if (config.avatar_renderer is not None) and (config.avatar_renderer.animator_running) and (config._idle_detector_overrides == 0) and (config.idle_timeout is not None):
                        time_now = time.monotonic_ns()
                        dt = (time_now - config._idle_detector_t0) / 10**9
                        if not config._avatar_speaking and dt > config.idle_timeout:
                            logger.info(f"emotion_autoreset_task: instance {task_env.task_name}: avatar idle for at least {config.idle_timeout} seconds; switching the avatar video off")
                            config._idle_detector_t0 = time_now
                            switching_off = True
                if switching_off:
                    self._switch_video_off_for_idle(config)
                time.sleep(0.1)
        # Save the env and the task handle for possible cancellation.
        config._emotion_autoreset_task_env = env()
        config._emotion_autoreset_task = self.emotion_autoreset_task_manager.submit(emotion_autoreset_task,
                                                                                    config._emotion_autoreset_task_env)

        return config

    def ping(self,
             config: env) -> None:
        """Reset the avatar instance idle-off timeout.

        `config`: Configuration for controlling a specific avatar instance and its GUI elements.
                  See `register_avatar_instance`.

        If `avatar_renderer` is provided in `config`: resume the avatar video, if currently paused.
        Except while the video is suppressed — there is no point resuming a stream nothing is showing,
        and doing so would undo `set_video_suppressed` at the next thing that counts as activity.
        """
        config._idle_detector_t0 = time.monotonic_ns()
        if config.avatar_renderer is not None:
            connected = (config.avatar_renderer.avatar_instance_id is not None)
            if connected and (not config._video_suppressed) and (not config.avatar_renderer.animator_running):
                config.avatar_renderer.pause(action="resume")
                config._idle_paused = False

    def _switch_video_off_for_idle(self,
                                   config: env) -> None:
        """Announce that this avatar instance has gone idle, then switch its video off.

        `config`: Configuration for controlling a specific avatar instance and its GUI elements.
                  See `register_avatar_instance`. Must have a renderer; the caller decides that.

        The order is the whole content of this method, and it is why the announcement exists at all: an app
        that shows something else in the avatar's place gets to do that *before* the renderer draws its
        "video is off" indicator into the panel it is losing.
        """
        # The flag before the announcement, so that a handler asking `video_available` — which is the
        # question it was woken to answer — is not told the video is still there, and does not decide to
        # keep showing an avatar that is about to stop.
        #
        # No lock: it is one bool store, and `ping` clears it without the lock either, so taking it here
        # would order this against nothing.
        config._idle_paused = True

        if config.on_idle is not None:
            try:
                config.on_idle(config)
            except Exception:  # noqa: BLE001 -- an app's handler must not be able to call off the pause it is being told about
                logger.exception(f"_switch_video_off_for_idle: instance '{config.avatar_instance_id}': `on_idle` handler raised")

        try:
            config.avatar_renderer.pause(action="pause")
        except Exception:
            logger.exception(f"_switch_video_off_for_idle: instance '{config.avatar_instance_id}': caught exception during `avatar_renderer.pause`")

    def set_video_suppressed(self,
                             config: env,
                             suppressed: bool) -> None:
        """Tell the controller whether anything is currently showing this avatar's video.

        `config`: Configuration for controlling a specific avatar instance and its GUI elements.
                  See `register_avatar_instance`.

        `suppressed`: `True` when the avatar's panel has been given to something else, `False` when
                      the avatar has it back.

        While suppressed, the animator is paused (both in the GUI and on the server) and stays paused:
        `ping` will not resume it, and the idle detector has nothing left to switch off. Un-suppressing
        resumes the video and restarts the idle countdown, an uncovering being activity by definition.

        Idempotent, and safe to call before the renderer has been started — a renderer with no avatar
        instance has no animator to pause, and the suppression still takes effect for whatever starts next.

        Not callable from the render thread: pausing the animator waits for a frame.
        """
        with config._idle_detector_lock:
            if suppressed == config._video_suppressed:
                return
            config._video_suppressed = suppressed
            if suppressed:
                renderer = config.avatar_renderer
                if (renderer is not None) and (renderer.avatar_instance_id is not None) and renderer.animator_running:
                    logger.info(f"set_video_suppressed: instance '{config.avatar_instance_id}': nothing is showing this avatar; pausing its video")
                    renderer.pause(action="pause")
            else:
                logger.info(f"set_video_suppressed: instance '{config.avatar_instance_id}': the avatar is on screen again; resuming its video")
                self.ping(config)

    def video_available(self,
                        config: env) -> bool:
        """Whether this avatar has live video to look at, ignoring whether anything is looking.

        `config`: Configuration for controlling a specific avatar instance and its GUI elements.
                  See `register_avatar_instance`.

        `False` while the stream is warming up — the renderer is started well before the first frame
        arrives, and the seconds in between are a blank panel — while the idle detector has the video
        switched off, and when there is no stream at all (never started, stopped, or lost).
        """
        # The obvious reading, `animator_running`, is the one that cannot be used: it also goes false
        # when the video is suppressed, so a caller switching panels on this answer would be reading
        # back its own decision. Every term below is a cause the caller cannot cause by hiding the avatar.
        #
        # Read without `_idle_detector_lock`: three bools, and a caller acting one tick late on any of
        # them shows the wrong panel for a fraction of a second. Taking the lock here would put a waiter
        # on whatever is mid-pause, which for this answer is a poor trade.
        renderer = config.avatar_renderer
        if renderer is None or renderer.avatar_instance_id is None:
            return False
        return renderer.first_frame_received and not config._idle_paused

    @contextlib.contextmanager
    def idle_override(self,
                      config: env) -> None:
        """Context manager. Temporarily override the idle-off mechanism for this avatar instance.

        `config`: Configuration for controlling a specific avatar instance and its GUI elements.
                  See `register_avatar_instance`.

        While overridden, the video auto-pause mechanism will not trigger. This can be used
        to keep the avatar active e.g. when the AI is processing, even if the prompt processing
        and/or tool calls are slow.

        When the override starts, this will ping once, so that the avatar video resumes if paused.

        When the override ends, this will also ping once, to reset the idle timeout.

        This is thread-safe, and acts as an OR gate across threads. The override is active
        as long as at least one dynamic extent with the override is active, in any thread.
        """
        # Possible scenarios.
        #
        # The brackets denote dynamic extents. The initial state is "not overridden".
        # Each dynamic extent wants to temporarily set the state to "overridden".
        #
        #    -+          -+
        #     |           | -+
        #     | -+        |  |
        #     |  |        | -+
        #    -+  |       -+
        #        |
        #       -+
        #
        with config._idle_detector_lock:
            config._idle_detector_overrides += 1
            self.ping(config)
        try:
            yield
        finally:
            with config._idle_detector_lock:
                config._idle_detector_overrides -= 1
                if config._idle_detector_overrides == 0:
                    self.ping(config)
            assert config._idle_detector_overrides >= 0  # contract: postcondition

    def update_emotion_from_text(self,
                                 config: env,
                                 text: str) -> str:
        """Update the emotion for the AI avatar from `text`, and reset the emotion autoreset (return-to-neutral) timer.

        `config`: Configuration for controlling a specific avatar instance and its GUI elements.
                  See `register_avatar_instance`.

        The analysis results are LRU cached (cache size 128) to facilitate running also on CPU setups, where the analysis can be slow,
        so that switching back and forth between the same AI messages won't cause slowdowns.

        For convenience, return the name of the emotion.
        """
        try:
            emotion = _avatar_get_emotion_from_text(config.emotion_blacklist,
                                                    text)
            logger.info(f"update_emotion_from_text: updating emotion to '{emotion}'")
            api.avatar_set_emotion(instance_id=config.avatar_instance_id,
                                   emotion_name=emotion)
            config._current_emotion = emotion  # so the autoreset knows whether it has anything to return from
            logger.info("update_emotion_from_text: emotion updated")
            return emotion
        finally:
            # Reset the timer last. If running on CPU, the emotion analysis may be slow.
            config._emotion_autoreset_t0 = time.monotonic_ns()

    def load_animator_settings(self, config: env, animator_settings: Dict) -> None:
        """Send animator settings to the server, and remember them as this instance's baseline.

        `animator_settings`: as `raven.client.api.avatar_load_animator_settings` takes them.

        Prefer this to calling the API directly. The server offers no getter for these, so anything that
        wants to change the avatar *temporarily* - `glitch`, below - has nothing to restore from unless
        somebody remembered what was in force. That somebody is here.
        """
        with config._animator_settings_lock:
            config._animator_settings = copy.deepcopy(animator_settings)
        api.avatar_load_animator_settings(config.avatar_instance_id, animator_settings)
        self.ping(config)

    def mark_discontinuity(self,
                           config: env,
                           effect: Optional[list] = None,
                           floor: float = 0.4,
                           ceiling: float = 1.5) -> None:
        """Mark a break in continuity by running a visual effect over the avatar. Returns immediately.

        For the moment when what the user is looking at is replaced by something else — in Raven-librarian,
        switching chat branches or rerolling a reply. Named for the occasion rather than the appearance,
        because which effect appears is the caller's to choose.

        `effect`: a postprocessor chain fragment, `[[filter_name, parameters_dict], ...]`, in the same
                  format as the `postprocessor_chain` of the animator settings. `None` uses
                  `DEFAULT_DISCONTINUITY_EFFECT`; an empty list does nothing at all.
        `floor`: seconds. Minimum time the effect stays up, so that a switch too fast to see still reads as
                 something having happened.
        `ceiling`: seconds from the *first* call in a run. Running the effect for longer than a moment stops
                   looking deliberate and starts looking broken, and holding a key down should not be able
                   to exceed this.

        Repeated calls extend the effect rather than restarting it, up to `ceiling` — which is what flicking
        through siblings does, and it should read as one continuous effect rather than a stutter of them.

        Does nothing if no animator settings have been loaded through `load_animator_settings`, there being
        no chain to overlay and nothing to restore afterwards.
        """
        if effect is None:
            effect = DEFAULT_DISCONTINUITY_EFFECT
        if not effect:  # configured off, or nothing to add
            return

        with config._animator_settings_lock:
            if config._animator_settings is None:
                logger.warning("mark_discontinuity: no animator settings loaded for this instance; skipping.")
                return

            now = time.monotonic()
            if config._effect_timer is not None:  # already running: extend, do not restart
                config._effect_timer.cancel()
            else:
                config._effect_started_at = now
                settings = copy.deepcopy(config._animator_settings)
                # Appended rather than inserted at a chosen index: the chain is the user's, and where their
                # own filters sit relative to each other is their business. Last means the effect is applied
                # to the finished frame, which is what "the transmission broke up" looks like.
                #
                # Deep-copied because the fragment is usually a module-level constant in someone's config,
                # and handing the same dicts to every call would let one mutation downstream edit the user's
                # configuration permanently.
                settings.setdefault("postprocessor_chain", []).extend(copy.deepcopy(effect))
                api.avatar_load_animator_settings(config.avatar_instance_id, settings)

            remaining = min(floor, max(0.0, config._effect_started_at + ceiling - now))
            config._effect_timer = threading.Timer(remaining, self._end_discontinuity_effect, args=(config,))
            config._effect_timer.daemon = True  # a pending restore must not hold the app open at exit
            config._effect_timer.start()

    def _end_discontinuity_effect(self, config: env) -> None:
        """Put the avatar's own settings back. Called from the effect timer; not part of the public API."""
        with config._animator_settings_lock:
            config._effect_timer = None
            config._effect_started_at = None
            settings = copy.deepcopy(config._animator_settings) if config._animator_settings is not None else None
        if settings is not None:
            api.avatar_load_animator_settings(config.avatar_instance_id, settings)

    def start_data_eyes(self, config: env) -> None:
        """Start the scifi "data eyes" cel effect, which says the system is consulting an external source.

        Semantics: the effect switches on instantly, and **calls nest**. Every `start_data_eyes` must be
        matched by a `stop_data_eyes`; the effect ends when the last one is stopped, not the first.

        `config`: Configuration for controlling a specific avatar instance and its GUI elements.
                  See `register_avatar_instance`.

        This only has any effect, if the character currently loaded to the avatar instance that `config`
        points to, supports the data eyes effect (per-character cels).
        """
        # Nesting is the whole point of the counter, and the reason it lives here rather than at the call
        # sites: the sources that light this run concurrently and know nothing about each other - a tool
        # call on the turn's thread, an attachment being read on a background one - so a caller cannot tell
        # whether it is the only one, and the naive pairing has the first `stop` cancel everyone's effect.
        with config._data_eyes_lock:
            config._data_eyes_users += 1
            if config._data_eyes_users > 1:  # already on
                return
        api.avatar_start_data_eyes(config.avatar_instance_id)
        self.ping(config)

    def stop_data_eyes(self, config: env) -> None:
        """Stop the scifi "data eyes" cel effect. See `start_data_eyes` for the nesting contract.

        Semantics: the effect fades out once every `start_data_eyes` has been matched. Fade duration is the
        animator setting `data_eyes_fadeout_duration`.

        `config`: Configuration for controlling a specific avatar instance and its GUI elements.
                  See `register_avatar_instance`.

        This only has any effect, if the character currently loaded to the avatar instance that `config`
        points to, supports the data eyes effect (per-character cels).
        """
        with config._data_eyes_lock:
            # Clamped at zero so an unmatched stop is harmless: teardown calls one unconditionally to be
            # sure the effect ends, and a negative count would then swallow the next real start.
            config._data_eyes_users = max(0, config._data_eyes_users - 1)
            if config._data_eyes_users > 0:  # someone else still needs it
                return
        api.avatar_stop_data_eyes(config.avatar_instance_id)
        self.ping(config)

    def send_text_to_tts(self,
                         config: env,
                         text: str,
                         video_offset: float,
                         on_audio_ready: Optional[Callable] = None,
                         on_start_speaking: Optional[Callable] = None,
                         on_stop_speaking: Optional[Callable] = None,
                         on_start_sentence: Optional[Callable] = None,
                         on_stop_sentence: Optional[Callable] = None) -> str:
        """Send a complete piece of text into the TTS queue.

        Returns a batch UUID. This is used in the events to identify which call to `send_text_to_tts`
        the triggered event refers to.

        `config`: Configuration for controlling a specific avatar instance and its GUI elements.
                  See `register_avatar_instance`.

                  Voice settings ("voice" and "voice_speed") are stored in `config`.

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
        if config.voice is None:
            logger.error("send_text_to_tts: 'voice' field of `config` is required (cannot be `None`)")
            raise ValueError("'voice' field of `config` is required (cannot be `None`)")
        if config.voice_speed is None:
            logger.error("send_text_to_tts: 'voice_speed' field of `config` is required (cannot be `None`)")
            raise ValueError("'voice_speed' field of `config` is required (cannot be `None`)")

        batch_uuid = str(gensym("tts_job"))
        logger.info(f"send_text_to_tts: adding text to TTS queue, batch {batch_uuid}.")
        # One atomic operation, no need for a lock.
        #
        # `speak_task` speaks the sentences in the order they arrive. The ordering of
        # TTS requests is nevertheless preserved, because a whole batch is queued in one go.
        # `preprocess_task` handles batches in the order they arrive, completing each batch
        # before moving on to the next - so all sentences from any given batch will be adjacent
        # in the output queue.
        self.tts_input_queue.put({"batch_uuid": batch_uuid,
                                  "batch_text": text,
                                  "video_offset": video_offset,
                                  "on_audio_ready": on_audio_ready,
                                  "on_start_speaking": on_start_speaking,
                                  "on_stop_speaking": on_stop_speaking,
                                  "on_start_sentence": on_start_sentence,
                                  "on_stop_sentence": on_stop_sentence,
                                  "config": config})
        return batch_uuid

    def stop_tts(self) -> None:
        """Stop the TTS, clearing also all speech pending in the queues.

        This triggers the `on_stop_speaking` event of the current batch.
        """
        logger.info("stop_tts: entered.")
        # Clear TTS input preprocess queue, so that no new preprocess jobs start.
        logger.info("stop_tts: clearing TTS input preprocess queue.")
        slurp(self.tts_input_queue)
        # Power-cycle the TTS input preprocessor, to cancel the current job.
        logger.info("stop_tts: power-cycling TTS input preprocessor.")
        self.input_queue_task_manager.clear()
        self.input_queue_task_manager.submit(self.preprocess_task, env())
        # Then clear the output queue, so that no new speak jobs start.
        logger.info("stop_tts: clearing TTS playback queue.")
        slurp(self.tts_output_queue)
        # Power-cycle the TTS playback controller, to cancel the current job.
        logger.info("stop_tts: power-cycling TTS playback controller.")
        self.output_queue_task_manager.clear()
        self.output_queue_task_manager.submit(self.speak_task, env())
        # We must still stop the TTS, to actually make the TTS playback task exit.
        # The current TTS task will end, and the old controller will then exit.
        logger.info("stop_tts: stopping TTS.")
        audio_player.require().stop()
        logger.info("stop_tts: all done.")

    # --------------------------------------------------------------------------------
    # Background task: TTS input preprocessor

    def preprocess_task(self, task_env: env) -> None:
        """Preprocess text for TTS (AI speech synthesizer) from the TTS input preprocess queue.

        The text should be a whole chat message (without the role name), or at least a complete paragraph.

        The text is cleaned, and then split into sentences. Depending on settings passed to `initialize`,
        each sentence may be translated for subtitling, closed-captioned (CC) as-is, or the subtitler may be skipped.

        Finally, the TTS audio and phonemes are precomputed.

        The resulting clean sentence, its possible subtitle, and the precomputed data, are added to the TTS playback queue.
        """
        logger.info(f"preprocess_task: instance {task_env.task_name}: TTS input preprocessor starting")

        def strip_emoji(text: str) -> str:
            return emoji.replace_emoji(text, replace="")

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
            if not common_text.is_speakable(batch_text):  # blank, or punctuation/symbols only
                logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}: Text has nothing speakable after post-stripping emoji and leading/trailing whitespace. Skipping.")
                return
            # Now we actually have some text that is worth sending to the TTS and to the translation/subtitling system.

            # Break into lines, and break each line into sentences.
            # TODO: This relies on the fact that LLMs don't insert newlines except as paragraph breaks.
            lines = batch_text.split("\n")
            # Filtering here (vs. doing it on the fly) buys us that we know when we are processing the last item.
            # Dropping unspeakable lines is what keeps a dangling Markdown bullet — an answer ending "...naked eye:\n*"
            # leaves a line that is just "*" — from becoming a sentence of its own, with no phonemes to synthesize.
            lines = [line.strip() for line in lines if common_text.is_speakable(line)]
            plural_s = "s" if len(lines) != 1 else ""
            logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}: detected {len(lines)} non-blank line{plural_s}.")
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
                sentences = [sentence.strip() for sentence in sentences if common_text.is_speakable(sentence)]  # Same here - now we know when we're processing the last item.
                for sentenceno, sentence in enumerate(sentences, start=1):
                    if task_env.cancelled:
                        return

                    sentence_uuid = str(gensym("tts_sentence"))
                    is_first_sentence = (sentenceno == 1)
                    is_last_sentence = (sentenceno == len(sentences))

                    logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): starting processing")

                    if self.subtitles_enabled and self.subtitle_text_gui_widget is not None:
                        if self.translator_source_lang is not None and self.translator_target_lang is not None:  # Call the AI translator on Raven-server
                            subtitle = _translate_sentence(sentence=sentence,
                                                           source_lang=self.translator_source_lang,
                                                           target_lang=self.translator_target_lang)
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
                    prep = self.tts.synthesize(text=sentence,
                                               voice=input_record["config"].voice,
                                               speed=input_record["config"].voice_speed,
                                               get_metadata=True,
                                               format="flac")
                    if not prep.audio_bytes:  # blank input or no-phoneme case — `tts_prepare` flags it via empty audio_bytes
                        logger.warning(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): no audio produced during precomputing, skipping sentence")
                        continue

                    logger.info(f"preprocess_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, line {lineno} out of {len(lines)}, sentence {sentenceno} out of {len(sentences)} ({sentence_uuid}): processing done")
                    if task_env.cancelled:  # IMPORTANT: don't queue the result (and trigger event) if cancelled
                        return

                    # The first part is for introspection/logging/debug; the second part is needed when speaking.
                    output_record = {"batch_uuid": input_record["batch_uuid"],
                                     "batch_text": input_record["batch_text"],
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
                                     "on_stop_sentence": input_record["on_stop_sentence"],
                                     "config": input_record["config"]}

                    if (on_audio_ready := input_record["on_audio_ready"]) is not None:
                        audio_bytes = prep.audio_bytes
                        on_audio_ready(output_record, audio_bytes)

                    self.tts_output_queue.put(output_record)

        # background task main loop
        try:
            while True:
                if task_env.cancelled:  # co-operative shutdown
                    return

                try:
                    input_record = self.tts_input_queue.get(block=False)
                except queue.Empty:
                    time.sleep(0.2)
                    continue

                try:
                    process_item(input_record)
                except Exception:
                    logger.exception(f"preprocess_task: instance {task_env.task_name}: caught exception during `process_item`")
        finally:
            logger.info(f"preprocess_task: instance {task_env.task_name}: TTS input preprocessor exiting")

    # --------------------------------------------------------------------------------
    # Background task: TTS playback controller

    def reposition_subtitle(self) -> None:
        """Reposition the current subtitle, if any.

        This should be done when the GUI is resized in a way that causes the
        avatar panel to change its size, and whenever the subtitle becomes
        visible again after a spell of not being laid out at all.
        """
        with guiutils.nonexistent_ok():
            if (self.subtitle_text_gui_widget is not None) and (dpg.get_value(self.subtitle_text_gui_widget) != ""):
                # Position the subtitle offscreen to measure it: the bottom edge is what is being placed,
                # so the height has to be known first, and a widget reports the size it was last laid out
                # at. Parked rather than hidden, a hidden item not being laid out at all.
                guiutils.park_offscreen(self.subtitle_text_gui_widget)
                # `required=False`: a misplaced caption is a much better outcome than a hung app.
                guiutils.split_frame(operation="measuring the subtitle to place it", required=False)
                w, h = guiutils.get_widget_size(self.subtitle_text_gui_widget)

                # position subtitle at bottom
                dpg.set_item_pos(self.subtitle_text_gui_widget, (self.subtitle_left_x0,
                                                                 self.subtitle_bottom_y0 - h))
                guiutils.split_frame(operation="showing the repositioned subtitle", required=False)

    def speak_task(self, task_env: env) -> None:
        """TTS, with avatar lipsync and subtitles (from AI translator)."""
        logger.info(f"speak_task: instance {task_env.task_name}: TTS playback controller starting")

        def process_item(output_record: Dict[str, Any]) -> None:
            batch_uuid = output_record["batch_uuid"]
            sentence_uuid = output_record["sentence_uuid"]
            config = output_record["config"]  # which avatar instance
            # sentence = output_record["sentence"]  # not actually used during speaking
            subtitle = output_record["subtitle"]
            logger.info(f"speak_task.process_item: instance {task_env.task_name}: batch {batch_uuid}, sentence {sentence_uuid}: starting processing")
            with task_env.lock:
                task_env.tts_speaking = True  # for `speak_task` main loop
                task_env.tts_idle_announced = False  # there is speech again, so there is a next fall-silent to announce
                config._avatar_speaking = True  # per-avatar-instance flag, for emotion autoreset

            def speak_task_on_start_speaking():
                logger.info(f"speak_task.process_item.speak_task_on_start_speaking: instance {task_env.task_name}: sentence {sentence_uuid}: TTS starting to speak.")
                self.ping(config)
                if output_record["is_first_sentence_in_batch"] and (custom_on_start_speaking := output_record["on_start_speaking"]) is not None:
                    custom_on_start_speaking(output_record)
                if (custom_on_start_sentence := output_record["on_start_sentence"]) is not None:
                    custom_on_start_sentence(output_record)
                if self.gui_updates_safe:
                    # Show subtitle if any
                    if self.subtitle_text_gui_widget is not None and subtitle is not None:
                        dpg.set_value(self.subtitle_text_gui_widget, subtitle)
                        dpg.show_item(self.subtitle_text_gui_widget)
                        self.reposition_subtitle()

                    # Allow the user to cancel the TTS
                    if self.stop_tts_button_gui_widget is not None:
                        dpg.enable_item(self.stop_tts_button_gui_widget)

            def speak_task_on_stop_speaking():
                logger.info(f"speak_task.process_item.speak_task_on_stop_speaking: instance {task_env.task_name}: sentence {sentence_uuid}: TTS finished.")
                if (custom_on_stop_sentence := output_record["on_stop_sentence"]) is not None:
                    custom_on_stop_sentence(output_record)
                # The `task_env.cancelled` check catches the case where `speak_task` is being power-cycled. In that case, we must emit the `on_stop_speaking` event (if configured).
                if (output_record["is_last_sentence_in_batch"] or task_env.cancelled) and (custom_on_stop_speaking := output_record["on_stop_speaking"]) is not None:
                    custom_on_stop_speaking(output_record)
                if self.gui_updates_safe:  # Be careful - the user might have closed the app while the TTS was speaking.
                    if self.subtitle_text_gui_widget is not None:
                        dpg.hide_item(self.subtitle_text_gui_widget)
                    if self.stop_tts_button_gui_widget is not None:
                        dpg.disable_item(self.stop_tts_button_gui_widget)
                with task_env.lock:
                    config._emotion_autoreset_t0 = time.monotonic_ns()  # reset the emotion autoreset timer, so that the last emotion stays for a couple more seconds once speaking ends.
                    self.tts_idle_check_t0 = time.monotonic_ns()  # and the quiet the TTS idle event waits out, which starts here rather than at the last announcement
                    self.ping(config)  # similarly, reset the idle countdown when speaking ends.
                    # Set the speaking state flags very last. These events are called from a different thread (the TTS client's background task),
                    # and our task threads (for `speak_task`, `emotion_autoreset_task`) monitor these flags and take action immediately.
                    config._avatar_speaking = False
                    task_env.tts_speaking = False

            logger.info(f"speak_task.process_item: instance {task_env.task_name}: sentence {sentence_uuid}: submitting TTS task.")
            self.tts.speak_lipsynced(instance_id=config.avatar_instance_id,
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
        task_env.tts_speaking = False
        # Starts armed-as-already-said: "the TTS fell silent" means nothing before anything has spoken,
        # and without this the event fires once, `tts_idle_check_interval` after the app starts.
        task_env.tts_idle_announced = True
        try:
            while True:
                if task_env.cancelled:  # co-operative shutdown
                    return
                with task_env.lock:
                    speaking = task_env.tts_speaking  # we must release the lock as soon as possible (so that `speak_task_on_stop_speaking` can lock it, if it happens to be run), so get the state into a temporary variable.
                if speaking:  # wait until TTS is free (previous speech ended)
                    time.sleep(0.1)
                    continue
                try:
                    output_record = self.tts_output_queue.get(block=False)
                except queue.Empty:  # wait until we have a sentence to speak
                    time_now = time.monotonic_ns()
                    # Once per fall-silent, not once per interval for as long as the silence lasts. The
                    # interval is the quiet a batch has to sit through before it counts as finished, which
                    # is what keeps the gap between two sentences of one reply from reading as the end of
                    # it — the queue really is empty in that gap, whenever preparing the next sentence
                    # takes longer than speaking the previous one.
                    if self.tts_idle_check_interval is not None and not task_env.tts_idle_announced:
                        dt = (time_now - self.tts_idle_check_t0) / 10**9
                        if not task_env.tts_speaking and dt > self.tts_idle_check_interval:
                            with task_env.lock:
                                task_env.tts_idle_announced = True
                            if self.on_tts_idle is not None:
                                self.on_tts_idle()
                    time.sleep(0.2)
                    continue

                try:
                    process_item(output_record)
                except Exception:
                    logger.exception(f"speak_task: instance {task_env.task_name}: caught exception during `process_item`")
        finally:
            logger.info(f"speak_task: instance {task_env.task_name}: TTS playback controller exiting")
