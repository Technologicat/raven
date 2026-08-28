"""A simple mono audio recorder for STT (speech to text), with background operation, autostop on silence, and VU metering in dBFS."""

__all__ = ["DEFAULT_FRAME_LENGTH",
           "DEFAULT_VU_PEAK_HOLD",

           "get_available_devices",
           "is_monitoring_device",
           "validate_capture_device",
           "Recorder",

           "instance",
           "initialize",
           "require"]

import logging
logger = logging.getLogger(__name__)

import concurrent.futures
import threading
import time
from typing import Callable, Optional, List, Tuple

import numpy as np

# `pvrecorder` is imported where a device is actually opened, not here. It is the only dependency in this
# module that a machine may not have, and importing it at module level made everything that *mentions* a
# recorder — a GUI panel, a test with a stub in place of the device — need it as well.
from unpythonic.env import env

from .. import bgtask

from . import silencegate
from . import utils as audio_utils

# Default audio recorder settings. The silence and autostop defaults live with the gate that reads them,
# in `raven.common.audio.silencegate`.
DEFAULT_FRAME_LENGTH = 512  # samples at device default sample rate
DEFAULT_VU_PEAK_HOLD = 1.0  # seconds

# How long to disbelieve the signal level at the start of a capture.
#
# A device opened cold hands over a few unusable frames first: measured 2026-08-28 on a USB webcam
# microphone at 16 kHz, two all-zero frames, then one at -18.8 dBFS against a room floor near -45, then
# one at -84, settling by 220 ms. On the same machine in Raven-librarian that spike reached full scale.
# A device already warm shows none of it.
#
# The frames are still recorded — it is only the *levels* that are wrong, and they feed two things that
# both take a maximum: the VU meter's peak, and the silence autodetection, which measures the first
# tenth of a second and would otherwise measure exactly this.
DEFAULT_SETTLING_TIME = 0.3  # seconds

# `pygame` doesn't support recording (although it can *list*
# which capture devices it sees), so Raven uses `pvrecorder`
# for recording audio for its STT (speech to text) features.
#
# To get a guaranteed-correct list of devices for each role (playback, capture),
# we query with the same library that is used for that role.
#
# There is at least one important difference in practice:
# `pygame` doesn't see monitoring devices (i.e. capture devices
# that record the audio that is going to an audio output),
# while `pvrecorder` does.
#
_available_devices = None  # cached answer for `get_available_devices`
_available_devices_lock = threading.Lock()

def get_available_devices(refresh: bool = False) -> List[str]:
    """Return a list of the names of available audio capture devices.

    The answer is cached, since setting a device up asks for it several times over.

    `refresh`: Ask the OS again instead of answering from the cache.

               Plugging a microphone in or unplugging one changes this list, so anything offering the
               user a choice of device wants a fresh answer rather than the one from app startup.
    """
    global _available_devices
    with _available_devices_lock:
        if refresh or _available_devices is None:
            import pvrecorder  # noqa: PLC0415 -- deferred on purpose; see the note at the imports
            _available_devices = list(pvrecorder.PvRecorder.get_available_devices())
        return list(_available_devices)  # a copy: the caller must not be able to edit the cache

def is_monitoring_device(device_name: str) -> bool:
    """Return whether `device_name` names a monitoring capture device.

    A monitoring device records what is being *played* rather than what a microphone hears, so it is
    never what someone means by "the microphone": pointed at one, speech recognition would transcribe
    the AI's own voice. `pvrecorder` lists them alongside real inputs — `pygame` does not see them at
    all — and the only thing distinguishing them is the name.
    """
    return "monitor of" in device_name.lower()

def validate_capture_device(device_name: Optional[str]) -> str:
    """Validate `device_name` against list of audio capture devices detected on the system.

    The return value is always the name of a valid audio capture device on which a `Recorder`
    can be instantiated.

    If `device_name` is given, return `device_name` if OK; raise `ValueError` if the specified
    device was not found on the system.

    If `device_name is None`, return the name of the first available NON-monitoring capture device.

    A monitoring capture device is a capture device that records the audio that is going
    to a playback device. Usually monitoring devices have "Monitor" in the device name.

    See the command-line utility `raven-check-audio-devices` to list audio devices on your system.
    """
    device_names = get_available_devices()
    if device_name is not None:  # User-specified device name -> device index as used by `pvrecorder`
        try:
            device_names.index(device_name)  # we just want to check if it's there
        except ValueError:  # what `list.index` raises for a name that is not in the list
            error_msg = f"validate_capture_device: No such audio capture device '{device_name}'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"validate_capture_device: Using audio capture device '{device_name}'.")
    else:  # Find first NON-monitoring audio capture device
        for device_name in device_names:
            if not is_monitoring_device(device_name):
                break
        else:
            error_msg = "validate_capture_device: No NON-monitoring audio capture device found on this system. If you want to use a MONITORING device for recording, please select it explicitly."
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"validate_capture_device: Using first available audio capture device '{device_name}'.")
    return device_name

class Recorder:
    def __init__(self,
                 frame_length: int,
                 device_name: Optional[str],
                 vu_peak_hold: float = 1.0,
                 silence_threshold: Optional[float] = None,
                 autostop_timeout: Optional[float] = 1.5,
                 executor: Optional[concurrent.futures.Executor] = None):
        """A simple audio recorder, mainly for STT purposes.

        `frame_length`: How many audio samples to process at once, at the device's default sample rate.
                        This affects the granularity of the VU meter.

                        If unsure, try e.g. `frame_length=512`.

        `device_name`: Which audio capture device to use, or `None` to use first available NON-monitoring input.

                       A monitoring input is a capture device that records the audio that is going to an audio output.
                       Usually monitoring devices have "Monitor" in the device name.

                       See the command-line utility `raven-check-audio-devices` to list available audio capture devices
                       on your system.

        `vu_peak_hold`: seconds. How long to hold the peak value in VU readout.

                        Digital hold; the peak value jumps down immediately after the hold time expires.

                        You'll only need this if you want to display a VU meter (input audio level)
                        in your GUI.

                        You can access the readout from the `vu` property (read-only),
                        or you can use the `connect_vu_readout` method to connect an event handler.

        `silence_threshold`: value in dBFS, or `None` to autodetect.

                             See `raven.common.audio.utils.linear_to_dBFS` for explanation of dBFS.

                             If `None`, autodetect background noise level from the first
                             `silencegate.DEFAULT_SILENCE_MEASUREMENT_TIME` seconds of audio, and set
                             the silence threshold to `background_noise + silencegate.DEFAULT_SILENCE_MARGIN` dB.

                             The idea of measuring the start of the audio is that when a human presses
                             the GUI button to start recording, they don't typically start speaking
                             straight away, but after a very short pause.

                             However, if the audio input has a noise gate, so that it only actually
                             starts capturing audio when a threshold level is exceeded (and sends zeroes
                             until that point), then the autodetection cannot work.

        `autostop_timeout`: seconds, or `None` to disable.

                            When specified, automatically stop recording if the input audio level
                            stays under `silence_threshold` for this long (i.e. we then consider
                            that the user has stopped speaking).

        `silence_threshold`, `autostop_timeout` and `vu_peak_hold` are plain attributes, and each is
        re-read while a recording runs — so a GUI can offer them as live controls. Changing
        `silence_threshold` after an autodetection has completed overrides the detected value;
        setting it back to `None` returns to it, rather than measuring again.
        """
        silence_threshold_msg = f"{silence_threshold:0.2f}dBFS" if silence_threshold is not None else "autodetection"
        logger.info("Recorder.__init__: Initializing audio recorder.")

        device_name = validate_capture_device(device_name)  # autodetect if `None`, and sanity check in any case
        device_names = get_available_devices()
        assert device_name in device_names  # we only get here if the validation succeeded
        self.device_name = device_name  # for information only
        logger.info(f"Recorder.__init__: Audio capture device '{device_name}', frame length {frame_length} samples, VU meter peak hold {vu_peak_hold:0.6g}s, silence threshold {silence_threshold_msg}, silence autostop timeout {autostop_timeout}s.")

        self.silence_threshold = silence_threshold  # dBFS
        self.autostop_timeout = autostop_timeout  # seconds

        import pvrecorder  # noqa: PLC0415 -- deferred on purpose; see the note at the imports

        # `pvrecorder` is always mono ( asked the author here: https://github.com/Picovoice/pvrecorder/issues/146 )
        self.frame_length = frame_length
        self.recorder = pvrecorder.PvRecorder(frame_length=self.frame_length,
                                              device_index=device_names.index(device_name))

        self.sample_rate = None  # sample rate (Hz) of last recording
        self.data = None  # last recording, as an s16 mono `np.array`; `get_recorded_audio` reads this

        # Populated later by users with `connect_vu_readout`. A list rather than one slot because the
        # level stream has several consumers at once: a VU meter in the toolbar, a tuning panel while
        # it is open, and — eventually — a wake-word detector.
        self._vu_listeners = []
        self._vu_listeners_lock = threading.Lock()

        self.vu_peak_hold = vu_peak_hold  # seconds
        self._vu_last_peak_timestamp = time.monotonic_ns()
        self._clear_vu_readout()

        self._is_capturing = False
        self._is_monitoring = False
        self._recording_state_lock = threading.RLock()

        # `TaskManager` requires an executor and now says so at construction. Making one here is what
        # this class has always documented, and it is what lets a script record without first building
        # a thread pool; remember that it is ours, so teardown does not close a caller's.
        self._own_executor = None
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                             thread_name_prefix=f"Recorder_0x{id(self):x}")
            self._own_executor = executor
        self._task_manager = bgtask.TaskManager(name=f"Recorder_0x{id(self):x}",
                                                mode="concurrent",
                                                executor=executor)
        logger.info("Recorder.__init__: Initialization complete.")

    def __del__(self) -> None:
        # Finalizers run at unpredictable times, including during interpreter
        # shutdown when dependent C state may already be gone. Wrap defensively.
        try:
            self.recorder.delete()
        except Exception:
            pass
        self.recorder = None
        # Only ours; an executor the caller lent us is the caller's to close, and shutting it down here
        # would take their other background work with it.
        if getattr(self, "_own_executor", None) is not None:
            try:
                self._own_executor.shutdown(wait=False)
            except Exception:
                pass
            self._own_executor = None

    def start(self,
              on_autostop: Optional[Callable] = None,
              monitor: bool = False) -> bool:
        """Start capturing audio. Return whether a capture was started.

        This automatically spawns a background task to handle the capture.

        **`False` means the device was already open**, and the caller's capture is not happening.
        Worth checking: a caller that assumes otherwise puts its GUI into a recording state over a
        recorder that is not recording, and the user finds out by getting a transcript of nothing.
        A `stop` on this recorder returns before its capture task has exited unless waited on, so
        the common way to arrive here is a `start` issued too soon after one — see `stop`.

        `on_autostop`: 0-argument callable. Return value is ignored.

                       Triggers when the silence detector autostops the recorder.

                       The main use case is so that the caller can do the same things
                       it would do when the recording is manually stopped.

        `monitor`: If `True`, capture for the VU readout only: keep no audio, and never autostop.

                   This is how a GUI shows the input level while nobody is dictating anything —
                   to let the user see the room's background noise while setting the silence
                   threshold, for example. Monitoring has no end of its own, which is why it
                   keeps nothing: an accumulating buffer would grow for as long as the window
                   is open.

                   `get_recorded_audio` is unaffected: a monitoring pass neither produces
                   audio nor discards what a previous recording produced.
        """
        logger.info(f"Recorder.start: Starting audio capture{' (monitoring only)' if monitor else ''}.")
        with self._recording_state_lock:
            if self._is_capturing:
                logger.warning(f"Recorder.start: This recorder is already capturing (monitoring: {self._is_monitoring}). Refusing to start another capture.")
                return False
            self._is_capturing = True
            self._is_monitoring = monitor

            def record_task(task_env: env) -> None:
                try:
                    logger.info(f"Recorder.start.record_task: instance {task_env.task_name}: Audio capture task starting.")
                    autostopped = False
                    if task_env.cancelled:  # while waiting in queue
                        logger.info(f"Recorder.start.record_task: instance {task_env.task_name}: Audio capture task cancelled while in queue.")
                        return
                    logger.info(f"Recorder.start.record_task: instance {task_env.task_name}: Starting audio recorder.")
                    if not monitor:
                        self.data = None
                    self.recorder.start()
                    self.sample_rate = self.recorder.sample_rate  # read-only property; not sure if it's available when not recording, so let's be safe.
                    # The gate is made fresh per recording, and its settings are refreshed from ours on
                    # every frame — so a threshold moved while the user is speaking takes effect on the
                    # next frame, which is what a live control in the GUI needs.
                    #
                    # Monitoring has no gate: the state it exists to show is silence, so autostopping
                    # on silence would end it the moment it became useful.
                    gate = None if monitor else silencegate.SilenceGate(threshold=self.silence_threshold,
                                                                        autostop_timeout=self.autostop_timeout,
                                                                        name=f"instance {task_env.task_name}")
                    self._vu_last_peak_timestamp = time.monotonic_ns()  # timestamp after the recorder is really up and running
                    settled_at_ns = self._vu_last_peak_timestamp + int(DEFAULT_SETTLING_TIME * 10**9)

                    logger.info(f"Recorder.start.record_task: instance {task_env.task_name}: Entering recording loop.")
                    while self.recorder.is_recording and not task_env.cancelled:
                        frame = self.recorder.read()  # -> List[int] (s16, mono)
                        array = np.array(frame, dtype=np.int16)
                        # The audio is kept from the very first frame; only its *level* is disbelieved
                        # while the device settles. See `DEFAULT_SETTLING_TIME` for what is being
                        # waited out, and why a maximum taken across it is the thing that suffers.
                        if not monitor:
                            if self.data is not None:
                                self.data = np.concatenate([self.data, array])
                            else:
                                self.data = array
                        if time.monotonic_ns() < settled_at_ns:
                            continue

                        self._update_vu_readout(array)
                        if monitor:
                            continue

                        # _vu_instant for the current audio frame is updated by `_update_vu_readout`, above
                        gate.threshold = self.silence_threshold
                        gate.autostop_timeout = self.autostop_timeout
                        if gate.update(self._vu_instant, time.monotonic_ns()):
                            logger.info(f"Recorder.start.record_task: instance {task_env.task_name}: Silence detected, autostopping. (Audio input level less than {gate.effective_threshold:0.2f}dBFS for {self.autostop_timeout:0.6g}s.)")
                            autostopped = True
                            self.stop()
                            break
                finally:
                    logger.info(f"Recorder.start.record_task: instance {task_env.task_name}: Audio capture task exiting.")
                    with self._recording_state_lock:
                        self._is_capturing = False
                        self._is_monitoring = False
                    if autostopped and on_autostop is not None:
                        logger.info(f"Recorder.start.record_task: instance {task_env.task_name}: Calling custom `on_autostop`.")
                        on_autostop()  # do this last, after no longer in recording state
                    logger.info(f"Recorder.start.record_task: instance {task_env.task_name}: Audio capture task exited.")
            self._task_manager.submit(record_task, env())
            logger.info("Recorder.start: Audio capture task submitted.")
            return True

    def set_device(self, device_name: Optional[str]) -> bool:
        """Switch to another audio capture device. Return whether the switch happened.

        `device_name`: as for the constructor — a name from `get_available_devices`, or `None` for the
                       first non-monitoring device.

        The recorder keeps its identity across the switch: connected VU readouts stay connected, and
        the silence and peak-hold settings carry over. Only the device handle underneath is replaced,
        which is what lets a GUI offer this as a control rather than as a restart.

        **Monitoring resumes on the new device; a recording is not resumed.** Splicing a message
        together from two microphones is not a thing anyone wants, so switching mid-recording is
        refused outright, and the caller gets `False`. `ValueError` still means the device is not there.
        """
        if self.is_recording():
            logger.warning(f"Recorder.set_device: a message is being recorded on '{self.device_name}'; not switching.")
            return False

        get_available_devices(refresh=True)  # switching should act on what is plugged in now
        device_name = validate_capture_device(device_name)  # raises if there is no such device
        if device_name == self.device_name:
            logger.debug(f"Recorder.set_device: already on '{device_name}'. Ignoring.")
            return True

        was_monitoring = self.is_monitoring()
        if was_monitoring:
            self.stop(wait=True)

        import pvrecorder  # noqa: PLC0415 -- deferred on purpose; see the note at the imports

        logger.info(f"Recorder.set_device: switching from '{self.device_name}' to '{device_name}'.")
        old_recorder = self.recorder
        self.recorder = pvrecorder.PvRecorder(frame_length=self.frame_length,
                                              device_index=get_available_devices().index(device_name))
        self.device_name = device_name
        # The old handle goes only once nothing can reach it — the capture task reads `self.recorder`,
        # and it is stopped above, but a `delete` before the rebind would leave a window where that
        # attribute names a freed device.
        old_recorder.delete()

        if was_monitoring:
            self.start(monitor=True)
        return True

    def connect_vu_readout(self, on_vu_update: Callable) -> None:
        """Connect a VU ("voltage units"; input audio level) readout to the recorder.

        You'll only need this if you want to display a VU meter (input audio level)
        in your GUI.

        Also, you don't necessarily have to use this; you can also access the same values
        in realtime from the `vu` property. This is just provided for the event-based
        push convenience: your event handler is called automatically when the values change.

        Any number of readouts can be connected at once; each is called in turn. To remove one,
        see `disconnect_vu_readout`.

        `on_vu_update`: 2-argument callable, signature `(instant: float, peak: float)`.
                        Values are in dBFS.

                            `instant` is the peak signal level from the current audio frame.

                            `peak` is the held peak signal level, with the `peak_vu_hold` time.

                        Return value is ignored, and an exception is logged and swallowed —
                        a readout that has gone away (a deleted GUI widget, say) must not take
                        the capture down with it.

                        When first connected, called once for initialization purposes.
                        While capturing, called once per audio frame with the new signal level data.
                        When the capture stops, called once more (with -∞ dBFS, indicating silence).

                        Connecting the same callable twice does nothing the second time.
        """
        with self._vu_listeners_lock:
            if on_vu_update in self._vu_listeners:
                return
            self._vu_listeners.append(on_vu_update)
        on_vu_update(self._vu_instant, self._vu_peak)

    def disconnect_vu_readout(self, on_vu_update: Callable) -> None:
        """Disconnect a VU readout connected by `connect_vu_readout`.

        Disconnecting something that is not connected does nothing.
        """
        with self._vu_listeners_lock:
            try:
                self._vu_listeners.remove(on_vu_update)
            except ValueError:
                pass

    def _notify_vu_readouts(self) -> None:
        """Send the current VU values to every connected readout."""
        # Snapshot rather than lock: this runs once per audio frame on the capture thread, and a
        # readout is typically a GUI widget whose owner connects and disconnects from another
        # thread. `tuple(...)` of a list is one C-level pass, so it cannot observe a half-mutated
        # list; what it can be is an instant out of date, which for a level meter is free.
        #
        # And a snapshot only makes the list safe to walk, not its entries safe to call: a readout
        # may be a widget that was deleted between the copy and the call, so each call is guarded.
        for on_vu_update in tuple(self._vu_listeners):
            try:
                on_vu_update(self._vu_instant, self._vu_peak)
            except Exception as exc:
                logger.warning(f"Recorder._notify_vu_readouts: VU readout {on_vu_update} raised, ignoring: {type(exc)}: {exc}")

    def _update_vu_readout(self, array: np.array) -> None:
        """Update the VU meter data. Called automatically once per audio frame when capturing."""
        peak = audio_utils.linear_to_dBFS(np.max(np.abs(array)))  # latest buffer (or whatever we were received)
        self._vu_instant = peak
        time_now = time.monotonic_ns()
        if (peak > self._vu_peak) or ((time_now - self._vu_last_peak_timestamp) / 10**9 >= self.vu_peak_hold):
            self._vu_peak = peak
            self._vu_last_peak_timestamp = time_now
        self._notify_vu_readouts()

    def _clear_vu_readout(self) -> None:
        """Clear the VU meter data, setting both instant and peak values to silence (-∞ dBFS).

        Called automatically when the recorder stops.
        """
        self._vu_instant = -np.inf  # dBFS
        self._vu_peak = -np.inf  # dBFS
        self._notify_vu_readouts()

    def stop(self,
             wait: bool = False,
             timeout: float = 1.0) -> bool:
        """Stop capturing, whether recording or monitoring. Return whether it has actually stopped.

        If not capturing, do nothing (and return `True`).

        `wait`: Whether to wait for the capture task to actually exit before returning.

                With `wait=False` (the default), the capture takes a small amount of time to wind
                down after this returns, so `get_recorded_audio` is not yet safe to call — poll
                `is_capturing` for that, or pass `wait=True` and read the return value.

                **Never wait from the capture task itself**, which is why this defaults to `False`:
                the autostop calls `stop` from inside the very task the wait would be waiting for.

        `timeout`: seconds, for `wait=True`. On expiry this returns `False` rather than raising,
                   because the caller is normally shutting down or about to report the failure —
                   in neither case is an exception the useful shape.
        """
        logger.info("Recorder.stop: Stopping audio capture.")
        with self._recording_state_lock:
            if not self._is_capturing:
                logger.info("Recorder.stop: This recorder is already stopped. Ignoring.")
                return True
            logger.info("Recorder.stop: Stopping audio recorder.")
            self.recorder.stop()
            self._clear_vu_readout()
            self._task_manager.clear()
        if not wait:
            logger.info("Recorder.stop: Done.")
            return not self.is_capturing()

        deadline = time.monotonic() + timeout
        while self.is_capturing():
            if time.monotonic() >= deadline:
                logger.error(f"Recorder.stop: Timed out after {timeout:0.6g}s waiting for the capture task to exit.")
                return False
            time.sleep(0.01)
        logger.info("Recorder.stop: Done.")
        return True

    def is_capturing(self) -> bool:
        """Return whether the audio device is open, in either mode.

        This is the one to wait on after `stop`, and the one that answers whether `start`
        would do anything.
        """
        with self._recording_state_lock:
            return self._is_capturing

    def is_recording(self) -> bool:
        """Return whether this audio recorder is currently keeping the audio it captures.

        `False` while monitoring, which captures for the VU readout only — so a GUI can ask this
        to decide whether a stop would produce a recording to transcribe.
        """
        with self._recording_state_lock:
            return self._is_capturing and not self._is_monitoring

    def is_monitoring(self) -> bool:
        """Return whether this audio recorder is currently capturing for the VU readout only."""
        with self._recording_state_lock:
            return self._is_capturing and self._is_monitoring

    def get_recorded_audio(self, clear: bool = True) -> Optional[np.array]:
        """Return the recorded audio as an `np.array`.

        If there is no recorded audio, returns `None`.

        `clear`: If `True` (default), release our reference to the audio recording,
                 causing it to be garbage-collected when you no longer need it.

                 If `False`, don't release the reference, so that calling `get_recorded_audio`
                 again returns the same recording.

                 In either case, when you start a new recording, the previous one
                 is cleared.

        The format is mono, s16 (signed 16-bit).

        The sample rate is available in the `sample_rate` attribute of this audio recorder.

        To encode it as an audio file, pass both the data and the sample rate to `encode`, which see.
        """
        data = self.data
        if self.data is not None:
            duration = len(data) / self.sample_rate  # -> seconds
            logger.info(f"Recorder.get_recorded_audio: returning {duration:0.6g}s of recorded audio.")
        else:
            logger.info("Recorder.get_recorded_audio: no audio recorded, returning `None`.")
        if clear:
            self.data = None
        return data

    def _get_vu(self) -> Tuple[float, float]:
        return self._vu_instant, self._vu_peak
    vu = property(fget=_get_vu, doc="VU (voltage units) meter, in dbFS. Tuple `[instant, peak]`. The `instant` value is for the current audio frame, `peak` is for the last `vu_peak_hold` seconds. Read-only.")


# The default (singleton) `Recorder` instance. `None` until `initialize` is called.
#
# Pre-populated to `None` so that apps can read the attribute and decide whether to
# initialize (or re-use) the recorder. Apps that don't need audio capture don't need
# to call `initialize`; they just leave this as `None`.
#
# Access via `raven.common.audio.recorder.instance` (read-only by convention).
instance: Optional["Recorder"] = None

def initialize(frame_length: int = DEFAULT_FRAME_LENGTH,
               device_name: Optional[str] = None,
               vu_peak_hold: float = DEFAULT_VU_PEAK_HOLD,
               silence_threshold: Optional[float] = silencegate.DEFAULT_SILENCE_THRESHOLD,
               autostop_timeout: Optional[float] = silencegate.DEFAULT_AUTOSTOP_TIMEOUT,
               executor: Optional[concurrent.futures.Executor] = None) -> "Recorder":
    """Initialize the default audio recorder singleton.

    Constructs a `Recorder` with the given parameters and assigns it to the module-level
    `instance`. Idempotent: subsequent calls return the existing instance.

    `device_name`: One of the Capture device names listed by `raven-check-audio-devices`,
                   or `None` for the first available non-monitoring capture device.

                   **A named device that is not present falls back to that first one, with a
                   warning, rather than raising.** This is an app's entry point, and a microphone
                   that was unplugged since the configuration was written should cost the user a
                   different microphone, not a program that will not start. Construct a `Recorder`
                   directly to get the strict behaviour, which is the right one for a library
                   caller who has a particular device in mind.

    `executor`: Used for the recording background task. If `None`, `Recorder` creates its own.

    Returns the `Recorder` instance.
    """
    global instance
    if instance is not None:
        logger.info("initialize: audio recorder already initialized. Using existing instance.")
        return instance

    if device_name is not None:
        logger.info(f"initialize: Validating audio capture device '{device_name}'.")
        try:
            validate_capture_device(device_name)
        except ValueError:
            logger.warning(f"initialize: Audio capture device '{device_name}' is not present on this system; falling back to the first available non-monitoring device. See `raven/client/config.py`, and run `raven-check-audio-devices` for the current choices.")
            device_name = None
    else:
        logger.info("initialize: Using first available audio capture device. If you want to use another device, see `raven/client/config.py`, and run `raven-check-audio-devices` to get available choices.")

    instance = Recorder(frame_length=frame_length,
                        device_name=device_name,
                        vu_peak_hold=vu_peak_hold,
                        silence_threshold=silence_threshold,
                        autostop_timeout=autostop_timeout,
                        executor=executor)
    return instance

def require() -> "Recorder":
    """Return the recorder, raising `RuntimeError` if not initialized.

    Use this at the entry point of any code that needs audio capture: it fails fast
    with a clear message, instead of letting an `AttributeError: 'NoneType'` surface
    deep inside a recording call.
    """
    if instance is None:
        raise RuntimeError("raven.common.audio.recorder.require: no recorder initialized. Call `raven.common.audio.initialize(...)` or `raven.common.audio.recorder.initialize(...)` first.")
    return instance
