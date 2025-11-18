"""A simple mono audio recorder for STT (speech to text), with background operation, autostop on silence, and VU metering in dBFS."""

__all__ = ["get_available_devices",
           "validate_capture_device",
           "Recorder"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import concurrent.futures
import threading
import time
from typing import Optional, List, Tuple

import numpy as np

import pvrecorder

from unpythonic import memoize
from unpythonic.env import env

from .. import bgtask

from . import utils as audio_utils

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
@memoize
def get_available_devices() -> List[str]:
    """Return a list of the names of available audio capture devices."""
    return list(pvrecorder.PvRecorder.get_available_devices())

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
        except IndexError:
            error_msg = f"validate_capture_device: No such audio capture device '{device_name}'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"validate_capture_device: Using audio capture device '{device_name}'.")
    else:  # Find first NON-monitoring audio capture device
        for device_name in device_names:
            if "monitor of" not in device_name.lower():
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

        `vu_peak_hold`: seconds.

                        How long to hold the peak value in `vu` readout.

                        Digital hold; the peak value jumps down immediately after the hold timeout.

                        You'll only need this if you want to display a VU meter in the GUI,
                        for monitoring the audio input level.

        `silence_threshold`: value in dBFS, or `None` to autodetect (preferred).

                             If `None`, autodetect background noise level from first 0.05s of audio,
                             and set the silence threshold to `background_noise + 6dB`.

                             The idea of using the first 1/20s of audio for silence level detection
                             is that when a human presses the GUI button to start recording, they
                             don't typically start speaking straight away, but after a very short pause.

                             See `raven.common.audio.utils.linear_to_dBFS` for explanation of dBFS.

        `autostop_timeout`: seconds, or `None` to disable.

                            When specified, automatically stop recording if the input audio level
                            stays under `silence_threshold` for this long (i.e. we then consider
                            that the user has stopped speaking).
        """
        silence_threshold_msg = f"{silence_threshold:0.2f}dBFS" if silence_threshold is not None else "autodetection"
        logger.info(f"Recorder.__init__: Initializing audio recorder on device '{device_name}', with frame length {frame_length} samples, VU meter peak hold {vu_peak_hold:0.6g}s, silence threshold {silence_threshold_msg}, and silence autostop timeout {autostop_timeout}s.")

        device_name = validate_capture_device(device_name)  # autodetect if `None`, and sanity check in any case
        device_names = get_available_devices()
        assert device_name in device_names  # we only get here if the validation succeeded
        self.device_name = device_name  # for information only

        self.silence_threshold = silence_threshold  # dBFS
        self.autostop_timeout = autostop_timeout  # seconds

        # `pvrecorder` is always mono ( asked the author here: https://github.com/Picovoice/pvrecorder/issues/146 )
        self.frame_length = frame_length
        self.recorder = pvrecorder.PvRecorder(frame_length=self.frame_length,
                                              device_index=device_names.index(device_name))

        self.sample_rate = None  # sample rate (Hz) of last recording
        self._start_timestamp = None  # recording start time, for initial background noise detection

        self.vu_peak_hold = vu_peak_hold
        self._vu_instant = -90.0
        self._vu_peak = -90.0
        self._vu_last_peak_timestamp = time.time_ns()

        self.is_recording = False
        self._recording_state_lock = threading.Lock()

        self._task_manager = bgtask.TaskManager(name=f"Recorder_0x{id(self):x}",
                                                mode="concurrent",
                                                executor=executor)
        logger.info("Recorder.__init__: Initialization complete.")

    def __del__(self) -> None:
        self.recorder.delete()
        self.recorder = None

    def start(self) -> None:
        """Start recording.

        This automatically spawns a background task to handle the recording.

        If already recording, do nothing.
        """
        with self._recording_state_lock:
            if self.is_recording:
                return
            self.is_recording = True

            def record_task(task_env: env) -> None:
                try:
                    if task_env.cancelled:  # while waiting in queue
                        return
                    self.data = None
                    self.recorder.start()
                    self.sample_rate = self.recorder.sample_rate  # read-only property; not sure if it's available when not recording, so let's be safe.
                    silence_level_available = False
                    silence_level_dBFS = -90.0
                    silence_measurement_timeout = 0.05  # seconds
                    self._start_timestamp = self._vu_last_peak_timestamp = last_signal_timestamp = time.time_ns()  # timestamp after the recorder is really up and running

                    while self.recorder.is_recording and not task_env.cancelled:
                        frame = self.recorder.read()  # -> List[int] (s16, mono)
                        array = np.array(frame, dtype=np.int16)
                        self._update_vu(array)
                        if self.data is not None:
                            self.data = np.concatenate([self.data, array])
                        else:
                            self.data = array

                        time_now = time.time_ns()
                        if not silence_level_available:  # start of recording
                            if self.silence_threshold is not None:
                                silence_level_dBFS = self.silence_threshold
                                silence_level_available = True
                                logger.info(f"Recorder.start.record_task: Silence level set by caller to {silence_level_dBFS:0.2f}dBFS.")
                            else:  # autodetect silence level at start
                                recording_time_elapsed = (time_now - self._start_timestamp) / 10**9
                                if recording_time_elapsed >= silence_measurement_timeout:
                                    silence_level_raw_dBFS = audio_utils.linear_to_dBFS(np.max(np.abs(self.data)))  # all data so far!
                                    silence_level_dBFS = silence_level_raw_dBFS + 6.0  # leave some margin to be safe, in case the very beginning was unusually silent
                                    silence_level_available = True
                                    logger.info(f"Recorder.start.record_task: Silence level measured from first {silence_measurement_timeout:0.6g}s of recorded audio as {silence_level_raw_dBFS:0.2f}dBFS.")
                        else:  # silence_level_available:  # normal operation
                            # _vu_instant for the current audio frame is updated by `_update_vu`, above
                            if self._vu_instant > silence_level_dBFS:
                                last_signal_timestamp = time_now

                            # Stop recording if the audio input stays silent and the autostop timeout is exceeded
                            time_elapsed_since_last_signal = (time_now - last_signal_timestamp) / 10**9
                            if time_elapsed_since_last_signal >= self.autostop_timeout:
                                self.stop(_clear_task=False)  # prevent deadlock; the task is already exiting, no need to wait until it exits.
                                break

                finally:
                    with self._recording_state_lock:
                        self.is_recording = False
            self._task_manager.submit(record_task, env())

    def _update_vu(self, array: np.array) -> None:
        """Update the VU meter data. Called automatically once per audio frame when recording."""
        peak = audio_utils.linear_to_dBFS(np.max(np.abs(array)))  # latest buffer (or whatever we were received)
        self._vu_instant = peak
        time_now = time.time_ns()
        if (peak > self._vu_peak) or ((time_now - self._vu_last_peak_timestamp) / 10**9 >= self.vu_peak_hold):
            self._vu_peak = peak
            self._vu_last_peak_timestamp = time_now

    def stop(self, _clear_task: bool = True) -> None:
        """Stop recording.

        If not recording, do nothing.

        When this function returns, you can `get_recorded_audio` to get your audio recording.

        `_clear_task`: Internal flag used by the autostop functionality.
        """
        with self._recording_state_lock:
            if not self.is_recording:
                return
            self.recorder.stop()
            self._vu_instant = -90.0
            if _clear_task:
                self._task_manager.clear(wait=True)  # cancel the recording task, and wait until it exits
            self._vu_peak = -90.0

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
        if self.data:
            duration = len(data) / self.sample_rate  # -> seconds
            logger.info(f"Recorder.get_recorded_audio: returning {duration:0.6g}s of recorded audio.")
        else:
            logger.info("Recorder.get_recorded_audio: no audio, returning `None`.")
        if clear:
            self.data = None
        return data

    def _get_vu(self) -> Tuple[float, float]:
        return self._vu_instant, self._vu_peak
    vu = property(fget=_get_vu, doc="VU (voltage units) meter, in dbFS. Tuple `[instant, peak]`. The `instant` value is for the current audio frame, `peak` is for the last `vu_peak_hold` seconds. Read-only.")
