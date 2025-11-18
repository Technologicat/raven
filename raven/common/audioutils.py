"""Audio-related utilities."""

__all__ = ["linear_to_dBFS", "dBFS_to_linear",
           "Recorder",
           "encode", "decode"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import concurrent.futures
import io
import threading
import time
from typing import BinaryIO, Generator, List, Optional, Tuple, Union

import numpy as np

import av
import pvrecorder

from unpythonic.env import env

from . import bgtask
from .numutils import si_prefix

from ..vendor.kokoro_fastapi.streaming_audio_writer import StreamingAudioWriter

def linear_to_dBFS(level: Union[int, float]) -> float:
    """Convert linear audio sample value to dBFS.

    `level`: one of:

        `int`: s16 (signed 16-bit) audio sample value, range [-32768, 32767].
        `float`: floating-point audio sample value, range [-1, 1].

        Note that if `level` is exactly zero, the corresponding dBFS is -∞,
        in which case this function returns `-np.inf`.

    dbFS = decibels full scale = a logarithmic scale, where 0.0 corresponds
    to the loudest representable signal. For example:

          0.00 dB -> maximum possible audio sample value
        -42.15 dB -> at this point the high byte in s16 (signed 16-bit) audio becomes zero
        -60.00 dB -> possibly useful silence threshold when there is some background noise
        -90.31 dB -> s16 audio is completely silent (in practice, will never happen in a recording)

    Returns the dBFS value corresponding to the linear audio sample `level`.
    Note that the sign of `level` is lost (this is an abs-logarithmic scale).
    """
    if isinstance(level, int):
        fs = 32767 if level > 0 else 32768
    elif isinstance(level, float):
        fs = 1.0
    else:
        raise TypeError(f"linear_to_dBFS: expected `level` to be float or s16 int; got {type(level)}.")
    # We use "20" here because `level` is the signal amplitude, which is a root-power quantity (P ~ level²).
    #     https://en.wikipedia.org/wiki/Decibel#Root-power_(field)_quantities
    dB = 20 * np.log10(abs(level) / fs)  # yes, -np.inf is fine (and the correct answer) if `level` is exactly zero
    return dB

def dBFS_to_linear(dB: float, format: type) -> Union[int, float]:
    """Convert dBFS value to linear scale.

    `dB`: the decibel level to convert, in dBFS (decibels full scale).
          See `linear_to_dBFS` for explanation.

    `format`: one of:

        `int`: for 16-bit integer output, range [0, 32767].
        `float`: for floating-point output, range [0, 1].

    Returns the linear audio sample value corresponding to `dB`.
    """
    if format is int:
        fs = 32767
    elif format is float:
        fs = 1.0
    else:
        raise TypeError(f"dBFS_to_linear: expected `format` to be exactly `int` (for s16 output) or `float` (for float output); got {format}.")
    level = fs * 10**(dB / 20)
    return level

class Recorder:
    @staticmethod
    def validate_device(cls, device_name: Optional[str]) -> str:
        """Validate `device_name` against list of audio capture devices detected on the system.

        The return value is always the name of a valid audio capture device on which a `Recorder`
        can be instantiated.

        If `device_name` is given, return `device_name` if OK; raise `ValueError` if the specified
        device was not found on the system.

        If `device_name is None`, return the name of the first available NON-monitoring capture device.

        A monitoring capture device is a capture device that records the audio that is going
        to a playback device. Usually monitoring devices have "Monitor" in the device name.

        See the command-line utility `raven-check-audio-devices` to see the list on your system.
        """
        device_names = pvrecorder.PvRecorder.get_available_devices()
        if device_name is not None:  # User-specified device name -> device index as used by `pvrecorder`
            try:
                device_names.index(device_name)  # we just want to check if it's there
            except IndexError:
                error_msg = f"Recorder.validate_device: No such audio capture device '{device_name}'."
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.info(f"Recorder.validate_device: Using audio capture device '{device_name}'.")
        else:  # Find first NON-monitoring audio capture device
            for device_name in device_names:
                if "monitor of" not in device_name.lower():
                    break
            else:
                error_msg = "Recorder.validate_device: No NON-monitoring audio capture device found on this system. If you want to use a MONITORING device for recording, please select it explicitly."
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.info(f"Recorder.validate_device: Using default audio capture device '{device_name}'.")
        return device_name

    def __init__(self,
                 frame_length: int,
                 device_name: Optional[str],
                 vu_peak_hold: float = 1.0,
                 silence_threshold: Optional[float] = None,
                 autostop_timeout: Optional[float] = 1.0,
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

                             See `linear_to_dBFS` for explanation of dBFS.

        `autostop_timeout`: seconds, or `None` to disable.

                            When specified, automatically stop recording if the input audio level
                            stays under `silence_threshold` for this long (i.e. we then consider
                            that the user has stopped speaking).
        """
        device_name = type(self).validate_device(device_name)  # autodetect if `None`, and sanity check in any case
        device_names = pvrecorder.PvRecorder.get_available_devices()
        assert device_name in device_names  # we only get here if the validation succeeded
        self.device_name = device_name  # for information only
        self.device_index = device_names.index(device_name)  # map device name to device index as used by `pvrecorder`

        self.silence_threshold = silence_threshold  # dBFS
        self.autostop_timeout = autostop_timeout  # seconds

        self.frame_length = frame_length
        self.recorder = pvrecorder.PvRecorder(frame_length=self.frame_length,
                                              device_index=self.device_index)

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
                                    silence_level_raw_dBFS = linear_to_dBFS(np.max(np.abs(self.data)))  # all data so far!
                                    silence_level_dBFS = silence_level_raw_dBFS + 6.0  # leave some margin to be safe, in case the very beginning was unusually silent
                                    silence_level_available = True
                                    logger.info(f"Recorder.start.record_task: Silence level measured from first {silence_measurement_timeout:0.6g}s of recorded audio as {silence_level_raw_dBFS:0.2f}dBFS.")
                        else:  # silence_level_available:  # normal operation
                            # _vu_instant for the current audio frame is updated by `_update_vu`, above
                            if self._vu_instant > silence_level_dBFS:
                                last_signal_timestamp = time_now

                            # Stop recording if the audio input stays silent and the autostop timeout is exceeded
                            time_since_last_signal = (time_now - last_signal_timestamp) / 10**9
                            if time_since_last_signal >= self.autostop_timeout:
                                self.stop(_clear_task=False)  # prevent deadlock; the task is already exiting, no need to wait until it exits.
                                break

                finally:
                    with self._recording_state_lock:
                        self.is_recording = False
            self._task_manager.submit(record_task, env())

    def _update_vu(self, array: np.array) -> None:
        """Update the VU meter data. Called automatically once per audio frame when recording."""
        peak = linear_to_dBFS(np.max(np.abs(array)))  # latest buffer (or whatever we were received)
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


def encode(audio_data: Union[np.array, List[np.array]],
           format: str,
           sample_rate: int,
           stream: bool = False) -> Union[bytes, Generator[bytes, None, None]]:
    """Encode audio data from an in-memory array into an audio file format.

    `audio_data`: Audio data in s16 format (signed 16-bit).

                  One chunk:

                  rank-1 `np.array` of shape `[n]` for mono,
                  rank-2 `np.array` of shape `[n, 2]` for stereo.

                  Several chunks:

                  List of rank-1 arrays, each may have different length.
                  List of rank-2 arrays, each may have different length,
                  as long as each has two channels (last dimension is length 2).

    `format`: Name of output format. Any audio format supported by PyAV, e.g. "mp3".

    `sample_rate`: Sample rate used for interpreting the audio data stored in the array.

    `stream`: If `False` (default), return a `bytes` object containing the encoded audio file.

              If `True`, return a generator, which yields a separate `bytes` object for
              the part of the encoded audio file that corresponds to one input array.
              Finally, it yields one more `bytes` object, resulting from finalization.
    """
    if not isinstance(audio_data, list):
        audio_data = [audio_data]

    # `StreamingAudioWriter` will reshape the input data array, but it needs to be of a compatible size, and something that reshapes sensibly.
    dims = np.shape(audio_data[0])
    if len(dims) == 1:
        channels = 1
    elif len(dims) == 2 and dims[-1] == 1:
        channels = 1
    elif len(dims) == 2 and dims[-1] == 2:
        channels = 2
    else:
        error_msg = f"encode only supports mono (rank-1 [n] or rank-2 [n, 1]) or stereo (rank-2 [n, 2]) arrays; got `audio_data` of shape {dims}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # TODO: Doesn't really need to be streaming here. We could just use PyAV directly, like `decode` does.
    audio_encoder = StreamingAudioWriter(format=format,
                                         sample_rate=sample_rate,
                                         channels=channels)
    if stream:
        def streamer():
            for audio_chunk in audio_data:
                yield audio_encoder.write_chunk(audio_chunk)
            yield audio_encoder.write_chunk(finalize=True)
        return streamer
    else:
        audio_buffer = io.BytesIO()
        for audio_chunk in audio_data:
            audio_buffer.write(audio_encoder.write_chunk(audio_chunk))
        audio_buffer.write(audio_encoder.write_chunk(finalize=True))
        audio_bytes = audio_buffer.getvalue()
        return audio_bytes

def decode(stream: BinaryIO,
           target_sample_format: Optional[str] = None,
           target_sample_rate: Optional[int] = None,
           target_layout: Optional[str] = None) -> Tuple[dict, np.array]:
    """Decode an audio stream into an in-memory array, with optional resampling.

    `stream`: Filelike (e.g. `io.BytesIO`) in any audio file format supported by PyAV
              (e.g. mp3 is fine).

    `target_sample_format`: Desired sample format for resampling.

                            If not specified, the output will be in the same sample format as `stream`.

                            NOTE: **sample** format, NOT file format. Some common sample formats are:

                              - "s16": signed 16-bit,
                              - "fltp": floating point.

                            https://pyav.org/docs/stable/api/audio.html

    `target_sample_rate`: Desired sample rate for resampling.

                          If not specified, the output will be at the same sample rate as `stream`.

                          E.g. some AI models have been trained at a specific sample rate,
                          and will only accept audio input at that sample rate.

    `target_layout`: Desired channel layout, e.g. "mono" or "stereo".

                     If not specified, the output will have the same layout as `stream`.

                     https://pyav.org/docs/stable/api/audio.html#module-av.audio.resampler

    Returns the tuple `(metadata, audio_data)`, where:

        - `metadata` is a dictionary with the following items:
            - `input_sample_format` is the original sample format of the input (e.g. "fltp").
            - `input_sample_rate` is the original sample rate of the input.
            - `input_layout` is the original layout of the input (e.g. "mono" or "stereo").

            These are useful mainly when you don't specify the corresponding `target_*` parameters,
            and want to get this information from the audio file instead.

        - `audio_data` is a rank-1 `np.array` containing PCM audio data with the desired sample format,
          sample rate, and layout.
    """
    logger.info("decode: Decoding received audio file.")

    audio_data = io.BytesIO()
    audio_data.write(stream.read())  # copy input into our own in-memory buffer, just in case
    audio_data.seek(0)

    # https://stackoverflow.com/questions/73198826/trying-to-get-audio-raw-data-and-print-it-using-pyav
    # https://pyav.org/docs/stable/api/audio.html
    # https://pyav.org/docs/stable/api/_globals.html?highlight=open#av.open
    # https://pyav.org/docs/stable/overview/caveats.html#garbage-collection
    data = None
    resampler = None
    input_sample_format = None
    input_sample_rate = None
    input_layout = None
    with av.open(audio_data) as container:
        # According to documentation, `av.time_base` is in fractional seconds:
        #     https://pyav.org/docs/stable/api/time.html
        # The name isn't uppercase; correct name (lowercase) mentioned here:
        #     https://pyav.org/docs/stable/api/container.html?highlight=time_base#av.container.InputContainer.seek
        # print(av.time_base)  # 1000000
        logger.info(f"decode: Detected container type '{container.format.name}', bitrate {si_prefix(container.bit_rate)}bps, duration {container.duration / av.time_base:0.6g}s, size {si_prefix(container.size)}B.")
        for packet in container.demux():
            for frame in packet.decode():
                if isinstance(frame, av.audio.frame.AudioFrame):
                    # # DEBUG
                    # print(frame,
                    #       frame.format,  # The audio sample format.
                    #       frame.sample_rate,  # Sample rate of the audio data, in samples per second.
                    #       frame.layout,  # The audio channel layout.
                    #       frame.layout.channels,  # A tuple of AudioChannel objects.
                    #       frame.samples)  # Number of audio samples (per channel).
                    if input_sample_rate is None:  # first audio frame in the stream?
                        # Detect sample format et al.
                        input_sample_format = frame.format.name
                        input_sample_rate = frame.sample_rate
                        input_layout = frame.layout.name

                        plural_s = "s" if frame.samples != 1 else ""
                        logger.info(f"decode: Detected input sample format '{frame.format.name}', sample rate {frame.sample_rate}, audio frame size {frame.samples} sample{plural_s} per channel, layout '{frame.layout.name}', channels: { {channel.name: channel.description for channel in frame.layout.channels} }.")

                        target_sample_format_str = f"'{target_sample_format}'" if (target_sample_format is not None) else "same as input"
                        target_sample_rate_str = f"{target_sample_rate} Hz" if (target_sample_rate is not None) else "same as input"
                        target_layout_str = f"'{target_layout}'" if (target_layout is not None) else "same as input"
                        logger.info(f"decode: Decoding to output sample format {target_sample_format_str}, sample rate {target_sample_rate_str}, layout {target_layout_str}.")

                    # Initialize optional parameters
                    # TODO: Add support for specifying these as `av.audio.format.AudioFormat` and `av.audio.layout.AudioLayout`, for more fine-grained control. The question is how to detect if they match the input.
                    if target_sample_format is None:
                        target_sample_format = input_sample_format
                    if target_sample_rate is None:  # `None`: keep input sample rate
                        target_sample_rate = input_sample_rate
                    if target_layout is None:
                        target_layout = input_layout

                    # Write to output array, with resampling if needed
                    if (target_sample_format != input_sample_format) or (target_sample_rate != input_sample_rate) or (target_layout != input_layout):
                        if not resampler:  # create resampler at first frame (we need the parameters, which we may autodetect from the stream)
                            resampler = av.audio.resampler.AudioResampler(format=target_sample_format,
                                                                          layout=target_layout,
                                                                          rate=target_sample_rate)
                        resampled_frames = resampler.resample(frame)
                        for resampled_frame in resampled_frames:
                            array = resampled_frame.to_ndarray()[0]
                            if data is not None:
                                data = np.concatenate([data, array])
                            else:
                                data = array
                    else:
                        array = frame.to_ndarray()[0]
                        if data is not None:
                            data = np.concatenate([data, array])
                        else:
                            data = array

    logger.info(f"decode: Decoded {len(data) / target_sample_rate:0.6g}s of audio.")
    metadata = {"input_sample_format": input_sample_format,
                "input_sample_rate": input_sample_rate,
                "input_layout": input_layout}
    return metadata, data
