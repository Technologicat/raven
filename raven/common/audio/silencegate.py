"""Deciding when a recording has stayed quiet long enough to stop itself.

Separate from `raven.common.audio.recorder` because the decision needs no audio device:
the recorder feeds it signal levels, and anything else can too — a test, a probe, or a GUI
tuning the threshold against recorded levels.
"""

__all__ = ["SilenceGate",

           "DEFAULT_SILENCE_THRESHOLD",
           "DEFAULT_AUTOSTOP_TIMEOUT",
           "DEFAULT_SILENCE_MEASUREMENT_TIME",
           "DEFAULT_SILENCE_MARGIN"]

import logging
logger = logging.getLogger(__name__)

import math

DEFAULT_SILENCE_THRESHOLD = -40.0  # dBFS
DEFAULT_AUTOSTOP_TIMEOUT = 1.5  # seconds
DEFAULT_SILENCE_MEASUREMENT_TIME = 0.1  # seconds of audio to measure the silence level from, when autodetecting
DEFAULT_SILENCE_MARGIN = 6.0  # dB above the measured background noise to place an autodetected silence level at

class SilenceGate:
    """Decide, frame by frame, when a recording has stayed quiet long enough to stop itself.

    Feed it one audio frame's signal level at a time with `update`, which answers whether the
    recorder should now stop. `raven.common.audio.recorder.Recorder` does this automatically.

    `threshold`: dBFS below which a frame counts as silence, or `None` to autodetect from the
                 start of the recording. Read at every `update`, so it can be changed while a
                 recording is running.

                 See `raven.common.audio.utils.linear_to_dBFS` for what dBFS means here.

    `autostop_timeout`: seconds of continuous silence after which `update` answers `True`, or
                        `None` to never stop. Also read at every `update`.

    `measurement_time`: seconds of audio the autodetection measures the background noise over.
                        Ignored when `threshold` is given.

    `margin`: dB placed between the measured background noise and the autodetected silence
              level. Ignored when `threshold` is given.

    `name`: identifies this gate in log messages.

    A gate holds the state of one recording; make a new one per recording.
    """

    def __init__(self,
                 threshold: float | None = DEFAULT_SILENCE_THRESHOLD,
                 autostop_timeout: float | None = DEFAULT_AUTOSTOP_TIMEOUT,
                 measurement_time: float = DEFAULT_SILENCE_MEASUREMENT_TIME,
                 margin: float = DEFAULT_SILENCE_MARGIN,
                 name: str = "silence gate"):
        self.threshold = threshold
        self.autostop_timeout = autostop_timeout
        self.measurement_time = measurement_time
        self.margin = margin
        self.name = name

        self._detected = None  # autodetected silence level, dBFS; `None` until the measurement completes
        self._measurement_max = -math.inf  # loudest frame seen so far during the measurement
        self._t0_ns = None  # first `update`, i.e. the start of the recording
        self._last_signal_ns = None  # most recent frame above the threshold

    def update(self,
               level: float,
               now_ns: int) -> bool:
        """Feed one audio frame's signal level, and answer whether to stop recording now.

        `level`: dBFS. The peak level of this audio frame — what a VU meter shows as its instant
                 reading.

        `now_ns`: when this frame was captured, from `time.monotonic_ns`.
        """
        if self._t0_ns is None:
            self._t0_ns = self._last_signal_ns = now_ns

        threshold = self.threshold
        if threshold is None:
            if self._detected is None:
                # A frame's level is the loudest sample in it, so the running maximum over frames is
                # the loudest sample in the measurement window — the same quantity a single pass over
                # the raw audio would give, computed as the audio arrives.
                self._measurement_max = max(self._measurement_max, level)
                if (now_ns - self._t0_ns) / 10**9 < self.measurement_time:
                    return False  # still measuring; nothing to compare against yet
                self._detected = self._measurement_max + self.margin
                logger.info(f"SilenceGate.update: {self.name}: silence level measured from first {self.measurement_time:0.6g}s of recorded audio as {self._measurement_max:0.2f}dBFS, gate set to {self._detected:0.2f}dBFS.")
            threshold = self._detected

        # A single frame above the threshold restarts the clock, so the threshold has to clear the
        # whole distribution of background-noise frames rather than its centre: in a room where one
        # transient per timeout window gets through, the recording never stops by itself.
        if level > threshold:
            self._last_signal_ns = now_ns
            return False
        if self.autostop_timeout is None:
            return False
        return (now_ns - self._last_signal_ns) / 10**9 >= self.autostop_timeout

    def _get_effective_threshold(self) -> float | None:
        return self.threshold if self.threshold is not None else self._detected
    effective_threshold = property(fget=_get_effective_threshold,
                                   doc="The dBFS level frames are actually being compared against, or `None` while an autodetection is still measuring. Read-only.")
