"""Unit tests for raven.common.audio.silencegate."""

import math

from raven.common.audio.silencegate import SilenceGate

SECOND = 10**9  # ns, the unit `SilenceGate.update` takes its timestamps in


def feed(gate, frames, t0=0):
    """Feed `(level, dt)` pairs to `gate`, returning the answer to each.

    `dt` is the seconds elapsed since the previous frame, so a test reads as a timeline
    rather than as a column of absolute timestamps.
    """
    answers = []
    t = t0
    for level, dt in frames:
        t += int(dt * SECOND)
        answers.append(gate.update(level, t))
    return answers


def silence(n, dt=0.03, level=-70.0):
    """`n` frames of quiet, `dt` seconds apart."""
    return [(level, dt)] * n


class TestStoppingOnSilence:
    def test_stops_only_after_the_timeout_has_elapsed(self):
        gate = SilenceGate(threshold=-40.0, autostop_timeout=1.0)
        # 0.1 s per frame, so the eleventh frame is the first at or past a second of silence.
        answers = feed(gate, [(-70.0, 0.1)] * 12)
        assert not any(answers[:10]), f"stopped before the timeout: {answers}"
        assert answers[10], "did not stop once a second of silence had elapsed"

    def test_never_stops_when_the_timeout_is_disabled(self):
        gate = SilenceGate(threshold=-40.0, autostop_timeout=None)
        assert not any(feed(gate, silence(200, dt=0.1)))

    def test_a_single_loud_frame_restarts_the_clock(self):
        # The property that decides how the threshold has to be chosen: one frame above it is
        # enough to keep the recording open, so the threshold must clear a room's transients
        # rather than its average level.
        loud_at_half_time = [(-70.0, 0.1)] * 5 + [(-10.0, 0.1)] + [(-70.0, 0.1)] * 6
        gate = SilenceGate(threshold=-40.0, autostop_timeout=1.0)
        assert not any(feed(gate, loud_at_half_time)), "the loud frame did not restart the clock"

        # The control: the same timeline without the loud frame does stop, so the sequence is long
        # enough for the two to disagree.
        all_quiet = [(-70.0, 0.1)] * 12
        assert any(feed(SilenceGate(threshold=-40.0, autostop_timeout=1.0), all_quiet)), \
            "this timeline is too short to stop at all, so it cannot tell a restarted clock from a running one"

    def test_the_comparison_is_strict(self):
        # A frame exactly at the threshold counts as silence.
        gate = SilenceGate(threshold=-40.0, autostop_timeout=0.5)
        assert feed(gate, [(-40.0, 0.1)] * 6)[-1]


class TestChangingTheThresholdWhileRecording:
    def test_a_raised_threshold_takes_effect_on_the_next_frame(self):
        # What the GUI's threshold slider needs: the recording is running, the level is steady, and
        # moving the threshold across it decides whether the recording stops.
        gate = SilenceGate(threshold=-70.0, autostop_timeout=0.5)
        steady = [(-50.0, 0.1)] * 6
        assert not any(feed(gate, steady)), "-50 dBFS is above a -70 dBFS threshold; nothing should have stopped"

        gate.threshold = -40.0  # the same audio is now below the threshold
        assert any(feed(gate, steady, t0=int(0.6 * SECOND))), "the raised threshold did not take effect"

    def test_a_lowered_threshold_keeps_the_recording_open(self):
        gate = SilenceGate(threshold=-40.0, autostop_timeout=0.5)
        rising = [(-50.0, 0.1)] * 3
        assert not any(feed(gate, rising))

        gate.threshold = -70.0
        assert not any(feed(gate, [(-50.0, 0.1)] * 10, t0=int(0.3 * SECOND))), \
            "the lowered threshold did not take effect; the recording stopped on audio that is now signal"

    def test_the_timeline_does_not_stop_of_its_own_accord(self):
        # The negative control for the two above: with the threshold left where it started, this
        # timeline never stops. So the stop they assert is the moved threshold's doing, and a gate
        # that read the threshold once at the first frame fails them rather than passing for the
        # wrong reason. (Checked that way, 2026-08-28: both fail against read-once semantics.)
        gate = SilenceGate(threshold=-70.0, autostop_timeout=0.5)
        assert not any(feed(gate, [(-50.0, 0.1)] * 12))


class TestAutodetectingTheSilenceLevel:
    def test_the_level_is_the_loudest_frame_of_the_window_plus_the_margin(self):
        gate = SilenceGate(threshold=None, autostop_timeout=None, measurement_time=0.1, margin=6.0)
        feed(gate, [(-70.0, 0.03), (-60.0, 0.03), (-70.0, 0.03), (-70.0, 0.03), (-70.0, 0.03)])
        assert gate.effective_threshold == -54.0

    def test_nothing_is_compared_while_the_measurement_runs(self):
        # The window is 10 s here, so every frame below falls inside it. A gate that armed
        # immediately would stop on this timeline.
        gate = SilenceGate(threshold=None, autostop_timeout=0.2, measurement_time=10.0)
        assert not any(feed(gate, silence(20, dt=0.1)))
        assert gate.effective_threshold is None, "the measurement should still be running"

    def test_the_level_is_measured_once_and_then_held(self):
        gate = SilenceGate(threshold=None, autostop_timeout=None, measurement_time=0.1, margin=6.0)
        feed(gate, silence(6, dt=0.03, level=-70.0))
        assert gate.effective_threshold == -64.0
        feed(gate, [(-10.0, 0.03)] * 10)  # loud audio, well after the measurement window
        assert gate.effective_threshold == -64.0, "a later loud frame moved a level that was already measured"

    def test_an_explicit_threshold_set_later_overrides_the_measured_one(self):
        gate = SilenceGate(threshold=None, autostop_timeout=None, measurement_time=0.1, margin=6.0)
        feed(gate, silence(6, dt=0.03, level=-70.0))
        assert gate.effective_threshold == -64.0
        gate.threshold = -30.0
        assert gate.effective_threshold == -30.0

    def test_returning_to_autodetect_returns_to_the_measured_level(self):
        # Rather than measuring again: the measurement is of the start of the recording, and that
        # audio is gone.
        gate = SilenceGate(threshold=None, autostop_timeout=None, measurement_time=0.1, margin=6.0)
        feed(gate, silence(6, dt=0.03, level=-70.0))
        gate.threshold = -30.0
        gate.threshold = None
        assert gate.effective_threshold == -64.0

    def test_digital_silence_measures_as_silence(self):
        # `linear_to_dBFS` answers -inf for an all-zero frame, which is what a muted or
        # not-yet-open input sends.
        gate = SilenceGate(threshold=None, autostop_timeout=0.5, measurement_time=0.1)
        answers = feed(gate, [(-math.inf, 0.1)] * 8)
        assert gate.effective_threshold == -math.inf
        assert answers[-1], "an input sending nothing at all never autostopped"
