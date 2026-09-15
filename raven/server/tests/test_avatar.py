"""Unit tests for raven.server.modules.avatar: the "data eyes" effect's state machine.

The `Animator` is built with `__new__` and only the fields the effect reads, so no model is loaded; the module
itself still imports torch, hence the `ml` marker. Time is driven by hand through `time.monotonic_ns`, which
both the animator and the cel compositor read.
"""

import pytest

pytestmark = pytest.mark.ml

avatar = pytest.importorskip("raven.server.modules.avatar", reason="the avatar module needs the ML stack")  # noqa: E402

SECOND = 10**9


@pytest.fixture
def clock(monkeypatch):
    """A settable `time.monotonic_ns`, starting well away from zero."""
    now = [1000 * SECOND]
    monkeypatch.setattr(avatar.time, "monotonic_ns", lambda: now[0])
    return now


@pytest.fixture
def animator(clock):
    animator = avatar.Animator.__new__(avatar.Animator)
    animator._settings = {"data_eyes_fps": 12.0,
                          "data_eyes_min_duration": 1.0,
                          "data_eyes_fadeout_duration": 0.5}
    animator.data_eyes_celnames = ["data1"]
    animator.data_eyes_epoch = clock[0]
    animator.data_eyes_state = "off"
    animator.data_eyes_on_since_ts = clock[0]
    animator.data_eyes_fadeout_start_ts = clock[0]
    return animator


def strength(animator):
    """The data eyes cel's opacity in the next rendered frame, or `None` if the effect drew nothing."""
    celstack = animator.animate_data_eyes([("data1", 0.0)])
    return dict(celstack)["data1"] if animator.data_eyes_state != "off" else None


class TestDataEyesMinimumDuration:
    def test_a_quick_stop_holds_full_strength_until_the_minimum_has_passed(self, animator, clock):
        animator.start_data_eyes()
        clock[0] += int(0.2 * SECOND)
        animator.stop_data_eyes()

        clock[0] += int(0.5 * SECOND)  # 0.7 s on: still inside the minimum
        assert strength(animator) == 1.0, "a stop before the minimum faded the effect at once"

        clock[0] += int(0.55 * SECOND)  # 1.25 s on: a quarter of a second into the fade
        assert 0.0 < strength(animator) < 1.0

        clock[0] += int(0.5 * SECOND)  # past the fade
        assert strength(animator) is None

    def test_a_stop_after_the_minimum_fades_at_once(self, animator, clock):
        animator.start_data_eyes()
        clock[0] += int(3.0 * SECOND)
        animator.stop_data_eyes()
        clock[0] += int(0.25 * SECOND)
        assert 0.0 < strength(animator) < 1.0, \
            "a stop long after the minimum still held, so the test above cannot tell a hold from a delay"

    def test_snapping_back_from_a_fade_does_not_restart_the_minimum(self, animator, clock):
        animator.start_data_eyes()
        clock[0] += int(3.0 * SECOND)
        animator.stop_data_eyes()
        clock[0] += int(0.1 * SECOND)
        animator.start_data_eyes()  # back on mid-fade: the same appearance, continued
        animator.stop_data_eyes()
        clock[0] += int(0.25 * SECOND)
        assert 0.0 < strength(animator) < 1.0, "the minimum was counted again from the snap-back"
