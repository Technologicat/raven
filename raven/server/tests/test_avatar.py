"""Unit tests for raven.server.modules.avatar: the "data eyes" effect's state machine, and anime effects played on
their own rather than through an emotion.

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


@pytest.fixture
def fx_animator(clock):
    animator = avatar.Animator.__new__(avatar.Animator)
    animator._settings = {"animefx": [["notice", {"enabled": True,
                                                  "emotions": ["surprise"],
                                                  "type": "sequence",
                                                  "duration": 0.5,
                                                  "cels": ["fx_notice1", "fx_notice2"]}]]}
    animator.emotion = "neutral"
    animator.last_emotion_change_timestamp = clock[0] - 60 * SECOND  # long settled
    animator.animefx_epochs = {}
    animator.animefx_trigger_timestamps = {}
    return animator


def notice_showing(animator):
    """Whether either notice cel is drawn in the next frame."""
    celstack = animator.animate_animefx([("fx_notice1", 0.0), ("fx_notice2", 0.0)])
    return any(value > 0.0 for _, value in celstack)


class TestTriggerAnimefx:
    def test_a_triggered_effect_plays_whatever_the_emotion_and_then_ends(self, fx_animator, clock):
        assert not notice_showing(fx_animator), "the effect shows untriggered, so this cannot tell a trigger apart"
        fx_animator.trigger_animefx("notice")
        clock[0] += int(0.1 * SECOND)
        assert notice_showing(fx_animator)
        clock[0] += int(0.5 * SECOND)
        assert not notice_showing(fx_animator), "the effect outlived its duration"

    def test_entering_a_trigger_emotion_still_plays_it(self, fx_animator, clock):
        fx_animator.emotion = "surprise"
        fx_animator.last_emotion_change_timestamp = clock[0]
        clock[0] += int(0.1 * SECOND)
        assert notice_showing(fx_animator)

    def test_an_unknown_effect_is_refused(self, fx_animator):
        with pytest.raises(ValueError):
            fx_animator.trigger_animefx("no such effect")


class _FakePart:
    """Stands in for an `Animator` or `Encoder`: `unload` only calls `exit` on them."""
    def __init__(self):
        self.exited = False

    def exit(self):
        self.exited = True


@pytest.fixture
def instances(monkeypatch, clock):
    """Two fake instances, loaded at the clock's start, in a module that believes it is initialized."""
    monkeypatch.setattr(avatar, "module_initialized", True)
    table = {iid: {"animator": _FakePart(), "encoder": _FakePart()} for iid in ("kept", "abandoned")}
    monkeypatch.setattr(avatar, "_avatar_instances", table)
    monkeypatch.setattr(avatar, "_last_seen_ns", {iid: clock[0] for iid in table})
    return table


class TestHeartbeat:
    def test_an_instance_kept_alive_survives_and_an_abandoned_one_is_unloaded(self, instances, clock):
        abandoned = instances["abandoned"]
        for _ in range(4):  # 4 × 50 s, well past a 120 s timeout, with "kept" checking in throughout
            clock[0] += 50 * SECOND
            assert avatar.heartbeat(["kept"]) == []
            avatar.reap_stale_instances(timeout=120.0)
        assert "abandoned" not in avatar._avatar_instances, "a client that never checked in kept its instance"
        assert abandoned["animator"].exited and abandoned["encoder"].exited
        assert "kept" in avatar._avatar_instances, "an instance whose client checked in every 50 s was unloaded"

    def test_nothing_is_reaped_before_the_timeout(self, instances, clock):
        clock[0] += 119 * SECOND
        assert avatar.reap_stale_instances(timeout=120.0) == []
        clock[0] += 2 * SECOND
        assert sorted(avatar.reap_stale_instances(timeout=120.0)) == ["abandoned", "kept"], \
            "past the timeout nothing was reaped, so this fixture cannot tell a reap from no reaper"

    def test_unknown_ids_are_reported_and_not_recorded(self, instances):
        assert avatar.heartbeat(["kept", "from-before-a-restart"]) == ["from-before-a-restart"]
        assert "from-before-a-restart" not in avatar._last_seen_ns

    def test_an_unload_forgets_the_instance_so_the_reaper_does_not_try_again(self, instances, clock):
        avatar.unload("kept")
        assert "kept" not in avatar._last_seen_ns
        clock[0] += 200 * SECOND
        assert avatar.reap_stale_instances(timeout=120.0) == ["abandoned"]

    def test_a_stalled_server_forgives_the_heartbeats_it_missed(self, instances, clock):
        clock[0] += 200 * SECOND  # the server was away for 200 s, and so heard from nobody
        avatar._reaper_step(stalled_for=190.0, timeout=120.0)
        assert sorted(avatar._avatar_instances) == ["abandoned", "kept"], "a stalled server reaped clients that never left"
        avatar._reaper_step(stalled_for=0.0, timeout=120.0)
        assert sorted(avatar._avatar_instances) == ["abandoned", "kept"], "the stall was not forgiven, only postponed"
        clock[0] += 121 * SECOND  # and silence after the server is back is counted again
        avatar._reaper_step(stalled_for=0.0, timeout=120.0)
        assert avatar._avatar_instances == {}, "after a stall, the reaper never reaped again"
