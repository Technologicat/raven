"""Two avatar effects whose contracts a caller cannot verify locally.

The **data eyes** say "the system is consulting an external source", and more than one thing can be doing
that at once: a turn's tool call runs on the turn's thread while an attachment is read on a background one.
The calls therefore nest — the effect ends when the last user stops it, not the first.

The **branch-switch glitch** overlays a transient filter on the avatar's own postprocessor chain. What
matters is the bookkeeping: the chain has to come back exactly as it was, and a run of switches has to read
as one glitch rather than a stutter of them.

No server and no GUI: the API calls are replaced, and only the bookkeeping is under test. How the glitch
*looks* is a matter for the eye, and its parameters are tuned by looking rather than asserted here.
"""

import threading
import time

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from unpythonic.env import env  # noqa: E402 -- after importorskip by design

from raven.client import avatar_controller  # noqa: E402 -- after importorskip by design


@pytest.fixture
def controller_and_config(monkeypatch):
    """A controller with the API stubbed out, and one registered avatar instance's worth of state.

    Built without `__init__`, which would want a GUI, a voice and a live server. Only the two methods
    under test are exercised, and neither reads anything the constructor sets.
    """
    calls = []
    monkeypatch.setattr(avatar_controller.api, "avatar_start_data_eyes", lambda instance_id: calls.append("start"))
    monkeypatch.setattr(avatar_controller.api, "avatar_stop_data_eyes", lambda instance_id: calls.append("stop"))

    controller = avatar_controller.DPGAvatarController.__new__(avatar_controller.DPGAvatarController)
    monkeypatch.setattr(controller, "ping", lambda config: None, raising=False)

    config = env(avatar_instance_id="test-instance",
                 _data_eyes_lock=threading.RLock(),
                 _data_eyes_users=0)
    return controller, config, calls


def test_one_user_switches_it_on_and_off(controller_and_config):
    controller, config, calls = controller_and_config
    controller.start_data_eyes(config)
    controller.stop_data_eyes(config)
    assert calls == ["start", "stop"]


def test_a_second_user_does_not_re_send_the_start(controller_and_config):
    """The effect is already on; telling the server again would restart its animation."""
    controller, config, calls = controller_and_config
    controller.start_data_eyes(config)
    controller.start_data_eyes(config)
    assert calls == ["start"]


def test_the_first_stop_does_not_end_it_for_the_second_user(controller_and_config):
    """The behaviour the counter exists for.

    Without it, a turn's tool call finishing would switch the eyes off while a background task was still
    reading a document — and the reader would see the effect flicker out mid-consultation.
    """
    controller, config, calls = controller_and_config
    controller.start_data_eyes(config)
    controller.start_data_eyes(config)
    controller.stop_data_eyes(config)
    assert calls == ["start"], "the effect should still be on for the remaining user"
    controller.stop_data_eyes(config)
    assert calls == ["start", "stop"]


def test_an_unmatched_stop_is_harmless(controller_and_config):
    """Teardown paths call stop defensively, and a count driven negative would swallow the next real start."""
    controller, config, calls = controller_and_config
    controller.stop_data_eyes(config)
    controller.stop_data_eyes(config)
    assert config._data_eyes_users == 0

    controller.start_data_eyes(config)
    assert "start" in calls, "a start after unmatched stops must still switch the effect on"


def test_the_count_survives_interleaving_from_two_threads(controller_and_config):
    """The two callers really are on different threads — a turn's, and a background task's."""
    controller, config, calls = controller_and_config

    def use_it():
        for _ in range(100):
            controller.start_data_eyes(config)
            controller.stop_data_eyes(config)

    threads = [threading.Thread(target=use_it) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert config._data_eyes_users == 0, "every start was matched, so nothing should still be holding it"


# --------------------------------------------------------------------------------
# The branch-switch glitch
#
# A transient filter overlaid on the avatar's own postprocessor chain. What is worth pinning is the
# bookkeeping around it: the chain has to come back, and a run of switches has to read as one glitch.

@pytest.fixture
def glitch_config(monkeypatch):
    """A controller and an instance with animator settings loaded, with the API recording what it is sent."""
    sent = []
    monkeypatch.setattr(avatar_controller.api, "avatar_load_animator_settings",
                        lambda instance_id, settings: sent.append(settings))

    controller = avatar_controller.DPGAvatarController.__new__(avatar_controller.DPGAvatarController)
    monkeypatch.setattr(controller, "ping", lambda config: None, raising=False)

    config = env(avatar_instance_id="test-instance",
                 _animator_settings_lock=threading.RLock(),
                 _animator_settings=None,
                 _glitch_timer=None,
                 _glitch_started_at=None)
    controller.load_animator_settings(config, {"postprocessor_chain": [["bloom", {"threshold": 0.5}]]})
    sent.clear()
    return controller, config, sent


def chain_of(settings):
    return [name for name, _parameters in settings["postprocessor_chain"]]


def test_the_glitch_is_added_on_top_of_the_existing_chain(glitch_config):
    """The chain is the user's; the glitch is a guest on it, and must not displace what is there."""
    controller, config, sent = glitch_config
    controller.glitch(config, floor=10.0)  # long, so it does not restore mid-test
    try:
        assert len(sent) == 1
        assert chain_of(sent[0]) == ["bloom", "digital_glitches"]
    finally:
        config._glitch_timer.cancel()


def test_the_chain_comes_back_when_the_glitch_ends(glitch_config):
    controller, config, sent = glitch_config
    controller.glitch(config, floor=0.01)
    time.sleep(0.2)
    assert chain_of(sent[-1]) == ["bloom"], f"the chain was left as {chain_of(sent[-1])}"
    assert config._glitch_timer is None


def test_the_users_own_settings_are_not_mutated(glitch_config):
    """The baseline is what every restore is built from, so a glitch that edited it in place would leave
    the avatar permanently glitching — and the second switch would stack another filter on the first."""
    controller, config, sent = glitch_config
    controller.glitch(config, floor=10.0)
    try:
        assert chain_of(config._animator_settings) == ["bloom"]
    finally:
        config._glitch_timer.cancel()


def test_a_second_switch_extends_rather_than_restarting(glitch_config):
    """Flicking through siblings should read as one continuous glitch, not a stutter of them — so a repeat
    call must not re-send the chain, which would restart the filter's own animation."""
    controller, config, sent = glitch_config
    controller.glitch(config, floor=10.0)
    controller.glitch(config, floor=10.0)
    try:
        assert len(sent) == 1, f"the chain was sent {len(sent)} times; a repeat should only move the deadline"
    finally:
        config._glitch_timer.cancel()


def test_the_ceiling_caps_a_held_key(glitch_config):
    """Holding a navigation key would otherwise extend the glitch forever, and a glitch that never stops
    reads as a broken avatar rather than as a transition."""
    controller, config, sent = glitch_config
    controller.glitch(config, floor=10.0, ceiling=0.05)
    time.sleep(0.02)
    controller.glitch(config, floor=10.0, ceiling=0.05)  # would push it 10 s out, but the ceiling holds
    time.sleep(0.3)
    assert chain_of(sent[-1]) == ["bloom"], "the ceiling did not end the glitch"


def test_no_settings_loaded_means_no_glitch(glitch_config):
    """A glitch before startup finished has no chain to overlay and nothing to restore, so it declines."""
    controller, config, sent = glitch_config
    config._animator_settings = None
    controller.glitch(config)
    assert sent == []
    assert config._glitch_timer is None
