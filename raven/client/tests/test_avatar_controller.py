"""The data-eyes effect's nesting contract.

The effect says "the system is consulting an external source", and more than one thing can be doing that at
once: a turn's tool call runs on the turn's thread while an attachment is read on a background one. The
calls therefore nest — the effect ends when the last user stops it, not the first — and that is a contract
a caller cannot verify locally, which is why it is pinned here.

No server and no GUI: the API calls are replaced, and only the counting is under test.
"""

import threading

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
