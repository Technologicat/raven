"""Unit tests for raven.client.api's record of each avatar instance's state, and `avatar_restore`. No server needed.

The HTTP layer is replaced by a recorder that answers every call as the server would, or fails it, as a test says.
"""

import pytest
import requests

from raven.client import api


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def server(monkeypatch):
    """A stand-in server: records each call as `(endpoint, data)`, and refuses all of them while `server.down`."""
    calls = []
    state = type("S", (), {"down": False, "next_id": 0})()

    def post(url, json=None, files=None, **kwargs):
        if state.down:
            raise requests.exceptions.ConnectionError("nothing listening")
        endpoint = url.rsplit("/api/avatar/", 1)[-1]
        calls.append((endpoint, json))
        if endpoint == "load":
            state.next_id += 1
            return FakeResponse({"instance_id": f"instance-{state.next_id}"})
        return FakeResponse({})

    monkeypatch.setattr(api.util, "require", lambda: None)
    monkeypatch.setattr(api.util, "post", post)
    monkeypatch.setattr(api.util.api_config, "raven_server_url", "http://fake", raising=False)
    monkeypatch.setattr(api.util.api_config, "raven_default_headers", {}, raising=False)
    monkeypatch.setattr(api.util.api_config, "network_timeout", None, raising=False)
    monkeypatch.setattr(api, "_load_file", lambda filename: b"")
    monkeypatch.setattr(api.avatarutil, "scan_addon_cels", lambda filename: {})
    monkeypatch.setattr(api, "_hold", lambda instance_id: None)  # no heartbeat thread in a test
    monkeypatch.setattr(api, "_avatar_states", {})
    state.calls = calls
    return state


def sent_after_load(calls, new_id):
    """The calls made to `new_id` after it was loaded, as `(endpoint, data without the instance ID)`."""
    out = []
    for endpoint, data in calls:
        if data is not None and data.get("instance_id") == new_id:
            out.append((endpoint, {k: v for k, v in data.items() if k != "instance_id"}))
    return out


class TestRestoringAnAvatar:
    def test_the_new_instance_gets_the_state_last_asked_for(self, server):
        old = api.avatar_load("aria.png")
        api.avatar_load_emotion_templates(old, {"joy": {"mouth": 1.0}})
        api.avatar_load_animator_settings(old, {"upscale": 2.0})
        api.avatar_set_emotion(old, "joy")
        api.avatar_start_data_eyes(old)
        api.avatar_start(old)
        new = api.avatar_restore(old)
        assert new != old
        assert sent_after_load(server.calls, new) == [("load_emotion_templates", {"emotions": {"joy": {"mouth": 1.0}}}),
                                                      ("load_animator_settings", {"animator_settings": {"upscale": 2.0}}),
                                                      ("set_emotion", {"emotion_name": "joy"}),
                                                      ("start_data_eyes", {}),
                                                      ("start", {})]

    def test_what_was_asked_while_the_server_was_down_counts(self, server):
        # Lipsync sets mouth overrides while speaking; the speech ends while the server is down, so the request to clear
        # them never arrives. The restored avatar must come back with its mouth as the client last asked: clear.
        old = api.avatar_load("aria.png")
        api.avatar_start(old)
        api.avatar_set_overrides(old, {"mouth_aaa": 0.8, "mouth_ooo": 0.2})
        server.down = True
        with pytest.raises(requests.exceptions.ConnectionError):
            api.avatar_modify_overrides(old, "unset", {"mouth_aaa": 0.0, "mouth_ooo": 0.0})
        server.down = False
        new = api.avatar_restore(old)
        assert ("set_overrides", {"overrides": {"mouth_aaa": 0.8, "mouth_ooo": 0.2}}) not in sent_after_load(server.calls, new)
        assert sent_after_load(server.calls, new) == [("start", {})]

    def test_overrides_still_set_are_replayed(self, server):
        """Control for the test above: overrides not cleared come back."""
        old = api.avatar_load("aria.png")
        api.avatar_set_overrides(old, {"mouth_aaa": 0.8})
        api.avatar_modify_overrides(old, "set", {"eye_wink": 1.0})
        new = api.avatar_restore(old)
        assert sent_after_load(server.calls, new) == [("set_overrides", {"overrides": {"mouth_aaa": 0.8, "eye_wink": 1.0}})]

    def test_a_paused_avatar_comes_back_paused(self, server):
        old = api.avatar_load("aria.png")
        api.avatar_start(old)
        api.avatar_stop(old)
        new = api.avatar_restore(old)
        assert sent_after_load(server.calls, new) == []

    def test_the_old_record_is_dropped_and_an_unloaded_instance_cannot_be_restored(self, server):
        old = api.avatar_load("aria.png")
        new = api.avatar_restore(old)
        with pytest.raises(KeyError):
            api.avatar_restore(old)
        api.avatar_unload(new)
        with pytest.raises(KeyError):
            api.avatar_restore(new)
