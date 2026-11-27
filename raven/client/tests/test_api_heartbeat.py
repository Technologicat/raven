"""Unit tests for raven.client.api's avatar heartbeat: what one beat tells its listeners. No server needed."""

import pytest
import requests

from raven.client import api


class TestHeartbeatListeners:
    """What one beat tells the listeners, without a server or the beat's ten-second loop."""

    @pytest.fixture
    def beat(self, monkeypatch):
        monkeypatch.setattr(api, "_held_instances", {"a", "b"})
        monkeypatch.setattr(api, "_heartbeat_listeners", [])
        heard = []
        api.avatar_add_heartbeat_listener(lambda known, lost: heard.append((known, lost)))
        return heard

    def test_a_beat_that_reaches_the_server_reports_known_and_lost(self, beat, monkeypatch):
        monkeypatch.setattr(api, "avatar_heartbeat", lambda instance_ids: ["b"])
        assert api._heartbeat_once(failing=True) is False
        assert beat == [(["a"], ["b"])]
        assert api._held_instances == {"a"}, "a lost instance is no longer reported"

    def test_a_beat_that_cannot_reach_the_server_tells_no_one(self, beat, monkeypatch):
        """Control for the test above: no answer, so nothing to say about any instance."""
        def refuse(instance_ids):
            raise requests.exceptions.ConnectionError("nothing listening")
        monkeypatch.setattr(api, "avatar_heartbeat", refuse)
        assert api._heartbeat_once(failing=False) is True
        assert beat == []

    def test_a_listener_that_raises_does_not_stop_the_others(self, beat, monkeypatch):
        monkeypatch.setattr(api, "avatar_heartbeat", lambda instance_ids: [])
        api._heartbeat_listeners.insert(0, lambda known, lost: 1 / 0)
        api._heartbeat_once(failing=False)
        assert beat == [(["a", "b"], [])]
