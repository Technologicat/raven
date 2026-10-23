"""Tests for `raven.client.mayberemote`'s choice of mode: which of local and remote a service may use.

Needs no server and loads no model: the base class decides the mode, and asking the server is the one thing
it does that these tests watch.
"""

import pytest

from raven.client import api, mayberemote


@pytest.fixture
def probe(monkeypatch):
    """Stands in for asking whether Raven-server answers, counting the times it was asked."""
    asked = []

    def raven_server_available():
        asked.append(True)
        return True
    monkeypatch.setattr(api, "raven_server_available", raven_server_available)
    monkeypatch.setattr(api, "modules", lambda: ["natlang"])
    return asked


class TestModes:
    def test_without_remote_the_server_is_never_asked(self, probe):
        service = mayberemote.MaybeRemoteService(allow_local=True, allow_remote=False)
        assert probe == [] and service.server_modules == [] and not service.server_available

    def test_by_default_the_server_is_asked(self, probe):
        service = mayberemote.MaybeRemoteService(allow_local=True)
        assert probe == [True] and service.server_modules == ["natlang"], \
            "the control: with remote allowed, the probe above is the one the service uses"

    def test_neither_mode_allowed_is_refused(self, probe):
        with pytest.raises(ValueError, match="cannot both be `False`"):
            mayberemote.MaybeRemoteService(allow_local=False, allow_remote=False)
        assert probe == []
