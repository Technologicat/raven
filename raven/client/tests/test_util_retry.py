"""Unit tests for raven.client.util: a request that cannot reach Raven-server is tried again, as `initialize_api` says.

Against a local port nothing listens on, so no server is needed. `requests.get` and `requests.post` are wrapped
to count the attempts, and still really try to connect.
"""

import socket

import pytest
import requests

from raven.client import api, util
from raven.client.config import Retry, Timeout


@pytest.fixture
def unreachable(monkeypatch):
    """A client initialized against a closed port; yields `(set_retry, attempts)`."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setattr(util, "api_initialized", True)
    monkeypatch.setattr(util.api_config, "raven_server_url", f"http://127.0.0.1:{port}", raising=False)
    monkeypatch.setattr(util.api_config, "network_timeout", Timeout(connect=2.0, read=2.0), raising=False)
    attempts = []
    for name in ("get", "post"):
        real = getattr(requests, name)

        def counted(*args, _real=real, **kwargs):
            attempts.append(1)
            return _real(*args, **kwargs)
        monkeypatch.setattr(requests, name, counted)

    def set_retry(maybe_retry):
        monkeypatch.setattr(util.api_config, "maybe_network_retry", maybe_retry, raising=False)
    return set_retry, attempts


class TestRetry:
    def test_a_request_that_cannot_connect_is_tried_again_and_then_fails(self, unreachable):
        set_retry, attempts = unreachable
        set_retry(Retry(attempts=2, delay=0.0))
        with pytest.raises(requests.exceptions.ConnectionError):
            util.get(f"{util.api_config.raven_server_url}/api/modules", timeout=util.api_config.network_timeout)
        assert len(attempts) == 3

    def test_none_sends_once(self, unreachable):
        set_retry, attempts = unreachable
        set_retry(None)
        with pytest.raises(requests.exceptions.ConnectionError):
            util.post(f"{util.api_config.raven_server_url}/api/modules", timeout=util.api_config.network_timeout)
        assert len(attempts) == 1


class TestProbesAreNotRetried:
    """The probes say whether the server is there; trying again would only make them slower to say it is not."""

    def test_the_availability_probe(self, unreachable):
        set_retry, attempts = unreachable
        set_retry(Retry(attempts=2, delay=0.0))
        assert api.raven_server_available() is False
        assert len(attempts) == 1

    def test_the_avatar_heartbeat(self, unreachable):
        set_retry, attempts = unreachable
        set_retry(Retry(attempts=2, delay=0.0))
        with pytest.raises(requests.exceptions.ConnectionError):
            api.avatar_heartbeat(["no-such-instance"])
        assert len(attempts) == 1
