"""Tests for `replserver`: that it stays off unless asked, binds localhost only, and never takes the app down.

No port is opened here. `unpythonic.net.server.start` is replaced by a recorder, because what these pin is
what `maybe_start` *asks for* — and asking for the wrong bind address is the failure that matters, which a
test that actually listened would be no better at catching.
"""

import argparse

import pytest

from unpythonic.net import server as repl_server

from raven.common import replserver


class Recorder:
    """Stands in for `unpythonic.net.server.start`, remembering how it was called."""

    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_with is not None:
            raise self.fail_with
        return (kwargs["bind"], kwargs["repl_port"], kwargs["control_port"])


@pytest.fixture
def recording_start(monkeypatch):
    """Install a `Recorder` in place of the real server, and hand it back."""
    def install(fail_with=None):
        recorder = Recorder(fail_with=fail_with)
        monkeypatch.setattr(repl_server, "start", recorder)
        return recorder
    return install


def test_no_port_means_no_server(recording_start):
    """The flag's absence is the off switch, and off must mean nothing was started."""
    recorder = recording_start()
    assert replserver.maybe_start(None, {}, "test app") is False
    assert recorder.calls == []

    # The control: without this, a `maybe_start` that never started anything under any circumstances would
    # pass the assertions above, and this fixture could not tell that from the behaviour it means to pin.
    assert replserver.maybe_start(1337, {}, "test app") is True
    assert len(recorder.calls) == 1, "a port given must actually start a server"


def test_it_binds_localhost_only(recording_start):
    """The one security property this module has: the port is not offered to the network."""
    recorder = recording_start()
    replserver.maybe_start(1337, {}, "test app")
    assert recorder.calls[0]["bind"] == "127.0.0.1", (
        "a REPL port is unauthenticated arbitrary code execution; it must never be bound to an address "
        "other than loopback, and there is deliberately no option to ask for one")


class TestTheControlChannel:
    """Two ports are taken, and the default pair is chosen rather than derived."""

    def test_the_default_port_keeps_its_own_partner(self, recording_start):
        """`unpythonic`'s client defaults to this pair, so deriving one of them breaks `client localhost`.

        The whole point of the default being the default is that neither end has to be told anything.
        """
        # The control: were the pair merely consecutive, this test would hold under the derivation it
        # exists to reject, and could not tell the two rules apart.
        assert replserver.DEFAULT_CONTROL_PORT != replserver.DEFAULT_PORT + 1, (
            "the default pair is consecutive, so this test no longer discriminates")

        recorder = recording_start()
        replserver.maybe_start(replserver.DEFAULT_PORT, {}, "test app")
        assert (recorder.calls[0]["repl_port"],
                recorder.calls[0]["control_port"]) == (replserver.DEFAULT_PORT,
                                                       replserver.DEFAULT_CONTROL_PORT)

    def test_any_other_port_takes_the_one_above_it(self, recording_start):
        """There is nothing else to derive it from, so a named port claims its neighbour too."""
        recorder = recording_start()
        replserver.maybe_start(9100, {}, "test app")
        assert (recorder.calls[0]["repl_port"], recorder.calls[0]["control_port"]) == (9100, 9101)


def test_the_session_sees_what_it_was_handed(recording_start):
    """An app hands over its namespace; a REPL that could not reach it would be an empty room."""
    recorder = recording_start()
    namespace = {"marker": object()}
    replserver.maybe_start(1337, namespace, "test app")
    assert recorder.calls[0]["locals"] is namespace


@pytest.mark.parametrize("failure", [OSError("address already in use"),
                                     RuntimeError("The current process already has a running REPL server.")],
                         ids=["port taken by another process", "server already running in this process"])
def test_a_server_that_will_not_start_does_not_take_the_app_down(recording_start, failure, caplog):
    """A debugging aid that can kill the app it is meant to debug is worse than no debugging aid."""
    recording_start(fail_with=failure)
    assert replserver.maybe_start(1337, {}, "test app") is False
    assert any(record.levelname == "ERROR" for record in caplog.records), (
        "declining silently would leave the user waiting to connect to a port nothing is listening on")


class TestTheFlag:
    """What `--repl` parses to, since `maybe_start` reads it straight from argparse."""

    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        replserver.add_argument(parser)
        return parser.parse_args(argv)

    def test_absent_is_none(self):
        assert self._parse([]).repl is None

    def test_bare_flag_is_the_default_port(self):
        assert self._parse(["--repl"]).repl == replserver.DEFAULT_PORT

    def test_a_port_may_be_given(self):
        # The control on the test above: a card that always answered `DEFAULT_PORT` would pass it, and
        # only a value that differs shows the port is read rather than assumed.
        assert self._parse(["--repl", "9999"]).repl == 9999
        assert replserver.DEFAULT_PORT != 9999
