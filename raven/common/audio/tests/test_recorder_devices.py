"""Tests for how `raven.common.audio.recorder` picks a capture device.

Constructing a `Recorder` opens a device, so these stub the enumeration and the construction, and
assert only the choosing: which name is accepted, which is refused, and what an app falls back to
when the one it asked for is not plugged in. That last is the part worth pinning — the failure it
prevents is an app that will not start because a microphone was unplugged since its configuration
was written.
"""

import sys
import types

import pytest

from raven.common.audio import recorder as audio_recorder

DEVICES = ["Built-in Microphone", "Webcam Microphone", "Monitor of Built-in Audio"]


@pytest.fixture
def devices(monkeypatch):
    """Answer a fixed device list, and forget any cached real one."""
    monkeypatch.setattr(audio_recorder, "_available_devices", None)
    monkeypatch.setattr(audio_recorder, "get_available_devices", lambda refresh=False: list(DEVICES))
    return DEVICES


class TestValidatingADeviceName:
    def test_a_present_device_is_accepted(self, devices):
        assert audio_recorder.validate_capture_device(DEVICES[0]) == DEVICES[0]

    def test_an_absent_device_is_refused(self, devices):
        # `list.index` raises `ValueError` rather than `IndexError`, so the module's own error was
        # unreachable for as long as it caught the wrong one.
        with pytest.raises(ValueError, match="No such audio capture device"):
            audio_recorder.validate_capture_device("Microphone That Is Not There")

    def test_none_picks_the_first_non_monitoring_device(self, devices):
        assert audio_recorder.validate_capture_device(None) == DEVICES[0]

    def test_a_monitoring_device_is_skipped_when_picking_automatically(self, monkeypatch):
        # A monitoring device records what is being *played*, so picking one by default would record
        # the AI's own voice back into the microphone.
        monkeypatch.setattr(audio_recorder, "_available_devices", None)
        monkeypatch.setattr(audio_recorder, "get_available_devices",
                            lambda refresh=False: ["Monitor of Built-in Audio", "Webcam Microphone"])
        assert audio_recorder.validate_capture_device(None) == "Webcam Microphone"

    def test_a_monitoring_device_can_still_be_asked_for_by_name(self, devices):
        assert audio_recorder.validate_capture_device(DEVICES[2]) == DEVICES[2]


@pytest.fixture
def enumerations(monkeypatch):
    """Count how often the OS is actually asked, with an empty cache to start from."""
    calls = []

    class FakePvRecorder:
        @staticmethod
        def get_available_devices():
            calls.append(1)
            return list(DEVICES)

    monkeypatch.setattr(audio_recorder, "_available_devices", None)
    monkeypatch.setitem(sys.modules, "pvrecorder", types.SimpleNamespace(PvRecorder=FakePvRecorder))
    return calls


class TestTheDeviceListIsCached:
    def test_the_second_call_does_not_ask_again(self, enumerations):
        assert audio_recorder.get_available_devices() == DEVICES
        assert audio_recorder.get_available_devices() == DEVICES
        assert len(enumerations) == 1

    def test_refresh_asks_again(self, enumerations):
        # The list changes when a microphone is plugged in, which is why the cache has to be escapable
        # — and why `memoize`, whose contract is a pure function, was the wrong tool for it.
        audio_recorder.get_available_devices()
        audio_recorder.get_available_devices(refresh=True)
        assert len(enumerations) == 2

    def test_the_caller_cannot_edit_the_cache(self, enumerations):
        got = audio_recorder.get_available_devices()
        got.append("Microphone Of The Imagination")
        assert audio_recorder.get_available_devices() == DEVICES
        assert len(enumerations) == 1, "this fixture re-enumerated, so it cannot tell a copy from a shared list"


class TestTheAppEntryPointFallsBack:
    """`initialize` must produce *a* microphone; `Recorder` must produce *the* one it was asked for."""

    @pytest.fixture
    def constructed_with(self, devices, monkeypatch):
        """Capture the `device_name` `initialize` ends up constructing a recorder with."""
        seen = {}

        def fake_recorder(**kwargs):
            seen["device_name"] = kwargs["device_name"]
            return object()

        monkeypatch.setattr(audio_recorder, "instance", None)
        monkeypatch.setattr(audio_recorder, "Recorder", fake_recorder)
        return seen

    def test_a_present_device_is_used(self, constructed_with):
        audio_recorder.initialize(device_name=DEVICES[1])
        assert constructed_with["device_name"] == DEVICES[1]

    def test_an_absent_device_falls_back_to_the_first_available(self, constructed_with):
        audio_recorder.initialize(device_name="Microphone That Was Unplugged")
        assert constructed_with["device_name"] is None, \
            "a configured microphone that is gone should cost the user a different one, not the app"

    def test_none_is_passed_through(self, constructed_with):
        audio_recorder.initialize(device_name=None)
        assert constructed_with["device_name"] is None
